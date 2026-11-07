"""Empirical-Bayes machine adaptation and paired CPU/GPU allocation experiments.

Measured phase demands train the source model. Hardware transfer and contention
are explicit assumptions of the demonstration, not measurements inferred from it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm, t

PHASES = ('cpu_pre_runtime_s', 'gpu_runtime_s',
          'cpu_post_locked_runtime_s', 'cpu_post_unlocked_runtime_s')
CPU = np.array([0, 2, 3])


@dataclass(frozen=True, order=True)
class Action:
    workers: int
    inflight: int

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1
               for v in (self.workers, self.inflight)):
            raise ValueError('workers and inflight must be positive integers')

    @property
    def label(self):
        return f'{self.workers} CPU / {self.inflight} WIP'


ACTIONS = tuple(Action(c, q) for c in (1, 2, 4, 8) for q in (c, 2*c))


def validate_events(frame):
    required = {'condition_id', 'run_id', 'replication', 'event_id', *PHASES}
    if missing := required.difference(frame.columns):
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError('Identifiers must be present and durations must be positive finite values')
    if frame.duplicated(['run_id', 'event_id']).any():
        raise ValueError('Found duplicate run/event identities')
    values = frame[list(PHASES)].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Phase durations must be positive finite values')
    if frame.groupby('run_id').replication.nunique().gt(1).any():
        raise ValueError('A run cannot occur in more than one replication')
    if 'is_warmup' in frame and frame.is_warmup.astype(str).str.lower().isin(['true', '1']).any():
        raise ValueError('Remove explicitly marked warm-up events before fitting')


def load_events(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'docs/study/events.csv.gz'
    frame = pd.read_csv(path)
    validate_events(frame)
    return frame


def _regular_cov(values, floor):
    covariance = np.cov(values, rowvar=False, ddof=1)
    return .8 * covariance + .2 * np.diag(np.diag(covariance)) + np.eye(4) * floor**2


@dataclass
class SourceModel:
    conditions: tuple[str, ...]
    location: np.ndarray
    baseline_cov: np.ndarray
    run_cov: np.ndarray
    noise: np.ndarray
    training_runs: tuple[str, ...]

    def indices(self, conditions):
        lookup = {k: i for i, k in enumerate(self.conditions)}
        unknown = sorted(set(conditions).difference(lookup))
        if unknown:
            raise ValueError(f'Unknown source conditions need a pilot/calibration: {unknown}')
        return np.array([lookup[k] for k in conditions], dtype=int)


def fit_source(frame):
    """Fit only replications 0/1; run means determine baseline uncertainty.

    Covariance estimates are regularized plug-ins (empirical Bayes). Their own
    uncertainty is not integrated. Two runs per condition cannot validate tails.
    """
    validate_events(frame)
    train = frame.loc[frame.replication.isin([0, 1])]
    conditions = tuple(sorted(train.condition_id.unique()))
    if not conditions:
        raise ValueError('Replications 0 and 1 are required for training')
    locations, baselines, runs, noises = [], [], [], []
    for condition in conditions:
        subset = train.loc[train.condition_id.eq(condition)]
        if set(subset.replication) != {0, 1}:
            raise ValueError(f'{condition}: both training replications are required')
        blocks = [np.log(group[list(PHASES)].to_numpy(float))
                  for _, group in subset.groupby('run_id', sort=True)]
        if any(len(block) < 2 for block in blocks):
            raise ValueError('Each training run needs at least two events')
        means = np.array([block.mean(axis=0) for block in blocks])
        residuals = np.concatenate([block-block.mean(axis=0) for block in blocks])
        # A floor of .03 log-seconds prevents a rank-one two-run covariance from
        # pretending that the other directions are known exactly.
        run_cov = _regular_cov(means, .03)
        precision = np.linalg.inv(run_cov) * len(means)
        covariance = np.linalg.inv(np.eye(4)/9 + precision)
        location = covariance @ (np.full(4, np.log(.01)/9) + precision @ means.mean(axis=0))
        locations.append(location)
        baselines.append(covariance)
        # Treat regularized run-mean covariance as a conservative proxy for a
        # new run's persistent shift. It includes finite run-mean sampling noise.
        runs.append(run_cov)
        noises.append(_regular_cov(residuals, .02))
    return SourceModel(conditions, np.array(locations), np.array(baselines),
                       np.array(runs), np.array(noises), tuple(sorted(train.run_id.unique())))


def source_diagnostics(model, frame, replication):
    """Frozen, before-observation predictions on an entire held-out replication."""
    rows = []
    heldout = frame.loc[frame.replication.eq(replication)]
    if heldout.empty:
        raise ValueError(f'Missing held-out replication {replication}')
    for condition, group in heldout.groupby('condition_id', sort=True):
        k = model.indices([condition])[0]
        y = group[list(PHASES)].to_numpy(float)
        loc = model.location[k]
        variance = np.diag(model.noise[k] + model.baseline_cov[k] + model.run_cov[k])
        lo, hi = np.exp(loc - norm.ppf(.95)*np.sqrt(variance)), np.exp(loc + norm.ppf(.95)*np.sqrt(variance))
        for p, phase in enumerate(PHASES):
            rows.append(dict(condition=condition, phase=phase, replication=replication,
                             events=len(y), median_ms=1000*np.exp(loc[p]),
                             mae_ms=1000*np.abs(y[:, p]-np.exp(loc[p])).mean(),
                             log_rmse=np.sqrt(np.mean((np.log(y[:, p])-loc[p])**2)),
                             coverage90=np.mean((y[:, p] >= lo[p]) & (y[:, p] <= hi[p])),
                             interval_width_ms=1000*(hi[p]-lo[p])))
    return pd.DataFrame(rows)


class MachineBelief:
    """Gaussian conditionals for log CPU/GPU speed, mixed over contention.

    State = [log CPU multiplier, log GPU multiplier, four baseline offsets per
    condition]. Baseline uncertainty and a new run's shift are persistent,
    rather than new independent noise for each event.
    """
    def __init__(self, source, prior_sd=.7, alphas=(0., .05, .15, .3, .6)):
        if not np.isfinite(prior_sd) or prior_sd <= 0:
            raise ValueError('prior_sd must be positive finite')
        self.source, self.prior_sd = source, prior_sd
        self.alphas = np.asarray(alphas, dtype=float)
        if self.alphas.ndim != 1 or not len(self.alphas) or not np.isfinite(self.alphas).all() or (self.alphas < 0).any():
            raise ValueError('alphas must be a nonempty nonnegative finite grid')
        d = 2 + 4*len(source.conditions)
        v = np.zeros((d, d))
        v[:2, :2] = np.eye(2) * prior_sd**2
        for k, cov in enumerate(source.baseline_cov + source.run_cov):
            v[2+4*k:6+4*k, 2+4*k:6+4*k] = cov
        self.means = np.zeros((len(self.alphas), d))
        self.covariances = np.repeat(v[None], len(self.alphas), axis=0)
        self.weights = np.full(len(self.alphas), 1/len(self.alphas))
        self.observed_batches = 0

    def design(self, k):
        h = np.zeros((4, self.means.shape[1]))
        h[CPU, 0] = 1
        h[1, 1] = 1
        h[:, 2+4*k:6+4*k] = np.eye(4)
        return h

    def update(self, conditions, durations, action):
        """Consume completed service times, never waiting times or future rows.

        Each condition's log mean is one observation. Effective sample size is
        capped at four per batch to temper the iid likelihood for contiguous data.
        This is a stated approximation, not an estimated autocorrelation model.
        """
        indices = self.source.indices(conditions)
        durations = np.asarray(durations, float)
        if durations.shape != (len(indices), 4) or not np.isfinite(durations).all() or (durations <= 0).any():
            raise ValueError('Completed durations must be positive finite n x 4 values')
        logweights = np.log(np.maximum(self.weights, np.finfo(float).tiny))
        for k in np.unique(indices):
            select = indices == k
            observation = np.log(durations[select]).mean(axis=0) - self.source.location[k]
            noise = self.source.noise[k] / min(int(select.sum()), 4)
            h = self.design(k)
            for j, alpha in enumerate(self.alphas):
                offset = np.zeros(4)
                offset[CPU] = np.log1p(alpha*(action.workers-1))
                m, v = self.means[j], self.covariances[j]
                error = observation - offset - h @ m
                predictive = h @ v @ h.T + noise
                _, logdet = np.linalg.slogdet(predictive)
                logweights[j] -= .5*(4*np.log(2*np.pi) + logdet + error @ np.linalg.solve(predictive, error))
                gain = np.linalg.solve(predictive, h @ v).T
                self.means[j] = m + gain @ error
                # Joseph form protects positive definiteness across repeated updates.
                remainder = np.eye(len(m)) - gain @ h
                updated = remainder @ v @ remainder.T + gain @ noise @ gain.T
                self.covariances[j] = (updated + updated.T)/2
        self.weights = np.exp(logweights-logsumexp(logweights))
        self.observed_batches += 1

    def draw(self, conditions, draws, rng):
        indices = self.source.indices(conditions)
        components = rng.choice(len(self.alphas), size=draws, p=self.weights)
        theta = np.empty((draws, self.means.shape[1]))
        for j in np.unique(components):
            selected = components == j
            theta[selected] = rng.multivariate_normal(self.means[j], self.covariances[j], size=selected.sum())
        log_base = np.repeat(self.source.location[indices][None], draws, axis=0)
        # One parameter vector is shared by the entire rollout. Phase noise is
        # correlated within each event and independent between modeled events.
        for k in np.unique(indices):
            selected = indices == k
            log_base[:, selected] += theta[:, None, 2+4*k:6+4*k]
            log_base[:, selected] += rng.multivariate_normal(np.zeros(4), self.source.noise[k], size=(draws, selected.sum()))
        return log_base, theta[:, :2], self.alphas[components]

    def mean_demands(self, conditions, action):
        indices = self.source.indices(conditions)
        result = np.zeros((len(indices), 4))
        for k in np.unique(indices):
            h = self.design(k)
            for j, alpha in enumerate(self.alphas):
                loc = self.source.location[k] + h @ self.means[j]
                loc[CPU] += np.log1p(alpha*(action.workers-1))
                variance = np.diag(h @ self.covariances[j] @ h.T + self.source.noise[k])
                result[indices == k] += self.weights[j]*np.exp(loc + variance/2)
        return result

    def summary(self):
        # Intervals for persistent multipliers at one worker, not event predictions.
        from scipy.optimize import brentq
        out = {'alpha_mean': float(self.weights @ self.alphas),
               'alpha_weights': self.weights.tolist(), 'observed_batches': self.observed_batches}
        for p, name in enumerate(('cpu', 'gpu')):
            m, sd = self.means[:, p], np.sqrt(self.covariances[:, p, p])
            out[f'{name}_mean'] = float(self.weights @ np.exp(m + sd**2/2))
            for q, suffix in ((.05, 'lo'), (.95, 'hi')):
                quantile = brentq(lambda x: self.weights @ norm.cdf((x-m)/sd)-q,
                                  float(np.min(m-10*sd)), float(np.max(m+10*sd)))
                out[f'{name}_{suffix}'] = float(np.exp(quantile))
        return out


def sampled_demands(sample, action):
    log_base, speeds, alphas = sample
    values = log_base.copy()
    values[:, :, CPU] += (speeds[:, 0] + np.log1p(alphas*(action.workers-1)))[:, None, None]
    values[:, :, 1] += speeds[:, 1, None]
    return np.exp(values)


def batch_makespan(durations, action):
    """Fast async replay with the same four-stage gate semantics as replay.py."""
    durations = np.asarray(durations, float)
    if durations.ndim != 2 or durations.shape[1] != 4 or not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError('durations must be a finite nonnegative n x 4 array')
    count = len(durations)
    pre, post = deque(), deque()
    gpu, ends = [], []
    free, active, admitted, completed = action.workers, 0, 0, 0
    gate, now, sequence = False, 0., 0

    def finish(index, stage):
        nonlocal sequence
        end = now + durations[index, stage]
        if not np.isfinite(end):
            raise ValueError('Batch timeline exceeds finite floating-point range')
        heapq.heappush(ends, (end, sequence, index, stage))
        sequence += 1

    while completed < count:
        while ends and ends[0][0] == now:
            _, _, i, stage = heapq.heappop(ends)
            if stage == 0:
                free += 1
                heapq.heappush(gpu, (now, i))
            elif stage == 1:
                post.append(i)
            elif stage == 2:
                gate = False
                finish(i, 3)
            else:
                free += 1
                active -= 1
                completed += 1
        while active < action.inflight and admitted < count:
            pre.append(admitted)
            admitted += 1
            active += 1
        if post and free:
            free -= 1
            finish(post.popleft(), 2)
            continue
        if gpu and not gate:
            _, i = heapq.heappop(gpu)
            gate = True
            finish(i, 1)
            continue
        if pre and free:
            free -= 1
            finish(pre.popleft(), 0)
            continue
        if completed < count:
            if not ends:
                raise RuntimeError('Batch replay deadlocked')
            now = ends[0][0]
    return now


def objective(makespan, action, cpu_price):
    if not np.isfinite(cpu_price) or cpu_price < 0:
        raise ValueError('cpu_price must be finite and nonnegative')
    return makespan * (1 + cpu_price*action.workers)


def select_fixed(frame, cpu_price=.05, batch_size=24):
    """Select one fixed action using replication 2 only, before test replay."""
    validation = frame.loc[frame.replication.eq(2)]
    if validation.empty:
        raise ValueError('Replication 2 is required to select the fixed baseline')
    blocks = [group.sort_values('event_id').head(batch_size)[list(PHASES)].to_numpy(float)
              for _, group in validation.groupby(['condition_id', 'run_id'], sort=True)]
    costs = [sum(objective(batch_makespan(block, a), a, cpu_price) for block in blocks)
             for a in ACTIONS]
    return ACTIONS[int(np.argmin(costs))]


def make_world(frame, seed=110, regime='measured', batches=7, batch_size=24):
    """Build the evaluator's hidden world from contiguous replication-3 blocks.

    This object is never passed to the policy's scoring function. All policies
    replay the same base demands, including identical synthetic perturbations.
    """
    if regime not in ('measured', 'transfer', 'stress'):
        raise ValueError('regime must be measured, transfer, or stress')
    if batches < 1 or batch_size < 1:
        raise ValueError('batches and batch_size must be positive')
    heldout = frame.loc[frame.replication.eq(3)]
    if heldout.empty:
        raise ValueError('Replication 3 is required for evaluation')
    rng = np.random.default_rng(seed)
    machine_rng = np.random.default_rng(seed + 500_000)
    conditions = sorted(heldout.condition_id.unique())
    order = []
    while len(order) < batches:
        order.extend(rng.permutation(conditions).tolist())
    data = []
    for condition in order[:batches]:
        groups = list(heldout.loc[heldout.condition_id.eq(condition)].groupby('run_id', sort=True))
        _, run = groups[int(rng.integers(len(groups)))]
        run = run.sort_values('event_id')
        if len(run) < batch_size:
            raise ValueError('Held-out run is smaller than the requested batch')
        start = int(rng.integers(len(run)-batch_size+1))
        block = run.iloc[start:start+batch_size]
        data.append(dict(conditions=[condition]*batch_size,
                         ids=list(zip(block.run_id, block.event_id)),
                         base=block[list(PHASES)].to_numpy(float).copy()))
    cpu, gpu, alpha = 1., 1., 0.
    if regime != 'measured':
        cpu, gpu = np.exp(machine_rng.uniform(np.log(.5), np.log(2.5), size=2))
        alpha = float(machine_rng.uniform(.03, .3))
    if regime == 'stress':
        for block in data:
            # Off-model joint event difficulty and rare bursts, fixed before
            # any policy runs. The controller is not told these perturbations.
            common = machine_rng.normal(0, .35, (batch_size, 1))
            burst = np.where(machine_rng.random((batch_size, 1)) < .08, 5., 1.)
            block['base'] *= np.exp(common - .35**2/2) * burst
    return dict(seed=seed, regime=regime, batches=data, cpu=float(cpu), gpu=float(gpu), alpha=alpha)


def realized_demands(world, batch, action):
    result = batch['base'].copy()
    power = 1.35 if world['regime'] == 'stress' else 1.
    cpu = world['cpu'] * (1 + world['alpha']*(action.workers-1)**power)
    result[:, CPU] *= cpu
    result[:, 1] *= world['gpu']
    return result


def score_actions(source, belief, conditions, policy, cpu_price, draws, rng):
    """The policy boundary accepts only visible source labels and current belief."""
    if policy not in ('nominal', 'mean', 'bayes'):
        raise ValueError('Unsupported predictive policy')
    if draws < 2:
        raise ValueError('At least two posterior draws are required')
    sample = belief.draw(conditions, draws, rng) if policy == 'bayes' else None
    idx = source.indices(conditions)
    nominal = np.exp(source.location[idx] + .5*np.diagonal(
        source.noise[idx] + source.baseline_cov[idx] + source.run_cov[idx], axis1=1, axis2=2))
    costs = []
    for action in ACTIONS:
        if policy == 'bayes':
            demands = sampled_demands(sample, action)
        elif policy == 'mean':
            demands = belief.mean_demands(conditions, action)[None]
        else:
            demands = nominal[None]
        costs.append([objective(batch_makespan(v, action), action, cpu_price) for v in demands])
    costs = np.array(costs)
    averages = costs.mean(axis=1)
    selected = int(np.argmin(averages))
    mc_se = costs.std(axis=1, ddof=1)/np.sqrt(draws) if policy == 'bayes' else np.zeros(len(ACTIONS))
    paired_se = (costs - costs[selected]).std(axis=1, ddof=1)/np.sqrt(draws) if policy == 'bayes' else np.zeros(len(ACTIONS))
    scores = pd.DataFrame([dict(action=a.label, workers=a.workers, inflight=a.inflight,
                                cost=averages[i], mc_se=mc_se[i], paired_se=paired_se[i],
                                selected=i == selected)
                           for i, a in enumerate(ACTIONS)])
    return ACTIONS[selected], scores


POLICIES = {'fixed': 'Fixed (validation)', 'nominal': 'Source model only',
            'mean': 'Adaptive mean demands', 'bayes': 'Bayesian expected cost',
            'oracle': 'Hindsight lower bound'}


def run_policy(source, world, policy, cpu_price=.05, draws=24, fixed=Action(4, 8), prior_sd=.7):
    if policy not in POLICIES:
        raise ValueError('Unknown policy')
    objective(0., fixed, cpu_price)
    belief = MachineBelief(source, prior_sd=prior_sd)
    rng = np.random.default_rng(world['seed'] + 900_000)
    total_time = core_seconds = planning = cost = 0.
    history = []
    for index, batch in enumerate(world['batches']):
        before = belief.summary()
        start = perf_counter()
        if policy == 'fixed':
            action, scores = fixed, pd.DataFrame()
        elif policy == 'oracle':
            values = [objective(batch_makespan(realized_demands(world, batch, a), a), a, cpu_price) for a in ACTIONS]
            action, scores = ACTIONS[int(np.argmin(values))], pd.DataFrame()
        else:
            action, scores = score_actions(source, belief, batch['conditions'], policy, cpu_price, draws, rng)
        planning_elapsed = perf_counter() - start
        # Environment reveals observations only AFTER this action is fixed.
        actual = realized_demands(world, batch, action)
        elapsed = batch_makespan(actual, action)
        if policy in ('mean', 'bayes'):
            start = perf_counter()
            belief.update(batch['conditions'], actual, action)
            planning_elapsed += perf_counter() - start
        batch_cost = objective(elapsed, action, cpu_price)
        history.append(dict(batch=index, condition=batch['conditions'][0], action=action,
                            before=before, after=belief.summary(), scores=scores,
                            makespan=elapsed, core_seconds=action.workers*elapsed,
                            objective=batch_cost, planning_seconds=planning_elapsed,
                            observed=actual, ids=batch['ids']))
        total_time += elapsed
        core_seconds += action.workers*elapsed
        planning += planning_elapsed
        cost += batch_cost
    return dict(policy=policy, seed=world['seed'], regime=world['regime'], makespan=total_time,
                core_seconds=core_seconds, objective=cost, planning_seconds=planning,
                elapsed_with_planning=total_time+planning, history=history)


def paired_summary(metrics):
    rows = []
    for regime, part in metrics.groupby('regime', sort=False):
        pivot = part.pivot(index='seed', columns='policy', values='objective')
        for reference in ('fixed', 'nominal', 'mean'):
            improvement = 100*(pivot[reference]-pivot.bayes)/pivot[reference]
            margin = t.ppf(.975, len(improvement)-1)*improvement.std(ddof=1)/np.sqrt(len(improvement)) if len(improvement)>1 else np.nan
            rows.append(dict(regime=regime, reference=POLICIES[reference], seeds=len(improvement),
                             saving_pct=improvement.mean(), ci_low=improvement.mean()-margin,
                             ci_high=improvement.mean()+margin,
                             wins=int((improvement > 1e-8).sum()),
                             ties=int((np.abs(improvement) <= 1e-8).sum())))
    return pd.DataFrame(rows)


def run_suite(frame, source=None, seeds=tuple(range(110, 118)), regimes=('measured', 'transfer', 'stress'),
              cpu_price=.05, draws=24, batches=7, batch_size=24, prior_sd=.7):
    source = fit_source(frame) if source is None else source
    fixed = select_fixed(frame, cpu_price, batch_size)
    results, metrics, worlds = {}, [], {}
    for regime in regimes:
        for seed in seeds:
            world = make_world(frame, seed, regime, batches, batch_size)
            worlds[(regime, seed)] = world
            for policy in POLICIES:
                result = run_policy(source, world, policy, cpu_price, draws, fixed, prior_sd)
                results[(regime, seed, policy)] = result
                metrics.append({k: v for k, v in result.items() if k != 'history'})
    metrics = pd.DataFrame(metrics)
    return dict(metrics=metrics, paired=paired_summary(metrics), results=results,
                worlds=worlds, fixed=fixed, source=source)
