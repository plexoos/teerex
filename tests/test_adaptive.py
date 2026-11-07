"""Behavioral checks for the adaptive allocation notebook's numerical engine."""
import numpy as np
import pandas as pd
import pytest

from teerex import adaptive as ad
from teerex.replay import replay


def source_frame():
    rng = np.random.default_rng(51)
    rows = []
    for condition, scale in [('cpu', [1., .1, .03, .01]), ('gpu', [.1, 1., .03, .01])]:
        for rep in range(4):
            for event, values in enumerate(np.array(scale) * np.exp(rng.normal(0, .08, (24, 4)))):
                rows.append(dict(condition_id=condition, run_id=f'{condition}-{rep}',
                                 replication=rep, event_id=event, **dict(zip(ad.PHASES, values))))
    return pd.DataFrame(rows)


def test_source_fit_uses_training_runs_only():
    frame = source_frame()
    first = ad.fit_source(frame)
    frame.loc[frame.replication.ge(2), list(ad.PHASES)] *= 100
    second = ad.fit_source(frame)
    np.testing.assert_array_equal(first.location, second.location)
    assert first.training_runs == ('cpu-0', 'cpu-1', 'gpu-0', 'gpu-1')
    with pytest.raises(ValueError, match='Unknown'):
        first.indices(['unseen'])


@pytest.mark.parametrize('workers,wip', [(1, 1), (1, 4), (2, 2), (2, 8), (4, 8)])
def test_fast_makespan_matches_existing_phase_replay(workers, wip):
    rng = np.random.default_rng(12)
    for values in [rng.lognormal(-2, 1, (15, 4)), np.zeros((6, 4))]:
        rows = [dict(event_id=i, **dict(zip(ad.PHASES, v))) for i, v in enumerate(values)]
        expected = replay(rows, workers=workers, max_inflight=wip, policy='async').end_time_offset_s.max()
        assert ad.batch_makespan(values, ad.Action(workers, wip)) == pytest.approx(expected)


def test_one_worker_supplies_no_contention_information():
    model = ad.fit_source(source_frame())
    belief = ad.MachineBelief(model)
    before = belief.weights.copy()
    observed = np.exp(np.tile(model.location[0], (8, 1))) * [1.8, .7, 1.8, 1.8]
    belief.update(['cpu'] * 8, observed, ad.Action(1, 2))
    np.testing.assert_allclose(belief.weights, before, atol=1e-12)
    assert belief.means[:, 0].mean() > 0
    assert belief.means[:, 1].mean() < 0
    assert np.all(belief.covariances[:, 0, 0] < belief.prior_sd**2)


def test_completed_observation_matches_gaussian_conditioning():
    model = ad.fit_source(source_frame())
    belief = ad.MachineBelief(model, alphas=(0.,))
    h = belief.design(0)
    v = belief.covariances[0].copy()
    residual = np.array([.3, -.2, .3, .3])
    noise = model.noise[0]  # One event, hence effective n=1.
    gain = np.linalg.solve(h @ v @ h.T + noise, h @ v).T
    expected_mean = gain @ residual
    expected_cov = v - gain @ h @ v
    belief.update(['cpu'], np.exp(model.location[[0]] + residual), ad.Action(1, 1))
    np.testing.assert_allclose(belief.means[0], expected_mean, atol=1e-12)
    np.testing.assert_allclose(belief.covariances[0], expected_cov, atol=1e-12)


def test_source_validation_rejects_duplicate_and_invalid_duration():
    frame = source_frame()
    with pytest.raises(ValueError, match='duplicate'):
        ad.fit_source(pd.concat([frame, frame.iloc[[0]]]))
    frame.loc[0, ad.PHASES[0]] = np.nan
    with pytest.raises(ValueError, match='positive finite'):
        ad.fit_source(frame)


def test_paired_experiment_is_reproducible_and_oracle_bounds_cost():
    frame = source_frame()
    model = ad.fit_source(frame)
    world = ad.make_world(frame, seed=23, regime='transfer', batches=3, batch_size=8)
    fixed = ad.select_fixed(frame, cpu_price=.05, batch_size=8)
    a = ad.run_policy(model, world, 'bayes', cpu_price=.05, draws=8, fixed=fixed)
    b = ad.run_policy(model, world, 'bayes', cpu_price=.05, draws=8, fixed=fixed)
    oracle = ad.run_policy(model, world, 'oracle', cpu_price=.05, draws=8, fixed=fixed)
    assert a['objective'] == b['objective']
    assert a['objective'] >= oracle['objective'] - 1e-12
    assert [h['before']['observed_batches'] for h in a['history']] == [0, 1, 2]
    assert a['history'][0]['before']['alpha_weights'] == [.2]*5
    assert all(h['before']['cpu_mean'] > 0 for h in a['history'])
    assert a['core_seconds'] >= a['makespan']


def test_first_decision_cannot_see_current_or_future_observed_durations():
    frame = source_frame()
    model = ad.fit_source(frame)
    world = ad.make_world(frame, seed=8, regime='measured', batches=2, batch_size=8)
    from copy import deepcopy
    changed = deepcopy(world)
    changed['batches'][0]['base'] *= 40
    changed['batches'][1]['base'] *= .01
    first = ad.run_policy(model, world, 'bayes', draws=8)
    second = ad.run_policy(model, changed, 'bayes', draws=8)
    assert first['history'][0]['action'] == second['history'][0]['action']
    np.testing.assert_array_equal(first['history'][0]['scores'].cost,
                                  second['history'][0]['scores'].cost)


def test_cost_is_completion_time_plus_allocated_cpu_seconds():
    assert ad.objective(3., ad.Action(4, 8), .1) == pytest.approx(4.2)
    with pytest.raises(ValueError):
        ad.objective(3., ad.Action(4, 8), -.1)


def test_new_run_shift_is_shared_uncertainty_not_independent_event_noise():
    model = ad.fit_source(source_frame())
    from dataclasses import replace
    shifted = replace(model, run_cov=np.repeat((np.eye(4)*.4)[None], 2, axis=0))
    belief = ad.MachineBelief(shifted)
    np.testing.assert_allclose(belief.covariances[0, 2:6, 2:6], shifted.baseline_cov[0] + shifted.run_cov[0])
    np.testing.assert_array_equal(shifted.noise, model.noise)
    before = ad.source_diagnostics(model, source_frame(), 3).interval_width_ms
    after = ad.source_diagnostics(shifted, source_frame(), 3).interval_width_ms
    assert (after > before).all()
    _, nominal_before = ad.score_actions(model, ad.MachineBelief(model), ['cpu']*8, 'nominal', .05, 8, np.random.default_rng(1))
    _, nominal_after = ad.score_actions(shifted, belief, ['cpu']*8, 'nominal', .05, 8, np.random.default_rng(1))
    assert (nominal_after.cost > nominal_before.cost).all()


def test_one_worker_calibration_can_resolve_previously_confounded_contention():
    model = ad.fit_source(source_frame())
    belief = ad.MachineBelief(model)
    baseline = np.exp(np.tile(model.location[0], (24, 1)))
    belief.update(['cpu']*24, baseline * [1.45, 1, 1.45, 1.45], ad.Action(4, 8))
    before = belief.weights.copy()
    belief.update(['cpu']*24, baseline, ad.Action(1, 2))
    assert not np.allclose(belief.weights, before)
    assert belief.weights[2] > before[2]  # α=.15 separates from CPU speed.
