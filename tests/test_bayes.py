"""Analytic and statistical checks for conjugate Bayesian regression."""

import copy
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import t

from teerex.bayes import BayesianLinearRegression


@pytest.fixture
def tiny_model():
    return BayesianLinearRegression(V0=np.eye(2)).fit([[1, 0], [1, 1]], [1, 3])


def test_hand_calculated_posterior_and_student_t_prediction(tiny_model):
    model = tiny_model
    assert_allclose(model.m_n, [1, 1], atol=1e-14)
    assert_allclose(model.V_n, [[0.4, -0.2], [-0.2, 0.6]], atol=1e-14)
    assert model.a_n == 3
    assert model.b_n == pytest.approx(2.5, rel=0, abs=5e-8)
    mean, scale, df = model.predict([[1, 0], [1, 2]])
    assert_allclose(mean, [1, 3], atol=1e-14)
    assert_allclose(scale**2, [7 / 6, 2.5], atol=1e-14)
    assert df == 6
    assert mean.shape == (2,)
    assert scale.shape == (2,)
    assert np.isscalar(df)


def test_default_prior_is_conditional_variance_ten():
    model = BayesianLinearRegression().fit([[1]], [2])
    assert_allclose(model.m_n, [20 / 11])
    assert_allclose(model.V_n, [[10 / 11]])
    assert model.a_n == 2.5
    assert model.b_n == pytest.approx(13 / 11, rel=0, abs=5e-8)


def test_nonzero_prior_mean_and_correlated_covariance():
    # V0 inverse = [[2/3, -1/3], [-1/3, 2/3]], m0 = [1, -1].
    # With X=I and y=[2, 0]: Vn=[[5/8,1/8],[1/8,5/8]], mn=[7/4,-1/4].
    model = BayesianLinearRegression(m0=[1, -1], V0=[[2, 1], [1, 2]], a0=3, b0=2)
    assert model.fit(np.eye(2), [2, 0]) is model
    assert_allclose(model.m_n, [7 / 4, -1 / 4])
    assert_allclose(model.V_n, [[5 / 8, 1 / 8], [1 / 8, 5 / 8]])
    assert model.a_n == 4
    assert model.b_n == pytest.approx(2.25, rel=0, abs=5e-8)


@pytest.mark.parametrize("X, y", [([[1, 1], [1, 1]], [0, 2]), ([[1, 1, 1]], [2])])
def test_rank_deficient_and_underdetermined_inputs_have_proper_posteriors(X, y):
    model = BayesianLinearRegression().fit(X, y)
    mean, scale, df = model.predict(X)
    assert np.isfinite(mean).all()
    assert (scale > 0).all()
    assert np.isfinite(scale).all()
    assert np.linalg.eigvalsh(model.V_n).min() > 0
    assert model.b_n > 0
    assert df > 0


def test_posterior_scale_does_not_cancel_for_large_exact_targets():
    model = BayesianLinearRegression(m0=[1e12], V0=[[1]]).fit([[1], [1]], [1e12, 1e12])
    assert model.b_n == pytest.approx(1, rel=0, abs=5e-8)
    assert np.isfinite(model.predict([[1]])[1]).all()


def test_predictive_intervals_cover_controlled_normal_observations():
    rng = np.random.default_rng(1847)
    train_x = rng.uniform(-2, 2, 600)
    model = BayesianLinearRegression().fit(
        np.column_stack((np.ones(600), train_x)),
        1.5 - 0.7 * train_x + rng.normal(0, 0.5, 600),
    )
    test_x = rng.uniform(-2, 2, 6000)
    truth = 1.5 - 0.7 * test_x + rng.normal(0, 0.5, 6000)
    mean, scale, df = model.predict(np.column_stack((np.ones(6000), test_x)))
    covered = np.abs(truth - mean) <= t.ppf(0.95, df) * scale
    assert covered.mean() > 0.87
    assert covered.mean() < 0.93


def test_predictive_draws_are_reproducible_with_expected_joint_covariance():
    model = BayesianLinearRegression().fit([[1]], [0])
    X = [[1], [1]]
    draws = model.sample_predictive(X, 80000, np.random.default_rng(42))
    assert draws.shape == (80000, 2)
    assert_array_equal(
        draws, model.sample_predictive(X, 80000, np.random.default_rng(42))
    )
    # df=5, b=1, V=10/11: Var(y)=14/11, Cov(y1,y2)=20/33.
    assert_allclose(draws.mean(axis=0), [0, 0], atol=0.025)
    assert_allclose(
        np.cov(draws.T), [[14 / 11, 20 / 33], [20 / 33, 14 / 11]], atol=0.06
    )
    assert not np.array_equal(draws[:, 0], draws[:, 1])


def test_predictive_noise_variance_is_shared_across_events_in_a_draw():
    model = BayesianLinearRegression().fit(np.ones((12, 1)), np.zeros(12))
    draws = model.sample_predictive([[0], [0]], 100000, np.random.default_rng(721))
    # With X=0 only event noise remains. Shared inverse-gamma variance gives
    # Corr(epsilon_1^2, epsilon_2^2)=1/(2*a_n-1)=1/15, not zero.
    correlation = np.corrcoef(draws[:, 0] ** 2, draws[:, 1] ** 2)[0, 1]
    assert correlation > 0.035
    assert correlation < 0.1


def test_round_trip_json_preserves_predictions_priors_and_draws():
    model = BayesianLinearRegression(m0=[1, -1], V0=[[2, 1], [1, 2]], a0=3, b0=2)
    model.fit([[1, 0], [1, 1]], [1, 3])
    state = model.to_dict()
    restored = BayesianLinearRegression.from_dict(
        json.loads(json.dumps(state, allow_nan=False))
    )
    assert restored.to_dict() == state
    for expected, actual in zip(model.predict([[1, 2]]), restored.predict([[1, 2]])):
        assert_allclose(actual, expected)
    assert_array_equal(
        model.sample_predictive([[1, 2]], 15, np.random.default_rng(93)),
        restored.sample_predictive([[1, 2]], 15, np.random.default_rng(93)),
    )
    state["m_n"][0] = -999
    assert model.m_n[0] != -999


@pytest.mark.parametrize(
    "operation",
    [
        lambda model: model.predict([[1]]),
        lambda model: model.to_dict(),
        lambda model: model.sample_predictive([[1]], 2, np.random.default_rng(0)),
    ],
    ids=["predict", "persist", "sample"],
)
def test_unfitted_prediction_sampling_and_persistence_are_rejected(operation):
    with pytest.raises(ValueError):
        operation(BayesianLinearRegression())


@pytest.mark.parametrize(
    "X, y",
    [
        ([], []),
        (np.empty((0, 2)), []),
        (np.empty((2, 0)), [1, 2]),
        ([1, 2], [1, 2]),
        ([[1], [2]], [1]),
        ([[1]], [[2]]),
        ([[np.nan]], [1]),
        ([[1]], [np.inf]),
        ([[1 + 1j]], [1]),
        ([[1]], [1 + 1j]),
        ([["bad"]], [1]),
        ([[1], [1, 2]], [1, 2]),
    ],
)
def test_invalid_training_data_is_rejected(X, y):
    with pytest.raises(ValueError):
        BayesianLinearRegression().fit(X, y)


@pytest.mark.parametrize(
    "prior",
    [
        {"a0": 0},
        {"a0": np.nan},
        {"a0": [2]},
        {"b0": -1},
        {"b0": np.inf},
        {"m0": [[0]]},
        {"m0": [np.nan]},
        {"m0": [1j]},
        {"V0": [[1, 0], [1, 1]]},
        {"V0": [[0]]},
        {"V0": [[-1]]},
        {"V0": [[np.inf]]},
        {"V0": [1]},
        {"m0": [0, 0], "V0": [[1]]},
    ],
)
def test_invalid_priors_are_rejected(prior):
    with pytest.raises(ValueError):
        BayesianLinearRegression(**prior).fit([[1]], [1])


@pytest.mark.parametrize("X", [[[1]], [1, 2], [[1, np.nan]], [[1, np.inf]], [[1, 1j]]])
def test_prediction_validates_feature_count_and_finite_values(tiny_model, X):
    model = tiny_model
    with pytest.raises(ValueError):
        model.predict(X)
    mean, scale, df = model.predict(np.empty((0, 2)))
    assert mean.shape == (0,)
    assert scale.shape == (0,)
    assert df == 6


@pytest.mark.parametrize("count", [-1, 0, 1.5, True])
def test_sampling_validates_draw_count_and_generator(tiny_model, count):
    model = tiny_model
    with pytest.raises(ValueError):
        model.sample_predictive([[1, 0]], count, np.random.default_rng(0))
    with pytest.raises(ValueError):
        model.sample_predictive([[1, 0]], 3, 42)
    draws = model.sample_predictive(np.empty((0, 2)), 3, np.random.default_rng(0))
    assert draws.shape == (3, 0)


def test_nonfinite_posterior_is_rejected_without_overflow_warnings():
    with np.errstate(all="raise"), pytest.raises(ValueError):
        BayesianLinearRegression().fit([[1], [1]], [1e308, -1e308])


def test_small_covariance_asymmetry_is_rejected_relative_to_its_scale():
    with pytest.raises(ValueError):
        BayesianLinearRegression(V0=[[1e-20, 1e-20], [0, 1e-20]])


@pytest.mark.parametrize("version", [True, 1.0])
def test_persistence_requires_the_integer_schema_version(tiny_model, version):
    state = tiny_model.to_dict()
    with pytest.raises(ValueError):
        BayesianLinearRegression.from_dict({**state, "version": version})


def test_default_prior_round_trip_retains_refit_behavior():
    original = BayesianLinearRegression().fit([[1]], [2])
    restored = BayesianLinearRegression.from_dict(
        json.loads(json.dumps(original.to_dict()))
    )
    assert restored.to_dict() == original.to_dict()
    restored.fit([[1, 0]], [2])
    assert restored.predict([[1, 0]])[0].shape == (1,)


@pytest.mark.parametrize(
    "changes",
    [
        None,
        {"version": 999},
        {"a_n": 0},
        {"b_n": np.nan},
        {"V_n": [[1]]},
        {"V_n": [[1, 0], [0, -1]]},
        {"V_n": [[1, 1], [0, 1]]},
        {"m_n": [np.inf, 0]},
        {"n_features": 3},
        {"n_observations": -1},
    ],
    ids=[
        "empty",
        "version",
        "shape",
        "scale",
        "covariance-size",
        "covariance-definiteness",
        "covariance-symmetry",
        "mean",
        "feature-count",
        "observation-count",
    ],
)
def test_malformed_persisted_models_are_rejected(tiny_model, changes):
    state = tiny_model.to_dict()
    invalid = {} if changes is None else {**state, **changes}
    with pytest.raises(ValueError):
        BayesianLinearRegression.from_dict(copy.deepcopy(invalid))
