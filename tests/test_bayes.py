"""Analytic and statistical checks for conjugate Bayesian regression."""

import copy
import json
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import t

from teerex.bayes import BayesianLinearRegression


class BayesianLinearRegressionTest(unittest.TestCase):
    model_class = BayesianLinearRegression

    def tiny_model(self):
        return self.model_class(V0=np.eye(2)).fit([[1, 0], [1, 1]], [1, 3])

    def test_hand_calculated_posterior_and_student_t_prediction(self):
        model = self.tiny_model()
        assert_allclose(model.m_n, [1, 1], atol=1e-14)
        assert_allclose(model.V_n, [[0.4, -0.2], [-0.2, 0.6]], atol=1e-14)
        self.assertEqual(model.a_n, 3)
        self.assertAlmostEqual(model.b_n, 2.5)
        mean, scale, df = model.predict([[1, 0], [1, 2]])
        assert_allclose(mean, [1, 3], atol=1e-14)
        assert_allclose(scale**2, [7 / 6, 2.5], atol=1e-14)
        self.assertEqual(df, 6)
        self.assertEqual(mean.shape, (2,))
        self.assertEqual(scale.shape, (2,))
        self.assertTrue(np.isscalar(df))

    def test_default_prior_is_conditional_variance_ten(self):
        model = self.model_class().fit([[1]], [2])
        assert_allclose(model.m_n, [20 / 11])
        assert_allclose(model.V_n, [[10 / 11]])
        self.assertEqual(model.a_n, 2.5)
        self.assertAlmostEqual(model.b_n, 13 / 11)

    def test_nonzero_prior_mean_and_correlated_covariance(self):
        # V0 inverse = [[2/3, -1/3], [-1/3, 2/3]], m0 = [1, -1].
        # With X=I and y=[2, 0]: Vn=[[5/8,1/8],[1/8,5/8]], mn=[7/4,-1/4].
        model = self.model_class(m0=[1, -1], V0=[[2, 1], [1, 2]], a0=3, b0=2)
        self.assertIs(model.fit(np.eye(2), [2, 0]), model)
        assert_allclose(model.m_n, [7 / 4, -1 / 4])
        assert_allclose(model.V_n, [[5 / 8, 1 / 8], [1 / 8, 5 / 8]])
        self.assertEqual(model.a_n, 4)
        self.assertAlmostEqual(model.b_n, 2.25)

    def test_rank_deficient_and_underdetermined_inputs_have_proper_posteriors(self):
        for X, y in [([[1, 1], [1, 1]], [0, 2]), ([[1, 1, 1]], [2])]:
            with self.subTest(X=X):
                model = self.model_class().fit(X, y)
                mean, scale, df = model.predict(X)
                self.assertTrue(np.isfinite(mean).all())
                self.assertTrue((scale > 0).all())
                self.assertTrue(np.isfinite(scale).all())
                self.assertGreater(np.linalg.eigvalsh(model.V_n).min(), 0)
                self.assertGreater(model.b_n, 0)
                self.assertGreater(df, 0)

    def test_posterior_scale_does_not_cancel_for_large_exact_targets(self):
        model = self.model_class(m0=[1e12], V0=[[1]]).fit([[1], [1]], [1e12, 1e12])
        self.assertAlmostEqual(model.b_n, 1)
        self.assertTrue(np.isfinite(model.predict([[1]])[1]).all())

    def test_predictive_intervals_cover_controlled_normal_observations(self):
        rng = np.random.default_rng(1847)
        train_x = rng.uniform(-2, 2, 600)
        model = self.model_class().fit(
            np.column_stack((np.ones(600), train_x)),
            1.5 - 0.7 * train_x + rng.normal(0, 0.5, 600),
        )
        test_x = rng.uniform(-2, 2, 6000)
        truth = 1.5 - 0.7 * test_x + rng.normal(0, 0.5, 6000)
        mean, scale, df = model.predict(np.column_stack((np.ones(6000), test_x)))
        covered = np.abs(truth - mean) <= t.ppf(0.95, df) * scale
        self.assertGreater(covered.mean(), 0.87)
        self.assertLess(covered.mean(), 0.93)

    def test_predictive_draws_are_reproducible_with_expected_joint_covariance(self):
        model = self.model_class().fit([[1]], [0])
        X = [[1], [1]]
        draws = model.sample_predictive(X, 80000, np.random.default_rng(42))
        self.assertEqual(draws.shape, (80000, 2))
        assert_array_equal(
            draws, model.sample_predictive(X, 80000, np.random.default_rng(42))
        )
        # df=5, b=1, V=10/11: Var(y)=14/11, Cov(y1,y2)=20/33.
        assert_allclose(draws.mean(axis=0), [0, 0], atol=0.025)
        assert_allclose(np.cov(draws.T), [[14 / 11, 20 / 33], [20 / 33, 14 / 11]], atol=0.06)
        self.assertFalse(np.array_equal(draws[:, 0], draws[:, 1]))

    def test_predictive_noise_variance_is_shared_across_events_in_a_draw(self):
        model = self.model_class().fit(np.ones((12, 1)), np.zeros(12))
        draws = model.sample_predictive([[0], [0]], 100000, np.random.default_rng(721))
        # With X=0 only event noise remains. Shared inverse-gamma variance gives
        # Corr(epsilon_1^2, epsilon_2^2)=1/(2*a_n-1)=1/15, not zero.
        correlation = np.corrcoef(draws[:, 0] ** 2, draws[:, 1] ** 2)[0, 1]
        self.assertGreater(correlation, 0.035)
        self.assertLess(correlation, 0.10)

    def test_round_trip_json_preserves_predictions_priors_and_draws(self):
        model = self.model_class(m0=[1, -1], V0=[[2, 1], [1, 2]], a0=3, b0=2)
        model.fit([[1, 0], [1, 1]], [1, 3])
        state = model.to_dict()
        restored = self.model_class.from_dict(json.loads(json.dumps(state, allow_nan=False)))
        self.assertEqual(restored.to_dict(), state)
        for expected, actual in zip(model.predict([[1, 2]]), restored.predict([[1, 2]])):
            assert_allclose(actual, expected)
        assert_array_equal(
            model.sample_predictive([[1, 2]], 15, np.random.default_rng(93)),
            restored.sample_predictive([[1, 2]], 15, np.random.default_rng(93)),
        )
        state["m_n"][0] = -999
        self.assertNotEqual(model.m_n[0], -999)

    def test_unfitted_prediction_sampling_and_persistence_are_rejected(self):
        model = self.model_class()
        for operation in [lambda: model.predict([[1]]), lambda: model.to_dict(),
                          lambda: model.sample_predictive([[1]], 2, np.random.default_rng(0))]:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                operation()

    def test_invalid_training_data_is_rejected(self):
        cases = [([], []), (np.empty((0, 2)), []), (np.empty((2, 0)), [1, 2]),
                 ([1, 2], [1, 2]), ([[1], [2]], [1]), ([[1]], [[2]]),
                 ([[np.nan]], [1]), ([[1]], [np.inf]), ([[1 + 1j]], [1]),
                 ([[1]], [1 + 1j]), ([["bad"]], [1]), ([[1], [1, 2]], [1, 2])]
        for X, y in cases:
            with self.subTest(X=X, y=y), self.assertRaises(ValueError):
                self.model_class().fit(X, y)

    def test_invalid_priors_are_rejected(self):
        cases = [{"a0": 0}, {"a0": np.nan}, {"a0": [2]}, {"b0": -1},
                 {"b0": np.inf}, {"m0": [[0]]}, {"m0": [np.nan]},
                 {"m0": [1j]}, {"V0": [[1, 0], [1, 1]]},
                 {"V0": [[0]]}, {"V0": [[-1]]}, {"V0": [[np.inf]]},
                 {"V0": [1]}, {"m0": [0, 0], "V0": [[1]]}]
        for prior in cases:
            with self.subTest(prior=prior), self.assertRaises(ValueError):
                self.model_class(**prior).fit([[1]], [1])

    def test_prediction_validates_feature_count_and_finite_values(self):
        model = self.tiny_model()
        for X in [[[1]], [1, 2], [[1, np.nan]], [[1, np.inf]], [[1, 1j]]]:
            with self.subTest(X=X), self.assertRaises(ValueError):
                model.predict(X)
        mean, scale, df = model.predict(np.empty((0, 2)))
        self.assertEqual(mean.shape, (0,))
        self.assertEqual(scale.shape, (0,))
        self.assertEqual(df, 6)

    def test_sampling_validates_draw_count_and_generator(self):
        model = self.tiny_model()
        for count in [-1, 0, 1.5, True]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                model.sample_predictive([[1, 0]], count, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            model.sample_predictive([[1, 0]], 3, 42)
        draws = model.sample_predictive(np.empty((0, 2)), 3, np.random.default_rng(0))
        self.assertEqual(draws.shape, (3, 0))

    def test_nonfinite_posterior_is_rejected_without_overflow_warnings(self):
        with np.errstate(all="raise"), self.assertRaises(ValueError):
            self.model_class().fit([[1], [1]], [1e308, -1e308])

    def test_small_covariance_asymmetry_is_rejected_relative_to_its_scale(self):
        with self.assertRaises(ValueError):
            self.model_class(V0=[[1e-20, 1e-20], [0, 1e-20]])

    def test_persistence_requires_the_integer_schema_version(self):
        state = self.tiny_model().to_dict()
        for version in [True, 1.0]:
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.model_class.from_dict({**state, "version": version})

    def test_default_prior_round_trip_retains_refit_behavior(self):
        original = self.model_class().fit([[1]], [2])
        restored = self.model_class.from_dict(json.loads(json.dumps(original.to_dict())))
        self.assertEqual(restored.to_dict(), original.to_dict())
        restored.fit([[1, 0]], [2])
        self.assertEqual(restored.predict([[1, 0]])[0].shape, (1,))

    def test_malformed_persisted_models_are_rejected(self):
        state = self.tiny_model().to_dict()
        cases = [{}, {**state, "version": 999}, {**state, "a_n": 0},
                 {**state, "b_n": np.nan}, {**state, "V_n": [[1]]},
                 {**state, "V_n": [[1, 0], [0, -1]]},
                 {**state, "V_n": [[1, 1], [0, 1]]},
                 {**state, "m_n": [np.inf, 0]}, {**state, "n_features": 3},
                 {**state, "n_observations": -1}]
        for invalid in cases:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.model_class.from_dict(copy.deepcopy(invalid))


if __name__ == "__main__":
    unittest.main()
