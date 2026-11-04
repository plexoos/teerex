"""Small conjugate normal/inverse-gamma Bayesian linear regression.

Features, including any intercept, and target transforms belong to the caller.
The predictive distribution is in the supplied target units; in particular,
log-target predictions are not silently exponentiated. Conditional on sigma²,
beta ~ Normal(m0, sigma² V0), and sigma² ~ InverseGamma(a0, b0), with b0
using the inverse-gamma scale convention exp(-b0 / sigma²).
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np


def _array(value, name, ndim):
    try:
        array = np.asarray(value)
        if np.iscomplexobj(array):
            raise ValueError(f"{name} must be real")
        array = np.array(array, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a real numeric array") from error
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite scalar")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _covariance(value, name):
    covariance = _array(value, name, 2)
    if (covariance.shape[0] == 0 or covariance.shape[0] != covariance.shape[1]
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0)):
        raise ValueError(f"{name} must be symmetric and positive definite")
    covariance = 0.5 * covariance + 0.5 * covariance.T
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be symmetric and positive definite") from error
    return covariance, factor


class BayesianLinearRegression:
    """Conjugate Gaussian regression with a proper normal/inverse-gamma prior.

    Args:
        m0: Prior coefficient mean; None supplies zeros for the fitted features.
        V0: Prior conditional coefficient covariance divided by sigma²; None
            supplies 10 I. Must be symmetric and positive definite.
        a0: Positive inverse-gamma shape (default 2).
        b0: Positive inverse-gamma scale (default 1).

    A proper prior permits collinear features and fewer observations than
    features. Extremely ill-scaled inputs whose posterior cannot be represented
    by finite positive-definite float64 parameters are rejected. Normalize
    features using training data before fitting when their scales differ.
    """

    def __init__(self, *, m0=None, V0=None, a0=2.0, b0=1.0):
        self.m0 = None if m0 is None else _array(m0, "m0", 1)
        if self.m0 is not None and self.m0.size == 0:
            raise ValueError("m0 must contain at least one coefficient")
        self.V0 = None if V0 is None else _covariance(V0, "V0")[0]
        if self.m0 is not None and self.V0 is not None and self.m0.size != self.V0.shape[0]:
            raise ValueError("m0 and V0 must have matching feature dimensions")
        self.a0 = _positive_scalar(a0, "a0")
        self.b0 = _positive_scalar(b0, "b0")
        self._fitted = False

    def _prior(self, n_features):
        mean = np.zeros(n_features) if self.m0 is None else self.m0
        covariance = np.eye(n_features) * 10 if self.V0 is None else self.V0
        if mean.shape != (n_features,) or covariance.shape != (n_features, n_features):
            raise ValueError("prior dimensions must match the number of features in X")
        return mean, covariance

    def fit(self, X, y):
        """Fit and return self; X is (observations, features), y is (observations,).

        QR on the prior-augmented least-squares system avoids normal equations.
        Residual sums of squares determine the inverse-gamma update without
        subtracting large nearly equal quadratic forms.
        """
        X = _array(X, "X", 2)
        y = _array(y, "y", 1)
        n_observations, n_features = X.shape
        if n_observations == 0 or n_features == 0 or y.shape != (n_observations,):
            raise ValueError("X and y need matching nonempty observations and at least one feature")
        m0, V0 = self._prior(n_features)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                prior_factor = np.linalg.cholesky(V0)
                whitening = np.linalg.solve(prior_factor, np.eye(n_features))
                centered_y = y - X @ m0
                augmented_X = np.vstack((X, whitening))
                augmented_y = np.concatenate((centered_y, np.zeros(n_features)))
                Q, R = np.linalg.qr(augmented_X, mode="reduced")
                delta = np.linalg.solve(R, Q.T @ augmented_y)
                m_n = m0 + delta
                inverse_R = np.linalg.solve(R, np.eye(n_features))
                V_n = inverse_R @ inverse_R.T
                residual = centered_y - X @ delta
                prior_residual = whitening @ delta
                b_n = self.b0 + 0.5 * (residual @ residual + prior_residual @ prior_residual)
                a_n = self.a0 + 0.5 * n_observations
                V_n, factor = _covariance(V_n, "posterior covariance")
                if not np.isfinite(m_n).all() or not np.isfinite(2 * a_n):
                    raise ValueError("posterior parameters must be finite")
                a_n = _positive_scalar(a_n, "posterior shape")
                b_n = _positive_scalar(b_n, "posterior scale")
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
            raise ValueError("inputs produce an unrepresentable posterior; rescale X and y") from error
        self.m_n, self.V_n, self.a_n, self.b_n = m_n, V_n, a_n, b_n
        self.n_features, self.n_observations = n_features, n_observations
        self._posterior_factor = factor
        self._fitted = True
        return self

    def _prediction_features(self, X):
        if not self._fitted:
            raise ValueError("fit the model before prediction or persistence")
        X = _array(X, "X", 2)
        if X.shape[1] != self.n_features:
            raise ValueError("X must have the same feature count as the fitted model")
        return X

    def predict(self, X):
        """Return per-event (mean, scale, df) for the Student-t observation law.

        mean and scale are arrays with one entry per row; df is a scalar. Scale
        is the Student-t scale, not its standard deviation. Observation noise
        and coefficient uncertainty are both included.
        """
        X = self._prediction_features(X)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                mean = X @ self.m_n
                projected = X @ self._posterior_factor
                scale = np.sqrt((self.b_n / self.a_n) * (1 + np.sum(projected**2, axis=1)))
        except FloatingPointError as error:
            raise ValueError("prediction is not representable; rescale X and y") from error
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("predictive means and positive scales must be finite")
        return mean, scale, 2 * self.a_n

    def sample_predictive(self, X, n_draws, rng):
        """Return (draw, event) samples with shared beta and sigma² per draw.

        rng must be a numpy.random.Generator. Every draw samples one inverse-
        gamma variance and conditional coefficient vector for the entire event
        stream, followed by conditionally independent observation noise.
        """
        X = self._prediction_features(X)
        n_draws = _positive_integer(n_draws, "n_draws")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                variance = self.b_n / rng.gamma(self.a_n, size=n_draws)
                standard_deviation = np.sqrt(variance)
                coefficients = self.m_n + (
                    rng.standard_normal((n_draws, self.n_features)) @ self._posterior_factor.T
                ) * standard_deviation[:, None]
                draws = coefficients @ X.T + (
                    rng.standard_normal((n_draws, X.shape[0])) * standard_deviation[:, None]
                )
        except FloatingPointError as error:
            raise ValueError("predictive samples are not representable; rescale X and y") from error
        if not np.isfinite(draws).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
            raise ValueError("predictive samples and positive noise variances must be finite")
        return draws

    def to_dict(self):
        """Return JSON-compatible prior and fitted posterior parameters."""
        if not self._fitted:
            raise ValueError("fit the model before prediction or persistence")
        return {
            "version": 1,
            "m0": None if self.m0 is None else self.m0.tolist(),
            "V0": None if self.V0 is None else self.V0.tolist(),
            "a0": self.a0,
            "b0": self.b0,
            "m_n": self.m_n.tolist(),
            "V_n": self.V_n.tolist(),
            "a_n": self.a_n,
            "b_n": self.b_n,
            "n_features": self.n_features,
            "n_observations": self.n_observations,
        }

    @classmethod
    def from_dict(cls, data):
        """Restore a fitted model, rejecting malformed/nonfinite parameters."""
        if not isinstance(data, Mapping):
            raise ValueError("Bayesian regression state must be a mapping")
        version = data.get("version")
        if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral) or version != 1:
            raise ValueError("unsupported or missing Bayesian regression state version")
        try:
            model = cls(m0=data["m0"], V0=data["V0"], a0=data["a0"], b0=data["b0"])
            n_features = _positive_integer(data["n_features"], "n_features")
            n_observations = _positive_integer(data["n_observations"], "n_observations")
            model._prior(n_features)
            mean = _array(data["m_n"], "m_n", 1)
            covariance, factor = _covariance(data["V_n"], "V_n")
            a_n = _positive_scalar(data["a_n"], "a_n")
            b_n = _positive_scalar(data["b_n"], "b_n")
            if mean.shape != (n_features,) or covariance.shape != (n_features, n_features):
                raise ValueError("posterior dimensions must match n_features")
            if not np.isfinite(2 * a_n) or a_n != model.a0 + 0.5 * n_observations:
                raise ValueError("posterior shape must match prior shape and observation count")
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError("incomplete or invalid Bayesian regression state") from error
        model.m_n, model.V_n, model.a_n, model.b_n = mean, covariance, a_n, b_n
        model.n_features, model.n_observations = n_features, n_observations
        model._posterior_factor = factor
        model._fitted = True
        return model
