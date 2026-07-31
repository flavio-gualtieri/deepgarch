import numpy as np
import pytest
import torch

from deepgarch.eval.static_garch import StaticGARCH
from deepgarch.eval.gjr_garch import GJRGARCH
from deepgarch.eval.egarch import EGARCH
from deepgarch.eval.ewma import EWMA
from deepgarch.models.vol.gjr_garch import GJRGARCH as GJRGARCHVariance

BASELINE_CLASSES = [StaticGARCH, GJRGARCH, EGARCH, EWMA]


@pytest.fixture
def train_returns() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(500) * 0.01


@pytest.fixture
def full_returns() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(700) * 0.01


@pytest.mark.parametrize("baseline_cls", BASELINE_CLASSES)
class TestBaselineContract:
    """Contract shared by every ArchBaseline subclass, independent of its
    specific volatility dynamics."""

    def test_unfitted_repr_does_not_raise(self, baseline_cls):
        baseline = baseline_cls()
        repr(baseline)

    def test_filter_before_fit_raises(self, baseline_cls, full_returns):
        baseline = baseline_cls()
        with pytest.raises(RuntimeError):
            baseline.filter(full_returns)

    def test_fit_returns_self_and_sets_fitted(self, baseline_cls, train_returns):
        baseline = baseline_cls()
        result = baseline.fit(train_returns)
        assert result is baseline
        assert baseline._fitted is True

    def test_fitted_repr_does_not_raise(self, baseline_cls, train_returns):
        baseline = baseline_cls().fit(train_returns)
        repr(baseline)

    def test_filter_output_shape_and_positivity(self, baseline_cls, train_returns, full_returns):
        baseline = baseline_cls().fit(train_returns)
        variances = baseline.filter(full_returns)
        assert isinstance(variances, np.ndarray)
        assert variances.shape == (len(full_returns),)
        assert np.all(variances > 0)

    def test_persistence_after_fit(self, baseline_cls, train_returns):
        baseline = baseline_cls().fit(train_returns)
        persistence = baseline.persistence
        assert isinstance(persistence, float)
        assert persistence > 0

    def test_name_is_string(self, baseline_cls):
        assert isinstance(baseline_cls().name, str)


class TestStaticGARCH:
    """Checks specific to the GARCH(1,1)-via-`arch` implementation."""

    def test_fit_recovers_positive_params(self, train_returns):
        model = StaticGARCH().fit(train_returns)
        assert model.omega > 0
        assert model.alpha >= 0
        assert model.beta >= 0

    def test_persistence_matches_alpha_plus_beta(self, train_returns):
        model = StaticGARCH().fit(train_returns)
        assert model.persistence == pytest.approx(model.alpha + model.beta)


class TestGJRGARCH:
    """Checks specific to the GJR-GARCH (asymmetric) implementation."""

    def test_fit_recovers_gamma(self, train_returns):
        model = GJRGARCH().fit(train_returns)
        assert model.gamma is not None

    def test_persistence_matches_formula(self, train_returns):
        model = GJRGARCH().fit(train_returns)
        expected = model.alpha + model.beta + model.gamma / 2
        assert model.persistence == pytest.approx(expected)

    def test_negative_shock_raises_variance_more_than_positive(self):
        """The defining property of GJR-GARCH: a negative past return should
        push variance up more than a same-magnitude positive one, given
        gamma > 0."""
        variance_model = GJRGARCHVariance(
            omega=torch.tensor(1e-6),
            alpha=torch.tensor([0.05]),
            beta=torch.tensor([0.90]),
            gamma=torch.tensor([0.10]),
            constraint="none",
        )
        past_variance = torch.tensor(1e-4)
        shock = torch.tensor(0.02)

        variance_after_negative = variance_model.variance_equation(1, -shock, past_variance)
        variance_after_positive = variance_model.variance_equation(1, shock, past_variance)

        assert variance_after_negative > variance_after_positive


class TestEGARCH:
    """Checks specific to the EGARCH (log-variance) implementation."""

    def test_fit_recovers_gamma(self, train_returns):
        model = EGARCH().fit(train_returns)
        assert model.gamma is not None

    def test_persistence_matches_abs_beta(self, train_returns):
        model = EGARCH().fit(train_returns)
        assert model.persistence == pytest.approx(abs(model.beta))

    def test_filtered_variance_is_scale_invariant(self, train_returns, full_returns):
        """The omega correction for EGARCH's log-space recursion is easy to
        get wrong in a way that only shows up as a magnitude error (not a
        crash, not a sign flip) — pin it down by checking that two
        differently-scaled fits agree on the actual-space variance."""
        model_scale_1 = EGARCH(scale=1.0).fit(train_returns)
        model_scale_100 = EGARCH(scale=100.0).fit(train_returns)

        variance_1 = model_scale_1.filter(full_returns)
        variance_100 = model_scale_100.filter(full_returns)

        np.testing.assert_allclose(variance_1, variance_100, rtol=0.05)


class TestEWMA:
    """Checks specific to the RiskMetrics-style EWMA implementation."""

    def test_rejects_invalid_decay(self):
        with pytest.raises(ValueError):
            EWMA(decay=1.5)

    def test_persistence_is_one_by_construction(self, train_returns):
        model = EWMA().fit(train_returns)
        assert model.persistence == pytest.approx(1.0)

    def test_uses_requested_decay(self, train_returns):
        model = EWMA(decay=0.9).fit(train_returns)
        assert model.decay == 0.9
