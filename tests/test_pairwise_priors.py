import numpy as np
import pytest
import pymc as pm

from bayespecfit.spec_line import (
    _add_pairwise_diff_potentials,
    _validate_pairwise_diff_constraints,
    _validate_prior_bounds,
    _validate_prior_inputs,
)


def test_validate_pairwise_diff_constraints_accepts_valid_pairs():
    assert _validate_pairwise_diff_constraints([(0, 1, 500.0)], 2, "vel_mean_diff") == [(0, 1, 500.0)]


@pytest.mark.parametrize(
    "constraints, error",
    [
        ([(0, 0, 500.0)], ValueError),
        ([(0, 2, 500.0)], IndexError),
        ([(0, 1, 0.0)], ValueError),
        ([(0, 1, 500.0), (1, 0, 500.0)], ValueError),
    ],
)
def test_validate_pairwise_diff_constraints_rejects_invalid_pairs(constraints, error):
    with pytest.raises(error):
        _validate_pairwise_diff_constraints(constraints, 2, "vel_mean_diff")


def test_add_pairwise_diff_potentials_adds_named_sparse_prior():
    with pm.Model() as model:
        values = pm.Uniform("v_mean", lower=[-2.0, -2.0], upper=[2.0, 2.0], shape=(2,))
        _add_pairwise_diff_potentials(values, [(0, 1, 500.0)], "v_mean")

    assert [potential.name for potential in model.potentials] == ["v_mean_diff_0_1"]


def test_validate_prior_bounds_accepts_valid_bounds():
    lower, upper = _validate_prior_bounds([-2, -1], [1, 2], 2, "vel_mean")

    assert np.array_equal(lower, np.array([-2.0, -1.0]))
    assert np.array_equal(upper, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "lower, upper, error",
    [
        (None, [1.0, 2.0], ValueError),
        ([-2.0], [1.0, 2.0], IndexError),
        ([-2.0, np.nan], [1.0, 2.0], ValueError),
        ([-2.0, 3.0], [1.0, 2.0], ValueError),
    ],
)
def test_validate_prior_bounds_rejects_invalid_bounds(lower, upper, error):
    with pytest.raises(error):
        _validate_prior_bounds(lower, upper, 2, "vel_mean")


def test_validate_prior_inputs_selects_bounded_flat_prior_from_bounds():
    prior_type, lower, upper = _validate_prior_inputs(None, None, [-2, -1], [1, 2], 2, "vel_mean")

    assert prior_type == "uniform"
    assert np.array_equal(lower, np.array([-2.0, -1.0]))
    assert np.array_equal(upper, np.array([1.0, 2.0]))


def test_validate_prior_inputs_selects_gaussian_prior_from_mean_scatter():
    prior_type, mu, sigma = _validate_prior_inputs([-1, 0], [1, 2], None, None, 2, "vel_mean")

    assert prior_type == "normal"
    assert np.array_equal(mu, np.array([-1.0, 0.0]))
    assert np.array_equal(sigma, np.array([1.0, 2.0]))


@pytest.mark.parametrize(
    "mu, sigma, lower, upper, error",
    [
        ([-1, 0], [1, 2], [-2, -1], [1, 2], ValueError),
        ([-1, 0], None, [-2, -1], [1, 2], ValueError),
        (None, None, None, None, ValueError),
        ([-1], [1, 2], None, None, IndexError),
        ([-1, 0], [1, 0], None, None, ValueError),
    ],
)
def test_validate_prior_inputs_rejects_invalid_or_mixed_inputs(mu, sigma, lower, upper, error):
    with pytest.raises(error):
        _validate_prior_inputs(mu, sigma, lower, upper, 2, "vel_mean")
