import warnings

import numpy as np

from bayespecfit.spec_line import SpecLine, lnlike_gaussian_abs, lnlike_student_t_abs
from bayespecfit.spec_sn import SpectrumSN


def test_force_pos_flux_filters_non_positive_flux(tmp_path):
    spec_path = tmp_path / "spec.dat"
    np.savetxt(
        spec_path,
        np.array(
            [
                [5000.0, 1.0, 0.1],
                [5001.0, 0.0, 0.1],
                [5002.0, -1.0, 0.1],
            ]
        ),
    )

    spec_all = SpectrumSN(str(spec_path), force_pos_flux=False)
    assert len(spec_all.fl) == 3

    spec_pos = SpectrumSN(str(spec_path), force_pos_flux=True)
    assert np.array_equal(spec_pos.fl, np.array([1.0]))


def test_masked_low_uncertainty_points_do_not_set_uncertainty_floor():
    wv = np.linspace(0.0, 10.0, 41)
    fl = np.ones_like(wv)
    fl_unc = np.full_like(wv, 0.1)
    masked = (wv > 4.0) & (wv < 6.0)
    fl_unc[masked] = 1e-8
    spec = np.column_stack([wv, fl, fl_unc])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        line = SpecLine(
            spec,
            lines=[5.0],
            line_regions=[(0.0, 10.0)],
            mask=[(4.0, 6.0)],
            spec_resolution=2.0,
        )

    assert not any("extremely low uncertainty" in str(item.message) for item in caught)
    assert np.allclose(line.fl_norm_unc, 0.1)


def test_continuum_bin_uncertainty_accounts_for_oversampling():
    wv = np.arange(0.0, 10.01, 0.1)
    fl = np.ones_like(wv)
    fl_unc = np.ones_like(wv)
    spec = np.column_stack([wv, fl, fl_unc])
    line = SpecLine(spec, lines=[5.0], line_regions=[(2.0, 8.0)], spec_resolution=1.0)

    mean, sigma = line.get_flux_at_lambda(5.0, delta_l=2.0, spec_resolution=1.0)
    selected = np.abs(wv - 5.0) < 2.0
    n_eff_res = (np.nanmax(wv[selected]) - np.nanmin(wv[selected])) / 1.0

    assert mean == 1.0
    assert np.isclose(sigma, 1 / np.sqrt(n_eff_res))


def test_likelihood_uncertainty_scale_accounts_for_oversampling():
    wv = np.arange(0.0, 10.01, 0.1)
    fl = np.ones_like(wv)
    fl_unc = np.ones_like(wv) * 0.1
    spec = np.column_stack([wv, fl, fl_unc])
    line = SpecLine(spec, lines=[5.0], line_regions=[(2.0, 8.0)], spec_resolution=1.0)

    assert np.allclose(line.oversampling_sigma_scale, np.sqrt(10.0))


def test_spectrum_without_uncertainty_column_keeps_missing_uncertainties(tmp_path):
    spec_path = tmp_path / "spec.dat"
    np.savetxt(spec_path, np.array([[5000.0, 1.0], [5001.0, 1.1], [5002.0, 1.0]]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spec = SpectrumSN(str(spec_path))

    assert np.all(np.isnan(spec.fl_unc))
    assert any("No flux uncertainty" in str(item.message) for item in caught)


def test_noise_floor_allows_likelihood_without_measurement_uncertainties():
    wv = np.linspace(0.0, 10.0, 41)
    fl = np.ones_like(wv)
    fl_unc = np.full_like(wv, np.nan)
    spec = np.column_stack([wv, fl, fl_unc])
    line = SpecLine(spec, lines=[5.0], line_regions=[(2.0, 8.0)], spec_resolution=1.0)

    theta = [0.0, 2.0, -2.0]

    assert lnlike_gaussian_abs(theta, line) == -np.inf
    assert np.isfinite(lnlike_gaussian_abs(theta, line, noise_floor=0.1))


def test_student_t_likelihood_is_finite_and_robust_to_an_outlier():
    wv = np.linspace(4990.0, 5010.0, 41)
    fl = np.ones_like(wv)
    fl[20] = 2.0
    fl_unc = np.full_like(wv, 0.1)
    line = SpecLine(
        np.column_stack([wv, fl, fl_unc]),
        lines=[5000.0],
        line_regions=[(4990.0, 5010.0)],
        spec_resolution=1.0,
    )

    theta = [0.0, 2.0, -2.0]
    student_t = lnlike_student_t_abs(theta, line, nu=4.0)
    gaussian = lnlike_gaussian_abs(theta, line)

    assert np.isfinite(student_t)
    assert student_t > gaussian


def test_continuum_bin_excludes_masked_points():
    wv = np.arange(0.0, 10.01, 0.5)
    fl = np.ones_like(wv)
    fl[wv == 5.0] = 100.0
    fl_unc = np.ones_like(wv)
    spec = np.column_stack([wv, fl, fl_unc])
    line = SpecLine(
        spec,
        lines=[5.0],
        line_regions=[(2.0, 8.0)],
        mask=[(4.75, 5.25)],
        spec_resolution=1.0,
    )

    mean, _ = line.get_flux_at_lambda(5.0, delta_l=1.0, spec_resolution=1.0)

    assert mean == 1.0
