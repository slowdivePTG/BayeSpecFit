# bayespecfit/flux_model.py

import numpy as np
from numpy.typing import ArrayLike

###################### conversion between wav and vel ##########################

SPEED_OF_LIGHT = 2.99792458e5


def velocity_rf(lambda_rf: ArrayLike, lambda_0: float) -> ArrayLike:
    """convert rest-frame wavelength to relative velocity"""
    lambda_rf = np.asarray(lambda_rf)
    v = SPEED_OF_LIGHT * ((lambda_rf / lambda_0) ** 2 - 1) / ((lambda_rf / lambda_0) ** 2 + 1)

    return v


def wv_rf(vel_rf: ArrayLike, lambda_0: float) -> ArrayLike:
    """convert relative velocity to rest-frame wavelength"""
    vel_rf = np.asarray(vel_rf)
    beta = vel_rf / SPEED_OF_LIGHT
    wv = lambda_0 * ((1 + beta) / (1 - beta)) ** 0.5

    return wv


def velocity_rf_line(lambda_0: float, lambda_1: float, vel: float) -> float:
    """get the relative velocity of a feature assuming another line

    Parameters
    ----------
    lambda_0 : float
        wavelength of the original line [angstrom]
    lambda_1 : float
        wavelength of the new line [angstrom]
    vel : float
        the velocity relative to lambda_0 [km/s]

    Returns
    -------
    vel_1 : float
        the velocity relative to lambda_1 [km/s]
    """

    lambda_rf = ((vel / SPEED_OF_LIGHT + 1) / (-vel / SPEED_OF_LIGHT + 1)) ** 0.5 * lambda_0
    vel_1 = velocity_rf(lambda_rf, lambda_1)

    return vel_1


###################### spectroscopic models ##########################


def _calc_gauss(mean_vel, sig_vel, amplitude, vel_rf):
    """Gaussian profile"""
    gauss = amplitude / np.sqrt(2 * np.pi * sig_vel**2) * np.exp(-0.5 * (vel_rf - mean_vel) ** 2 / sig_vel**2)
    return gauss


def _calc_lorentzian(mean_vel, gamma_vel, amplitude, vel_rf):
    """Lorentzian profile"""
    lorentz = amplitude / np.sqrt(np.pi * gamma_vel) / (1 + (vel_rf - mean_vel) ** 2 / gamma_vel**2)
    return lorentz


def calc_model_flux(
    theta: list,
    wv_rf: ArrayLike,
    lines: list,
    rel_strength: list,
    line_regions: list[tuple],
    vel_resolution: list[float],
    model: str = "Gauss",
):
    """Calculate normalized flux based on a Gaussian/Lorentzian model

    Parameters
    ----------
    theta : array_like
        fitting parameters:
            (flux at the blue edge, flux at the red edge) * Number of the line regions,
            (mean of relative velocity, log standard deviation, log amplitude) * Number of velocity components,
            log10 line ratio for each velocity components (if set free)

    wv_rf : ArrayLike
        rest-frame wavelength [angstrom]

    lines : list
        wavelength of each line component

    rel_strength : list
        the relative strength between each line in the series

    line_regions : list
        the blue and red edges of each line region to fit

    vel_resolution : list[float]
        the spectral resolution of the spectrograph in terms of
        the velocity - will broaden the lines

    model : string, default="Gauss"
        "Gauss" for a Gaussian profile, "Lorentz" for a Lorentzian profile

    Returns
    -------
    model_flux : array_like
        predicted (normalized) flux at each relative radial
        velocity
    """

    # create a 2D mask: regions x wv_values
    wv_blue = np.array([line[0] for line in line_regions])
    wv_red = np.array([line[1] for line in line_regions])
    in_line_region = (wv_rf >= wv_blue[:, np.newaxis]) & (wv_rf <= wv_red[:, np.newaxis])

    ##################### get the continuum flux (a piecewise linear function) ############################

    n_cont = 2 * len(line_regions)
    theta_cont = theta[:n_cont]

    fl_blue = np.array([theta_cont[2 * i] for i in range(len(line_regions))])
    fl_red = np.array([theta_cont[2 * i + 1] for i in range(len(line_regions))])

    # calculate slopes for all segments at once
    slopes = np.divide(fl_red - fl_blue, wv_red - wv_blue, out=np.zeros_like(fl_red), where=(wv_red != wv_blue))

    # calculate all interpolated values
    continuum_2d = fl_blue[:, np.newaxis] + slopes[:, np.newaxis] * (wv_rf - wv_blue[:, np.newaxis])

    # take the first non-NaN value for each x (assumes non-overlapping regions)
    continuum = np.nanmin(np.where(in_line_region, continuum_2d, np.nan), axis=0)

    ###################################### get the absorption lines ########################################

    n_lines = 3 * len(lines)
    theta_abs = theta[n_cont : n_cont + n_lines]

    # the spectral resolution of the spectrograph limits the line width
    vel_resolution_2d = np.array(vel_resolution)[:, np.newaxis]
    vel_res = np.nanmin(np.where(in_line_region, vel_resolution_2d, np.nan), axis=0)

    absorption = np.zeros_like(wv_rf)

    for k in range(len(lines)):
        mean_vel, log_sig_vel, log_amp = theta_abs[3 * k : 3 * k + 3]
        sig_vel = 10 ** log_sig_vel
        amp = 10 ** log_amp
        sig_instru = (sig_vel**2 + vel_res**2) ** 0.5

        for rel_s, li in zip(rel_strength[k], lines[k]):
            vel_rf = velocity_rf(wv_rf, li)
            if model == "Gauss":
                calc = _calc_gauss
            elif model == "Lorentz":
                calc = _calc_lorentzian
            else:
                raise NameError("Line model not supported (optional: Gauss & Lorentz)")
            absorption += rel_s * calc(mean_vel, sig_instru, amp, vel_rf)

    model_flux = continuum * (1 - absorption)

    return model_flux
