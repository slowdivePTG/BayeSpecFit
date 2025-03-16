# bayespecfit/tools/data_binning.py

import numpy as np
from numpy.typing import ArrayLike


def data_binning(
    data: ArrayLike, size: int, min_bin: int = 1, spec_resolution: float = 0, sigma_clip: float = 3
) -> ArrayLike:
    """
    binning spectroscopic data with a finite spectral resolution (FWHM)
    within the FWHM, the noise is expected to be correlated

    data : array_like
        np.array([[wavelength], [flux], [flux_err]]).T
    size : float
        bin size in same units as that of the wavelength
    min_bin : int, default = 1
        minimum number of data points with a bin
    spec_resolution : float, default = 0
        the typical scale of noise correlation
        roughly the spectroscopic resolution
        default - noise is independent
    sigma_clip : float, default = 3
        treat data points beyond this threshold as outliers
    """
    from astropy.stats import mad_std

    data_bin = []
    i = 0
    while i < len(data):
        j = i
        while j < len(data):
            if data[j, 0] < data[i, 0] + size:
                j += 1
            else:
                break
        temp = data[i:j, :]
        if len(temp) >= min_bin:
            if len(temp) > 1:  # if there are more than 1 data points in the bin
                arg = np.arange(len(temp))
                x, y, yerr = (
                    temp[arg, 0].reshape(-1, 1),
                    temp[arg, 1].reshape(-1, 1),
                    temp[arg, 2].reshape(-1, 1),
                )
                if sigma_clip != None:
                    clip = np.abs(y - np.median(y)) <= sigma_clip * mad_std(y)
                    x, y, yerr = x[clip], y[clip], yerr[clip]

                # mu = weight^T * y
                # var = weight^T * Cov * weight
                weight = yerr ** (-2.0) / (yerr ** (-2.0)).sum()

                if spec_resolution > 0:
                    X_A = np.repeat(x, len(x)).reshape(len(x), len(x))
                    X_B = X_A.T
                    rho = np.exp(
                        -((X_A - X_B) ** 2) / (2 * (spec_resolution / 2.355) ** 2)
                    )  # correlation coefficient = exp(-r^2/2 sigma^2)
                    cov = rho * np.outer(yerr, yerr)
                else:
                    cov = np.diag(yerr.flatten() ** 2)
                X_bin = x.mean()
                Y_bin = weight.T @ y
                Yerr_bin = (weight.T @ cov @ weight) ** 0.5
                i = j
            else:  # if there is only 1 data point in the bin
                X_bin = data[i, 0]
                Y_bin = data[i, 1]
                Yerr_bin = data[i, 2]
                i += 1
            data_bin.append([X_bin, Y_bin, Yerr_bin])
        else:  # if there are less than min_bin data points in the bin
            for t in temp.reshape(len(temp.flatten()) // 3, 3):
                data_bin.append(t)
            i = j
    return np.array(data_bin)


def plot_box_spec(wave: ArrayLike, flux: ArrayLike) -> ArrayLike:
    flux_plot = np.repeat(flux, 2)
    wv_plot = wave.copy()
    wv_plot[:-1] += np.diff(wave) / 2
    wv_plot = np.append(
        wave[0] - (wave[1] - wave[0]) / 2,
        np.append(np.repeat(wv_plot[0:-1], 2), wave[-1] + (wave[-1] - wave[-2]) / 2),
    )

    return wv_plot, flux_plot
