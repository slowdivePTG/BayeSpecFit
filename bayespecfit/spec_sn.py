# bayespecfit/spec_sn.py

import pandas as pd
import numpy as np
import warnings

from .spec_line import SpecLine
from .tools.dust_extinction import calALambda

from ._utils import plt

from numpy.typing import ArrayLike
from typing import Optional

##################### SpectrumSN class ##########################


class SpectrumSN:
    """1D optical spectrum

    Attributes
    ----------

    fl : array_like
        flux (in arbitrary units)

    wv_rf : array_like
        wavelength [angstrom] in the host galaxy's rest frame:
        wv_rf = wavelength / (1 + z)

    fl_unc : array_like
        uncertainty in flux (in arbitrary units)

    line : dict
        a dictionary of various lines (SpecLine objects)

    Methods
    -------
    add_line(name, blue_edge, red_edge, lines=[])) :
        Add one (series of) absorption line(s)

    plot_line_region(blue_edge, red_edge) :
        plot the spectrum in the line region

    get_flux_at_lambda(lambda_0, delta_l=50) :
        pget the flux and its uncertainty at some given wavelength
    """

    def __init__(
        self,
        spec1D: str,
        z: float = 0,
        ebv: float = 0,
        snr: Optional[float] = None,
        spec_resolution: Optional[float] = None,
        force_pos_flux: bool = False,
    ):
        """Constructor

        Parameters
        ----------
        spec1D : str
            the spectrum file (directory + filename)
            the inputs should include 3 columns (wavelengths, fluxes, flux uncertainties)
            if only the first two columns are provided, the uncertainties will be estimated with an S/N

        z : float (default=0)
            host galaxy redshift

        ebv : float (default=0)
            E(B-V), Galactic reddening

        snr : float, default=None
            the assigned S/N for spectra with no flux errors

        spec_resolution : float, default=5
            the spectral resolution of the spectrum in Angstrom

        force_pos_flux : bool, default=False
            remove all the non-positive flux measurements
        """

        spec_df = pd.read_csv(spec1D, comment="#", delim_whitespace=True, header=None)

        wv = spec_df[0].values
        wv_rf = wv / (1 + z)
        ALambda = calALambda(wv, RV=3.1, EBV=ebv)
        fl = spec_df[1].values * 10 ** (0.4 * ALambda)

        if snr == None:
            try:
                fl_unc = spec_df[2].values * 10 ** (0.4 * ALambda)
            except:
                warnings.warn("No flux uncertainty in the datafile!")
                warnings.warn(f"Please assign the S/N manually.")
        else:
            # the same uncertainty is assigned to all the flux measurements
            warnings.warn("snr = {:.1f} assigned.".format(snr))
            fl_unc = (
                np.ones_like(fl)
                * (np.nanmedian(fl) / snr)
                * (np.where(fl > np.nanmedian(fl) / 10, fl, np.nanmedian(fl) / 10) / np.nanmedian(fl)) ** -0.5
            )

        # make sure flux measurements are positive
        pos_flux = (fl > 0) | (force_pos_flux)
        self.fl = fl[pos_flux]
        self.wv_rf = wv_rf[pos_flux]
        self.fl_unc = fl_unc[pos_flux]

        # self.snr = snr
        if spec_resolution == None:
            warnings.warn("The spectral resolution is not provided: assume infinite resolution.")
            self.spec_resolution = 0
        else:
            self.spec_resolution = spec_resolution

        self.line: dict[str, SpecLine] = {}

    def add_line(
        self,
        name: str,
        lines: list[list[float]] | list[float],
        line_regions: list[tuple[float, float]],
        lines_id: Optional[list[int]] = None,
        rel_strength: Optional[list[list[float]] | list[float]] = None,
        line_model: str = "Gauss",
        mask: Optional[list[tuple, tuple]] = None,
        plot_region: bool = False,
    ):
        """Add one (series of) absorption line(s)

        Construct a new AbsorbLine object, and save it in self.line

        Parameters
        ----------
        name : str
            the name of the absorption line

        lines : list
            wavelength of each absorption line

        line_regions : list
            the blue and red edges of each line region to fit

        lines_id : list, default=None
            the ID of each line in the series

        rel_strength : list, default=None
            the relative strength between each line in the series
            default: all lines are of the equal strength

        line_model : str, default="Gauss"
            the line profile model: "Gauss" or "Lorentzian"

        mask : list, default=None
            the mask to exclude certain regions in the line fitting

        plot_region : bool, default=False
            whether to plot the line region
        """

        self.line[name] = SpecLine(
            np.array([self.wv_rf, self.fl, self.fl_unc]).T,
            lines=lines,
            line_regions=line_regions,
            lines_id=lines_id,
            rel_strength=rel_strength,
            line_model=line_model,
            spec_resolution=self.spec_resolution,
            mask=mask,
        )
        if plot_region:
            self.plot_line_region(line_regions)
            plt.show()

    def plot_line_region(self, blue_edge, red_edge):
        """plot the spectrum in the line region

        Parameters
        ----------
        blue_edge, red_edge : float
            the wavelength [angstrom] (host galaxy frame)
            at the blue/red edge

        Returns
        -------
        ax : matplotlib.axes
            the axes
        """
        line_region = np.where((self.wv_rf < red_edge) & (self.wv_rf > blue_edge))[0]

        _, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.plot(self.wv_rf[line_region], self.fl[line_region], color="0.5", lw=2)
        ax.errorbar(
            self.wv_rf[line_region],
            self.fl[line_region],
            yerr=self.fl_unc[line_region],
            fmt="o",
            capsize=2,
            elinewidth=1,
        )
        ax.set_xlabel(r"$\mathrm{Wavelength}\ [\mathrm{\r{A}}]$")
        ax.set_ylabel(r"$\mathrm{Flux}$")
        return ax
