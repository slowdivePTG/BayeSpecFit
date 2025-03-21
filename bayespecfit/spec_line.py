# bayespecfit/spec_line.py

from scipy.optimize import minimize
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import corner

import warnings

from .tools.data_binning import data_binning
from .flux_model import velocity_rf, calc_model_flux

from ._utils import plt

from numpy.typing import ArrayLike
from typing import Optional
from matplotlib.axes import Axes
from arviz.data.inference_data import InferenceData
from pymc.model import Model

SPEED_OF_LIGHT = 2.99792458e5


class SpecLine:
    """amp (series of) absorption line(s) in a 1D optical spectrum

    Methods
    -------
    LS_estimator(guess) :
        Least square point estimation

    MCMC_sampler(vel_mean_mu=[], vel_mean_sig=[],
                 vel_var_lim=[2e1, 1e8],
                 amp_lim=[-1e5, 1e5],
                 sampler='NUTS',
                 nburn=2000,
                 target_accept=0.8,
                 initial=[],
                 plot_structure=False,
                 plot_model=True,
                 plot_mcmc=False) :
        An NUTS sampler based on the package pymc

    plot_model(theta) :
        plot the predicted absorption features

    """

    def __init__(
        self,
        spec: ArrayLike,
        lines: list[list[float]] | list[float],
        line_regions: list[tuple[float, float]] | tuple[float, float],
        lines_id: Optional[list[str]] = None,
        rel_strength: Optional[list[list[float]] | list[float]] = None,
        free_rel_strength: Optional[list[bool]] = None,
        line_model: str = "Gauss",
        mask: Optional[list[tuple, tuple]] = None,
        spec_resolution: float = 0,
    ):
        """Constructor

        Parameters
        ----------
        spec: ArrayLike, shape=(n, 3)
            [wavelength in the host galaxy rest frame, flux, flux_uncertainty]

        z: float, default=0
            host galaxy redshift

        lines: list[list[float]] | list[float]
            the central wavelength(s) [angstrom] of this (series) of line(s)
                1D/2D for single components:
                    Si II: [[6371.359, 6347.103]] or [6371.359, 6347.103]
                2D for multiple components":
                    different vel components of one element:
                        Ca II IRT [[8498.018, 8542.089, 8662.140], [8498.018, 8542.089, 8662.140]]
                    multiple elements:
                        He I/Fe II [[10830], [9998, 10500, 10863]]

        line_regions: list[tuple[float, float]] | tuple[float, float]
            the blue and red edges [angstrom] of each line region to fit

        lines_id : list[str], optional
            the name of each line component

        rel_strength : 2D array_like, optional
            the relative strength between each line in the series
            1D/2D for single component:
                Si II: [] or [[]] - empty for equal strength
            2D for multiple components:
                Ca II IRT [[], []]
                He I/Fe II [[], [0.382, 0.239, 0.172]]

        free_rel_strength : array_like, default=[]
            whether to set the relative strength of each line series as
            another free parameter in MCMC fit

        line_model : string
            ["Gauss", "Lorentz"]

        mask : list[tuple], optional
            line regions to be excluded in the fitting
        """

        self.wv_rf, self.fl, self.fl_unc = spec[:, 0], spec[:, 1], spec[:, 2]
        self.line_model = line_model

        ##################### Initialize the flux data ######################
        # line region
        in_line_region = np.zeros_like(self.wv_rf, dtype=bool)
        for edge in line_regions:
            in_line_region |= (self.wv_rf > edge[0]) & (self.wv_rf < edge[1])
        self.line_regions = line_regions
        wv_line = self.wv_rf[in_line_region]

        # exclude masked regions
        in_unmasked_region = np.ones_like(wv_line, dtype=bool)
        if mask is not None:
            for mk in mask:
                in_unmasked_region &= (wv_line < mk[0]) | (wv_line > mk[1])
            self.mask = mask
        else:
            self.mask = []

        self.wv_line = wv_line[in_unmasked_region]
        self.wv_line_unmasked = wv_line

        # normalized flux
        self._fl_med = np.nanmedian(self.fl[in_line_region][in_unmasked_region])
        fl_norm = self.fl[in_line_region] / self._fl_med
        fl_norm_unc = self.fl_unc[in_line_region] / self._fl_med

        # check if there are points with relative uncertainty
        # two orders of magnitude lower than the median
        rel_unc = fl_norm_unc / fl_norm
        med_rel_unc = np.nanmedian(rel_unc)
        if rel_unc.min() < med_rel_unc / 1e2:
            warnings.warn("Some flux with extremely low uncertainty!")
            rel_unc[rel_unc < med_rel_unc / 1e2] = med_rel_unc
            warnings.warn("New uncertainty assigned!")

        fl_norm_unc = rel_unc * fl_norm

        self.fl_norm = fl_norm[in_unmasked_region]
        self.fl_norm_unmasked = fl_norm

        self.fl_norm_unc = fl_norm_unc[in_unmasked_region]
        self.fl_norm_unc_unmasked = fl_norm_unc

        # specify the spectral resolution in each line region
        # if not provided, assume infinite resolution
        if isinstance(spec_resolution, (int, float, np.number)):
            self.spec_resolution = np.ones(len(self.line_regions)) * spec_resolution
        else:
            self.spec_resolution = spec_resolution
        # spectral resolution to velocity resolution
        self.vel_resolution = []
        # estimated flux at each edge (used as the initial guess & prior)
        self.fl_blue, self.fl_red = np.empty(len(self.line_regions)), np.empty(len(self.line_regions))
        self.blue_fl_unc, self.red_fl_unc = np.empty(len(self.line_regions)), np.empty(len(self.line_regions))
        for k, (edge, spec_res) in enumerate(zip(self.line_regions, self.spec_resolution)):
            vel_res = SPEED_OF_LIGHT * spec_res / 2.355 / ((edge[0] + edge[1]) / 2)
            print(f"Velocity resolution for line region {edge}: {vel_res:.0f} km/s")
            self.vel_resolution.append(vel_res)

            range_l = edge[1] - edge[0]
            delta_l = min(spec_res * 3, range_l / 10)
            fl_blue = self.get_flux_at_lambda(edge[0], delta_l=delta_l) / self._fl_med
            fl_red = self.get_flux_at_lambda(edge[1], delta_l=delta_l) / self._fl_med
            self.fl_blue[k] = fl_blue[0]
            self.fl_red[k] = fl_red[0]
            self.blue_fl_unc[k] = fl_blue[1]
            self.red_fl_unc[k] = fl_red[1]

        ################ Initialize the absorption lines #####################
        # if only one line is provided and the input is a 1D list
        if isinstance(lines[0], (int, float, np.number)):
            lines = [lines]

        # check if the shapes of lines and rel_strength match
        if rel_strength is None:
            rel_strength = [[]] * len(lines)
        elif isinstance(rel_strength[0], (int, float, np.number)):
            rel_strength = [rel_strength]

        if len(rel_strength) != len(lines):
            raise IndexError("The number of lines and their relative strength do not match in shape")
        for k in range(len(lines)):
            # if rel_strength is not provided, assume equal strength
            if len(rel_strength[k]) == 0:
                rel_strength[k] = np.ones_like(lines[k])
            elif len(lines[k]) != len(rel_strength[k]):
                raise IndexError("The number of line components and their relative strength do not match in shape")

        # check if the shapes of lines and free_rel_strength match
        if free_rel_strength is None:
            free_rel_strength = np.zeros(len(lines), dtype=bool)
        elif len(free_rel_strength) != len(lines):
            raise IndexError("The number of lines and the free_rel_strength indicator do not match in shape")

        # reorder the lines based on their relative strength
        self.rel_strength = []
        self.lines = []
        
        for li, rs in zip(lines, rel_strength):
            idx = np.argsort(rs)
            self.lines.append(np.array(li)[idx])
            self.rel_strength.append(np.array(rs)[idx][:-1] / np.max(rs))

        if lines_id is None:
            lines_id = [f"line_{k}" for k in range(len(lines))]
        elif len(lines_id) != len(lines):
            raise IndexError(f"The number of lines ({len(lines)}) and their ID ({len(lines_id)}) do not match in shape")
        self.lines_id = lines_id

        self.free_rel_strength = free_rel_strength

        # initialize the fitting results
        self.theta_LS = None
        self.chi2_LS = None

        self.theta_MCMC = None
        self.sig_theta_MCMC = None

    def LS_estimator(self, guess):
        """Least square point estimation

        Parameters
        ----------
        guess: tuple
            an initial guess for the fitting parameter theta (not including the flux at the blue and red edges)

        plot_model : bool, default=False
            whether to plot the best fit result
        """

        LS_res = minimize(
            neg_lnlike_gaussian_abs,
            guess,
            method="Powell",  # Powell method does not need derivatives
            args=(self),
        )

        theta_cont = []
        for k in range(len(self.line_regions)):
            theta_cont.append(self.fl_blue[k])
            theta_cont.append(self.fl_red[k])

        self.theta_LS = np.append(theta_cont, LS_res["x"])
        self.neg_lnL_LS = LS_res["fun"]

        print("LS estimation:")
        for k in range(len(self.lines)):
            print("Velocity {}: {:.0f} km/s".format(k + 1, self.theta_LS[2 * len(self.line_regions) + 3 * k]))
        # convert amplitude to equivalent width
        # self.EW = 0
        # for k, rs in enumerate(self.rel_strength):
        #     ratio = (
        #         np.sum(rs)
        #         / (self.red_vel - self.blue_vel)
        #         / ((self.fl_red[0] + self.fl_blue[0]) / 2)
        #         * (self.wv_line[-1] - self.wv_line[0])
        #     )
        #     self.EW += self.theta_LS[4 + 3 * k] * -ratio
        # self.sig_EW = np.nan

    def MCMC_sampler(
        self,
        vel_mean_mu: list[float],
        vel_mean_sig: list[float],
        log_vel_sig_mu: list[float],
        log_vel_sig_sig: list[float],
        initial: list[float] = None,
        vel_mean_diff: list[float] = None,
        log_vel_sig_diff: list[float] = None,
        log_vel_sig_min: list = None,
        log_vel_sig_max: list = None,
        log_amp_lim=[0, 5],
        sampler="NUTS",
        nburn=2000,
        target_accept=0.8,
        plot_structure=False,
        plot_mcmc=False,
    ) -> tuple[InferenceData, Model]:
        """MCMC sampler with pymc

        Parameters
        ----------

        vel_mean_mu, vel_mean_sig : list[float]
            means/standard deviations of the velocity priors

        log_vel_sig_mu, log_vel_sig_sig : list[float]
            means/standard deviations of the logarithmic velocity dispersion priors

        initial : array_like, optional
             initial values for the MCMC sampler
                if not provided, the initial values will be set to the least square estimation

        vel_mean_diff : array_like, optional
            standard deviations of the difference between velocity components
            list of tuples - (j, k, v_diff) : Var(v_j - v_k) = v_diff**2

        log_vel_sig_diff : array_like, optional
            standard deviations of the difference between the logarithmic velocity dispersions
            list of tuples - (j, k, log_v_sig_diff) : Var(log_v_sig_j - log_v_sig_k) = log_v_sig_diff**2

        log_vel_sig_min, log_vel_sig_max : array_like, optional
            minimum/maximum logarithmic velocity dispersions
                if not None, a softplux function will be added the posterior to punish velocity
                dispersions greater than this value

        log_amp_lim : float, default=[-1e5, 1e5]
            allowed range of the log_amplitude

        sampler : ['NUTS', 'MH'], default='NUTS'
            amp step function or collection of functions
                'NUTS' : The No-U-Turn Sampler
                'MH' : Metropolis–Hastings Sampler

        nburn : int, default=2000
            number of "burn-in" steps for the MCMC chains

        target_accept : float in [0, 1], default=0.8
            the step size is tuned such that we approximate this acceptance rate
            higher values like 0.9 or 0.95 often work better for problematic posteriors

        plot_mcmc : bool, default=False
            whether to plot the MCMC chains and corner plots

        Returns
        -------
        trace : arviz.data.inference_data.InferenceData
            the samples drawn by the NUTS sampler
        Profile : pymc.model.Model
            the Bayesian model
        ax : matplotlib.axes
            the axes with the plot
        """

        n_line_regions = len(self.line_regions)
        n_lines = len(self.lines)

        with pm.Model() as Profile:
            # continuum fitting
            # model flux at the blue edge
            fl1 = pm.TruncatedNormal(
                "fl_blue",
                mu=self.fl_blue,
                sigma=self.blue_fl_unc,
                lower=self.fl_blue - self.blue_fl_unc * 2,
                upper=self.fl_blue + self.blue_fl_unc * 2,
            )
            # model flux at the red edge
            fl2 = pm.TruncatedNormal(
                "fl_red",
                mu=self.fl_red,
                sigma=self.red_fl_unc,
                lower=self.fl_red - self.red_fl_unc * 2,
                upper=self.fl_red + self.red_fl_unc * 2,
            )

            # absorption profile
            # amplitude
            log_amp = pm.Uniform("log_amp", lower=log_amp_lim[0], upper=log_amp_lim[1], shape=(n_lines,))
            amp = pm.Deterministic("amp", 10**log_amp)

            if (len(vel_mean_mu) == n_lines) and (len(log_vel_sig_mu) == n_lines):
                # optional priors
                vel_mean_diff = [] if vel_mean_diff is None else vel_mean_diff
                log_vel_sig_diff = [] if log_vel_sig_diff is None else log_vel_sig_diff

                # mean velocity
                vel_mean_cov = np.diag(vel_mean_sig) ** 2  # covariance matrix
                for j, k, mean_diff in vel_mean_diff:
                    vel_mean_cov[j, k] = vel_mean_cov[k, j] = (
                        vel_mean_sig[j] ** 2 + vel_mean_sig[k] ** 2 - mean_diff**2
                    ) / 2
                if np.any(np.linalg.eigvals(vel_mean_cov) < 0):
                    raise ValueError("Covariance matrix not positive semi-definite!")
                v_mean = pm.MvNormal("v_mean", mu=vel_mean_mu, cov=vel_mean_cov)

                # velocity dispersion
                log_vel_sig_cov = np.diag(log_vel_sig_sig) ** 2  # covariance matrix
                for j, k, log_sig_diff in log_vel_sig_diff:
                    log_vel_sig_cov[j, k] = log_vel_sig_cov[k, j] = (
                        log_vel_sig_sig[j] ** 2 + log_vel_sig_sig[k] ** 2 - log_sig_diff**2
                    ) / 2
                log_v_sig = pm.MvNormal(
                    "log_v_sig",
                    mu=log_vel_sig_mu,
                    cov=log_vel_sig_cov,
                )

                # velocity dispersion limits
                if log_vel_sig_min is not None:
                    if len(log_vel_sig_min) != len(log_vel_sig_mu):
                        raise IndexError(
                            f"The number of the velocity priors {len(log_vel_sig_min)} does not match the number of lines {len(log_vel_sig_mu)}"
                        )
                    print("There is a vel_sig_min lim...")
                    pm.Potential(
                        "vel_sig_min_lim",
                        -pm.math.log1pexp(-(log_v_sig - log_vel_sig_min) * 5**2),
                    )
                if log_vel_sig_max is not None:
                    if len(log_vel_sig_max) != len(log_vel_sig_mu):
                        raise IndexError(
                            f"The number of the velocity priors {len(log_vel_sig_max)} does not match the number of lines {len(log_vel_sig_mu)}"
                        )
                    print("There is a vel_sig_max lim...")
                    pm.Potential(
                        "vel_sig_max_lim",
                        -pm.math.log1pexp((log_v_sig - log_vel_sig_max) * 5**2),
                    )
            else:
                raise IndexError("The number of the velocity priors does not match the number of lines")

            pm.Deterministic("v_sig", 10**log_v_sig)
            theta = []
            for k in range(n_line_regions):
                theta += [fl1[k], fl2[k]]
            for k in range(n_lines):
                theta += [v_mean[k], log_v_sig[k], amp[k]]
            # relative intensity of lines
            rel_strength = []
            ratio_index = []
            for k, free in enumerate(self.free_rel_strength):
                if free:
                    ratio_index.append(k)
                    log_ratio_0 = np.log10(self.rel_strength[k])
                    # weaker lines are always weaker (log ratio <= 0)
                    log_rel_strength_k = pm.TruncatedNormal(f"log_ratio_{k}", mu=log_ratio_0, sigma=1, upper=0)
                    rel_strength.append(pm.Deterministic(f"ratio_{k}", 10**log_rel_strength_k))
                else:
                    # rel_strength.append(pm.Deterministic(f"ratio_{k}", pt.as_tensor_variable(self.rel_strength[k])))
                    rel_strength.append(self.rel_strength[k])
                # equivalent width
                # EW_k = pm.Deterministic(
                #     f"EW_{k}",
                #     -amp[k]
                #     / (self.red_vel - self.blue_vel)
                #     / ((fl1 + fl2) / 2)
                #     * (self.wv_line[-1] - self.wv_line[0])
                #     * pm.math.sum(rel_strength[k]),
                # )

            # flux expectation
            mu = pm.Deterministic(
                "mu",
                calc_model_flux(
                    fl_blue=fl1,
                    fl_red=fl2,
                    vel_mean=v_mean,
                    log_vel_sig=log_v_sig,
                    log_amp=log_amp,
                    wv_rf=self.wv_line,
                    lines=self.lines,
                    rel_strength=rel_strength,
                    line_regions=self.line_regions,
                    vel_resolution=self.vel_resolution,
                    model=self.line_model,
                ),
            )

            # uncertainty normalization
            # typical_unc = np.median(self.fl_norm_unc)
            # sigma_0 = pm.HalfCauchy("sigma_0", beta=typical_unc)
            # sigma = pm.Deterministic("sigma", (sigma_0**2 + self.fl_norm_unc**2) ** 0.5)

            sigma = self.fl_norm_unc

            pm.Normal("Flux", mu=mu, sigma=sigma, observed=self.fl_norm)

        if plot_structure:
            pm.model_to_graphviz(Profile)
            # return Profile

        # initialization
        if initial is None:
            start = None
        else:
            initial = initial[:2 * n_line_regions + 3 * n_lines]
            start = {}
            start["fl_blue"] = self.fl_blue
            start["fl_red"] = self.fl_red
            start["v_mean"] = initial[2 * len(self.line_regions) :: 3]
            start["log_v_sig"] = initial[2 * len(self.line_regions) + 1 :: 3]
            start["amp"] = initial[2 * len(self.line_regions) + 2 :: 3]
            # start["sigma_0"] = 1e-3
            for k, free in enumerate(self.free_rel_strength):
                if free:
                    start[f"ratio_{k}"] = self.rel_strength[k]
                    start[f"log_ratio_{k}"] = np.log10(self.rel_strength[k])

        with Profile:
            if sampler == "NUTS":
                trace = pm.sample(
                    return_inferencedata=True,
                    initvals=start,
                    target_accept=target_accept,
                    tune=nburn,
                )
            elif sampler == "MH":
                trace = pm.sample(
                    return_inferencedata=True,
                    initvals=start,
                    step=pm.Metropolis(),
                    tune=nburn,
                )
        self.trace = trace
        var_names_summary = ["v_mean", "v_sig", "amp"]
        for k in ratio_index:
            var_names_summary.append(f"log_ratio_{k}")
        # for k in range(n_lines):
        #     var_names_summary.append(f"EW_{k}")
        summary = az.summary(
            trace,
            var_names=var_names_summary,
            stat_focus="mean",
            round_to=3,
            hdi_prob=0.68,
        )
        print(summary)

        all = az.summary(trace, kind="stats")

        theta = []
        sig_theta = []
        self.EW = []
        self.sig_EW = []
        for k in range(n_line_regions):
            theta.append(all["mean"][f"fl_blue[{k}]"])
            theta.append(all["mean"][f"fl_red[{k}]"])
            sig_theta.append(all["sd"][f"fl_blue[{k}]"])
            sig_theta.append(all["sd"][f"fl_red[{k}]"])
        for k in range(n_lines):
            theta.append(all["mean"][f"v_mean[{k}]"])
            theta.append(all["mean"][f"log_v_sig[{k}]"])
            theta.append(all["mean"][f"log_amp[{k}]"])
            # self.EW.append(all["mean"][f"EW_{k}"])
            # self.sig_EW.append(all["sd"][f"EW_{k}"])

            sig_theta.append(all["sd"][f"v_mean[{k}]"])
            sig_theta.append(all["sd"][f"log_v_sig[{k}]"])
            sig_theta.append(all["sd"][f"log_amp[{k}]"])
        for j in ratio_index:
            for k in range(len(self.lines[j]) - 1):
                if f"ratio_{j}[{k}]" in all["mean"].index:
                    theta.append(all["mean"][f"log_ratio_{j}[{k}]"])
                    sig_theta.append(all["sd"][f"log_ratio_{j}[{k}]"])
        self.theta_MCMC = theta
        self.sig_theta_MCMC = sig_theta

        # find the maximum a posteriori point
        neg_log_posterior = -np.array(trace.sample_stats.lp)
        ind = np.unravel_index(np.argmin(neg_log_posterior, axis=None), neg_log_posterior.shape)
        theta_MAP = []
        for k in range(n_line_regions):
            theta_MAP.append(np.array(trace.posterior["fl_blue"])[ind][k])
            theta_MAP.append(np.array(trace.posterior["fl_red"])[ind][k])
        
        for k in range(n_lines):
            theta_MAP.append(np.array(trace.posterior[f"v_mean"])[ind][k])
            theta_MAP.append(np.array(trace.posterior[f"log_v_sig"])[ind][k])
            theta_MAP.append(np.array(trace.posterior[f"log_amp"])[ind][k])

        for j in ratio_index:
            for k in range(len(self.lines[j]) - 1):
                theta_MAP.append(np.array(trace.posterior[f"log_ratio_{j}"])[ind][k])
        self.theta_MAP = theta_MAP

        if plot_mcmc:
            # by default, show the mean velocity, velocity dispersion, pseudo-EW, and line ratios
            var_names_plot = ["v_mean", "v_sig"]
            # for k in range(n_lines):
            #     var_names_plot.append(f"EW_{k}")
            for k in ratio_index:
                var_names_plot.append(f"ratio_{k}")
            corner.corner(trace, var_names=var_names_plot)

        return trace, Profile

    def plot_model(
        self,
        theta: list,
        lambda_0: Optional[float | list[float]] = None,
        return_ax: bool = False,
        ax: Optional[Axes | list[Axes]] = None,
        bin: bool = True,
        bin_size: Optional[int] = None,
    ) -> Optional[Axes | list[Axes]]:
        """plot the predicted absorption features

        Parameters
        ----------
        theta : array_like
            fitting parameters: flux at the blue edge, flux at the
            red edge, (mean of relative velocity, log standard deviation,
            amplitude) * Number of velocity components

        return_ax : boolean, default=False
            whether to return the axes
            if return_ax == True, a matplotlib axes will be returned

        ax : matplotlib axes
            if it is not None, plot on it

        bin : bool, default=False
            whether to bin the spectrum (for visualization)

        bin_size : int, default=None
            wavelength bin size (km s^-1)
        """

        n_line_regions = len(self.line_regions)

        if lambda_0 is None:
            warnings.warn("No reference wavelength is provided. Using the red edge of each line region.")
            lambda_0 = [edge[-1] for edge in self.line_regions]
        elif isinstance(lambda_0, (int, float, np.number)):
            lambda_0 = [lambda_0]
        if len(lambda_0) != n_line_regions:
            raise IndexError(
                f"The number of reference wavelengths ({len(lambda_0)}) and line regions ({n_line_regions}) do not match"
            )

        if ax == None:
            _, ax = plt.subplots(1, n_line_regions, figsize=(6 * n_line_regions, 6), constrained_layout=True)
        ax = np.atleast_1d(ax)
        if len(ax) != n_line_regions:
            raise IndexError("The number of axes and line regions do not match")

        params = self.decode_theta(theta)
        log_ratio = params.pop("log_ratio")
        rel_strength = self.get_rel_strength(log_ratio=log_ratio)
        model_flux = calc_model_flux(
            **params,
            wv_rf=self.wv_line,
            lines=self.lines,
            rel_strength=rel_strength,
            line_regions=self.line_regions,
            vel_resolution=self.vel_resolution,
            model=self.line_model,
        )

        model_flux_elem = []
        for k in range(len(self.lines)):
            theta_elem = np.append(
                    theta[: 2 * n_line_regions],
                    theta[2 * n_line_regions + 3 * k : 2 * n_line_regions + 3 * (k + 1)],
                )
            params_elem = self.decode_theta(theta_elem)
            params_elem.pop("log_ratio")

            model_flux_elem.append(
                calc_model_flux(
                    **params_elem,
                    wv_rf=self.wv_line,
                    lines=[self.lines[k]],
                    rel_strength=[rel_strength[k]],
                    line_regions=self.line_regions,
                    vel_resolution=self.vel_resolution,
                    model=self.line_model,
                )
            )

        spec = np.array([self.wv_line_unmasked, self.fl_norm_unmasked, self.fl_norm_unc_unmasked]).T

        for k, edges in enumerate(self.line_regions):
            # trim the spectrum
            in_line_region = (spec[:, 0] > edges[0]) & (spec[:, 0] < edges[1])
            spec_plot = spec[in_line_region]
            vel_rf = velocity_rf(spec_plot[:, 0], lambda_0[k])
            spec_resolution = self.spec_resolution[k]

            # plot the model
            model_plot = ax[k].plot(vel_rf, model_flux[in_line_region], linewidth=5, color="k")

            # bin the spectrum for visualization purposes
            if bin:
                if bin_size == None:
                    bin_size = spec_resolution
                print("binning spectrum for visualization...")
                print(
                    f"bin size: {bin_size:.0f} Ang = {bin_size / ((edges[0] + edges[1])/2) * SPEED_OF_LIGHT:.0f} km/s"
                )
                spec_plot = data_binning(
                    spec_plot,
                    size=bin_size,
                    spec_resolution=spec_resolution,
                    sigma_clip=2,
                )

            vel_rf_plot = velocity_rf(spec_plot[:, 0], lambda_0[k])

            # plot the observations
            ax[k].errorbar(
                vel_rf_plot,
                spec_plot[:, 1],
                yerr=spec_plot[:, 2],
                alpha=0.5,
                elinewidth=0.5,
                marker="o",
                zorder=-100,
            )

            # plot the residuals
            model_res = (
                calc_model_flux(
                    **params,
                    wv_rf=spec_plot[:, 0],
                    lines=self.lines,
                    rel_strength=rel_strength,
                    line_regions=self.line_regions,
                    vel_resolution=self.vel_resolution,
                    model=self.line_model,
                )
                - spec_plot[:, 1]
            )
            ax[k].plot(vel_rf_plot, model_res, color="grey")

            # plot the edges of the line region
            ax[k].errorbar(
                [vel_rf[0], vel_rf[-1]],
                [theta[2 * k], theta[2 * k + 1]],
                yerr=[self.blue_fl_unc[k], self.red_fl_unc[k]],
                color=model_plot[0].get_color(),
                fmt="s",
                markerfacecolor="w",
                capsize=5,
            )

            # if there are multiple lines, plot each of them
            if len(self.lines) > 1:
                colors_elem = [
                    "#66c2a5",
                    "#fc8d62",
                    "#8da0cb",
                    "#e78ac3",
                    "#a6d854",
                    "#ffd92f",
                    "#e5c494",
                ]
                for l in range(len(self.lines)):
                    ax[k].plot(
                        vel_rf,
                        model_flux_elem[l][in_line_region],
                        linewidth=2,
                        label=self.lines_id[l],
                        color=colors_elem[l % len(colors_elem)],
                    )

                ax[0].legend()

            ax[k].set_xlabel(r"$v\ [\mathrm{km/s}]$")
        ax[0].set_ylabel(r"$\mathrm{Normalized\ Flux}$")

        # mask
        for mk in self.mask:
            # find out in which line region the mask is
            idx = int(np.where([mk[0] < edge[1] and mk[1] > edge[0] for edge in self.line_regions])[0][0])
            v_mk_1 = velocity_rf(mk[0], lambda_0[idx])
            v_mk_2 = velocity_rf(mk[1], lambda_0[idx])
            ax[idx].axvspan(v_mk_1, v_mk_2, color="0.8", alpha=0.5)

        if return_ax:
            return ax
        else:
            plt.show()

    def get_flux_at_lambda(self, lambda_0, delta_l=None):
        """Get the flux and uncertainty at some given wavelength

        Returns the mean and uncertainty of flux in the
        wavelength range: [lambda_0 - delta_l, lambda_0 + delta_l]

        Parameters
        ----------
        lambda_0 : float
            the central wavelength [angstrom]

        delta_l : float, default=50
            the size of the wavelength interval [angstrom]

        Returns
        -------
        mean : float
            mean flux around the central wavelength

        std : float
            multiple measurements:
                standard deviation in flux around the central wavelength
            single measurement:
                flux uncertainty given
        """

        if delta_l == None:
            delta_l = self.spec_resolution * 3
        region = np.where(np.abs(self.wv_rf - lambda_0) < delta_l)[0]
        try:
            if len(region) == 0:
                raise IndexError("No data within this range!")
            elif len(region) == 1:
                warnings.warn("Too few points within the wavelength range!")
                return (self.fl[region[0]], self.fl_unc[region[0]])
            else:
                mean = np.sum(self.fl[region] / self.fl_unc[region] ** 2) / np.sum(1 / self.fl_unc[region] ** 2)
                from astropy.stats import mad_std

                std = mad_std(self.fl[region])
                if len(region) <= 5:
                    warnings.warn("<=5 points within the wavelength range!")
                    std = np.nanmedian(self.fl_unc[region])

                # std = np.nanmin(self.fl_unc[region])
                # std = np.std(self.fl[region], ddof=1)
            return (mean, std)
        except IndexError as e:
            repr(e)
            return None, None

    def get_rel_strength(self, log_ratio: Optional[list] = None) -> list[list[float]]:
        """Get the relative strength of the absorption lines"""

        rel_strength = self.rel_strength.copy()

        # return the relative strength if not set free
        if log_ratio is None:
            return rel_strength

        # update the relative strength if set free
        idx_rel = 0
        for k, rel in enumerate(self.free_rel_strength):
            if rel:
                for rel_s in range(len(self.rel_strength[k])):
                    rel_strength[k][rel_s] = 10 ** log_ratio[idx_rel]
                    idx_rel += 1
        if idx_rel != len(log_ratio):
            raise IndexError(f"Number of free parameters ({len(log_ratio)}) and relative strength ({idx_rel}) do not match")
        return rel_strength

    def decode_theta(self, theta: list) -> dict:
        """Decode the fitting parameters"""
        n_cont = 2 * len(self.line_regions)
        n_lines = 3 * len(self.lines)

        fl_blue = theta[0:n_cont:2]
        fl_red = theta[1:n_cont:2]

        vel_mean = theta[n_cont : n_cont + n_lines : 3]
        log_vel_sig = theta[n_cont + 1 : n_cont + n_lines : 3]
        log_amp = theta[n_cont + 2 : n_cont + n_lines : 3]

        log_ratio = theta[n_cont + n_lines :]

        return dict(
            fl_blue=fl_blue,
            fl_red=fl_red,
            vel_mean=vel_mean,
            log_vel_sig=log_vel_sig,
            log_amp=log_amp,
            log_ratio=log_ratio,
        )


###################### Likelihood ##########################


def lnlike_gaussian_abs(theta, spec_line: SpecLine) -> float:
    """Log likelihood function assuming Gaussian profile

    Parameters
    ----------
    theta : array_like
        fitting parameters:
            (flux at the blue edge, flux at the red edge) * Number of the line regions,
            (mean of relative velocity, log standard deviation, amplitude) * Number of velocity components,
            log10 line ratio for each velocity components (if set free)

    spec_line : sn_line_vel.SpecLine.SpecLine
        the SpecLine object

    Returns
    -------
    lnl : float
        the log likelihood function
    """

    theta_cont = []
    for k in range(len(spec_line.line_regions)):
        theta_cont.append(spec_line.fl_blue[k])
        theta_cont.append(spec_line.fl_red[k])
    theta0 = np.append(theta_cont, theta)

    params = spec_line.decode_theta(theta0)
    log_ratio = params.pop("log_ratio")

    rel_strength = spec_line.get_rel_strength(log_ratio=log_ratio)

    model_flux = calc_model_flux(
        **params,
        wv_rf=spec_line.wv_line,
        lines=spec_line.lines,
        rel_strength=rel_strength,
        line_regions=spec_line.line_regions,
        vel_resolution=spec_line.vel_resolution,
        model=spec_line.line_model,
    )
    lnl = (
        -0.5 * len(model_flux) * np.log(2 * np.pi)
        - np.sum(np.log(spec_line.fl_norm_unc))
        - 0.5 * np.sum((spec_line.fl_norm - model_flux) ** 2 / spec_line.fl_norm_unc**2)
    )

    return lnl


def neg_lnlike_gaussian_abs(theta, spec_line):
    """negative log-likelihood function"""

    lnl = lnlike_gaussian_abs(theta, spec_line)
    return -lnl

# Add a debugging function to check shapes
def debug_shapes(*args, names=None):
    """Print shapes of all arguments to help debug shape issues."""
    if names is None:
        names = [f"arg{i}" for i in range(len(args))]
    
    for name, arg in zip(names, args):
        if isinstance(arg, (pt.TensorVariable, pt.TensorConstant)):
            print(f"{name}: TensorVariable with shape {arg.shape.eval() if hasattr(arg.shape, 'eval') else arg.shape}")
        elif isinstance(arg, np.ndarray):
            print(f"{name}: NumPy array with shape {arg.shape}")
        elif hasattr(arg, '__len__') and not isinstance(arg, str):
            print(f"{name}: Sequence with length {len(arg)}")
        else:
            print(f"{name}: Scalar or other type")