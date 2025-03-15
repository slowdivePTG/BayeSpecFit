# bayespecfit/spec_line.py

from scipy.optimize import minimize
import numpy as np
import warnings

from .tools.data_binning import data_binning
from .flux_model import velocity_rf, calc_model_flux

from ._utils import plt


class SpecLine(object):
    """A (series of) absorption line(s) in a 1D optical spectrum

    Attributes
    ----------

    wv_line : array_like
        the wavelength range [angstrom] of the line
        (host galaxy frame)

    rel_strength : array_like, default=[]
        the relative strength between each line in the series
        rel_strength = []: all lines are of the equal strength

    free_rel_strength : array_like, default=[]
        whether to set the relative strength of each line series as
        another free parameter in MCMC fit

    lambda_0 : float
        wavelength as a reference for velocity

    norm_fl : array_like
        flux normalized by the median value within the line region

    norm_fl_unc : array_like
        normalized flux uncertainty within the line region

    blue_fl, red_fl : list
        normalized flux and uncertainty at the blue/red edge

    vel_rf : array_like
        relative velocities of each point with respect
        to the center of absorption

    blue_vel, red_vel : float
        relative velocity at the blue/red edge

    lines : 2D array_like, default=[]
            wavelength of each absorption line

    theta_LS : array_like
        best fit with least square methods, powered by
        scipy.optimize.minimize()

    neg_lnL_LS : float
        the minimized negative log-likelihood with least square methods

    chi2_MCMC : float
        the minimized residual with MCMC

    theta_MCMC : array_like
        best fit with an MCMC sampler, approximated by median
        values in the MCMC chains

    sig_theta_MCMC : array_like
        1 sigma uncertainty for each parameter, approximated
        by the half the range of 16 and 84 percentile values
        in the MCMC chains

    EW : float
        effective width [angstrom]

    sig_EW : float
        uncertainty in effective width [angstrom]

    Methods
    -------
    LS_estimator(guess) :
        Least square point estimation

    MCMC_sampler(vel_mean_mu=[], vel_mean_sig=[],
                 vel_var_lim=[2e1, 1e8],
                 A_lim=[-1e5, 1e5],
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
        spec,
        spec_resolution,
        blue_edge=-np.inf,
        red_edge=np.inf,
        lines=[],
        rel_strength=[],
        free_rel_strength=[],
        line_model="Gauss",
        mask=[],
    ):
        """Constructor

        Parameters
        ----------
        spec: array-like
            [wv_rf, fl, fl_unc]

        z: float, default=0
            host galaxy redshift

        blue_edge, red_edge: float
            the wavelength [angstrom] (host galaxy frame) at the blue/red edge

        lines: 2D array_like, default=[]
            the central wavelength(s) [angstrom] of this (series) of line(s)
            1D/2D for single components:
                Si II: [[6371.359, 6347.103]] or [6371.359, 6347.103]
            2D for multiple components":
                different vel components of one element:
                    Ca II IRT [[8498.018, 8542.089, 8662.140], [8498.018, 8542.089, 8662.140]]
                multiple elements:
                    He I/Fe II [[10830], [9998, 10500, 10863]]

        rel_strength : 2D array_like, default=[]
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

        mask : array of tuples, default=[]
            line regions to be masked in the fit
        """

        # if bin:
        #     print("binning spectrum...")
        #     dat = data_binning(spec, size=bin_size, spec_resolution=spec_resolution)
        # else:
        dat = spec
        self.wv_rf, self.fl, self.fl_unc = dat[:, 0], dat[:, 1], dat[:, 2]
        self.spec_resolution = spec_resolution
        self.line_model = line_model

        # line region
        line_region = (self.wv_rf < red_edge) & (self.wv_rf > blue_edge)
        self.wv_line = self.wv_rf[line_region]

        # normalized flux
        norm_fl = self.fl[line_region] / np.nanmedian(self.fl[line_region])
        norm_fl_unc = self.fl_unc[line_region] / np.nanmedian(self.fl[line_region])

        # check if there are points with relative uncertainty
        # two orders of magnitude lower than the median
        rel_unc = norm_fl_unc / norm_fl
        med_rel_unc = np.nanmedian(rel_unc)
        if rel_unc.min() < med_rel_unc / 1e2:
            warnings.warn("Some flux with extremely low uncertainty!")
            warnings.warn("New uncertainty assigned!")
        rel_unc[rel_unc < med_rel_unc / 1e2] = med_rel_unc
        norm_fl_unc = rel_unc * norm_fl

        # mask certain regions
        self.mask = mask
        line_region_masked = np.ones_like(self.wv_line, dtype=bool)
        for mk in mask:
            line_region_masked &= (self.wv_line < mk[0]) | (self.wv_line > mk[1])
        self.wv_line_masked = self.wv_line[line_region_masked]

        self.norm_fl = norm_fl[line_region_masked]
        self.norm_fl_unmasked = norm_fl

        self.norm_fl_unc = norm_fl_unc[line_region_masked]
        self.norm_fl_unc_unmasked = norm_fl_unc

        # calculate the covariance matrix & its determinant
        # wv_A = np.repeat(self.wv_line_masked, len(self.wv_line_masked)).reshape(
        #     len(self.wv_line_masked), -1
        # )
        # wv_B = wv_A.T
        # if self.spec_resolution > 0:
        #     rho = np.exp(
        #         -((wv_A - wv_B) ** 2) / (2 * (self.spec_resolution / 2.355 / 2) ** 2)
        #     )
        # else:
        #     rho = np.diag(np.ones_like(self.norm_fl_unc))
        # self.norm_fl_cov = np.outer(self.norm_fl_unc, self.norm_fl_unc) * rho

        # flux at each edge
        range_l = red_edge - blue_edge
        delta_l = min(self.spec_resolution * 3, range_l / 10)
        blue_fl = self.get_flux_at_lambda(blue_edge, delta_l=delta_l)
        red_fl = self.get_flux_at_lambda(red_edge, delta_l=delta_l)
        self.blue_fl = blue_fl / np.nanmedian(self.fl[line_region])
        self.red_fl = red_fl / np.nanmedian(self.fl[line_region])

        # velocity
        try:
            if len(lines[0]) > 0:
                pass
        except:
            lines = np.atleast_2d(lines)
            rel_strength = np.atleast_2d(rel_strength)

        self.rel_strength = []
        self.lines = []

        for k in range(len(lines)):
            if len(rel_strength[k]) == 0:
                rs = np.ones_like(lines[k])
            else:
                rs = rel_strength[k]
            li = np.array(lines[k])[np.argsort(rs)]
            rs = np.sort(rs) / np.max(rs)
            self.lines.append(li)
            self.rel_strength.append(rs)

        if len(free_rel_strength) == 0:
            free_rel_strength = np.array([False] * len(self.rel_strength))
        self.free_rel_strength = free_rel_strength

        self.lambda_0 = self.lines[0][-1]
        vel_rf = velocity_rf(self.wv_rf, self.lambda_0)
        self.vel_rf_unmasked = vel_rf[line_region]
        self.vel_rf = vel_rf[line_region][line_region_masked]

        self.blue_vel = velocity_rf(blue_edge, self.lambda_0)
        self.red_vel = velocity_rf(red_edge, self.lambda_0)

        self.vel_resolution = (
            2.99792458e5 * self.spec_resolution / 2.355 / ((blue_edge + red_edge) / 2)
        )  # spectral resolution in terms of velocity (FWHM = 2.355 sigma for Gaussian dist.)

        self.theta_LS = []
        self.chi2_LS = np.nan

        self.theta_MCMC = []
        self.sig_theta_MCMC = []

    def LS_estimator(self, guess, plot_model=False):
        """Least square point estimation

        Parameters
        ----------
        guess: tuple, default=(1, 1, -10000, 15, -1000)
            an initial guess for the fitting parameter theta

        plot_model : bool, default=False
            whether to plot the best fit result
        """

        LS_res = minimize(
            neg_lnlike_gaussian_abs,
            guess,
            method="Powell",  # Powell method does not need derivatives
            args=(self),
        )

        self.theta_LS = LS_res["x"]
        ndim = len(self.theta_LS)
        self.neg_lnL_LS = LS_res["fun"]

        ndim = len(self.theta_LS)
        if plot_model:
            self.plot_model(self.theta_LS)

        print("LS estimation:")
        for k in range(ndim // 3):
            print("Velocity {}: {:.0f} km/s".format(k + 1, self.theta_LS[2 + 3 * k]))
        # convert amplitude to equivalent width
        self.EW = 0
        for k, rs in enumerate(self.rel_strength):
            ratio = (
                np.sum(rs)
                / (self.red_vel - self.blue_vel)
                / ((self.red_fl[0] + self.blue_fl[0]) / 2)
                * (self.wv_line[-1] - self.wv_line[0])
            )
            self.EW += self.theta_LS[4 + 3 * k] * -ratio
        self.sig_EW = np.nan

    def MCMC_sampler(
        self,
        initial=[],
        vel_mean_mu=[],
        vel_mean_sig=[],
        vel_mean_diff=[],
        ln_vel_sig_mu=[],
        ln_vel_sig_sig=[],
        ln_vel_sig_diff=[],
        ln_vel_sig_min=[],
        ln_vel_sig_max=[],
        A_lim=[-1e5, 1e5],
        fix_continuum=None,
        sampler="NUTS",
        nburn=2000,
        target_accept=0.8,
        find_MAP=False,
        plot_structure=False,
        plot_model=True,
        plot_mcmc=False,
    ):
        """MCMC sampler with pymc

        Parameters
        ----------

        initial : array_like, default=[]
             initial values for the MCMC sampler

        vel_mean_mu, vel_mean_sig : array_like, default=[]
            means/standard deviations of the velocity priors

        vel_mean_diff : array_like, default=[]
            standard deviations of the difference between velocity components
            list of tuples - (j, k, v_diff) : Var(v_j - v_k) = v_diff**2

        ln_vel_sig_mu, ln_vel_sig_sig : float, default=[]
            means/standard deviations of the logarithmic velocity dispersion priors

        ln_vel_sig_diff : array_like, default=[]
            standard deviations of the difference between the logarithmic velocity dispersions
            list of tuples - (j, k, ln_v_sig_diff) : Var(ln_v_sig_j - ln_v_sig_k) = ln_v_sig_diff**2

        ln_vel_sig_min, ln_vel_sig_max : array_like, default=[]
            minimum/maximum logarithmic velocity dispersions
            if not None, a softplux function will be added the posterior to punish velocity
            dispersions greater than this value

        A_lim : float, default=[-1e5, 1e5]
            allowed range of the amplitude

        sampler : ['NUTS', 'MH'], default='NUTS'
            A step function or collection of functions
            'NUTS' : The No-U-Turn Sampler
            'MH' : Metropolis–Hastings Sampler

        nburn : int, default=2000
            number of "burn-in" steps for the MCMC chains

        target_accept : float in [0, 1], default=0.8
             the step size is tuned such that we approximate this acceptance rate
             higher values like 0.9 or 0.95 often work better for problematic posteriors

        find_MAP : bool, default=False
             whether to find the local maximum a posteriori point given a model

        plot_model : bool, default=True
            whether to plot the model v.s. data

        plot_mcmc : bool, default=False
            whether to plot the MCMC chains and corner plots

        Returns
        -------
        trace : arviz.data.inference_data.InferenceData
            the samples drawn by the NUTS sampler
        GaussianProfile : pymc.model.Model
            the Bayesian model
        ax : matplotlib.axes
            the axes with the plot
        """
        import pymc as pm
        import arviz as az
        import corner

        n_lines = len(self.lines)

        with pm.Model() as GaussProfile:
            # continuum fitting
            if fix_continuum is not None:
                fl1 = fl2 = fix_continuum
            else:
                # model flux at the blue edge
                fl1 = pm.TruncatedNormal(
                    "blue_fl",
                    mu=self.blue_fl[0],
                    sigma=self.blue_fl[1],
                    lower=self.blue_fl[0] - self.blue_fl[1] * 2,
                    upper=self.blue_fl[0] + self.blue_fl[1] * 2,
                )
                # model flux at the red edge
                fl2 = pm.TruncatedNormal(
                    "red_fl",
                    mu=self.red_fl[0],
                    sigma=self.red_fl[1],
                    lower=self.red_fl[0] - self.red_fl[1] * 2,
                    upper=self.red_fl[0] + self.red_fl[1] * 2,
                )

            # Gaussian profile
            # amplitude
            A = pm.Uniform("A", lower=A_lim[0], upper=A_lim[1], shape=(n_lines,))

            if (len(vel_mean_mu) == n_lines) and (len(ln_vel_sig_mu) == n_lines):
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
                ln_vel_sig_cov = np.diag(ln_vel_sig_sig) ** 2  # covariance matrix
                for j, k, ln_sig_diff in ln_vel_sig_diff:
                    ln_vel_sig_cov[j, k] = ln_vel_sig_cov[k, j] = (
                        ln_vel_sig_sig[j] ** 2 + ln_vel_sig_sig[k] ** 2 - ln_sig_diff**2
                    ) / 2
                ln_v_sig = pm.MvNormal(
                    "ln_v_sig",
                    mu=ln_vel_sig_mu,
                    cov=ln_vel_sig_cov,
                )
                if len(ln_vel_sig_min) == len(ln_vel_sig_mu):
                    print("There is a vel_sig_min lim...")
                    pm.Potential(
                        "vel_sig_min_lim",
                        -pm.math.log1pexp(-(ln_v_sig - ln_vel_sig_min) * 5**2),
                    )
                if len(ln_vel_sig_max) == len(ln_vel_sig_mu):
                    print("There is a vel_sig_max lim...")
                    pm.Potential(
                        "vel_sig_max_lim",
                        -pm.math.log1pexp((ln_v_sig - ln_vel_sig_max) * 5**2),
                    )
            else:
                raise IndexError("The number of the velocity priors does not match the number of lines")

            v_sig = pm.Deterministic("v_sig", np.exp(ln_v_sig))
            theta = [fl1, fl2]
            for k in range(n_lines):
                theta += [v_mean[k], ln_v_sig[k], A[k]]
            # relative intensity of lines
            rel_strength = []
            ratio_index = []
            for k, free in enumerate(self.free_rel_strength):
                if free:
                    ratio_index.append(k)
                    log_ratio_0 = np.log10(self.rel_strength[k])
                    log_rel_strength_k = pm.Normal(f"log_ratio_{k}", mu=log_ratio_0, sigma=0.1)
                    rel_strength.append(pm.Deterministic(f"ratio_{k}", 10**log_rel_strength_k))
                else:
                    rel_strength.append(self.rel_strength[k])
                # equivalent width
                EW_k = pm.Deterministic(
                    f"EW_{k}",
                    -A[k]
                    / (self.red_vel - self.blue_vel)
                    / ((fl1 + fl2) / 2)
                    * (self.wv_line[-1] - self.wv_line[0])
                    * pm.math.sum(rel_strength[k]),
                )

            # flux expectation
            mu = pm.Deterministic(
                "mu",
                calc_model_flux(
                    theta,
                    self.vel_resolution,
                    rel_strength,
                    self.lambda_0,
                    self.blue_vel,
                    self.red_vel,
                    self.vel_rf,
                    self.lines,
                    self.line_model,
                ),
            )

            # uncertainty normalization
            # typical_unc = np.median(self.norm_fl_unc)
            # sigma_0 = pm.HalfCauchy("sigma_0", beta=typical_unc)
            # sigma = pm.Deterministic("sigma", (sigma_0**2 + self.norm_fl_unc**2) ** 0.5)

            sigma = self.norm_fl_unc

            Flux = pm.Normal("Flux", mu=mu, sigma=sigma, observed=self.norm_fl)
            # Flux = pm.MvNormal(
            #     "Flux", mu=mu, cov=self.norm_fl_cov, observed=self.norm_fl
            # )

        if plot_structure:
            pm.model_to_graphviz(GaussProfile)
            plt.show()

        # initialization
        if len(initial) == 0:
            start = None
        else:
            start = {}
            start["blue_fl"], start["red_fl"] = self.blue_fl[0], self.red_fl[0]
            start["v_mean"] = initial[2::3]
            start["ln_v_sig"] = initial[3::3]
            start["A"] = initial[4::3]
            # start["sigma_0"] = 1e-3
            for k, free in enumerate(self.free_rel_strength):
                if free:
                    start[f"ratio_{k}"] = self.rel_strength[k]
                    start[f"log_ratio_{k}"] = np.log10(self.rel_strength[k])

        with GaussProfile:
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
        var_names_summary = ["v_mean", "v_sig", "A"]  # , "sigma_0"]
        for k in ratio_index:
            var_names_summary.append(f"ratio_{k}")
        for k in range(n_lines):
            var_names_summary.append(f"EW_{k}")
        summary = az.summary(
            trace,
            var_names=var_names_summary,
            stat_focus="mean",
            round_to=3,
            hdi_prob=0.68,
        )
        print(summary)

        all = az.summary(trace, kind="stats")
        if fix_continuum is not None:
            theta = []
            sig_theta = []
        else:
            theta = [all["mean"]["blue_fl"], all["mean"]["red_fl"]]
            sig_theta = [all["sd"]["blue_fl"], all["sd"]["red_fl"]]
        self.EW = []
        self.sig_EW = []
        for k in range(n_lines):
            theta.append(all["mean"][f"v_mean[{k}]"])
            theta.append(all["mean"][f"ln_v_sig[{k}]"])
            theta.append(all["mean"][f"A[{k}]"])
            self.EW.append(all["mean"][f"EW_{k}"])
            self.sig_EW.append(all["sd"][f"EW_{k}"])

            sig_theta.append(all["sd"][f"v_mean[{k}]"])
            sig_theta.append(all["sd"][f"ln_v_sig[{k}]"])
            sig_theta.append(all["sd"][f"A[{k}]"])
        self.theta_MCMC = theta
        self.sig_theta_MCMC = sig_theta

        if find_MAP:
            neg_log_posterior = -np.array(trace.sample_stats.lp)
            ind = np.unravel_index(np.argmin(neg_log_posterior, axis=None), neg_log_posterior.shape)
            theta_MAP = [
                np.array(trace.posterior["blue_fl"])[ind],
                np.array(trace.posterior["red_fl"])[ind],
            ]
            for k in range(n_lines):
                theta_MAP.append(np.array(trace.posterior[f"v_mean"])[ind][k])
                theta_MAP.append(np.array(trace.posterior[f"ln_v_sig"])[ind][k])
                theta_MAP.append(np.array(trace.posterior[f"A"])[ind][k])
            self.theta_MAP = theta_MAP

        if plot_mcmc:
            # by default, show the mean velocity, velocity dispersion, pseudo-EW, and line ratios
            var_names_plot = ["v_mean", "v_sig"]
            for k in range(n_lines):
                var_names_plot.append(f"EW_{k}")
            for k in ratio_index:
                var_names_plot.append(f"ratio_{k}")
            corner.corner(trace, var_names=var_names_plot)

        if plot_model:
            if find_MAP:
                warnings.warn("The model from the MAP estimators are shown.")
                warnings.warn("The corresponding parameters:")
                print(self.theta_MAP)
                if fix_continuum is not None:
                    theta_MAP = [fix_continuum] * 2 + self.theta_MAP
                else:
                    theta_MAP = self.theta_MAP
                ax = self.plot_model(theta_MAP, return_ax=True)
            else:
                if fix_continuum is not None:
                    theta_MCMC = [fix_continuum] * 2 + self.theta_MCMC
                else:
                    theta_MCMC = self.theta_MCMC
                ax = self.plot_model(theta_MCMC, return_ax=True)
            return trace, GaussProfile, ax
        else:
            return trace, GaussProfile

    def plot_model(
        self,
        theta,
        return_ax=False,
        ax=None,
        bin=True,
        bin_size=None,
    ):
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

        print(f"Lambda_0 : {self.lambda_0} Ang")

        if ax == None:
            _, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
        # ensure high resolution in predicted model
        if len(self.vel_rf) < 200:
            vel_rf = np.linspace(self.vel_rf[0], self.vel_rf[-1], 200)
        else:
            vel_rf = self.vel_rf_unmasked

        # calculate the total flux from multiple lines
        num = len(self.rel_strength)
        theta0 = theta[: 2 + 3 * num]
        j = 2 + 3 * num
        rel_strength = self.rel_strength.copy()
        for k, rel in enumerate(self.free_rel_strength):
            if rel:
                for rel_s in range(len(rel_strength[k]) - 1):
                    rel_strength[k][rel_s] = 10 ** theta[j]
                    j += 1
        if j != len(theta):
            raise IndexError("Number of free parameters and relative strength do not match")
        model_flux = calc_model_flux(
            theta0,
            self.vel_resolution,
            rel_strength,
            self.lambda_0,
            self.blue_vel,
            self.red_vel,
            vel_rf,
            self.lines,
            model=self.line_model,
        )

        # bin the spectrum for visualization purposes
        spec_plot = np.array([self.vel_rf_unmasked, self.norm_fl_unmasked, self.norm_fl_unc_unmasked]).T
        if bin:
            if bin_size == None:
                bin_size = self.vel_resolution * 2.355
            print("binning spectrum for visualization...")
            print(f"bin size: {bin_size:.0f} km/s")
            spec_plot = data_binning(
                spec_plot,
                size=bin_size,
                spec_resolution=self.vel_resolution,
                sigma_clip=2,
            )
        ax.errorbar(
            spec_plot[:, 0],
            spec_plot[:, 1],
            yerr=spec_plot[:, 2],
            alpha=0.5,
            elinewidth=0.5,
            marker="o",
            zorder=-100,
        )

        # residual
        model_res = (
            calc_model_flux(
                theta0,
                self.vel_resolution,
                rel_strength,
                self.lambda_0,
                self.blue_vel,
                self.red_vel,
                spec_plot[:, 0],
                self.lines,
                model=self.line_model,
            )
            - spec_plot[:, 1]
        )

        model_plot = plt.plot(vel_rf, model_flux, linewidth=5, color="k")
        ax.errorbar(
            [self.vel_rf_unmasked[0], self.vel_rf_unmasked[-1]],
            [model_flux[0], model_flux[-1]],
            yerr=[self.blue_fl[1], self.red_fl[1]],
            color=model_plot[0].get_color(),
            fmt="s",
            markerfacecolor="w",
            capsize=5,
        )
        ax.plot(spec_plot[:, 0], model_res, color="grey")

        if len(rel_strength) > 1:
            colors = [
                "#66c2a5",
                "#fc8d62",
                "#8da0cb",
                "#e78ac3",
                "#a6d854",
                "#ffd92f",
                "#e5c494",
            ]
            for k in range(len(rel_strength)):
                model_flux = calc_model_flux(
                    np.append(theta0[:2], theta0[2 + 3 * k : 5 + 3 * k]),
                    self.vel_resolution,
                    [rel_strength[k]],
                    self.lambda_0,
                    self.blue_vel,
                    self.red_vel,
                    self.vel_rf_unmasked,
                    [self.lines[k]],
                    model=self.line_model,
                )
                ax.plot(
                    self.vel_rf_unmasked,
                    model_flux,
                    linewidth=2,
                    label=f"line_{k}",
                    color=colors[k % len(colors)],
                )
            ax.legend()

        ax.set_xlabel(r"$v\ [\mathrm{km/s}]$")
        ax.set_ylabel(r"$\mathrm{Normalized\ Flux}$")

        # mask
        for mk in self.mask:
            v_mk_1 = velocity_rf(mk[0], self.lambda_0)
            v_mk_2 = velocity_rf(mk[1], self.lambda_0)
            ax.axvspan(v_mk_1, v_mk_2, color="0.8", alpha=0.5)
        # print(ax.get_ylim())
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


###################### Likelihood ##########################


def lnlike_gaussian_abs(theta, spec_line):
    """Log likelihood function assuming Gaussian profile

    Parameters
    ----------
    theta : array_like
        fitting parameters: flux at the blue edge, flux at the
        red edge, (mean of relative velocity, log standard deviation,
        amplitude) * Number of velocity components, log10 line ratio
        for each velocity components (if set free)

    spec_line : sn_line_vel.SpecLine.SpecLine
        the SpecLine object

    Returns
    -------
    lnl : float
        the log likelihood function
    """

    rel_strength = spec_line.rel_strength.copy()
    num = len(rel_strength)
    theta0 = theta[: 2 + 3 * num]
    j = 2 + 3 * num
    for k, rel in enumerate(spec_line.free_rel_strength):
        if rel:
            for rel_s in range(len(spec_line.rel_strength[k]) - 1):
                rel_strength[k][rel_s] = 10 ** theta[j]
                j += 1
    if j != len(theta):
        print(j, theta)
        raise IndexError("Number of free parameters and relative strength do not match")

    model_flux = calc_model_flux(
        theta0,
        vel_resolution=spec_line.vel_resolution,
        rel_strength=spec_line.rel_strength,
        lambda_0=spec_line.lambda_0,
        blue_vel=spec_line.blue_vel,
        red_vel=spec_line.red_vel,
        vel_rf=spec_line.vel_rf,
        lines=spec_line.lines,
        model=spec_line.line_model,
    )
    lnl = (
        -0.5 * len(model_flux) * np.log(2 * np.pi)
        - np.sum(np.log(spec_line.norm_fl_unc))
        - 0.5 * np.sum((spec_line.norm_fl - model_flux) ** 2 / spec_line.norm_fl_unc**2)
    )

    return lnl


def neg_lnlike_gaussian_abs(theta, spec_line):
    """negative log-likelihood function"""

    lnl = lnlike_gaussian_abs(theta, spec_line)
    return -1 * lnl
