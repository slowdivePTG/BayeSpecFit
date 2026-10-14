# BayeSpecFit

A Python package to fit supernova absorption features with correlated
Gaussian/Lorentzian profiles and infer the expanding velocity and equivalent
width of multiple species.

The package is developed based on
[`sn_line_velocities`](https://github.com/slowdivePTG/sn_line_velocities), but
works specifically for absorption lines. It allows modeling multiple line
regions (i.e., different pseudo-continua) simultaneously in search of specific
elements.

Features:

- Correlated Gaussian or Lorentzian absorption profiles, with instrumental
  broadening (spectral resolution) added in quadrature to the intrinsic
  velocity dispersion.
- Simultaneous fitting of several line regions, each with its own piecewise
  linear pseudo-continuum.
- A least-squares point estimator (`Powell`) and a full Bayesian sampler
  (`SpecLine.MCMC_sampler`) using PyMC's NUTS or Metropolis-Hastings.
- Configurable priors, including Gaussian/ordered pairwise constraints on
  velocity components.
- A Student-t flux likelihood that down-weights outliers (cosmic rays,
  sky-residual spikes) by default.

## Package prerequisites

- `pymc`>=4
- `arviz`
- `corner`

We recommend using Anaconda (or Miniforge) to install Python on your local
machine, which allows for packages to be installed using its conda utility.
Once you have conda installed, the packages required can be installed into a
new conda environment as follows:

```shell
conda create -c conda-forge -n MY_ENV "pymc>=4" arviz corner
conda activate MY_ENV
```

The default sampler is the No-U-Turn sampler embedded in `pymc`.

## Installation

```shell
git clone https://github.com/slowdivePTG/BayeSpecFit.git
cd BayeSpecFit
pip install -e .
```

## Quickstart

A spectrum is read through `SpectrumSN`, which handles redshift correction and
Galactic reddening; individual absorption features are then registered with
`add_line`. Each registered line becomes a `SpecLine` object stored in
`spec.line`.

```python
import numpy as np
from bayespecfit import SpectrumSN

# spec.dat holds rest-frame wavelengths [A], flux, and flux uncertainty.
# Pass the observed-frame redshift via z.
spec = SpectrumSN("spec.dat", z=0.05, spec_resolution=2.0)

# Si II lambda 6355: a single line region with one velocity component.
spec.add_line(
    name="Si II",
    lines=[[6355.0]],
    line_regions=[(6280.0, 6430.0)],
    lines_id=["Si II 6355"],
)

line = spec.line["Si II"]
```

### 1. Least-squares point estimate

`LS_estimator` minimizes the Gaussian negative log-likelihood with the `Powell`
method. It provides a fast starting point (and the `theta_LS` vector) for the
sampler. The guess is `(v_mean, log10 v_sig, log10 amp)` per velocity
component:

```python
line.LS_estimator(guess=[0.0, 2.0, -1.0])
```

### 2. Bayesian sampling

`MCMC_sampler` builds the PyMC model and runs the sampler. The posterior trace
is returned (and stored on the object), together with the model itself:

```python
trace, model = line.MCMC_sampler(
    vel_mean_mu=[0.0], vel_mean_sig=[3000.0],       # v_mean ~ Normal(0, 3000)
    log_vel_sig_mu=[2.0], log_vel_sig_sig=[0.5],    # log10 v_sig ~ Normal(2, 0.5)
    initial=line.theta_LS,                          # start from the LS solution
    nburn=1000, draws=1000, chains=4,
    likelihood="StudentT", student_t_nu=4.0,        # robust outliers (default)
    random_seed=1,
)
```

### 3. Inspecting and plotting results

```python
import corner
import matplotlib.pyplot as plt

# Best-fit parameters and their uncertainties (MCMC posterior means).
print(line.theta_MCMC)
print(line.sig_theta_MCMC)

# Maximum a posteriori sample.
print(line.theta_MAP)

# Posterior corner plot.
corner.corner(trace, var_names=["v_mean", "v_sig"])

# Model over the data (binned for display).
ax = line.plot_model(line.theta_MAP, lambda_0=[6355.0], return_ax=True)
plt.show()
```

`line.get_model_flux(theta, wv)` evaluates the model at arbitrary wavelengths,
optionally returning the contribution of each line in the series separately
(`get_indivial=True`).

## Priors

Every parameter is assigned an explicit prior inside the PyMC model. The table
below summarizes the defaults and the keyword arguments that change them. All
priors are validated before sampling.

| Parameter | Prior | Set by |
|---|---|---|
| `fl_blue`, `fl_red` | `TruncatedNormal` centred on the locally measured edge flux, truncated to ±2σ | automatic (pseudo-continuum) |
| `log_amp` | `Uniform(log_amp_lim[0], log_amp_lim[1])` (default `[0, 5]`) | `log_amp_lim` |
| `v_mean` | `Uniform(vel_mean_min, vel_mean_max)` **or** `Normal(vel_mean_mu, vel_mean_sig)` | `vel_mean_min`/`vel_mean_max` xor `vel_mean_mu`/`vel_mean_sig` |
| `log_v_sig` | `Uniform(log_vel_sig_min, log_vel_sig_max)` **or** `Normal(log_vel_sig_mu, log_vel_sig_sig)` | `log_vel_sig_min`/`log_vel_sig_max` xor `log_vel_sig_mu`/`log_vel_sig_sig` |
| line ratios `log_ratio` | `Uniform(-3, 0)` (weaker lines stay weaker) | `free_rel_strength` |

### Continuum (pseudo-continuum)

The flux at the blue and red edge of every line region is measured directly
from the data with `get_flux_at_lambda`, which returns an inverse-variance
weighted mean and a robust (MAD) local scatter. The scatter accounts for the
number of *independent* resolution elements in the bin, so oversampled pixels
do not underestimate the continuum uncertainty. These measurements become the
mean and standard deviation of a `TruncatedNormal` prior on `fl_blue` and
`fl_red`, truncated to ±2σ. The linear interpolation between the two edges
defines the pseudo-continuum across the region.

### Mean velocity and velocity dispersion

For each velocity component, `v_mean` and `log10 v_sig` are assigned either a
bounded flat prior (using `vel_mean_min`/`vel_mean_max` and
`log_vel_sig_min`/`log_vel_sig_max`) or a Gaussian prior (using
`vel_mean_mu`/`vel_mean_sig` and `log_vel_sig_mu`/`log_vel_sig_sig`). The two
families are mutually exclusive; supplying both raises a `ValueError`.

Three optional constraints further shape the velocity prior. They are added as
`pm.Potential` factors on top of the base prior, so they never replace it:

- **Gaussian difference priors** — `vel_mean_diff=[(j, k, sigma), ...]` adds
  `v_mean[j] - v_mean[k] ~ Normal(0, sigma)`. Use
  `log_vel_sig_diff=[(j, k, sigma), ...]` for the same constraint on
  `log_v_sig`.

- **Soft ordering priors** — `vel_mean_order=[(j, k, sigma), ...]` enforces
  `v_mean[j] <= v_mean[k]` (component `j` more blue-shifted than `k`) through a
  half-Gaussian penalty of width `sigma` applied only when the components are
  out of order:

  ```python
  wrong_order = max(v_mean[j] - v_mean[k], 0.0)
  logp = -0.5 * (wrong_order / sigma) ** 2
  ```

  Correctly ordered draws cost nothing, and each component keeps its own
  (wide) velocity range — no hard, disjoint bounds are imposed. This is useful
  for a physical blueshift sequence across components.

- **Per-component Gaussian factors** — `vel_mean_prior=[(mu, sigma) or None,
  ...]` (one entry per component) multiplies an extra `Normal(mu, sigma)`
  factor onto that component's velocity. `None` leaves a component unchanged.
  This can tie a component to an independently known velocity, e.g. anchoring a
  Ca II component to a measured Si II velocity:

  ```python
  line.MCMC_sampler(
      vel_mean_mu=[0.0, 0.0], vel_mean_sig=[3000.0, 3000.0],
      vel_mean_prior=[None, (-11000.0, 500.0)],  # anchor component 1
      ...
  )
  ```

### Amplitude and line ratios

The amplitude prior is `Uniform` on `log10 amp` over `log_amp_lim` (default
`[0, 5]`), i.e. `amp = 10**log_amp`. When a line series has relative strengths
flagged as free via `free_rel_strength`, each free `log10` ratio is assigned
`Uniform(-3, 0)`, which keeps the weaker lines weaker by construction.

## Likelihood and outlier treatment

The observed normalized flux `fl_norm` is modeled with an additive
continuum-absorption form and evaluated against the data with a flux
likelihood. The default is a **Student-t** distribution; a **Normal** option is
also available.

### Observation model

The expectation is

```
mu = continuum * (1 - sum(absorption))
```

where `absorption` sums the Gaussian/Lorentzian profiles of every line in
every series. The instrumental velocity resolution is added in quadrature to
the intrinsic dispersion, `sig_instru = sqrt(v_sig**2 + vel_resolution**2)`, so
a higher spectral resolution (smaller `vel_resolution`) yields narrower
features.

### Measurement uncertainty

The per-pixel scale entering the likelihood is

```python
measurement_sigma = fl_norm_unc * oversampling_sigma_scale
```

The `oversampling_sigma_scale` factor inflates pixels that are oversampled
relative to the spectral resolution: within one resolution element the noise is
correlated, so adjacent pixels do not count as independent measurements. The
per-pixel scale for a region is `sqrt(R / dlambda)`, where `R` is the spectral
FWHM resolution and `dlambda` is the median pixel spacing (floored at 1 when
the data are not oversampled).

### Outlier treatment (Student-t likelihood)

The default `likelihood="StudentT"` uses a Student-t distribution with **fixed**
degrees of freedom `student_t_nu=4`:

```python
residual = (fl_norm - mu) / sigma
loglike = sum(
    gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * log(nu * pi)
    - log(sigma)
    - ((nu + 1) / 2) * log1p(residual**2 / nu)
)
```

Compared with a Normal likelihood, the heavy tails make individual discrepant
pixels — cosmic rays, hot pixels, sky-residual spikes — far less influential:
their log-likelihood penalty grows only logarithmically instead of
quadratically with the residual. Outliers are thus effectively down-weighted
rather than driving the fit, while the bulk of the data is still modeled
tightly. This is the recommended setting and the default.

A Gaussian likelihood is available when the data are known to be
well-behaved (or when a Gaussian likelihood is wanted for consistency with the
least-squares estimator):

```python
line.MCMC_sampler(..., likelihood="Normal")
```

Setting a smaller `student_t_nu` makes the tails even heavier (more robust to
larger deviations); larger values approach a Normal distribution.

### Optional noise floor

By default `use_noise_floor=False`, and the likelihood scale is exactly the
oversampling-rescaled measurement uncertainty above. Set
`use_noise_floor=True` only when the input spectrum has no uncertainty column
(or the reported uncertainties are not trustworthy); a single global additive
noise floor is then inferred and combined in quadrature:

```python
sigma = sqrt(measurement_sigma**2 + noise_floor**2)
noise_floor ~ HalfNormal(noise_floor_scale)
```

`noise_floor_scale` defaults to the median finite positive measurement
uncertainty (or `1.0` if none are available) and can be overridden with the
`noise_floor_scale` argument.

### Relationship to the least-squares estimator

`LS_estimator` maximizes the *Gaussian* log-likelihood
(`lnlike_gaussian_abs`) using the same oversampling-rescaled `sigma`, with an
optional fixed additive `noise_floor` argument. It is intended for fast point
estimates and initialization, not as a robust sampler. Outlier robustness
during inference comes from the Student-t likelihood in `MCMC_sampler`.

## Outputs

After `MCMC_sampler` returns, the following are available on the `SpecLine`
object:

- `trace` — the ArviZ `InferenceData` posterior.
- `theta_MCMC` / `sig_theta_MCMC` — posterior means and standard deviations, in
  the same ordering as the least-squares `theta`.
- `theta_MAP` — the maximum-a-posteriori sample (minimum `lp` in the trace).
- `likelihood`, `student_t_nu`, `use_noise_floor`, `noise_floor_prior_scale` —
  the likelihood configuration actually used.

## Tests

```shell
pytest tests/
```
