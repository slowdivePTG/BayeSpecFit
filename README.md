# BayeSpecFit
A Python package to fit supernova absorption features with correlated Gaussian/Lorentzian profiles and infer the expanding velocity and equivalent width of multiple species.

The package is developed based on [`sn_line_velocities`](https://github.com/slowdivePTG/sn_line_velocities), but works specifically for absorption lines. It allows modeling multiple line regions (i.e., different pseudo-continua) simultaneously in search of specific elements.

## Package prerequisites
- `pymc`>=4
- `arviz`
- `corner`

We recommend using Anaconda (or Miniforge) to install Python on your local machine, which allows for packages to be installed using its conda utility. Once you have conda installed, the packages required can be installed into a new conda environment as follows:

```shell
conda create -c conda-forge -n MY_ENV "pymc>=4" arviz corner
conda activate MY_ENV
```

The default sampler is the No-U-Turn sampler embedded in `pymc`.

## Installation (under developement)

```shell
git clone https://github.com/slowdivePTG/BayeSpecFit.git
cd BayeSpecFit
pip install -e .
```

