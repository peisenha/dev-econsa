# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import chaospy as cp
import numpy as np
import pytest
from scipy.stats import norm

from econsa.copula import _cov2corr
from econsa.correlation import *
from econsa.correlation import _find_positive_definite


# -

# ## Testing functions

def get_strategies(name):
    dim = np.random.randint(2, 10)
    means = np.random.uniform(-100, 100, dim)

    if name == "test_gc_correlation_functioning":
        # List of distributions to draw from.
        distributions = [
            cp.Exponential,
            cp.Gilbrat,
            cp.HyperbolicSecant,
            cp.Laplace,
            cp.LogNormal,
            cp.LogUniform,
            cp.LogWeibull,
            cp.Logistic,
            cp.Maxwell,
            cp.Normal,
            cp.Rayleigh,
            cp.Uniform,
            cp.Wigner,
        ]
        marginals = list()
        for mean in means:
            dist = distributions[np.random.choice(len(distributions))](mean)
            marginals.append(dist)

        cov = np.random.uniform(-1, 1, size=(dim, dim))
        cov = cov @ cov.T
        cov = _find_positive_definite(cov)
        # The rounding is necessary to prevent ValueError("corr must be between 0 and 1")
        corr = _cov2corr(cov).round(8)
    elif (
        name == "test_gc_correlation_2d" or name == "test_gc_correlation_2d_force_calc"
    ):
        dim = 2
        means = np.random.uniform(-100, 100, dim)
        distributions = [
            cp.Normal,
            cp.Uniform,
            cp.Exponential,
            cp.Rayleigh,
            cp.LogWeibull,
        ]
        marginals = [cp.Normal(means[0])]
        dist2 = distributions[np.random.choice(len(distributions))](means[1])
        marginals.append(dist2)

        corr = np.identity(2)
        corr[0, 1] = corr[1, 0] = np.random.uniform(-0.75, 0.75)

    elif name == "test_gc_correlation_exception_marginals":
        marginals = list()
        for i in range(dim):
            marginals.append(norm())

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    elif name == "test_gc_correlation_exception_corr_symmetric":
        distributions = [cp.Normal, cp.Uniform, cp.LogNormal]
        marginals = list()
        for mean in means:
            dist = distributions[np.random.choice(len(distributions))](mean)
            marginals.append(dist)

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    else:
        raise NotImplementedError

    strategy = (marginals, corr)
    return strategy


# ## Theoretical value does not meet real value (`test_gc_correlation_2d`)

# ### Example 1

# +
_ = 1054648162

np.random.seed(_)
marginals, corr = get_strategies("test_gc_correlation_2d")

print(marginals)
corr_transformed = gc_correlation(marginals, corr)
copula = cp.Nataf(cp.J(*marginals), corr_transformed)
corr_copula = np.corrcoef(copula.sample(10000000))
np.testing.assert_allclose(corr, corr_copula, rtol=0.01, atol=0.01)
