from scipy.stats import multivariate_normal as multivariate_norm
from scipy import optimize
import chaospy as cp
import numpy as np


def gc_correlation(marginals, corr):

    corr = np.atleast_2d(corr)

    # TODO: Tests for the correlation matrix, i.e. ones on diagonal, all other values between -1
    #  and 1, symmetric. Test that marginals are all cp.distributions. Can we exploit the
    #  material in P.L. Liu, A. Der Kiureghian, Probab. Eng. Mech. 1 (1986) 10 for the special
    #  cases where the transformation for selected pairwise correlations.
    dim = len(corr)

    indices = np.tril_indices(dim, -1)
    gc_corr = np.identity(dim)

    for i, j in list(zip(*indices)):
        subset = [marginals[i], marginals[j]]
        distributions, rho = cp.J(*subset), corr[i, j]
        gc_corr[i, j] = _gc_correlation_pairwise(distributions, rho)
    
    # Align upper triangular with lower triangular.
    gc_corr = gc_corr + gc_corr.T - np.diag(np.diag(gc_corr))
    
    return gc_corr


def _gc_correlation_pairwise(distributions, rho, seed=123, num_draws=100000):
    
    assert len(distributions) == 2

    arg_1 = np.prod(cp.E(distributions))
    arg_2 = np.sqrt(np.prod(cp.Var(distributions)))
    arg = (rho * arg_2 + arg_1)

    kwargs = dict()
    kwargs["args"] = (arg, distributions, seed, num_draws)
    kwargs["bounds"] = (-0.99, 0.99)
    kwargs["method"] = "bounded"

    out = optimize.minimize_scalar(_criterion, **kwargs)
    assert out["success"] 
    
    return out["x"]

    
def _criterion(rho_c, arg, distributions, seed, num_draws):
    
    cov = np.identity(2)
    cov[1, 0] = cov[0, 1] = rho_c

    np.random.seed(seed)

    # TODO: Here we need to use proper quadrature rules instead of Monte Carlo integration.
    x_1, x_2 = multivariate_norm([0, 0], cov).rvs(num_draws).T

    standard_norm_cdf = cp.Normal().cdf
    arg_1 = distributions[0].inv(standard_norm_cdf(x_1))
    arg_2 = distributions[1].inv(standard_norm_cdf(x_2))
    point = arg_1 * arg_2

    return (np.mean(point) - arg) ** 2
