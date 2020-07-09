from scipy.stats import multivariate_normal as multivariate_norm
from scipy.stats import norm
import numpy as np

from econsa.sampling import cond_mvn


def cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u):

    given_value_u = np.atleast_1d(given_value_u)

    assert np.all((given_value_u >= 0) & (given_value_u <= 1)), "sanitize your inputs!"

    given_value_y = [float(norm().ppf(u)) for u in given_value_u]

    means = np.zeros(cov.shape[0])
    cond_mean, cond_cov = cond_mvn(means, cov, dependent_ind, given_ind, given_value_y)

    cond_dist = multivariate_norm(cond_mean, cond_cov)
    cond_draw = np.atleast_1d(cond_dist.rvs())
    cond_quan = [cond_dist.cdf(draw) for draw in cond_draw]

    return np.atleast_1d(cond_quan)


def cov2corr(cov, return_std=False):
    """
    Convert covariance matrix to correlation matrix

    This function does not convert subclasses of ndarrays. This requires
    that division is defined elementwise. np.ma.array and np.matrix are allowed.

    Parameters
    ----------
    cov: (N, N) array like
        Covariance matrix

    return_std: bool, optional
         If True then the standard deviation is also returned. By default only the correlation matrix is returned.

    Returns
    -------
    corr: (N, N) ndarray
        Correlation matrix

    std: (1, N) ndarray
        Standard deviation
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr