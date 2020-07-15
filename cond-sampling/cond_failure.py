"""This module shows a failure of our test case that requires further investigation."""
from scipy.stats import multivariate_normal as multivariate_norm
import chaospy as cp
import numpy as np

from econsa.sampling import cond_mvn

from cond_auxiliary import cond_gaussian_copula


for _ in range(17):
    print("seed ", _)
    np.random.seed(_)

    dim = 5
    means = np.random.uniform(-100, 100, dim)

    sigma = np.random.normal(size=(dim, dim))
    cov = sigma @ sigma.T


    marginals = list()
    for i in range(dim):
        mean, sigma = means[i], np.sqrt(cov[i, i])
        marginals.append(cp.Normal(mu=mean, sigma=sigma))
    distribution = cp.J(*marginals)

    sample = distribution.sample(1).T[0]

    full = list(range(0, dim))
    dependent_ind = [np.random.choice(full)]

    given_ind = full[:]

    # TODO: This test always treats the first element as given.
    # We need it to be more flexible and select random subsets.
    given_ind.remove(dependent_ind[0])

    given_value = sample[given_ind]

    np.testing.assert_almost_equal(np.linalg.inv(np.linalg.inv(cov)), cov)

    np.random.seed(123)
    given_value_u = [distribution[ind].cdf(given_value[i]) for i, ind in enumerate(given_ind)]
    condi_value_u = cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u)
    gc_value = distribution[int(dependent_ind[0])].inv(condi_value_u)

    np.random.seed(123)
    cond_mean, cond_cov = cond_mvn(means, cov, dependent_ind, given_ind, given_value)
    cond_dist = multivariate_norm(cond_mean, cond_cov)
    cn_value = np.atleast_1d(cond_dist.rvs())

    np.testing.assert_almost_equal(cn_value, gc_value)
