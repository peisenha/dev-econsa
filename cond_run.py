from scipy.stats import multivariate_normal as multivariate_norm
from econsa.sampling import cond_mvn
import chaospy as cp
import numpy as np

from cond_auxiliary import cond_gaussian_copula, cov2corr


for _ in range(100):
    np.random.seed(_)

    # TODO: SOme test failures when increaseing dimeniosn further.
    dim = 2
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

