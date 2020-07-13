from scipy.stats import multivariate_normal as multivariate_norm
import chaospy as cp
import numpy as np

from econsa.sampling import cond_mvn
from scipy.stats import invwishart, invgamma
from cond_auxiliary import cond_gaussian_copula
#
for _ in range(1000):
    print(_)
    np.random.seed(_)

    dim = np.random.randint(2, 10)
    means = np.random.uniform(-100, 100, dim)

    sigma = np.random.normal(size=(dim, dim))
    cov = sigma @ sigma.T

    while True:
        cov = invwishart(dim, np.identity(dim)).rvs()
        try:
            np.testing.assert_almost_equal(np.linalg.inv(np.linalg.inv(cov)), cov)
            break
        except AssertionError:
            pass

    marginals = list()
    for i in range(dim):
        mean, sigma = means[i], np.sqrt(cov[i, i])
        marginals.append(cp.Normal(mu=mean, sigma=sigma))
    distribution = cp.J(*marginals)

    sample = distribution.sample(1).T[0]

    full = list(range(0, dim))
    dependent_ind = [np.random.choice(full)]

    given_ind = full[:]

    # TODO: This test always treats the first element as given. We need it to be more
    #  flexible and select random subets.
    given_ind.remove(dependent_ind[0])

    given_value = sample[given_ind]


    np.random.seed(123)
    given_value_u = [distribution[ind].cdf(given_value[i]) for i, ind in enumerate(given_ind)]
    condi_value_u = cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u)
    gc_value = distribution[int(dependent_ind[0])].inv(condi_value_u)

    np.random.seed(123)
    cond_mean, cond_cov = cond_mvn(means, cov, dependent_ind, given_ind, given_value)
    cond_dist = multivariate_norm(cond_mean, cond_cov)
    cn_value = np.atleast_1d(cond_dist.rvs())

    print(dim)
    np.testing.assert_allclose(cn_value, gc_value, rtol=1e-4)

