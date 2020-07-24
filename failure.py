import chaospy as cp
import numpy as np
import pytest
from scipy.stats import multivariate_normal as multivariate_norm

from econsa.copula import cond_gaussian_copula
from econsa.sampling import cond_mvn

# code from get_strategies

np.random.seed(12345)
dim = np.random.randint(2, 10)

full = list(range(0, dim))
given_ind = full[:]

np.random.seed(12345)
dependent_n = np.random.randint(low=1, high=dim)
dependent_ind = np.random.choice(full, replace=False, size=dependent_n)

for i in dependent_ind:
    given_ind.remove(i)

means = np.random.uniform(-100, 100, dim)

# Draw new sigma until cov is well-conditioned
while True:
    sigma = np.random.normal(size=(dim, dim))
    cov = sigma @ sigma.T
    if np.linalg.cond(cov) < 100:
        break

marginals = list()
for i in range(dim):
    mean, sigma = means[i], np.sqrt(cov[i, i])
    marginals.append(cp.Normal(mu=mean, sigma=sigma))

distribution = cp.J(*marginals)

sample = distribution.sample(1).T[0]
given_value = sample[given_ind]

np.random.seed(123)
given_value_u = [
    distribution[ind].cdf(given_value[i]) for i, ind in enumerate(given_ind)
]
strategy_gc = (cov, dependent_ind, given_ind, given_value_u, distribution)
strategy_cn = (means, cov, dependent_ind, given_ind, given_value)
strategy = (strategy_gc, strategy_cn)

# code from test_cond_gaussian_copula

args_gc, args_cn = strategy
cov, dependent_ind, given_ind, given_value_u, distribution = args_gc

np.random.seed(123)
condi_value_u = cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u)
dist_subset = list()
for ind in dependent_ind:
    dist_subset.append(distribution[int(ind)])
dist_subset = cp.J(*dist_subset)
gc_value = dist_subset.inv(condi_value_u)

np.random.seed(123)
cond_mean, cond_cov = cond_mvn(*args_cn)
cond_dist = multivariate_norm(cond_mean, cond_cov)
cn_value = np.atleast_1d(cond_dist.rvs())
np.testing.assert_almost_equal(cn_value, gc_value)


