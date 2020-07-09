import numpy as np
import chaospy as cp

from cond_auxiliary import cond_gaussian_copula

# We specify the marginals for our variables.
marginals = list()
for center in [1230, 0.0135, 2.15]:
    lower, upper = 0.9 * center, 1.1 * center
    marginals.append(cp.Uniform(lower, upper))
distribution = cp.J(*marginals)

np.random.seed(123)

# Specification of  gaussian copula
corr = np.array([[1.0, 0.5, 0.2],
                 [0.5, 1.0, 0.4],
                 [0.2, 0.4, 1.0]])

sample = np.array([None, 1.30536798e-02, 1.95678248e+00])
dependent_ind, given_value, given_ind = [0], sample[1:], [1, 2]

given_value_u = [distribution[ind].cdf(given_value[i]) for i, ind in enumerate(given_ind)]
condi_value_u = cond_gaussian_copula(corr, dependent_ind, given_ind, given_value_u)

# Now we apply the variable's marginal distribution.
rslt = np.tile(np.nan, len(dependent_ind))
for i, ind in enumerate(dependent_ind):
    dist, value_u = distribution[ind], condi_value_u[i]
    stat = float(dist.inv(value_u))
    rslt[i] = stat

np.testing.assert_almost_equal(rslt[0], 1137.5819290382278)
