# https://dfm.io/posts/mixture-models/
# in the original code, the blobs_dtype argument is missing and therefore the code is not working


import numpy as np
import matplotlib.pyplot as pl

# We'll choose the parameters of our synthetic data.
# The outlier probability will be 80%:
true_frac = 0.8

# The linear model has unit slope and zero intercept:
true_params = [1.0, 0.0]

# The outliers are drawn from a Gaussian with zero mean and unit variance:
true_outliers = [0.0, 1.0]

# For reproducibility, let's set the random number seed and generate the data:
np.random.seed(12)
x = np.sort(np.random.uniform(-2, 2, 15))
yerr = 0.2 * np.ones_like(x)
y = true_params[0] * x + true_params[1] + yerr * np.random.randn(len(x))

# Those points are all drawn from the correct model so let's replace some of
# them with outliers.
m_bkg = np.random.rand(len(x)) > true_frac
y[m_bkg] = true_outliers[0]
y[m_bkg] += np.sqrt(true_outliers[1]+yerr[m_bkg]**2) * np.random.randn(sum(m_bkg))

# First, fit the data and find the maximum likelihood model ignoring outliers.
A = np.vander(x, 2)
p = np.linalg.solve(np.dot(A.T, A / yerr[:, None]**2), np.dot(A.T, y / yerr**2))

# Then save the *true* line.
x0 = np.linspace(-2.1, 2.1, 200)
y0 = np.dot(np.vander(x0, 2), true_params)


import emcee

# Define the probabilistic model...
# A simple prior:
# m,b,Q,M,lnV
bounds = [(0.1, 1.9), (-0.9, 0.9), (0, 1), (-2.4, 2.4), (-7.2, 5.2)]
def lnprior(p):
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0

# The "foreground" linear likelihood:
def lnlike_fg(p):
    m, b, _, M, lnV = p
    model = m * x + b
    return -0.5 * (((model - y) / yerr) ** 2 + 2 * np.log(yerr))

# The "background" outlier likelihood:
def lnlike_bg(p):
    _, _, Q, M, lnV = p
    var = np.exp(lnV) + yerr**2
    return -0.5 * ((M - y) ** 2 / var + np.log(var))

# Full probabilistic model.
def lnprob(p):
    m, b, Q, M, lnV = p
    
    # First check the prior.
    lp = lnprior(p)
    if not np.isfinite(lp):
        return -np.inf, None
    
    # Compute the vector of foreground likelihoods and include the q prior.
    ll_fg = lnlike_fg(p)
    arg1 = ll_fg + np.log(Q)
    
    # Compute the vector of background likelihoods and include the q prior.
    ll_bg = lnlike_bg(p)
    arg2 = ll_bg + np.log(1.0 - Q)
    
    # Combine these using log-add-exp for numerical stability.
    ll = np.sum(np.logaddexp(arg1, arg2))
    
    # We're using emcee's "blobs" feature in order to keep track of the
    # foreground and background likelihoods for reasons that will become
    # clear soon.
    return lp + ll, (arg1, arg2)

# Initialize the walkers at a reasonable location.
ndim, nwalkers = 5, 10
p0 = np.array([1.0, 0.0, 0.7, 0.0, np.log(2.0)])
p0 = np.array([p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)])

# Set up the sampler.
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,blobs_dtype=[('args',tuple)])


# Run a burn-in chain and save the final location.
pos, _, _, _ = sampler.run_mcmc(p0, 500)

# Run the production chain.
sampler.reset()
sampler.run_mcmc(pos, 5000);


# Compute the quantiles of the predicted line and plot them.
A = np.vander(x0, 2)
lines = np.dot(sampler.flatchain[:, :2], A.T)
quantiles = np.percentile(lines, [16, 84], axis=0)

norm = 0.0
post_prob = np.zeros(len(x))
for i in range(sampler.chain.shape[1]):
    for j in range(sampler.chain.shape[0]):
        ll_fg, ll_bg = sampler.blobs[i][j][0]
        post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
        norm += 1
post_prob /= norm


# Plot the predition.
plt.fill_between(x0, quantiles[0], quantiles[1], color="#8d44ad", alpha=0.5)

# Plot the data points.
plt.errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
# Plot the (true) outliers.
plt.scatter(x[m_bkg], y[m_bkg], marker="s", s=22, c=post_prob[m_bkg], cmap="gray_r", vmin=0, vmax=1, zorder=1000)
# Plot the (true) good points.
plt.scatter(x[~m_bkg], y[~m_bkg], marker="o", s=22, c=post_prob[~m_bkg], cmap="gray_r", vmin=0, vmax=1, zorder=1000)

# Plot the true line.
plt.plot(x0, y0, color="k", lw=1.5)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.ylim(-2.5, 2.5)
plt.xlim(-2.1, 2.1);
