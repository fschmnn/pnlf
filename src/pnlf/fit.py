import numpy as np 
import matplotlib.pyplot as plt 
import emcee
from IPython.display import display, Math

class linearMLE:
    '''Define the probabilistic model...

    this class provides all the necessary routines to do a MLE fit 
    with a mixture model that allows for bad points that are marked 
    as outliers.

    based on the blog post by
    https://dfm.io/posts/mixture-models/
    
    p = m,b,Q,M,lnV
    '''
    def __init__(self,x,y,yerr,bounds):
        self.x = x
        self.y = y
        self.yerr = yerr
        self.bounds = bounds

    def lnprior(self,p):
        '''A simple prior
        
        we'll just put reasonable uniform priors on all the parameters'''
        
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        return 0

    def lnlike_fg(self,p):
        '''The "foreground" linear likelihood'''
        
        m, b, _, M, lnV = p
        model = m * self.x + b
        return -0.5 * (((model - self.y) / self.yerr) ** 2 + 2 * np.log(self.yerr))

    def lnlike_bg(self,p):
        '''The "background" outlier likelihood'''
        
        _, _, Q, M, lnV = p
        var = np.exp(lnV) + self.yerr**2
        return -0.5 * ((M - self.y) ** 2 / var + np.log(var))

    def lnprob(self,p):
        '''Full probabilistic model'''

        m, b, Q, M, lnV = p

        # First check the prior.
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf, None

        # Compute the vector of foreground likelihoods and include the q prior.
        ll_fg = self.lnlike_fg(p)
        arg1 = ll_fg + np.log(Q)

        # Compute the vector of background likelihoods and include the q prior.
        ll_bg = self.lnlike_bg(p)
        arg2 = ll_bg + np.log(1.0 - Q)

        # Combine these using log-add-exp for numerical stability.
        ll = np.sum(np.logaddexp(arg1, arg2))

        # We're using emcee's "blobs" feature in order to keep track of the
        # foreground and background likelihoods for reasons that will become
        # clear soon.
        return lp + ll, (arg1, arg2)

    def fit(self,p0):
        '''fit using emcee'''
        
        ndim, nwalkers = 5, 64
        p0 = np.array([p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)])

        # Set up the sampler.
        self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob,blobs_dtype=[('args',tuple)])

        # Run a burn-in chain and save the final location.
        pos, _, _, _ = self.sampler.run_mcmc(p0, 500)

        # Run the production chain.
        self.sampler.reset()
        self.sampler.run_mcmc(pos, 2000)

        flat_samples = self.sampler.get_chain(discard=100, thin=15, flat=True)
        labels = ['m','b','Q','M','lnV']
        self.results = {}
        for i in range(ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            q = np.diff(mcmc)
            self.results[labels[i]] = (mcmc[1], q[0], q[1])
       
    def outlier(self,names=None):
        norm = 0.0
        self.post_prob = np.zeros(len(self.x))
        for i in range(self.sampler.chain.shape[1]):
            for j in range(self.sampler.chain.shape[0]):
                ll_fg, ll_bg = self.sampler.blobs[i][j][0]
                self.post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        self.post_prob /= norm
        if names:
            print('\n'.join([f'{n}: {q:.2f}' for n,q in zip(names,self.post_prob)]))

        return self.post_prob
        
    def plot(self,xlim,ax=None,**kwargs):
        '''
        
        '''
        
        # Compute the quantiles of the predicted line and plot them.
        x0 = np.linspace(*xlim,200)
        A = np.vander(x0, 2)
        lines = np.dot(self.sampler.flatchain[:, :2], A.T)
        quantiles = np.percentile(lines, [16, 84], axis=0)
        if not ax:
            fig,ax=plt.subplots()

        ax.fill_between(x0, quantiles[0], quantiles[1], color="#E05659", alpha=0.7)    
        ax.plot(x0,self.results['b'][0]+x0*self.results['m'][0],color='black')

        # Plot the data.
        ax.errorbar(self.x, self.y, yerr=self.yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
        ax.scatter(self.x, self.y, marker="o", s=22, c=self.post_prob, cmap="gray_r", 
                   edgecolors='black',vmin=0, vmax=1, zorder=1000)

        ax.set(xlim=xlim,**kwargs)

        return ax
        
    def __repr__(self):
        for k,v in self.results.items():
            txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
            txt = txt.format(*v, k)
            display(Math(txt))
        return ''