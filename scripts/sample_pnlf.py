from pnlf.analyse import sample_pnlf
from pnlf.analyse import MaximumLikelihood1D, pnlf
import pickle
import logging
from tqdm import tqdm


logging.basicConfig(format='%(levelname)s: %(message)s',level=logging.WARNING)

N_iter = 1000
mu   = 30
Mmax = -4.47

mu_dict = {}
for cl in tqdm([26.5,27,27.5,28],position=0,leave=False,colour='green'):
    for N_PN in tqdm([20,50,100,150],position=1,leave=False,colour='red'):
        mu_dict[(N_PN,cl)] = []
        for i in range(N_iter):
            sample = sample_pnlf(N_PN,mu,cl)
            fitter = MaximumLikelihood1D(pnlf,sample,mhigh=cl,Mmax=Mmax)
            mu_fit,mu_p,mu_m = fitter([29])
            mu_dict[(N_PN,cl)].append(mu_fit)

with open('sampled_pnlf.pkl','wb') as f:
    pickle.dump(mu_dict,f,pickle.HIGHEST_PROTOCOL)