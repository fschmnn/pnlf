import numpy as np
from scipy.special import hyp2f1


def search_table(table,string):
    
    mask = np.zeros(len(table),dtype=bool)
    
    for i, row in enumerate(table):
        if string in ','.join(map(str,row)):
            mask[i]= True
    return  table[mask]

correct_PSF = lambda lam: 1- 4.7e-5*(int(lam[-4:])-6450)

def fwhm_moffat(alpha,gamma):
    '''calculate the FWHM of a Moffat'''

    return 2*gamma * np.sqrt(2**(1/alpha)-1) 

def light_in_moffat(x,alpha,gamma):
    '''theoretical growth curve for a moffat PSF

    f(x;alpha,gamma) ~ [1+r^2/gamma^2]^-alpha

    Parameters
    ----------
    x : float
        Radius of the aperture in units of pixel
    '''

    return 1-(1+x**2/gamma**2)**(1-alpha)

def light_in_gaussian(x,fwhm):
    '''theoretical growth curve for a gaussian PSF

    Parameters
    ----------
    x : float
        Radius of the aperture in units of pixel

    fwhm : float
        FWHM of the Gaussian in units of pixel
    '''

    return 1-np.exp(-4*np.log(2)*x**2 / fwhm**2)



def light_in_moffat_old(x,alpha,gamma):
    '''
    without r from rdr one gets an hpyerfunction ...
    '''
    r_inf = 105
    f_inf = r_inf*hyp2f1(1/2,alpha,3/2,-r_inf**2/gamma**2)
    return x*hyp2f1(1/2,alpha,3/2,-x**2/gamma**2) / f_inf


def test_convergence(array,length=4,threshold=0.05):
    '''test if the given array approches a constant limit
    
    Parameters
    ----------
    array : list
    
    length : int
        number of elements that must approach the limit
        
    threshold : float
        maximum deviation from the limit
    '''
    
    if len(array) < length:
        return False

    array = np.atleast_1d(array)
    mean  = np.mean(array[-length:])
    return np.all((array[-length:]-mean)/mean < threshold)