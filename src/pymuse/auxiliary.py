import numpy as np


correct_PSF = lambda lam: 1- 4.7e-5*(int(lam[-4:])-6450)


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