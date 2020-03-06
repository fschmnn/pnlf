import numpy as np
from scipy.special import hyp2f1


class Distance:
    def __init__(self,value,unit):
        '''save distance in 

        '''

        if unit == 'cm':
            self.value = value
        elif unit == 'm':
            self.value = value * 100
        elif unit in {'pc','parsec'}:
            self.value = value * 3.085678e18
        elif unit in {'ly','lightyear'}:
            self.value = value * 9.460730472e17
        elif unit in {'mag','distance_modulus','mu'}:
            self.value = 10**(1+value/5) *  3.085678e18 
        else:
            raise ValueError(f'unkown unit: {unit}')

    def to_cm(self):
        return self.value

    def to_parsec(self):
        return self.value / 3.085678e18 

    def to_lightyear(self):
        return self.value / 9.460730472e17

    def to_distance_modulus(self):
        return 5*np.log10(self.value/3.085678e18 ) - 5
        

def search_table(table,string):
    
    mask = np.zeros(len(table),dtype=bool)
    
    for i, row in enumerate(table):
        if string in ','.join(map(str,row)):
            mask[i]= True
    return  table[mask]

correct_PSF = lambda lam: 1- 4.7e-5*(lam-6450)

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


def circular_mask(h, w, center=None, radius=None):
    '''Create a circular mask for a numpy array

    from
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    '''

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask