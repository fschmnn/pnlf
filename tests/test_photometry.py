import numpy as np


from astropy.modeling import models, fitting 
from astropy.stats import gaussian_fwhm_to_sigma

from photutils import aperture_photometry, CircularAperture

from pymuse.photometry import growth_curve, light_in_aperture


def test_growth_curve():
    '''
    test if the growth curve fit finds the correct FWHM

    we create a mock source with known FWHM and apply the growth_curve
    function to see if it finds the correct value
    '''
    size = 64

    fwhm = np.random.uniform(4,12)
    std  = fwhm * gaussian_fwhm_to_sigma
    gaussian = models.Gaussian2D(x_mean=size/2,y_mean=size/2,x_stddev=std,y_stddev=std)
    img = gaussian(*np.indices((size,size)))

    fwhm_fit = growth_curve(img,size/2,size/2,r_aperture=0.4*size)[0]

    assert abs(fwhm-fwhm_fit)/fwhm < 0.05



def test_aperture_correction():
    '''
    measure the flux with a large aperture that should contain and a
    second time with a smaller aperture. The flux from the second 
    measurement is corrected and then compared to the larger aperture.
    '''

    size = 64

    fwhm = np.random.uniform(4,12)
    std  = fwhm * gaussian_fwhm_to_sigma
    gaussian = models.Gaussian2D(x_mean=size/2,y_mean=size/2,x_stddev=std,y_stddev=std)
    img = gaussian(*np.indices((size,size)))


    aperture = CircularAperture((size/2,size/2),r=4*fwhm)
    total_flux = aperture_photometry(img,aperture)[0]['aperture_sum']

    r = np.random.uniform(fwhm,2*fwhm)
    aperture = CircularAperture((size/2,size/2),r=r)
    partial_flux = aperture_photometry(img,aperture)[0]['aperture_sum']

    corrected_flux =  partial_flux / light_in_aperture(r,fwhm)

    assert abs(corrected_flux-total_flux) / total_flux < 0.05