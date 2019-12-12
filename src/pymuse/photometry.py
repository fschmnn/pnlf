import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations

import matplotlib.pyplot as plt  # plot growth curve

import astropy.units as u        # handle units
from astropy.coordinates import SkyCoord              # convert pixel to sky coordinates

from astropy.table import vstack

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.stats import gaussian_fwhm_to_sigma

from photutils import CircularAperture         # define circular aperture
from photutils import CircularAnnulus          # define annulus
from photutils import aperture_photometry      # measure flux in aperture

import scipy.optimize as optimization          # fit Gaussian to growth curve

from .io import ReadLineMaps
from .auxiliary import correct_PSF

basedir = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)

def light_in_aperture(x,fwhm):
    '''theoretical growth curve for a gaussian PSF

    Parameters
    ----------
    x : float
        Radius of the aperture in units of pixel

    fwhm : float
        FWHM of the Gaussian in units of pixel
    '''

    return 1-np.exp(-4*np.log(2)*x**2 / fwhm**2)
    #return 1-np.exp(-x**2 / (2*gaussian_fwhm_to_sigma**2*fwhm**2))

def measure_flux(self,lines=None,aperture_size=1.5,oversize_PSF=1.0):
    '''
    measure flux for all lines in lines
    
    Parameters
    ----------
    
    self : Galaxy
       Galaxy object with detected sources
    
    lines : list
       list of lines that are measured
    
    aperture_size : float
       size of the aperture in multiples of the fwhm
    '''
    
    # convertion factor from arcsec to pixel (used for the PSF)
    input_unit = 1e-20 * u.erg / u.cm**2 / u.s
    
    # self must be of type Galaxy
    if not isinstance(self,ReadLineMaps):
        raise TypeError('input must be of type ReadLineMaps')
    
    # if no line is specified, we measure the flux in all line maps
    if not lines:
        lines = self.lines
    else:
        # make sure lines is a list
        lines = [lines] if not isinstance(lines, list) else lines
    
        # check if all required lines exist
        missing = set(lines) - set(self.lines)
        if missing:
            raise AttributeError(f'{self.name} has no attribute {", ".join(missing)}')
    
    if not hasattr(self,'peaks_tbl'):
        raise AttributeError(f'run `detect_unresolved_sources` to find sources first')
    else:
        sources = getattr(self,'peaks_tbl')
        
    logger.info(f'measuring fluxes in {self.name} for {len(sources)} sources')    
    
    out = {}
    # we need to do this for each line
    for line in lines:
        
        logger.info(f'measuring fluxes in [{line}] line map')
        
        # select data and error (copy in case we want to modify it)
        data  = getattr(self,f'{line}').copy()
        error = getattr(self,f'{line}_err').copy()
        v_disp = np.sqrt(getattr(self,f'{line}_SIGMA')**2 - getattr(self,f'{line}_SIGMA_CORR')**2)

        PSF_correction = correct_PSF(line)
        
        for fwhm in np.unique(sources['fwhm']):

            source_part = sources[sources['fwhm']==fwhm]
            positions = np.transpose((source_part['x'], source_part['y']))

            # define size of aperture and annulus and create a mask for them
            r = aperture_size * fwhm / 2 * oversize_PSF * PSF_correction 
            r_in  = 3. * fwhm / 2 * oversize_PSF * PSF_correction
            #r_in = r
            r_out = np.sqrt(3*r**2+r_in**2)

            aperture = CircularAperture(positions, r=r)
            annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
            annulus_masks = annulus_aperture.to_mask(method='center')
            
            # for each source we calcualte the background individually 
            bkg_median = []
            for mask in annulus_masks:
                # select the pixels inside the annulus and calulate sigma clipped median
                annulus_data = mask.multiply(data)
                annulus_data_1d = annulus_data[mask.data > 0]
                _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)])          
                bkg_median.append(median_sigclip)
            
            phot = aperture_photometry(data, 
                                       aperture, 
                                       error = error,
                                      )

            # save bkg_median in case we need it again
            phot['bkg_median'] = np.array(bkg_median) 
            # multiply background with size of the aperture
            phot['aperture_bkg'] = phot['bkg_median'] * aperture.area

            # calculate the average of the velocity dispersion
            aperture = CircularAperture(positions, r=fwhm/2*oversize_PSF)
            SIGMA = aperture_photometry(v_disp,aperture)
            phot['SIGMA'] = SIGMA['aperture_sum'] / aperture.area

            # we don't subtract the background from OIII because there is none
            if line == 'OIII5006_DAP':
                phot['flux'] = phot['aperture_sum']
            else:
                phot['flux'] = phot['aperture_sum'] - phot['aperture_bkg']
                
            # correct for flux that is lost outside of the aperture
            phot['flux'] /= light_in_aperture(r,fwhm*oversize_PSF*correct_PSF)
            
            # save fwhm in an additional column
            phot['fwhm'] = fwhm
            
            # concatenate new sources with output table
            if 'flux' in locals():
                phot['id'] += np.amax(flux['id'],initial=0)
                flux = vstack([flux,phot])
            else:
                flux = phot
            
        # for consistent table output
        for col in flux.colnames:
            flux[col].info.format = '%.8g'  
        flux['fwhm'].info.format = '%.3g' 
   
        out[line] = flux
        
        # we need an empty table for the next line
        del flux
      
    for k,v in out.items():
        
        # first we create the output table with 
        if 'flux' not in locals():
            flux = v[['id','xcenter','ycenter','fwhm']]

        flux[k] = v['flux']
        flux[f'{k}_apsum'] = v['aperture_sum']
        flux[f'{k}_apbkg'] = v['aperture_bkg']

        flux[f'{k}_err'] = v['aperture_sum_err']
        flux[f'{k}_bkg'] = v['bkg_median']
        flux[f'{k}_SIGMA'] = v['SIGMA']

    flux.rename_column('xcenter','x')
    flux.rename_column('ycenter','y')
    flux['x'] = flux['x'].value
    flux['y'] = flux['y'].value

    logger.info('all flux measurements completed')

    return flux


def growth_curve(data,x,y,r_aperture=10,plot=False):
    '''do a growth curve analysis on the given star
    
    measure the amount of light as a function of radius and tries
    to fit a Gaussian to the measured profile. Returns the FWHM
    of the Gaussian.
    '''
    
    # we measure the flux for apertures of different radii
    radius = np.arange(1,r_aperture-2,1)
    flux = []

    for r in radius:
        aperture = CircularAperture((x,y), r=r)
        r_in = r_aperture
        r_out = 1.5*r_in
        annulus_aperture = CircularAnnulus((x,y), r_in=r_in, r_out=r_out)
        mask = annulus_aperture.to_mask(method='center')
        annulus_data = mask.multiply(data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, bkg_median, _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)])
        phot = aperture_photometry(data,aperture)
        flux.append(phot['aperture_sum'][0]-aperture.area*bkg_median)
    flux = np.array(flux)   
    flux = flux/flux[-1]

    guess = 4
    fit = optimization.curve_fit(light_in_aperture, radius,flux , guess)
    fwhm = fit[0]

    if plot:
        plt.plot(radius,flux,label='observed')
        plt.plot(radius,light_in_aperture(radius,fwhm),label='fit',ls='--')
        plt.xlabel('radius in px')
        plt.ylabel('light in aperture')
        plt.legend()
        plt.grid()

    
    return fit