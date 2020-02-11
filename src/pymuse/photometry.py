import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations

import matplotlib.pyplot as plt  # plot growth curve

import astropy.units as u        # handle units
from astropy.coordinates import SkyCoord              # convert pixel to sky coordinates

from astropy.table import vstack

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm

from photutils import CircularAperture         # define circular aperture
from photutils import CircularAnnulus          # define annulus
from photutils import aperture_photometry      # measure flux in aperture

from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground, MMMBackground, SExtractorBackground

import scipy.optimize as optimization          # fit Gaussian to growth curve

from .io import ReadLineMaps
from .auxiliary import correct_PSF, test_convergence, \
                      light_in_gaussian, light_in_moffat, fwhm_moffat

basedir = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)


def measure_flux(self,peak_tbl,alpha,lines=None,aperture_size=1.5,background='local'):
    '''
    measure flux for all lines in lines
    
    Parameters
    ----------
    
    self : Galaxy
       Galaxy object with detected sources
    
    peak_tbl : astropy table
        Table with columns `x` and `y` (position of the sources)
    
    alpha : float
        power index of the moffat

    lines : list
       list of lines that are measured
    
    aperture_size : float
       size of the aperture in multiples of the fwhm

    background : string
        `local` (default) or `global` or None
    '''

    #del self.peaks_tbl['SkyCoord']

    # convertion factor from arcsec to pixel (used for the PSF)
    input_unit = 1e-20 * u.erg / u.cm**2 / u.s
    
    '''
    check the input parameters
    '''

    # self must be of type Galaxy
    if not isinstance(self,ReadLineMaps):
        raise TypeError('input must be of type ReadLineMaps')
    
    if background not in ['global','local',None]:
        raise TypeError(f'unknown Background estimation: {background}')

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
            
    logger.info(f'measuring fluxes in {self.name} for {len(peak_tbl)} sources')    
    

    '''
    loop over all lines to measure the fluxes for all sources
    '''
    out = {}
    for line in lines:
        
        logger.info(f'measuring fluxes in [{line}] line map')
        
        # select data and error (copy in case we need to modify it)
        data  = getattr(self,f'{line}').copy()
        error = getattr(self,f'{line}_err').copy()
        v_disp = np.sqrt(getattr(self,f'{line}_SIGMA')**2 - getattr(self,f'{line}_SIGMA_CORR')**2)

        # the fwhm varies slightly with wavelength
        PSF_correction = correct_PSF(line)
    
        if background == 'global':
            '''method global:
            We create a background image by excluding sources (sigma clipped). 
            The resulting background is then subtracted from the data.
            '''
            sigma_clip = SigmaClip(sigma=3.,maxiters=None)
            bkg_estimator = SExtractorBackground()
            mask = np.isnan(data)
            box_size = (40,40)
            filter_size = (5,5)

            bkg = Background2D(data, box_size, 
                            filter_size=filter_size,
                            sigma_clip=sigma_clip, 
                            bkg_estimator=bkg_estimator,
                            mask=mask).background
            bkg[mask] = np.nan
            data -= bkg

        '''
        loop over the individual pointings (they have different fwhm)
        '''
        for fwhm in np.unique(peak_tbl['fwhm']):

            source_part = peak_tbl[peak_tbl['fwhm']==fwhm]
            positions = np.transpose((source_part['x'], source_part['y']))

            gamma = fwhm * PSF_correction / (2*np.sqrt(2**(1/alpha)-1))

            r = aperture_size * fwhm / 2 * PSF_correction 
            aperture = CircularAperture(positions, r=r)

            # measure the flux for each source
            phot = aperture_photometry(data, 
                                       aperture, 
                                       error = error,
                                      )

            if background == 'local':
                # the local background subtraction estimates the background for 
                # each source individually 
                if aperture_size > 3:
                    logger.warning('aperture > 3 FWHM')
                r_in  = 5 * fwhm / 2  * PSF_correction
                r_out = 1.*np.sqrt(3*r**2+r_in**2)
                annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
                annulus_masks = annulus_aperture.to_mask(method='center')
                
                bkg_median = []
                for mask in annulus_masks:
                    # select the pixels inside the annulus and calulate sigma clipped median
                    annulus_data = mask.multiply(data)
                    annulus_data_1d = annulus_data[mask.data > 0]
                    _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)],sigma=5,maxiters=None)          
                    bkg_median.append(median_sigclip)

                # save bkg_median in case we need it again
                phot['bkg_median'] = np.array(bkg_median) 
                # multiply background with size of the aperture
                phot['aperture_bkg'] = phot['bkg_median'] * aperture.area

                # we don't subtract the background from OIII because there is none
                if line == 'OIII5006_DAP':
                    phot['flux'] = phot['aperture_sum']
                else:
                    phot['flux'] = phot['aperture_sum'] - phot['aperture_bkg']
            
            else:
                # in case of the global background, bkg is already subtracted
                phot['flux'] = phot['aperture_sum']

            # calculate the average of the velocity dispersion
            aperture = CircularAperture(positions, r=fwhm)
            SIGMA = aperture_photometry(v_disp,aperture)
            phot['SIGMA'] = SIGMA['aperture_sum'] / aperture.area

            # correct for flux that is lost outside of the aperture
            phot['flux'] /= light_in_moffat(r,alpha,gamma)
            
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


def growth_curve(data,x,y,model,rmax=30,plot=False,**kwargs):
    '''do a growth curve analysis on the given star
    
    measure the amount of light as a function of radius and tries
    to fit a Gaussian to the measured profile. Returns the FWHM
    of the Gaussian.

    Parameters
    ----------

    model : str
        shape of the PSF. Must be either `gaussian` or `moffat`.
    
    rmax : float
        maximum radius for the growth curve
    '''
    
    # -----------------------------------------------------------------
    # determine background (we use same bkg_median for all apertures)
    # -----------------------------------------------------------------

    r_in  = 0.7*rmax
    r_out = rmax
    annulus_aperture = CircularAnnulus((x,y), r_in=r_in, r_out=r_out)
    mask = annulus_aperture.to_mask(method='center')
    annulus_data = mask.multiply(data)
    annulus_data_1d = annulus_data[mask.data > 0]
    _, bkg_median, _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)],maxiters=None)

    # -----------------------------------------------------------------
    # measure flux for different aperture radii
    # -----------------------------------------------------------------

    radius = []
    flux   = []

    r = 0.5
    while True:
        if r > rmax:
            logger.warning(f'no convergence within a radius of {rmax}')
            break

        aperture = CircularAperture((x,y), r=r)
        phot = aperture_photometry(data,aperture)
        flux.append(phot['aperture_sum'][0]-aperture.area*bkg_median)
        radius.append(r)

        if test_convergence(flux,**kwargs):
            break
        
        r += 0.5

    radius = np.array(radius)
    flux = np.array(flux)   
    flux = flux/flux[-1]

    # -----------------------------------------------------------------
    # fit moffat or gaussian
    # -----------------------------------------------------------------

    if model == 'moffat':
        guess = np.array([2,2])
        func = light_in_moffat
        fit,sig = optimization.curve_fit(func, radius,flux , guess)
        alpha, gamma = fit[0], fit[1]
        fwhm = 2*gamma * np.sqrt(2**(1/alpha)-1)
        #print(f'alpha={alpha:.2f}, gamma={gamma:.2f}, fwhm={fwhm:.2f}')


    elif model == 'gaussian':
        guess =5
        func = light_in_gaussian
        fit,sig = optimization.curve_fit(func, radius,flux , guess)
        fwhm = fit[0]
    else:
        raise TypeError('model must be `moffat` or `gaussian`')

    
    if plot:
        p = plt.plot(radius,flux,label='observed')
        plt.plot(radius,func(radius,*fit),label='fit',ls='--',color=p[0].get_color())

        plt.xlabel('radius in px')
        plt.ylabel('light in aperture')
        #plt.legend()
        plt.grid()

    return fit



