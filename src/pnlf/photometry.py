import logging              # use instead of print for more control
from pathlib import Path    # filesystem related stuff
import numpy as np          # numerical computations

import re

import matplotlib.pyplot as plt  # plot growth curve

import astropy.units as u        # handle units
from astropy.coordinates import SkyCoord              # convert pixel to sky coordinates

from astropy.table import vstack

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.stats import SigmaClip
from astropy.stats import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
import astropy.units as u        # handle units

from photutils import CircularAperture         # define circular aperture
from photutils import CircularAnnulus          # define annulus
from photutils import aperture_photometry      # measure flux in aperture

from photutils import Background2D, MedianBackground, MMMBackground, SExtractorBackground

import pyneb as pn

import scipy.optimize as optimization          # fit Gaussian to growth curve

from .io import ReadLineMaps
from .auxiliary import correct_PSF, test_convergence, \
                      light_in_gaussian, light_in_moffat, fwhm_moffat

basedir = Path(__file__).parent.parent.parent
logger = logging.getLogger(__name__)


def measure_flux(LineMaps,peak_tbl,alpha,Rv,Ebv,lines=None,aperture_size=1.5,background='local',extinction='MW'):
    '''measure flux for all lines in lines

    for each position in peak_tbl, the flux inside an aperture of 
    `aperture_size` is measured for each line `lines` (if `LineMaps`
    has an extinsion). The background is estimated from an annulus
    with a sigma clipped median (both median and mean are reported).
    The [OIII] fluxes are Milky Way extinction corrected and the 
    internal extinction is estimated if the Halpha and Hbeta lines
    are present.
    
    Parameters
    ----------
    
    LineMaps : Galaxy
       Galaxy object with detected sources
    
    peak_tbl : astropy table
        Table with columns `x` and `y` (position of the sources)
    
    alpha : float
        power index of the moffat

    lines : list
       list of lines that are measured
    
    aperture_size : float
       size of the aperture in multiples of the fwhm

    background : no longer used

    extinction : no longer used
    '''

    #del self.peaks_tbl['SkyCoord']

    # convertion factor from arcsec to pixel (used for the PSF)
    input_unit = 1e-20 * u.erg / u.cm**2 / u.s
    
    '''
    check the input parameters
    '''

    # self must be of type Galaxy
    if not isinstance(LineMaps,ReadLineMaps):
        logger.warning('input should be of type ReadLineMaps')
    
    if background not in ['global','local',None]:
        raise TypeError(f'unknown Background estimation: {background}')

    # if no line is specified, we measure the flux in all line maps
    if not lines:
        lines = LineMaps.lines
    else:
        # make sure lines is a list
        lines = [lines] if not isinstance(lines, list) else lines
    
        for line in lines:
            if not hasattr(LineMaps,line):
                raise AttributeError(f'{LineMaps.name} has no attribute {line}')
            
    logger.info(f'measuring fluxes in {LineMaps.name} for {len(peak_tbl)} sources\naperture = {aperture_size} fwhm')    
    

    '''
    loop over all lines to measure the fluxes for all sources
    '''
    out = {}
    for line in lines:
        
        logger.info(f'measuring fluxes in {line} line map')
        
        # select data and error (copy in case we need to modify it)
        data  = getattr(LineMaps,f'{line}').copy()
        error = getattr(LineMaps,f'{line}_err').copy()
        
        try:
            v_disp = getattr(LineMaps,f'{line}_SIGMA')
        except:
            logger.warning('no maps with velocity dispersion for ' + line)
            v_disp = np.zeros(data.shape) 

        # the fwhm varies slightly with wavelength
        wavelength = int(re.findall(r'\d{4}', line)[0])
        PSF_correction = correct_PSF(wavelength)

        # calculate a global background map
        mask = np.isnan(data)

        '''
        bkg = Background2D(data,(10,10), 
                        #filter_size=(15,15),
                        sigma_clip= None,#SigmaClip(sigma=3.,maxiters=None), 
                        bkg_estimator=MedianBackground(),
                        mask=mask).background
        bkg[mask] = np.nan

        from astropy.convolution import convolve, Gaussian2DKernel, Box2DKernel

        kernel = Box2DKernel(10) #Gaussian2DKernel(10) 
        bkg_convolve = convolve(data,kernel,nan_treatment='interpolate',preserve_nan=True)
    
        # this is too slow and the masks ignore bright HA emitter etc.
        source_mask = np.zeros(self.shape,dtype=bool)

        for fwhm in np.unique(peak_tbl['fwhm']):
            source_part = peak_tbl[peak_tbl['fwhm']==fwhm]
            positions = np.transpose((source_part['x'], source_part['y']))
            r = 4 * (fwhm-PSF_correction) / 2 
            aperture = CircularAperture(positions, r=r)
            for m in aperture.to_mask(method='center'):
                source_mask |= m.to_image(self.shape).astype(bool)
        '''


        '''
        loop over the individual pointings (they have different fwhm)
        '''
        for fwhm in np.unique(peak_tbl['fwhm']):

            source_part = peak_tbl[peak_tbl['fwhm']==fwhm]
            positions = np.transpose((source_part['x'], source_part['y']))

            gamma = (fwhm - PSF_correction) / (2*np.sqrt(2**(1/alpha)-1))

            if aperture_size > 3:
                logger.warning('aperture > 3 FWHM')
            r = aperture_size * (fwhm-PSF_correction) / 2 
            aperture = CircularAperture(positions, r=r)

            # measure the flux for each source
            phot = aperture_photometry(data, aperture, error = error)

            # the local background subtraction estimates the background for 
            # each source individually (annulus with 5 times the area of aperture) 
            r_in  = 4 * (fwhm-PSF_correction) / 2 
            r_out = np.sqrt(5*r**2+r_in**2)
            annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
            annulus_masks = annulus_aperture.to_mask(method='center')

            # background from annulus with sigma clipping
            bkg_median = []
            bgk_mean   = []
            for mask in annulus_masks:
                # select the pixels inside the annulus and calulate sigma clipped median
                annulus_data = mask.multiply(data)
                annulus_data_1d = annulus_data[mask.data > 0]
                _, median_sigclip , _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)],sigma=3,maxiters=10,cenfunc='median')          
                mean_sigclip, _ , _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)],sigma=3,maxiters=10,cenfunc='mean')          
                bkg_median.append(median_sigclip)
                bgk_mean.append(mean_sigclip)

            # save bkg_median in case we need it again and multiply background with size of the aperture
            phot['bkg_median'] = np.array(bkg_median) * aperture.area
            phot['bkg_mean'] = np.array(bgk_mean) * aperture.area
            
            '''
            # background from annulus with masked sources
            ones = np.ones(self.shape)
            # calculate flux in annulus where other sources are masked
            bkg_phot = aperture_photometry(data,annulus_aperture,mask=source_mask)
            # calculate area of the annulus (parts can be masked)
            bkg_area = aperture_photometry(ones,annulus_aperture,mask=source_mask)
            # save bkg_median in case we need it again
            phot['bkg_median'] = bkg_phot['aperture_sum'] / bkg_area['aperture_sum'] 
            # multiply background with size of the aperture
            phot['bkg_local'] = phot['bkg_median'] * aperture.area
            '''
            
            phot[f'{line}_flux'] = phot['aperture_sum'] - phot['bkg_median']
            phot[f'{line}_flux_raw'] = phot['aperture_sum'] 

            # we don't subtract the background from OIII because there is none
            #if line == 'OIII5006_DAP':
            #    phot[f'{line}_flux'] = phot['aperture_sum']

            # correct for flux that is lost outside of the aperture
            phot[f'{line}_flux'] /= light_in_moffat(r,alpha,gamma)
            phot[f'{line}_flux_raw'] /= light_in_moffat(r,alpha,gamma)
            phot[f'{line}_flux_err'] = phot['aperture_sum_err'] / light_in_moffat(r,alpha,gamma)
            phot['bkg_median'] /= light_in_moffat(r,alpha,gamma)
            phot['bkg_mean'] /= light_in_moffat(r,alpha,gamma)
            #print(f'{fwhm}: {light_in_moffat(r,alpha,gamma):.2f}')

            # calculate the average of the velocity dispersion
            aperture = CircularAperture(positions, r=4)
            SIGMA = aperture_photometry(v_disp,aperture)
            phot['SIGMA'] = SIGMA['aperture_sum'] / aperture.area

            # calculate stellar mass (for mass specific PN number, not used)
            #aperture = CircularAperture(positions, r=2)
            #stellar_mass = aperture_photometry(LineMaps.stellar_mass,aperture)
            #phot['stellar_mass'] = stellar_mass['aperture_sum'] / aperture.area
            
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

    # so far we have an individual table for each emission line
    for line,v in out.items():
        
        # find the wavelength for the extinction correction
        wavelength = re.findall(r'\d{4}', line)
        if len(wavelength) != 1:
            logger.error('line name must contain wavelength as 4 digit number in angstrom')
        wavelength = int(wavelength[0])

        # first we create the output table with 
        if 'flux' not in locals():
            flux = v[['id','xcenter','ycenter','fwhm']]
            flux.rename_columns(['xcenter','ycenter'],['x','y'])
            flux['x'] = flux['x'].value   # we don't want them to be in pixel units
            flux['y'] = flux['y'].value   #

        flux[f'{line}_flux'] = v[f'{line}_flux'] 
        flux[f'{line}_flux_err']  = v[f'{line}_flux_err']
        flux[f'{line}_flux_raw'] = v[f'{line}_flux_raw'] 
        flux[f'{line}_bkg_median'] = v[f'bkg_median'] 
        flux[f'{line}_bkg_mean'] = v[f'bkg_mean'] 

        # linemaps are already MW extinction corrected (OIII sum is not) 
        # the new [OIII] fluxes use the DAP [OIII] errors and hence are already extinction corrected
        if line=='OIII5006':
            rc = pn.RedCorr(R_V=3.1,E_BV=Ebv,law='CCM89')
            flux['OIII5006_flux'] *= rc.getCorr(5006)
            flux['OIII5006_flux_raw'] *= rc.getCorr(5006)
            flux['OIII5006_bkg_median'] *= rc.getCorr(5006)
            flux['OIII5006_bkg_mean'] *= rc.getCorr(5006)
            logger.info(f'lambda{wavelength}: Av={-2.5*np.log10(1/rc.getCorr(5006)):.2f}')

        # those columns are only needed for tests
        if False:
            flux[f'{line}_aperture_sum'] = v['aperture_sum']
            flux[f'{line}_bkg_local']  = v['bkg_local']
            flux[f'{line}_bkg_median'] = v['bkg_median']
            #flux[f'{k}_bkg_convole'] = v['bkg_convolve']
        flux[f'{line}_SIGMA'] = v['SIGMA']

    # the internal extinction correction based on the balmer decrement
    # we do not calculate an error of E(B-V) and hance also do not account for this in the corrected errors
    if 'HB4861' in lines and 'HA6562' in lines:
        logger.info('correction for internal extinction with balmer decrement')
        rc = pn.RedCorr(R_V=3.1,law='CCM89')
        rc.setCorr(obs_over_theo= flux['HA6562_flux']/flux['HB4861_flux'] / 2.86, wave1=6562.81, wave2=4861.33)
        rc.E_BV[(rc.E_BV<0) | (flux['HB4861_flux']<3*flux['HB4861_flux_err']) |  (flux['HA6562_flux']<3*flux['HA6562_flux_err'])] = 0
        flux['EBV_balmer'] = rc.E_BV
        for line in lines:
            wavelength = int(re.findall(r'\d{4}', line)[0])
            flux[f'{line}_flux_corr'] = flux[f'{line}_flux'] * rc.getCorr(wavelength)
            flux[f'{line}_flux_corr_err'] = flux[f'{line}_flux_err'] * rc.getCorr(wavelength)
            flux[f'{line}_bkg_median_corr'] = flux[f'{line}_bkg_median'] * rc.getCorr(wavelength)
            flux[f'{line}_bkg_mean_corr'] = flux[f'{line}_bkg_mean'] * rc.getCorr(wavelength)

    logger.info('all flux measurements completed')

    return flux

def measure_single_flux(img,positions,aperture_size,model='Moffat',alpha=None,gamma=None,fwhm=None,bkg=True,plot=False):
    '''Measure the flux for a single object
    
    The Background is subtracted from an annulus. No error is reported

    Parameters
    ----------

    img : ndarray
        array with the image data

    position : tuple
        position (x,y) of the source

    aperture_size : float
        aperture size in units of FWHM
    
    alpha : float
        power index of the Moffat
    
    gamma : float
        other parameter for the Moffat

    bkg : bool
        determines if the background is subtracted
    '''

    if model=='Moffat':
        fwhm = 2 * gamma * np.sqrt(2**(1/alpha)-1)
    
    r = aperture_size * (fwhm) / 2 
    aperture = CircularAperture(positions, r=r)
    phot = aperture_photometry(img,aperture)

    r_in  = 4 * fwhm / 2 
    r_out = np.sqrt(3*r**2+r_in**2)
    annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    mask = annulus_aperture.to_mask(method='center')

    annulus_data = mask.multiply(img)
    annulus_data_1d = annulus_data[mask.data > 0]
    _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)],sigma=3,maxiters=1)          

    phot['bkg_median'] = np.array(median_sigclip) 
    phot['bkg_local'] = phot['bkg_median'] * aperture.area
    if bkg:
        phot['flux'] = phot['aperture_sum'] - phot['bkg_local']
    else:
        phot['flux'] = phot['aperture_sum']

    if model=='Moffat':
        correction = light_in_moffat(r,alpha,gamma)
    elif model=='Gaussian':
        correction = light_in_gaussian(r,fwhm)
    else:
        raise ValueError(f'unkown model {model}')
    phot['flux'] /= correction
    
    if plot:
        from astropy.visualization import simple_norm

        norm = simple_norm(img, 'sqrt', percent=99)
        plt.imshow(img, norm=norm)
        aperture.plot(color='orange', lw=2)
        annulus_aperture.plot(color='red', lw=2)
        plt.show()

    #return phot
    return phot['flux'][0]


from astropy.modeling.models import custom_model
from astropy.modeling import models, fitting

@custom_model
def _moffat_model(x,alpha=2.5,gamma=3.0):
    return light_in_moffat(x,alpha,gamma)

@custom_model
def _gaussian_model(x,fwhm=2):
    return light_in_gaussian(x,fwhm)

def growth_curve(data,x,y,model,rmax=30,alpha=None,plot=False,**kwargs):
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

    if np.isnan(bkg_median):
        logger.warning('background contains NaN')
        if model == 'moffat':
            return np.array([np.inf,np.inf]), np.atleast_1d(np.inf)
        else:
            return np.array([np.inf]), np.atleast_1d(np.inf)        

    # -----------------------------------------------------------------
    # measure flux for different aperture radii
    # -----------------------------------------------------------------

    radius = []
    flux   = []

    r = 0.5
    while True:
        if r > rmax:
            logger.debug(f'no convergence within a radius of {rmax:.2f}')
            break

        aperture = CircularAperture((x,y), r=r)
        phot = aperture_photometry(data,aperture)
        flux.append(phot['aperture_sum'][0]-aperture.area*bkg_median)
        radius.append(r)

        #if test_convergence(flux,**kwargs):
        #    pass
        #    break
        
        r += 0.5

    radius = np.array(radius)
    flux = np.array(flux)   
    flux = flux/flux[-1]

    if np.any(np.isnan(flux)):
        if model == 'moffat':
            return np.array([np.inf,np.inf]), np.atleast_1d(np.inf)
        else:
            return np.array([np.inf]), np.atleast_1d(np.inf)

    # -----------------------------------------------------------------
    # fit moffat or gaussian
    # -----------------------------------------------------------------

    if model == 'moffat':
        '''
        guess = np.array([2,2])
        func = light_in_moffat
        fit,sig = optimization.curve_fit(func, radius,flux , guess)
        alpha, gamma = fit[0], fit[1]
        fwhm = 2*gamma * np.sqrt(2**(1/alpha)-1)
        #print(f'alpha={alpha:.2f}, gamma={gamma:.2f}, fwhm={fwhm:.2f}')
        '''
        func = light_in_moffat

        if alpha:
            model = _moffat_model(alpha=alpha)
            model.alpha.fixed=True
        else:
            model = _moffat_model()

        fitter = fitting.LevMarLSQFitter() 
        fitted_line = fitter(model, radius,flux)
        alpha,gamma = fitted_line.parameters
        fit = [alpha,gamma]
        fwhm = 2*gamma * np.sqrt(2**(1/alpha)-1)
        print(f'alpha={alpha:.2f}, gamma={gamma:.2f}, fwhm={fwhm:.2f}')

    elif model == 'gaussian':
        '''
        guess =5
        func = light_in_gaussian
        fit,sig = optimization.curve_fit(func, radius,flux , guess)
        fwhm = fit[0]
        '''
        func = light_in_gaussian

        model = _gaussian_model()
        fitter = fitting.LevMarLSQFitter() 
        fitted_line = fitter(model, radius,flux)
        fwhm = fitted_line.parameters
        fit = [fwhm]    
    else:
        raise TypeError('model must be `moffat` or `gaussian`')

    
    if plot:
        from astropy.visualization import simple_norm

        fig = plt.figure(figsize=(6.9,6.9/2))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        norm = simple_norm(data,percent=99,clip=False)#, percent=99.)
        yslice = slice(int(x-rmax/2),int(x+rmax/2))
        xslice = slice(int(y-rmax/2),int(y+rmax/2))
        im1 = ax1.imshow(data[xslice,yslice], norm=norm, origin='lower', cmap='Greens')

        p = ax2.plot(radius,flux,label='observed')
        ax2.plot(radius,func(radius,*fit),label='fit',ls='--',color=p[0].get_color())

        plt.xlabel('radius in px')
        plt.ylabel('light in aperture')
        plt.legend()
        plt.grid()

    return fit



