'''
this file contains old functions that are no longer used

 !!! this file is not meant to be run !!!

1. Create and search for mock sources
2. Source detection in old line maps
'''

if __name__ == '__main__':
    raise RuntimeError('this file cannot be executed (import functions you want instead)')

# ---------------------------------------------------------------------
# create mock sources and search for them
# ---------------------------------------------------------------------

from collections import OrderedDict
from photutils.datasets import (make_random_gaussians_table,
                                make_noise_image,
                                make_gaussian_sources_image)

from photutils import CircularAperture
from astropy.stats import gaussian_sigma_to_fwhm

from astropy.visualization import simple_norm


def test_detection(StarFinder_Algorithm,sigma_psf,amplitude,PSF_size=1):
    '''create an image with mock sources and try to detect them
    
    Parameters
    ----------
        
    StarFinder_Algorithm:
         Class to detect stars
    
    sigma_psf:
        standard deviation of the PSF of the mock sources 
    
    amplitude:
        amplitude of the mock sources
        
    PSF_size: 
        The StarFinder_Algorithm need to know the sigma of the sources
        they try to detect. This parameter changes the provided size 
        compared to the sigma of the mock sources.
    '''
    
    # create mock data
    n_sources = 20
    tshape = (256,256)

    param_ranges = OrderedDict([
                    ('amplitude', [amplitude, amplitude*1.2]),
                    ('x_mean', [0,tshape[0]]),
                    ('y_mean', [0,tshape[1]]),
                    ('x_stddev', [sigma_psf,sigma_psf]),
                    ('y_stddev', [sigma_psf, sigma_psf]),
                    ('theta', [0, 0]) ])

    sources = make_random_gaussians_table(n_sources, param_ranges,
                                          random_state=1234)

    image = (make_gaussian_sources_image(tshape, sources) +
             make_noise_image(tshape, type='poisson', mean=6.,
                              random_state=1) +
             make_noise_image(tshape, type='gaussian', mean=0.,
                              stddev=2., random_state=34234))

    fwhm = gaussian_sigma_to_fwhm * sigma_psf

    mean, median, std = sigma_clipped_stats(image, sigma=3.0)

    StarFinder = StarFinder_Algorithm(fwhm=fwhm*PSF_size, 
                                      threshold=3.*std,
                                      sharplo=0.1, 
                                      sharphi=1.0,
                                      roundlo=-.2,
                                      roundhi=.2)

    sources_mock = StarFinder(image)

    # for consistent table output
    for col in sources_mock.colnames:
        sources_mock[col].info.format = '%.8g'  

    string = str(StarFinder_Algorithm).split('.')[-1][:-2] + f' sig={sigma_psf} A={amplitude}'
    print(f'{string}: {len(sources_mock):} of {n_sources} sources found')

    positions = np.transpose([sources_mock['xcentroid'],sources_mock['ycentroid']])
    apertures = CircularAperture(positions, r=fwhm)    
    
    return image, apertures, sources, string

def run_test_dectecion():
    nrows = 2
    ncols = 4
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10),
                            squeeze=True)
    ax = ax.ravel()


    amplitude_lst = [40,200]
    sigma_lst = [2.,4.]
    StarFinder_lst = [DAOStarFinder,IRAFStarFinder]


    settings = []
    for f in StarFinder_lst:
        for s in sigma_lst:
            for a in amplitude_lst:
                settings.append((f,s,a))
        
        
    for i in range(nrows*ncols):
        f,s,a = settings[i]
        img, ap, sc, string = test_detection(f,s,a,PSF_size=1)
        
        norm = simple_norm(img, 'log', percent=99.)
        ax[i].imshow(img, norm=norm, origin='lower', cmap='viridis')
        ap.plot(color='red', lw=1., alpha=0.9,ax=ax[i])
        ax[i].scatter(sc['x_mean'],sc['y_mean'],color='red',s=1)
        ax[i].set_title(string)
        
    plt.show()


# ---------------------------------------------------------------------
# search for sources in old line maps (from Kreckel et al. 2017)
# ---------------------------------------------------------------------

from photutils import find_peaks

def search_in_old():
    data_folder = os.path.join('d',os.sep,'downloads','MUSEDAP')
    with fits.open(os.path.join(data_folder,'NGC628p','NGC628p_MAPS.fits')) as hdul:
        data = hdul[0].data
        header = hdul[0].header

    with fits.open(os.path.join(data_folder,'NGC628p','NGC628p_MAPS_err.fits')) as hdul:
        err = hdul[0].data

        mean, median, std = sigma_clipped_stats(err[~np.isnan(err)], sigma=3.0)

    # initialize daofind 
    # FWHM is given in arcsec. one pixel is 0.2" 
    StarFinder = DAOStarFinder(fwhm=0.8/0.17, 
                            threshold=8.*median,
                            sharplo=0.1, 
                            sharphi=1.0,
                            roundlo=-1,
                            roundhi=1)

    # for the source detection we subtract the sigma clipped median
    #sources_old = find_peaks(data,12*median,box_size=6)
    sources_old = StarFinder(data)

    print(f'{len(sources_old):>5.0f}{mean:>8.3f}{median:>8.3f}{std:>8.3f}')

    # for consistent table output
    for col in sources_old.colnames:
        sources_old[col].info.format = '%.8g'  

    sources_old.rename_column('xcentroid','x')
    sources_old.rename_column('ycentroid','y')
        
    # calculate astronomical coordinates
    sources_old['SkyCoord'] = SkyCoord.from_pixel(sources_old['x'],sources_old['y'],WCS(header))
    #sources['RaDec'] = sources['SkyCoord'].to_string(style='hmsdms',precision=2)

    ID, angle, Quantity  = match_coordinates_sky(pn_bright['SkyCoord'],sources_old['SkyCoord'])
    within_1_arcsec = len(angle[angle.__lt__(Angle("0.5s"))])

    print(f'{within_1_arcsec} of {len(angle)} match within 0.5": {within_1_arcsec / len(angle)*100:.1f} %')
    print(f'mean seperation is {angle.mean().to_string(u.arcsec,decimal=True)}"')



    file = Path.cwd() / '..' / 'reports' / 'figures' / 'sources_old.pdf'

    position = np.transpose((sources_old['x'], sources_old['y']))
    references = np.transpose(pn_bright['SkyCoord'].to_pixel(wcs=WCS(header)))
    positions = (position,references)

    plot_sources(data=data,wcs=WCS(header),positions=positions)
                            
    plt.xlim([1300,2900])
    plt.ylim([1800,3300])

    plt.savefig(file)




class MaximumLikelihood:
    '''

    for uncertainties 
    https://erikbern.com/2018/10/08/the-hackers-guide-to-uncertainty-estimates.html
    
    Parameters
    ----------
    func : function
        PDF of the form `func(data,params)`. `func` must accept a
        ndarray for `data` and can have any number of additional
        parameters (at least one).
        
    data : ndarray
        Measured data that are feed into `func`.

    err : ndarray
        Error associated with data.

    prior : function
        Prior probabilities for the parameters of func.

    method : 
        algorithm that is used for the minimization.

    **kwargs
       additional fixed key word arguments that are passed to func.
    '''
    
    def __init__(self,func,data,err=None,prior=None,method='Nelder-Mead',**kwargs):
        
        logger.warning('this function is deprecated. Use MaximumLikelihood1D instead')

        if len(signature(func).parameters)-len(kwargs)<2:
            raise ValueError(f'`func` must have at least one free argument')
        self.func = func

        self.data   = data
        self.err    = err
        if prior:
            self.prior = prior
        self.method = method
        self.kwargs = kwargs

    def prior(self,*args):
        '''uniform prior'''
        return 1/len(self.data)


    def _loglike(self,params,data):
        '''calculate the log liklihood of the given parameters
        
        This function takes the previously specified PDF and calculates
        the sum of the logarithmic probabilities. If key word arguments
        were initially passed to the class, they are also passed to the
        function
        '''
        
        return -np.sum(np.log(self.func(data,*params,**self.kwargs))) - np.log(self.prior(*params)) 

    def fit(self,guess):
        '''use scipy minimize to find the best parameters'''
        
        logger.info(f'searching for best parameters with {len(self.data)} data points')

        self.result = minimize(self._loglike,guess,args=(self.data),method=self.method)
        self.x = self.result.x
        if not self.result.success:
            raise RuntimeError('fit was not successful')

        self.dx = np.zeros((len(self.x),2))
        if np.any(self.err):
            
            B = 100
            #bootstrapping
            result_bootstrap = np.zeros((B,len(self.x)))
            for i in range(B):
                sample = np.random.normal(self.data,self.err)
                result_bootstrap[i,:] = minimize(self._loglike,guess,args=(sample),method=self.method).x
            err_boot = np.sqrt(np.sum((result_bootstrap-self.x)**2,axis=0)/B)
            self.dx[:,0] = err_boot 
            self.dx[:,1] = err_boot  
        
            '''
            self.result_m = minimize(self._loglike,guess,args=(self.data-self.err),method=self.method)
            self.result_p = minimize(self._loglike,guess,args=(self.data+self.err),method=self.method)

            if not self.result_m.success or not self.result_p.success:
                raise RuntimeError('fit for error was not successful')
            
            self.dx[:,0] = self.x - self.result_m.x
            self.dx[:,1] = self.result_p.x - self.x
            '''

        else:
            B = 500
            #bootstrapping
            result_bootstrap = np.zeros((B,len(self.x)))
            for i in range(B):
                sample = np.random.choice(self.data,len(self.data))
                result_bootstrap[i,:] = minimize(self._loglike,guess,args=(sample),method=self.method).x
            err_boot = np.sqrt(np.sum((result_bootstrap-self.x)**2,axis=0)/B)
            self.dx[:,0] = err_boot 
            self.dx[:,1] = err_boot  

        for name,_x,_dx in zip(list(signature(self.func).parameters)[1:],self.x,self.dx):
            print(f'{name} = {_x:.3f} + {_dx[1]:.3f} - {_dx[0]:.3f} ')

        return self.x

    def plot(self,limits):
        '''plot the likelihood
        
        plot the evidence, prior and likelihood for the given data over
        some parameters space.
        '''
        
        mu = np.linspace(*limits,500)
        evidence   = np.exp([np.sum(np.log(self.func(self.data,*[_],**self.kwargs))) for _ in mu])
        prior      = np.array([self.prior(_) for _ in mu])
        likelihood = np.exp(np.array([-self._loglike([_],self.data) for _ in mu]))
 
        valid = ~np.isnan(evidence) &  ~np.isnan(likelihood) 
        evidence /= np.abs(np.trapz(evidence[valid],mu[valid]))
        prior /= np.abs(np.trapz(prior[valid],mu[valid]))
        likelihood /= np.abs(np.trapz(likelihood[valid],mu[valid]))

        print(np.nanmean(likelihood))
        print(np.nanstd(likelihood))


        fig = figure()
        ax  = fig.add_subplot()

        ax.plot(mu,evidence,label='evidence')
        ax.plot(mu,prior,label='prior')
        ax.plot(mu,likelihood,label='likelihood')
        ax.legend()

        ax.set_ylabel('likelihood')
        ax.set_xlabel('mu')

    def __call__(self,guess):
        '''use scipy minimize to find the best parameters'''

        return self.fit(guess)



# ------------------------------------------------------
# compare with old linemaps
# ------------------------------------------------------

def old_vs_new_linemaps():
    with fits.open(data_raw / name / 'ngc628_ha.fits') as hdul:
        HA6562_old = hdul[0].data
        HA6562_old_header = hdul[0].header
        
    with fits.open(data_raw / name / 'ngc628_ha_err.fits') as hdul:
        HA6562_old_err = hdul[0].data
        
    galaxy.lines.append('HA6562_old')

    x,y = sources['SkyCoord'].to_pixel(WCS(HA6562_old_header))

    sources['x_big'] = x
    sources['y_big'] = y

    peak_tbl = sources


    from pymuse.photometry import light_in_moffat, correct_PSF
    from photutils import CircularAnnulus, CircularAperture, aperture_photometry

    alpha = galaxy.alpha
    aperture_size=galaxy.aperturesize
    wavelength = 6562
    PSF_correction = correct_PSF(wavelength)
        
    del flux_HA

    for fwhm in np.unique(peak_tbl['fwhm']):

        source_part = peak_tbl[peak_tbl['fwhm']==fwhm]
        positions = np.transpose((source_part['x_big'], source_part['y_big']))

        gamma = (fwhm - PSF_correction) / (2*np.sqrt(2**(1/alpha)-1))

        r = aperture_size * (fwhm-PSF_correction) / 2
        aperture = CircularAperture(positions, r=r)

        # measure the flux for each source
        phot = aperture_photometry(HA6562_old, 
                                aperture, 
                                error = HA6562_old_err,
                                )


        r_in  = 5 * (fwhm-PSF_correction) / 2 
        r_out = 1.*np.sqrt(3*r**2+r_in**2)
        annulus_aperture = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
        annulus_masks = annulus_aperture.to_mask(method='center')

        bkg_median = []
        for mask in annulus_masks:
            # select the pixels inside the annulus and calulate sigma clipped median
            annulus_data = mask.multiply(HA6562_old)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d[~np.isnan(annulus_data_1d)],sigma=5,maxiters=None)          
            bkg_median.append(median_sigclip)

        # save bkg_median in case we need it again
        phot['bkg_median'] = np.array(bkg_median) 
        # multiply background with size of the aperture
        phot['aperture_bkg'] = phot['bkg_median'] * aperture.area


        phot['flux'] = phot['aperture_sum'] - phot['aperture_bkg']


        # correct for flux that is lost outside of the aperture
        phot['flux'] /= light_in_moffat(r,alpha,gamma)

        # save fwhm in an additional column
        phot['fwhm'] = fwhm

        # concatenate new sources with output table
        if 'flux_HA' in locals():
            phot['id'] += np.amax(flux_HA['id'],initial=0)
            flux_HA = vstack([flux_HA,phot])
        else:
            flux_HA = phot
            

    from dust_extinction.parameter_averages import CCM89

    # initialize extinction model
    extinction_model = CCM89(Rv=Rv)
    k = lambda lam: ext_model.evaluate(wavelength*u.angstrom,Rv) * Rv


    extinction_mw = extinction_model.extinguish(wavelength*u.angstrom,Ebv=Ebv)


    flux_HA['flux'] /= extinction_mw 

    flux['HA6562_old'] = flux_HA['flux']

    fig,ax = plt.subplots(figsize=(4,4))
    plt.scatter(flux[tbl['type']=='PN']['HA6562'],flux[tbl['type']=='PN']['HA6562_old'],label='PN')
    plt.scatter(flux[tbl['type']=='SNR']['HA6562'],flux[tbl['type']=='SNR']['HA6562_old'],label='SNR')
    plt.legend()
    plt.plot([-1e5,3e4],[-1e5,3e4])
    plt.xlim([-15000,20000])
    plt.ylim([-15000,20000])

    plt.xlabel(r'H$\alpha$ old / (erg/s$^2$ / cm$^2$ / \AA)')
    plt.ylabel(r'H$\alpha$ DR1 / (erg/s$^2$ / cm$^2$ / \AA)')

    plt.show()

def save_PN():
    tmp = tbl[c_shape &  c_detec & c_limit & ~tbl['exclude'] & ((tbl['type']=='PN') | (tbl['type']=='SNR'))]
    tbl_out = tmp[['id','type','x','y','mOIII','dmOIII','HA6562','HA6562_err',
                            'NII6583','NII6583_err','SII','SII_err']]
    tbl_out.sort('mOIII')
    for col in tbl_out.columns[2:]:
        tbl_out[col].info.format = '%.3f'

    with open(basedir/'data'/f'{galaxy.name}_nebulae.txt','w',newline='\n') as f:
        ascii.write(tbl_out,f,format='fixed_width_two_line',overwrite=True,delimiter_pad=' ',position_char='=')