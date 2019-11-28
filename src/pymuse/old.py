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