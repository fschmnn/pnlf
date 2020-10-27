from astropy.visualization import AsymmetricPercentileInterval, simple_norm

def _data_stretch(image, vmin=None, vmax=None, pmin=0.25, pmax=99.75,
                  stretch='linear', vmid=None, exponent=2):

    if vmin is None or vmax is None:
        interval = AsymmetricPercentileInterval(pmin, pmax, n_samples=10000)
        try:
            vmin_auto, vmax_auto = interval.get_limits(image)
        except IndexError:  # no valid values
            vmin_auto = vmax_auto = 0

    if vmin is None:
        vmin = vmin_auto

    if vmax is None:
        vmax = vmax_auto


    normalizer = simple_norm(image, stretch=stretch, power=exponent,
                             asinh_a=vmid, min_cut=vmin, max_cut=vmax, clip=False)

    data = normalizer(image, clip=True).filled(0)
    data = np.nan_to_num(data)
    data = np.clip(data * 255., 0., 255.)

    return data.astype(np.uint8)


def make_rgb_image(image_r,image_g,image_b, vmin=None, vmax=None, vmid=1,
                   pmin=0.25, pmax=99.75, stretch='linear',exponent=2,filename=None):
    """
    Make an RGB image from a FITS RGB cube or from three FITS files.

    adopted from 
    https://aplpy.readthedocs.io/en/stable/_modules/aplpy/rgb.html#make_rgb_cube

    Parameters
    ----------

    data : str or tuple or list
        If a string, this is the filename of an RGB FITS cube. If a tuple
        or list, this should give the filename of three files to use for
        the red, green, and blue channel.

    output : str
        The output filename. The image type (e.g. PNG, JPEG, TIFF, ...)
        will be determined from the extension. Any image type supported by
        the Python Imaging Library can be used.

    vmin_r, vmin_g, vmin_b : float, optional
        Minimum pixel value to use for the red, green, and blue channels.
        If set to None for a given channel, the minimum pixel value for
        that channel is determined using the corresponding pmin_x argument
        (default).

    vmax_r, vmax_g, vmax_b : float, optional
        Maximum pixel value to use for the red, green, and blue channels.
        If set to None for a given channel, the maximum pixel value for
        that channel is determined using the corresponding pmax_x argument
        (default).

    pmin_r, pmin_r, pmin_g : float, optional
        Percentile values used to determine for a given channel the
        minimum pixel value to use for that channel if the corresponding
        vmin_x is set to None. The default is 0.25% for all channels.

    pmax_r, pmax_g, pmax_b : float, optional
        Percentile values used to determine for a given channel the
        maximum pixel value to use for that channel if the corresponding
        vmax_x is set to None. The default is 99.75% for all channels.

    stretch_r, stretch_g, stretch_b : { 'linear', 'log', 'sqrt', 'asinh', 'power' }
        The stretch function to use for the different channels.

    vmid_r, vmid_g, vmid_b : float, optional
        Baseline values used for the log and arcsinh stretches. If
        set to None, this is set to zero for log stretches and to
        vmin - (vmax - vmin) / 30. for arcsinh stretches

    exponent_r, exponent_g, exponent_b : float, optional
        If stretch_x is set to 'power', this is the exponent to use.

    """

    try:
        from PIL import Image
    except ImportError:
        try:
            import Image
        except ImportError:
            raise ImportError("The Python Imaging Library (PIL) is required to make an RGB image")

    if isinstance(vmin,(float,int)):
        vmin_r,vmin_g,vmin_b = vmin,vmin,vmin
    elif isinstance(vmin,(list,tuple)) and len(vmin) == 3:
        vmin_r,vmin_g,vmin_b = vmin
        
    if isinstance(vmax,(float,int)):
        vmax_r,vmax_g,vmax_b = vmax,vmax,vmax
    elif isinstance(vmax,(list,tuple)) and len(vmax) == 3:
        vmax_r,vmax_g,vmax_b = vmax
     
    if isinstance(pmin,(float,int)):
        pmin_r,pmin_g,pmin_b = pmin,pmin,pmin
    elif isinstance(pmin,(list,tuple)) and len(pmin) == 3:
        pmin_r,pmin_g,pmin_b = pmin
        
    if isinstance(pmax,(float,int)):
        pmax_r,pmax_g,pmax_b = pmax,pmax,pmax
    elif isinstance(pmax,(list,tuple)) and len(pmax) == 3:
        pmax_r,pmax_g,pmax_b = pmax       
    
    if isinstance(stretch,str):
        stretch_r,stretch_g,stretch_b = stretch,stretch,stretch
    elif isinstance(stretch,(list,tuple)) and len(stretch) == 3:
        stretch_r,stretch_g,stretch_b = stretch   
        
    if isinstance(exponent,(float,int)):
        exponent_r,exponent_g,exponent_b = exponent,exponent,exponent
    elif isinstance(exponent,(list,tuple)) and len(exponent) == 3:
        exponent_r,exponent_g,exponent_b = exponent       
    
    if isinstance(vmid,(float,int)):
        vmid_r,vmid_g,vmid_b = vmid,vmid,vmid
    elif isinstance(vmid,(list,tuple)) and len(vmid) == 3:
        vmid_r,vmid_g,vmid_b = vmid
    
    image_r = Image.fromarray(_data_stretch(image_r, vmin=vmin_r, vmax=vmax_r,
                                            pmin=pmin_r, pmax=pmax_r, stretch=stretch_r,
                                            vmid=vmid_r, exponent=exponent_r))

    image_g = Image.fromarray(_data_stretch(image_g,
                                            vmin=vmin_g, vmax=vmax_g,
                                            pmin=pmin_g, pmax=pmax_g,
                                            stretch=stretch_g,
                                            vmid=vmid_g,
                                            exponent=exponent_g))

    image_b = Image.fromarray(_data_stretch(image_b,
                                            vmin=vmin_b, vmax=vmax_b,
                                            pmin=pmin_b, pmax=pmax_b,
                                            stretch=stretch_b,
                                            vmid=vmid_b,
                                            exponent=exponent_b))

    #rgb = Image.merge("RGB", (image_r, image_g, image_b))
    #rgb = rgb.transpose(Image.FLIP_TOP_BOTTOM)
    
    rgb = create_RGB(image_r,image_g,image_b,percentile=1)
    
    return rgb



def combine_catlaogues():
    filename = data_ext / 'HST' / 'cluster catalogue' / f'ngc628c_phangshst_base_catalog.fits'
with fits.open(filename) as hdul:
        clustersc = Table(hdul[1].data)
    SkyCoord_c = SkyCoord(clustersc['PHANGS_RA']*u.degree,clustersc['PHANGS_DEC']*u.degree)
    x_c,y_c = SkyCoord_c.to_pixel(hst_whitelight.wcs)
    clustersc['PHANGS_X'] = x_c
    clustersc['PHANGS_Y'] = y_c

    filename = data_ext / 'HST' / 'cluster catalogue' / f'ngc628e_phangshst_base_catalog.fits'
    with fits.open(filename) as hdul:
        clusterse = Table(hdul[1].data) 
    SkyCoord_e = SkyCoord(clusterse['PHANGS_RA']*u.degree,clusterse['PHANGS_DEC']*u.degree)
    x_e,y_e = SkyCoord_e.to_pixel(hst_whitelight.wcs)
    clusterse['PHANGS_X'] = x_e
    clusterse['PHANGS_Y'] = y_e

    clustersc['pointing'] = 'central'
    clusterse['pointing'] = 'east'

    clusters = vstack([clustersc,clusterse])
    hdu = fits.BinTableHDU(clusters)
    filename = data_ext / 'HST' / 'cluster catalogue' / f'ngc0628_phangshst_base_catalog.fits'
    hdu.writeto(filename,overwrite=True)
