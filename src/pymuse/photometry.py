import numpy as np

import astropy.units as u        # handle units
from astropy.coordinates import SkyCoord              # convert pixel to sky coordinates

from astropy.table import vstack

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images

from photutils import CircularAperture         # define circular aperture
from photutils import CircularAnnulus          # define annulus
from photutils import aperture_photometry      # measure flux in aperture

from pymuse.io import ReadLineMaps

def measure_flux(self,lines=None,aperture_size=1.5):
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
    arcsec_to_pixel = 0.2
    input_unit = 1e-20 * u.erg / u.cm**2 / u.s
    
    # self must be of type Galaxy
    #if not isinstance(self,MUSEDAP):
    #    raise TypeError('input must be of type Galaxy')
    
    if not lines:
        lines = self.lines
    else:
        # make sure lines is a list
        lines = [lines] if not isinstance(lines, list) else lines
    
    # check if all required lines exist
    if set(lines) - set(self.lines):
        raise AttributeError(f'{self.name} has no attribute {str(lines)}')
    
    if not hasattr(self,'peaks_tbl'):
        raise AttributeError(f'use "detect_sources" to find sources first')
    else:
        sources = getattr(self,'peaks_tbl')
        
    print(f'measuring in {self.name} for {len(sources)} sources')    
    
    out = {}
    
    # we need to do this for each line
    for line in lines:
        
        print(f'measuring fluxes in [{line}] line map')
        
        # select data and error (copy in case we want to modify it)
        data  = getattr(self,f'{line}').copy()
        error = getattr(self,f'{line}_err').copy()
        
        #_, _, std = sigma_clipped_stats(data)
        #data[data<3*std] = 0
        
        for fwhm in np.unique(sources['fwhm']):

            source_part = sources[sources['fwhm']==fwhm]
            positions = np.transpose((source_part['x'], source_part['y']))

            # define size of aperture and annulus and create a mask for them
            r     = aperture_size * fwhm / 2 / arcsec_to_pixel
            r_in  = 3 * fwhm / arcsec_to_pixel
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
            
            #bkg_median = np.array(bkg_median)
            
            phot = aperture_photometry(data, 
                                       aperture, 
                                       error = error,
                                      )
            
            # save bkg_median in case we need it again
            phot['bkg_median'] = bkg_median 
            #phot['bkg_median'].unit = input_unit
                        # multiply background with size of the aperture
            phot['aperture_bkg'] = bkg_median 
            phot['aperture_bkg'] *= aperture.area
            #phot['aperture_bkg'].unit = input_unit


            # we don't subtract the background from OIII because there is none
            if line != 'OIII5006_depends':
                phot['flux'] = phot['aperture_sum'] - phot['aperture_bkg']
            else:
                phot['flux'] = phot['aperture_sum']
                
            # correct for flux that is lost outside of the aperture
            phot['flux'] /= (1-np.exp(-np.log(2)*aperture_size**2))
            
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
        flux[f'{k}_err'] = v['aperture_sum_err']

    
    # calculate astronomical coordinates for comparison
    flux['SkyCoord'] = SkyCoord.from_pixel(flux['xcenter'],flux['ycenter'],self.wcs)
   
    flux['mOIII'] = -2.5*np.log10(flux['OIII5006']*1e-20) - 13.74
    flux['dmOIII'] = np.abs( 2.5/np.log(10) * flux['OIII5006_err'] / flux['OIII5006'] )

    print('done')
    return flux