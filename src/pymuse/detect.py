import numpy as np

from astropy.table import vstack # combine multiple tables

from astropy.stats import sigma_clipped_stats  # calcualte statistics of images
from astropy.coordinates import SkyCoord

from photutils import DAOStarFinder            # DAOFIND routine to detect sources
from photutils import IRAFStarFinder           # IRAF starfind routine to detect stars

from pymuse.io import MUSEDAP


np.warnings.filterwarnings('ignore')

def detect_sources(self,line,StarFinder,threshold=4,arcsec_to_pixel=0.2):
    '''detect unresolved sources in a Galaxy object
    
    Parameters
    ----------
    self : Galaxy
        contains the line maps that are used for the detection.
        
    line : string
        name of a line map in self.
        
    threshold : 
        detection threshold in terms of background median
        
    arcsec_to_pixel :
        convert fwhm from arcsec to pixel
    '''
    
    #if not isinstance(self,MUSEDAP):
    #    raise TypeError('input must be of type Galaxy')
    
    # for convenience only, to make accessing the data easier
    data = getattr(self,line)
    #err  = getattr(self,f'{line}_err')
    
    print(f'searching for sources in {self.name} with [{line}] line map (using ' + \
          str(StarFinder).split('.')[-1][:-2] + ')\n' )
    
    # header for the print information
    print(f'{"fwhm":>9}{"#N":>5}{"mean":>8}{"median":>8}{"std":>8}')
    
    # loop over all pointings with different PSFs
    for fwhm in np.unique(self.PSF[~np.isnan(self.PSF)]):
                
        # we create a mask for the current pointing (must be inverted)
        mask = ~(self.PSF == fwhm)

        mean, median, std = sigma_clipped_stats(data[(~np.isnan(data)) & (~mask)], sigma=3.0)

        # initialize daofind 
        # FWHM is given in arcsec. one pixel is 0.2" 
        finder = StarFinder(fwhm = fwhm/arcsec_to_pixel, 
                            threshold = threshold*std,
                            sharplo   = 0.5, 
                            sharphi   = 1.0,
                            roundlo   = -0.4,
                            roundhi   = 0.4)
        
        peaks_part = finder(data, mask=mask)
            
        # save fwhm in an additional column
        peaks_part['fwhm'] = fwhm
        
        n_sources = len(peaks_part)
        print(f'{fwhm:>8.3f}"{n_sources:>5.0f}{mean:>8.3f}{median:>8.3f}{std:>8.3f}')
        
        # concatenate new sources with output table
        if 'peak_tbl' in locals():
            peaks_part['id'] += np.amax(peak_tbl['id'],initial=0)
            peak_tbl = vstack([peak_tbl,peaks_part])
        # if no output table exists we create a new one
        else:
            peak_tbl = peaks_part
        
    # for consistent table output
    for col in peak_tbl.colnames:
        peak_tbl[col].info.format = '%.8g'  
    peak_tbl['fwhm'].info.format = '%.3g' 

    peak_tbl.rename_column('xcentroid','x')
    peak_tbl.rename_column('ycentroid','y')
          
    # calculate astronomical coordinates
    peak_tbl['SkyCoord'] = SkyCoord.from_pixel(peak_tbl['x'],peak_tbl['y'],self.wcs)
    peak_tbl['RaDec'] = peak_tbl['SkyCoord'].to_string(style='hmsdms',precision=2)

    # save the result to the object
    setattr(self,'peaks_tbl',peak_tbl)

    # we save the found positions to a file
    #with open(os.path.join('catalogue',f'peaks_{self.name}.txt'),'w',newline='\n') as f:
    #    peak_tbl[['id','x','y','RaDec']].write(f,format='ascii.no_header',overwrite=True)
    
    print(f'{len(peak_tbl)} sources detected')
    
    return peak_tbl
    

if __name__ == '__main__':
    
    from pymuse.io import MUSEDAP
    NGC628 = MUSEDAP('NGC628')
    sources = detect_sources(NGC628,'OIII5006',StarFinder=IRAFStarFinder,threshold=3,arcsec_to_pixel=0.2)