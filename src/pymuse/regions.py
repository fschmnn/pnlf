import logging

import numpy as np 
from matplotlib.pyplot import figure, savefig, show

from astropy.wcs import WCS
from astropy.visualization import simple_norm

from skimage.measure import regionprops, find_contours

logger = logging.getLogger(__name__)

class Regions:
    def __init__(self,mask=None,coords=None,labels=None,shape=None,projection=None,bkg=0):
        '''Class to store regions (mask, coordinates, labels)
        
        Can be initialized either with a mask (=data) where each region
        has a unique id (integer >=1) or with the coordinates of the 
        regions. For the former, regionpros is used to extract the 
        coordinates of the regions. `regionprops` considers 0 as 
        background and only searches for regions with >=1. Therefore 
        regions that are labeld with 0 should use the `bkg` 
        keyword to shift the smallest region number to 1.
        For the second case, the coordinates of the regions must be 
        provided together with their labels and the shape of the image.
        
        Parameters
        ----------
        data : ndarray 
            Array with labeld regions
        regions : list of tules
            list that contains the coordinates of the regions
        labels : list 
            list with the labels associated with the regions
        projection : Header or WCS
            header or wcs object to which the coordinates are projected  
        bkg : int
            used to shift the mask to avoid regions that are labeld 0
        '''
        
        if isinstance(projection,fits.header.Header):
            self.wcs = WCS(projection)
            self.shape = (projection['NAXIS2'],projection['NAXIS1'])
        elif isinstance(projection,WCS):
            self.wcs = projection
        else:
            raise ValueError('projection not understood')
        
        # if data is specified we ignore the other parameters  
        if isinstance(mask,np.ndarray):
            self.mask    = mask
            self.shape   = mask.shape
            self.coords = []
            self.labels  = []
            for prop in regionprops(mask.astype(int)-bkg):
                x,y = prop.coords.T
                self.coords.append((x,y))
                self.labels.append(prop.label+bkg)
        # if not we check if all three of the other parameters are given
        elif any([coords,labels,shape]) and not all([coords,labels,shape]):
            raise ValueError('you need to specify regions, labels and shape together')
        elif all([coords,labels,shape]):
            self.coords = coords
            self.labels  = labels
            self.shape   = shape
            self.mask_from_regions()
        else:
            raise ValueError('you need to provide some input')
        
        self.contours = self.find_contours()
        logger.info('initialized with {} regions'.format(len(self.labels)))
        
       
    def find_contours(self,level=0.5):
        '''Find contours in data
        
        Neighboring regions that touch are detected as one region.
        Therefore the number of contours does not equal the number
        of regions.
        '''
        mask = self.mask
        mask[np.isnan(mask)]=-1        
        contours = find_contours(mask,level)
        mask[mask==-1] = np.nan
        return contours
    
    def mask_from_regions(self):
        '''Create an image that contains only the regions in regions_id
        
        Because we loop over each region this is slow.
        
        Parameters
        ----------
        regions : list
            list of tuples (x,y) with coordinates of the regions
        labels : list
            list of the labels of the regions
        shape : tuple
            tuple with the shape of the output image
            
        Returns 
        -------
        image : ndarray
            Image from the original data where only the regions in 
            regions_id are contained
        '''
                
        mask = np.zeros(self.shape)*np.nan
        for r,l in zip(self.coords,self.labels):
            mask[r] = l
        self.mask = mask
    
    def reproject(self,output_projection,shape=None):
        '''Reproject coordinates to output_projection

        This function takes a list of tuples with coordinates and 
        projects them from the wcs of the `input_header` (of this class)
        to the `output_projection`. 

        Parameters 
        ----------
        input_pix : list of tuples
            List of tuples where each tuple (x,y) contains the coordinates
        output_projection : Header or WCS
            header or wcs object to which the coordinates are projected  

        Returns
        -------
        output_pix : list of coordinates
            list of projected coordinates of the regions
        ''' 
        
        if not self.wcs:
            raise AttributeError('no wcs information for original data')
        if isinstance(output_projection,fits.header.Header):
            wcs = WCS(output_projection)
            shape = (output_projection['NAXIS2'],output_projection['NAXIS1'])
        elif isinstance(output_projection,WCS):
            wcs = output_projection
            if not shape:
                raise ValueError('you need to specify a shape')
        else:
            raise ValueError('output_projection not understood')
        
        # those transformations are weird. we need to switch the columns before and after
        world = [self.wcs.all_pix2world(np.array(c[::-1]).T,0) for c in self.coords]
        output_pix = []
        for c in world:
            x,y = np.unique(wcs.all_world2pix(c,0).astype(int),axis=0).T[::-1]
            output_pix.append((x,y))
        return Regions(coords=output_pix,labels=self.labels,shape=shape,projection=wcs)
      
    def plot(self,image=None,regions=True,contours=True,filename=None,**kwargs):
        '''Plot the regions in regions_id

        This uses the built in function `select_regions` to select
        the regions in `regions_id` to create an image with those
        regions and plots them. If a filename is specified, the 
        plot is also saved
        '''

        if not isinstance(image,np.ndarray) and not regions and not contours:
            raise ValueError('you must select something to plot')

        fig = plt.figure(figsize=(7,7))
        ax  = fig.add_subplot(111,projection=self.wcs)
        ax.coords[0].set_major_formatter('hh:mm:ss.s')
        ax.coords[1].set_major_formatter('dd:mm:ss')

        if isinstance(image,np.ndarray):
            norm = simple_norm(image,'linear',clip=False,percent=99)
            ax.imshow(image,norm=norm,cmap=plt.cm.gray)  

        if regions:
            ax.imshow(self.mask,alpha=0.6,cmap=plt.cm.Reds)

        if contours:
            for cont in self.contours:
                ax.plot(cont[:,1], cont[:,0], 'k-',lw=0.2)

        ax.set(**kwargs)

        if filename:
            plt.savefig(filename,dpi=600)
        plt.show()        

    
    def __repr__(self):
        pass