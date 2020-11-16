'''import frequently used packages

instead of importing all those packages each time by hand, simply use
    
    `from pnlf.packages import *`
'''

# ---------------------------------------------------------------------
# basic os stuff
# ---------------------------------------------------------------------
import os                      # filesystem related stuff
from pathlib import Path       # use instead of os.path and glob
import sys                     # mostly replaced by pathlib

import errno                   # more detailed error messages
import warnings                # handles warnings
import logging                 # use logging instead of print

# ---------------------------------------------------------------------
# datastructures and scientific computing
# ---------------------------------------------------------------------
from collections import OrderedDict      # dict that remembers order
import json                              # load and write json files
import yaml
import numpy as np                       # arrays
import scipy as sp                       # useful functions


# ---------------------------------------------------------------------
# plotting routines
# ---------------------------------------------------------------------
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

# ---------------------------------------------------------------------
# astronomy related stuff
# ---------------------------------------------------------------------
import astropy

from astropy.table import Table,QTable   # useful datastructure
from astropy.table import vstack         # combine multiple tables
from astropy.nddata import NDData, StdDevUncertainty, Cutout2D  

from astropy.io import ascii,fits        # open text and fits files

from astropy.wcs import WCS              # handle coordinates
from astropy.coordinates import Angle    # work with angles (e.g. 1°2′3″)
from astropy.coordinates import SkyCoord # convert pixel to sky coordinates
from astropy.coordinates import Distance # convert between pc and m-M

from astropy.stats import sigma_clipped_stats  

import astropy.units as u
import astropy.constants as c

from photutils import CircularAperture

# ---------------------------------------------------------------------
# useful directories
# ---------------------------------------------------------------------
basedir = Path(__file__).parent.parent.parent
pathdir = Path(__file__).parent

plt.style.use( str(pathdir / 'TeX.mplstyle'))