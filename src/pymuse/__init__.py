# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound


from configparser import ConfigParser
import os
import json

# we reference everything relative to this path
basedir = os.path.dirname(__file__)


# dictionary with the default values
default = {
'data_folder': ''
}

# ======================================================================
# make sure all chosen options exist. this doesn't work with default values
# ======================================================================

config_test = ConfigParser()
configfile = os.path.join(basedir,'config.ini')
config_test.read(configfile)

options = {
'general': ['data_folder']
}

for sec in config_test._sections:
    for var in config_test.items(sec):
        if var[0] not in options[sec]:
            print('no option {}={} in section {}'.format(var[0],var[1],sec))

# ======================================================================
# assign parameters to variables
# ======================================================================

# initialize the configparser with the default values defined above
config = ConfigParser(default)
configfile = os.path.join(basedir,'config.ini')
config.read(configfile)

data_folder = config.get('general','data_folder')
