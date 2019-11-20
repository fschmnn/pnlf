# load useful packages
import logging              # log errors
from pathlib import Path    # handle path to files

# parent folder with data, notebook, src etc.
basedir = Path(__file__).parent.parent.parent

# set up logger for debugging and additional informations
logger = logging.getLogger(__name__)

from configparser import ConfigParser

# configuration file
configfile = basedir / 'config.ini'

# dictionary with the default values
default = {
'data_folder': ''
}

# test if the chosen options exist
config_test = ConfigParser()
config_test.read(configfile)

options = {
'general': ['data_folder']
}

for sec in config_test._sections:
    for var in config_test.items(sec):
        if var[0] not in options[sec]:
            logger.warning('no option {}={} in section {}'.format(var[0],var[1],sec))
del config_test

# initialize the configparser with the default values defined above
config = ConfigParser(default)
config.read(configfile)

data_raw = Path(config.get('general','data_folder'))

# some basic information for debugging
logger.debug(f'Project location {basedir}')
logger.debug(f'Project location {data_raw}')