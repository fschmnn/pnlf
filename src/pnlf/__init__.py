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


import sys
import logging

logger = logging.getLogger(__name__)
env = sys.executable.split('\\')[-2]

if env != 'pymuse':
    logger.warning(f'\nYou are currently in the enviorment `{env}`.\n' +
                   f'It is recommendet to use this package in a dedicated enviornment `pymuse`.\n' + 
                   f'To do this type `conda activate pymuse`.'
                   )