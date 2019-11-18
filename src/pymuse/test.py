import logging    # use logging instead of print

from pymuse.io import MUSEDAP


if __name__ == '__main__':

    logging.basicConfig(filename='info.log',
                    filemode='a',
                    format='%(levelname)s %(name)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

    NGC628 = MUSEDAP('NGC628')