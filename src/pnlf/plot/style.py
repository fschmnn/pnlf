'''
This is the main script for the different plotting routines. It defines 
the common style (pyTeX) and imports parent moduls

if final is set to True, pgf is used as the backend and a copy is saved
in the folder img
'''

final = False 

from pathlib import Path
import numpy as np

cwd = Path(__file__).parent


def figsize(scale=1):
    '''Create nicely proportioned figure

    This function calculates the optimal figuresize for any given scale
    (the ratio between figuresize and textwidth. A figure with scale 1
    covers the entire writing area). Therefor it is important to know 
    the textwidth of your target document. This can be obtained by using
    the command "\the\textwidth" somewhere inside your document.
    '''

    width_pt  = 355.6595                      # textwidth from latex
    in_per_pt = 1.0/72.27                     # Convert pt to inch
    golden    = 1.61803398875                 # Aesthetic ratio 
    width  = width_pt * in_per_pt * scale     # width in inches
    height = width / golden                   # height in inches
    return [width,height]

import matplotlib as mpl


import matplotlib.pyplot as plt

if final:
    plt.style.use(str(cwd / 'TeX.mplstyle'))

# create dictionary for colors
#names = ['blue','orange','red','cyan','green','yellow','purple','pink','brown','gray']
#colors = dict(zip(names,plt.rcParams['axes.prop_cycle'].by_key()['color']))

def newfig(scale=1,ratio=None):
    '''Create a new figure object

    We use the function figsize to create a figure of corresponding size.
    If the option ratio is choosen, the width of the plot is still taken
    from figsize but the ratio of the figure is determined by ratio.
    '''

    # we using jupyter this is required to close open plots
    #plt.clf()
    if not final:
        scale*=2

    size = figsize(scale)
    if not ratio:
        fig = plt.figure(figsize=size)
    else:
        fig = plt.figure(figsize=(size[0],ratio*size[0]))

    return fig



