from .utils import figsize, create_RGB, quick_plot, add_scale
from .map import classification_map
from .pnlf import plot_pnlf, plot_emission_line_ratio
from .distances import compile_distances, plot_distances

__all__ = ['figsize','create_RGB','quick_plot','add_scale','classification_map',
           'plot_pnlf', 'plot_emission_line_ratio', 'compile_distances','plot_distances']