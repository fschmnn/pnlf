# pymuse

Python package to manipulate, analyse and plot MUSE data.

## Description

For my PhD thesis I'm using spectroscopic data from the [Multi Unit Spectroscopic Explorer](https://www.eso.org/sci/facilities/develop/instruments/muse.html) (MUSE) instrument of the VLT. This data has been observed as part of the [PHANGS](https://sites.google.com/view/phangs/home) collaboration. 

The data is already reduced and advanced data products like emission line maps are available from other members of the collaboration. This package aims to provide a structured starting point for exploring and analysing this dataset.

## Project structure

The structure of this project was inspired by [this blog post](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/) and was set up using [PyScaffold 3.2.3](https://pyscaffold.org/). 


```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── config.ini              <- 
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── notebooks               <- Jupyter notebooks. 
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── catalogues          <- 
│   └── figures             <- Generated plots and figures for reports.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Install for development or create a distribution.
├── src
│   └── pymuse              <- Python package where the main functionality goes.
└── tests                   <- Unit tests which can be run with `py.test`.
```



## Installation


The following instructions show how to set up everything initially:

1.  create an *conda environment* with all required packages

   ```bash
   conda env create -f .\environment.yaml
   ```

    the new environment can then be activated with

   ```bash
   conda activate pymuse
   ```

   `pymuse` was chosen as the name of the environment in `environment.yaml`. In case new packages need to be installed, the file can be altered an updated with

   ```bash
   conda env update -f environment.yaml --prune
   ```

   (this requires an active environment). In case `photutils` isn't installed, use

   ```bash
   conda install photutils -c astropy
   ```

   also useful when working with jupyter notebooks:

   ```bash
   conda install -c conda-forge jupyter_contrib_nbextensions
   conda install -c conda-forge jupyter_nbextensions_configurator
   ```

   

2. Now that we have a new 

   setup for development

   ```bash
   python setup.py develop
   ```

   install distribution with

   ```bash
   python setup.py bdist_wheel
   ```

   



