# pnlf

*last updated 2020.08.25*

A Python package to analyse MUSE data and measure the Planetary Nebula Luminosity Function (PNLF).



## Description

This repository contains the code for Scheuermann et al. (submitted). 

The data used in this project has been observed for the [PHANGS](https://sites.google.com/view/phangs/home) collaboration (Emsellem et al. [submitted](https://arxiv.org/abs/2110.03708)).

The *planetary nebula luminosity function* (PNLF) is an empirical relation that can be used to measure the distance to nearby galaxies. 

![PNLF](https://raw.githubusercontent.com/fschmnn/pnlf/master/references/pnlf.png)

A detailed description of the functionality is either provided by the docstrings of the functions and classes or in the jupyter notebooks. They are annotated with additional background information on what what's happening.



## Installation

In principle one could clone this repository from [github](https://github.com/fschmnn/pnlf) and use it right away. However to ensure that everything works as intended, a few additional steps are recommended.

1. **Set up conda environment**: It is highly advised to run data science projects in a dedicated environment. This has the advantage that any third party packages have the correct version installed which helps to make the results reproducible. We use *conda* to do this. The required packages are listed in `environment.yml` and a new environment, called `pymuse` is created with

   ```bash
   conda env create -f .\environment.yml
   ```

    Every time one opens a new shell, the environment must be activated with

   ```bash
   conda activate pymuse
   ```

   New packages can either be installed by altering the installation file and running

   ```bash
   conda env update -f environment.yml --prune
   ```

   or by typing

   ```bash
   conda install photutils -c astropy
   ```

   Both cases require an active environment. Lastly, a useful addition when working with *jupyter notebooks* are extensions which can be activated with

   ```bash
   conda install -c conda-forge jupyter_contrib_nbextensions
   conda install -c conda-forge jupyter_nbextensions_configurator
   ```

   The extensions can then be activated in the `Nbextensions` tab of the jupyter explorer

2. **Install the package**: with the dependencies installed, we still need to setup the actual package. To develop the package, simply type

   ```bash
   python setup.py develop
   ```

And that's it. You may have noticed that the project already contains folders and files for unit test and documentations. However neither are currently used but both should eventually be added.

The raw data for the project is supposed to be stores in `/data/raw`. However since I do not have enough space on my hard drive, I keep those files on an external drive. For easy access I created a symbolic link between the two folders like so 

```
mklink /J data\raw g:\Archive
```

(see this [link](https://www.howtogeek.com/howto/16226/complete-guide-to-symbolic-links-symlinks-on-windows-or-linux/) for more information on symbolic links)



## Project structure

The structure of this project was inspired by [this blog post](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/) and was set up using [PyScaffold 3.2.3](https://pyscaffold.org/). It consists of the following files and folders: 

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
├── scripts                 <- Python script that are used for final run
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Install for development or create a distribution.
├── src
│   └── pnlf                <- Python package where the main functionality goes.
└── tests                   <- Unit tests which can be run with `py.test`.
```

## Usage

Here are a few examples on what to do with this package

1. Read a fits file from the MUSE data release pipeline (MUSEDAP). This assumes that you have a folder `NGC628` inside your `raw` folder (for more details, see the documentation of `ReadLineMaps`)

   ```python
   from pathlib import Path
   from pnlf.io import ReadLineMaps
   
   data_folder = Path('../data/raw')
   NGC628 = ReadLineMaps(data_folder / 'NGC628')
   ```

2. Search for sources in this file

   ```python
   from photutils import DAOStarFinder
   from pnlf.detection import detect_unresolved_sources
   
   sources = detect_unresolved_sources(NGC628,['OIII5006'],DAOStarFinder)
   ```
