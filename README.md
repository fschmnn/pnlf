# pymuse

A Python package to manipulate, analyze and plot MUSE data. The name is preliminary and will be changed later to avoid confusion with the existing [PyMUSE](https://github.com/ismaelpessa/PyMUSE) package.

## Description

For my PhD thesis I'm using spectroscopic data from the [Multi Unit Spectroscopic Explorer](https://www.eso.org/sci/facilities/develop/instruments/muse.html) (MUSE) instrument of the VLT. This data has been observed as part of the [PHANGS](https://sites.google.com/view/phangs/home) collaboration. 

The data is already reduced and advanced data products like emission line maps are available from other members of the collaboration. This package aims to provide a structured tool for exploring and analyzing this dataset.

A more detailed description of the functionality is either provided by the docstrings of the functions and classes or by the jupyter notebooks. They are annotated with additional background information on what what's happening.

## Installation

In principle one could clone this repository from [github](https://github.com/fschmnn/pymuse) and use it right away. However to ensure that everything works as intended, a few additional steps are recommended.

1. **Set up conda environment**: It is highly advised to run data science projects in a dedicated environment. This has the advantage that any third party packages have the correct version installed which helps to make the results reproducible. We use *conda* to do this. The required packages are listed in `environment.yaml` and a new environment, called `pymuse` is created with

   ```bash
   conda env create -f .\environment.yaml
   ```

    Every time one opens a new shell, the environment must be activated with

   ```bash
   conda activate pymuse
   ```

   New packages can either be installed by altering the installation file and running

   ```bash
   conda env update -f environment.yaml --prune
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

   

2. **Install the package**: with the dependencies installed, we still need to setup the actual package. To develop the package, simply type

   ```bash
   python setup.py develop
   ```

And that's it. You may have noticed that the project already contains folders and files for unit test and documentations. However neither are currently used but both should eventually be added.

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
│   └── pymuse              <- Python package where the main functionality goes.
└── tests                   <- Unit tests which can be run with `py.test`.
```

## Usage

Here are a few examples on what to do with this package

1. Read a fits file from the MUSE data release pipeline (MUSEDAP). This assumes that you have a folder `NGC628` inside your `raw` folder (for more details, see the documentation of `ReadLineMaps`)

   ```python
   from pathlib import Path
   from pymuse.io import ReadLineMaps
   
   data_folder = Path('../data/raw')
   NGC628 = ReadLineMaps(data_folder / 'NGC628')
   ```

2. Search for sources in this file

   ```python
   form photutils import DAOStarFinder
   from pymuse.detection import detect_unresolved_sources
   
   sources = detect_unresolved_sources(NGC628,['OIII5006'],DAOStarFinder)
   ```


## Todo list



* [ ] PSF size
* [x] Mock sources for completeness limit (insert into real data, ignoring overlap), improve matching algorithm
* [ ] Errors (estimate error from CUBE maps)
* [ ] improve maximum likelihood (avoid binning with least square) Error?
* [ ] Emission line diagnostics: more careful with NaN values
