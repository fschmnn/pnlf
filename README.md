# pymuse

Add a short description here!


## Description

A longer description of your project goes here...



## Installation

the structure of this project was inspired by [this blog post](https://florianwilhelm.info/2018/11/working_efficiently_with_jupyter_lab/) and PyScaffold. The following instructions show how to set up everything initially:

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

   also usefull when working with jupyter notebooks:

   ```bash
   conda install -c conda-forge jupyter_contrib_nbextensions
   conda install -c conda-forge jupyter_nbextensions_configurator
   ```

   

2. Now that we have a new 





## Project structure

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.

```
├── AUTHORS.rst             <- List of developers and maintainers.
├── CHANGELOG.rst           <- Changelog to keep track of new features and fixes.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yaml        <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- Use `python setup.py develop` to install for development or
|                              or create a distribution with `python setup.py bdist_wheel`.
├── src
│   └── PYTHON_PKG          <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `py.test`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```