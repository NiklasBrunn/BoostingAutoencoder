# Infusing structural assumptions into dimension reduction for single-cell RNA sequencing data to identify small gene sets

## Overwiev
This GitHub repository contains all the code and scripts for defining and training a boosting autoencoder (BAE), and to reproduce the results presented in our manuscript *Infusing structural assumptions into dimension reduction for single-cell RNA sequencing data to identify small gene sets*.


## Author information for the manuscript
> First/corresponding authors: Niklas Brunn<sup>12</sup>, Maren Hackenberg<sup>12</sup><br>
> Senior author: Harald Binder<sup>123</sup>
>
> <sup>1</sup> Institute of Medical Biometry and Statistics, Faculty of Medicine and Medical Center, University of Freiburg, Germany<br>
> <sup>2</sup> Freiburg Center for Data Analysis and Modeling, University of Freiburg, Germany<br>
> <sup>3</sup> Centre for Integrative Biological Signaling Studies (CIBSS), University of Freiburg, Germany

## Covered topics

## Repository structure

## Instructions for running code
To run the julia scripts, Julia (we used v1.6.7) has to be [downloaded](https://julialang.org/downloads/) and installed. The required packages and their versions are specified in the `Project.toml` and `Manifest.toml` files in the main folder and automatically loaded/installed at the beginning of each script with the `Pkg.activate()` and `Pkg.instantiate()` commands. See [here](https://pkgdocs.julialang.org/v1.2/environments/) for more info on Julia environments. 

To run the python scripts, we included details about a [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment in (`Environment.yml`) consisting of information about the python version and used packages. A new conda environment can be created from this file. See [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for more details about managing and creating conda environments.
