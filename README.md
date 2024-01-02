# Infusing structural assumptions into dimension reduction for single-cell RNA sequencing data to identify small gene sets

## Overwiev
This GitHub repository contains all the code and scripts for defining and training a boosting autoencoder (BAE), and to reproduce the results presented in our [manuscript](https://github.com/NiklasBrunn/BoostingAutoencoder/tree/main) *Infusing structural assumptions into dimension reduction for single-cell RNA sequencing data to identify small gene sets*.

## Author information for the manuscript
> First/corresponding authors: Niklas Brunn<sup>12</sup>, Maren Hackenberg<sup>12</sup><br>
> Senior author: Harald Binder<sup>123</sup>
>
> <sup>1</sup> Institute of Medical Biometry and Statistics, Faculty of Medicine and Medical Center, University of Freiburg, Germany<br>
> <sup>2</sup> Freiburg Center for Data Analysis and Modeling, University of Freiburg, Germany<br>
> <sup>3</sup> Centre for Integrative Biological Signaling Studies (CIBSS), University of Freiburg, Germany

## Covered topics


![](figures/ModelOverview.png)

## Repository structure
The `scripts` subfolder consists of scripts for:
  * 1. Preprocessing: Julia and Python scripts for downloading and preprocessing the scRNA-seq datasets.
  * 2. Simulation: Julia scripts for generating two different scRNA-seq like datasets.
  * 3. BAE application: Julia scripts for application of the BAE and timeBAE to the preprocessed simulated and real-world scRNA-seq datasets.

The `tutorials` subfolder consists of a Julia Jupyter notebook illustrating the functionality of the BAE on simulated scRNA-seq data.

The `src` subfolder consists of Julia source code files for the BAE approach. 

All plots and data that are downloaded or generated during running the scripts is stored in the subfolder `figures` or `data` respectively. 

## Instructions for running code
To run the julia scripts, Julia (we used v1.6.7) has to be [downloaded](https://julialang.org/downloads/) and installed. The required packages and their versions are specified in the `Project.toml` and `Manifest.toml` files in the main folder and automatically loaded/installed at the beginning of each script with the `Pkg.activate()` and `Pkg.instantiate()` commands. See [here](https://pkgdocs.julialang.org/v1.2/environments/) for more info on Julia environments. 

To run the python scripts, we included details about a [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment in (`Environment.yml`) consisting of information about the python version and used packages. A new conda environment can be created from this file. See [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for more details about managing and creating conda environments.

For running the BAE and timeBAE analysis on the simulated scRNA-seq data, you can directly run the files `main_sim3cellgroups3stagessScRNAseq.jl`, `main_sim10stagesScRNAseq.jl`, `modelcomparison_sim10stagesScRNAseq.jl`. For running the BAE analysis on the cortical mouse data from [*Tasic et al.*](https://www.nature.com/articles/nn.4216) first run the script `get_corticalMouseData.jl` followed by `preprocess_corticalMouseData.py` for downloading and preprocessing the data. Subsequently, analysises can be performed by running the scripts `main_corticalMouseData.jl` and `subgroupAnalysis_corticalMouseData.jl`. For running the timeBAE analysis on the embyoid body data from [*Moon et al.*](https://www.nature.com/articles/s41587-019-0336-3) first run the preprocessing scripts `get_and_preprocess_embryoidBodyData.py` and `cluster_and_filter_embryoidBodyData.py` followed by `main_embryoidBodyData.jl` for computing the results.
