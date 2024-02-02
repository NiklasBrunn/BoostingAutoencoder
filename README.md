# Infusing structural assumptions into dimension reduction for single-cell RNA sequencing data to identify small gene sets

## Overwiev
This GitHub repository contains the code and scripts to define and train a boosting autoencoder (BAE) and to reproduce the results presented in our [manuscript](https://github.com/NiklasBrunn/BoostingAutoencoder/tree/main).

## Author information for the manuscript
> First/corresponding authors: Niklas Brunn<sup>12*+</sup>, Maren Hackenberg<sup>12*</sup><br>
> Co-author: Tanja Vogel <sup>3</sup><br>
> Senior author: Harald Binder<sup>124</sup>
>
> <sup>1</sup> Institute of Medical Biometry and Statistics, Faculty of Medicine and Medical Center, University of Freiburg, Germany<br>
> <sup>2</sup> Freiburg Center for Data Analysis and Modeling, University of Freiburg, Germany<br>
> <sup>3</sup> Institute of Anatomy and Cell Biology, Department of Molecular Embryology, Faculty of Biology, University of Freiburg, Germany<br>
> <sup>4</sup> Centre for Integrative Biological Signaling Studies (CIBSS), University of Freiburg, Germany<br>
> <sup>*</sup> Corresponding author<br>
> <sup>+</sup> Repository owner

## What is it all about? 
Dimension reduction approaches are widely used for exploring cellular heterogeneity in single-cell RNA sequencing (scRNA-seq) data, e.g., for identifying two-dimensional visual representations where cell groups can be disentangled, followed by post-hoc analyses. While most approaches are data-driven or are challenging to interprete, it might still be useful to incorporate assumptions that reflect intuition on the underlying structure or the experimental design already as part of the dimension reduction. E.g., dimensions that help to distinguish between cell groups intuitively should be characterized by distinct small sets of genes, or the design in a time-series experiment should be incorporated such that temporal changes of cell states are characterized by gradual changes in corresponding gene sets.  
We combine the advantages of two machine learning approaches, namely autoencoders for dimension reduction via deep learning and boosting for formalizing assumptions. Specifically, we use a componentwise boosting approach, which selects small sets of characteristic genes for each dimension, and allows for tailoring the selection logic to encode further assumptions, such as distinct cell groups or temporal patterns. Our approach facilitates interpretability by selecting different small sets of genes during optimization, where the gene sets explain the learned patterns in latent dimensions.

![](figures/ModelOverview.png)

We illustrate the approach in a scRNA-seq dataset of cortical neurons, where it captures different cell types in distinct dimensions and identifies corresponding marker genes. In particular, we could also capture very small cell groups. Similarly, encoding assumptions that reflect the experimental design allowed for extracting temporal development patterns and corresponding gene programs in an application to time-series data. These examples demonstrate the general benefit of incorporating structural knowledge into dimension reduction for scRNA-seq data.  

## Repository structure
The `scripts` subfolder consists of scripts for:
  * 1. Preprocessing: Julia and Python scripts for downloading and preprocessing the scRNA-seq datasets.
  * 2. Simulation: Julia scripts for generating two different scRNA-seq-like datasets.
  * 3. BAE application: Julia scripts for the BAE and timeBAE application to the preprocessed simulated and real-world scRNA-seq datasets.

The `tutorials` subfolder consists of a Julia Jupyter notebook illustrating the functionality of the BAE on simulated scRNA-seq data.

The `src` subfolder consists of Julia source code files for the BAE approach. 

All plots and data downloaded or generated while running the scripts are stored in the subfolder `figures` or `data`, respectively. 

## Installation
- Git should be installed on your computer. You can download and install it from [Git's official website](https://git-scm.com/downloads).

0.1. **Open your terminal**
   - On macOS or Linux, open the Terminal application.
   - On Windows, you can use Command Prompt, PowerShell, or Git Bash.

0.2. **Navigate to your desired directory**
   - Use the `cd` command to change to the directory where you want to clone the repository.
   - Example (macOS): To change to a directory named `MyProjects` on your desktop, you would use:
     ```bash
     cd ~/Desktop/MyProjects
     ```
   - Example (Windows): To change to a directory named `MyProjects` on your desktop, you would use:
     ```bash
     cd C:\Users\[YourUsername]\Desktop\MyProjects
     ```
     
0.3. **Clone the repository**
   - Use the `git clone` command followed by the URL of the repository.
   - You can find the URL on the repository's GitHub page.
   - Example:
     ```bash
     git clone https://github.com/NiklasBrunn/BoostingAutoencoder/tree/main
     ```

1. **Install Julia**
   - To run the Julia scripts, [Julia v1.6.7](https://julialang.org/downloads/) has to be downloaded and installed manually by the user. The required packages and their versions are specified in the `Project.toml` and `Manifest.toml` files in the main folder and automatically loaded/installed at the beginning of each script with the `Pkg.activate()` and `Pkg.instantiate()` commands. See [here](https://pkgdocs.julialang.org/v1.2/environments/) for more info on Julia environments. 

2. **Install Python**
   - To run the Python scripts, we included details about a [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) environment in (`Environment.yml`) consisting of information about the Python version and used packages. A new conda environment can be created from this file. See [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) for more details about managing and creating conda environments. Follow these steps to set up your development environment:

2.1. **Navigate to the project directory**
   - Navigate to the directory of the cloned GitHub repository:
     ```bash
     cd ~/BoostingAutoencoder
     ```
       
2.2. **Create the conda environment**
   - Create a new conda environment that is named as specified in the `Environment.yml` file (in this case it is named `BAE-env`):
     ```bash
     conda env create -f Environment.yml
     ```

2.3. **Use the BAE conda environment for running python code**
   - Once the environment is created, select it as the kernel for running the python code in the repository.


## Instructions for running scripts
1. **Simulated scRNA-seq data**
   - For running the BAE and timeBAE analysis on the simulated scRNA-seq data, you can directly run the files `main_sim10stagesScRNAseq.jl`, `modelcomparison_sim10stagesScRNAseq.jl`, `main_sim3cellgroups3stagessScRNAseq.jl`.

2. **Cortical mouse scRNA-seq data**
   - For running the BAE analysis on the cortical mouse data from [Tasic et al.](https://www.nature.com/articles/nn.4216) first, run the script `get_corticalMouseData.jl` followed by `preprocess_corticalMouseData.py` for downloading and preprocessing the data. Subsequently, analysis can be performed by running the scripts `main_corticalMouseData.jl` and `subgroupAnalysis_corticalMouseData.jl`.
3. **Embryoid body scRNA-seq data**
   - For running the timeBAE analysis on the embryoid body data from [Moon et al.](https://www.nature.com/articles/s41587-019-0336-3) first run the preprocessing scripts `get_and_preprocess_embryoidBodyData.py` and `cluster_and_filter_embryoidBodyData.py` followed by `main_embryoidBodyData.jl` for generating the results.
