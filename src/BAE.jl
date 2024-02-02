module BoostingAutoEncoder

#check which ones are required ...
using Flux;
using Random; 
using Statistics;
using StatsBase;
using DelimitedFiles;
using GZip;
using XLSX;
using Plots;
using LinearAlgebra;
using DataFrames;
using VegaLite;
using UMAP;
using ProgressMeter;
using ColorSchemes;
using Distributions;

include("model.jl");
include("utils.jl");
include("losses.jl");
include("training.jl");
include("boosting.jl");
include("preprocessing.jl");
include("simulations.jl");
include("plotting.jl");

export 
    # training function for BAE and timeBAE:
    trainBAE, 
    # loss functions for BAE, timeBAE, and AE versions:
    loss_z, loss, loss_wrapper, jointLoss, jointLoss_wrapper, loss_L1reg, loss_wrapper_L1reg, corpen, loss_correg, loss_wrapper_correg,
    # BAE model architecture:
    LinearLayer, Autoencoder,
    # componentwise boosting functions:
    seq_constr_compL2Boost_jointLoss, seq_constr_compL2Boost, seq_compL2Boost, compL2Boost!, compL2Boost_jointLoss, calcunibeta, calcunibeta_jointLoss,
    # plotting functions:
    vegaheatmap, create_colored_umap_plot, create_latent_umaps, normalized_scatter_top_values, 
    # data simulation functions:
    simulate_10StagesScRNAseq, simulate_3cellgroups3stagesScRNAseq, addstages!,
    # utility functions:
    get_latdim_grads, prcomps, generate_umap, find_zero_columns, split_traintestdata, create_sorted_numlabels_and_datamat, onehotcelltypes,
    # preprocessing functions:
    standardize, log1transform, downloadcountsandload, phenodata, expressiondata, estimatesizefactorsformatrix, normalizecountdata
#


end