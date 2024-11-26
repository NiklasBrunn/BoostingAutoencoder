module BoostingAutoEncoder

#check which ones are required ...
using Flux;
using Random; 
using Statistics;
using StatsBase;
using DelimitedFiles;
using Distances;
using GZip;
using CSV;
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
    trainBAE, train_gaußianVAE_fixedvariance!, train_gaußianVAE!,
    # loss functions for BAE, timeBAE, and AE versions:
    loss_z, loss, loss_wrapper, jointLoss, jointLoss_wrapper, loss_L1reg, loss_wrapper_L1reg, corpen, loss_correg, loss_wrapper_correg, vae_loss_gaußian_fixedvariance, vae_loss_gaußian, VAE_loss_wrapper, VAE_loss_wrapper_fixedvariance,
    # BAE model architecture:
    LinearLayer, Autoencoder,
    # componentwise boosting functions:
    seq_constr_compL2Boost_jointLoss, seq_constr_compL2Boost, seq_compL2Boost, compL2Boost!, compL2Boost_jointLoss, calcunibeta, calcunibeta_jointLoss,
    # plotting functions:
    vegaheatmap, create_colored_umap_plot, create_latent_umaps, normalized_scatter_top_values, 
    # data simulation functions:
    simulate_10StagesScRNAseq, simulate_3cellgroups3stagesScRNAseq, addstages!,
    # utility functions:
    get_latdim_grads, prcomps, generate_umap, find_zero_columns, split_traintestdata, create_sorted_numlabels_and_datamat, onehotcelltypes, quantile_elements, find_matching_type, find_matching_type_per_BAEdim, get_nonzero_rows, divide_dimensions, reparameterize, get_VAE_latentRepresentation, get_top_selected_genes, adjust_pvalues, coefficients_tTest, predict_celllabels,
    # preprocessing functions:
    standardize, log1transform, downloadcountsandload, phenodata, expressiondata, estimatesizefactorsformatrix, normalizecountdata
#


end