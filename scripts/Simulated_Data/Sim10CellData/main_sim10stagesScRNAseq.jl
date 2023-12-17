#-------------------------------------------------------
# Modelcomparison on simulated 10 stages scRNA-seq data: 
#-------------------------------------------------------


#------------------------------
# Setup: 
#------------------------------
#---Activate the enviroment:
using Pkg;

# All paths are relative to the repository main folder

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

#---Load packages:
using Flux;
using Random; 
using Statistics;
using DelimitedFiles;
using Plots;
using LinearAlgebra;
using DataFrames;
using VegaLite;
using UMAP;
using ProgressMeter;


#------------------------------
# Define paths and include functions:
#------------------------------
#---Set paths:
projectpath = joinpath(@__DIR__, "../../../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/sim10stagesScRNAseq/";
figurespath = projectpath * "figures/sim10stagesScRNAseq/";
if !isdir(figurespath)
    # Create the folder if it does not exist
    mkdir(figurespath)
end

#---Include functions:
include(srcpath * "utility.jl");
include(srcpath * "model.jl");
include(srcpath * "losses.jl");
include(srcpath * "training.jl");
include(srcpath * "boosting.jl");
include(srcpath * "preprocessing.jl");
include(srcpath * "simulations.jl");
include(srcpath * "plotting.jl");



#------------------------------
# Generate data:
#------------------------------
#---Seed for reproducibility:
dataseed = 1; 

#---Hyperparameter for data generation:
rescale_factor = 1.5; 

#---Data generation:
(X, X_dicho) = simulate_10StagesScRNAseq(dataseed; rescale_val=rescale_factor);
X_st = simulate_10StagesScRNAseq(dataseed; rescale_val=1)[1];

p = size(X, 2);


#---Save the data matrices:
#writedlm(datapath * "sim10stagesScRNAseqData_binarized.txt", X_dicho);
#writedlm(datapath * "sim10stagesScRNAseqData_standardized_rescaleVal$(rescale_factor).txt", X);
#writedlm(datapath * "sim10stagesScRNAseqData_standardized", X_st);

#---Create and save plots of the data matrices:
vegaheatmap(X_dicho; 
    path=figurespath * "sim10stagesScRNAseqData_binary.pdf", xlabel="Gene", ylabel="Observation",
    legend_title="Gene expression", color_field="value:o", scheme="paired", save_plot=true
);
vegaheatmap(X; 
    path=figurespath * "sim10stagesScRNAseqData_st_rescaledNoiseFeatures.pdf", xlabel="Gene", ylabel="Observation",
    legend_title="Gene expression", color_field="value", scheme="inferno", save_plot=true, set_domain_mid=true
);
vegaheatmap(X_st; 
    path=figurespath * "sim10stagesScRNAseqData_st.pdf", xlabel="Gene", ylabel="Observation",
    legend_title="Gene expression", color_field="value",scheme="inferno", save_plot=true, set_domain_mid=true
);



#------------------------------
# Define and train BAE:
#------------------------------
#---Seeds for reproducibility:
#seeds for comparison of different decoder parameter initializations: 
#5, 12, 27, 357, 468, 700, 819, 937, 1923, 34825
modelseed = 700; 
batchseed = 1;

#---Hyperparameters for training:
mode = "alternating"; #options are: "alternating", "jointLoss"

zdim = 10;

epochs = 15; #epochs used for generating the plots: 1, 15, 150

batchsize = 800;

ϵ = 0.01; 



#---Build BAE:
encoder = LinearLayer(zeros(p, zdim));

Random.seed!(modelseed);
decoder = Chain(
            Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
            Dense(p, p, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Train BAE (alternating version):
Random.seed!(batchseed);
B = trainBAE(X, BAE; mode=mode, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);
B_perm = hcat(B[:, 8], B[:, 4], B[:, 6], B[:, 10], B[:, 2],
              B[:, 5], B[:, 9], B[:, 7], B[:, 1], B[:, 3]
); #permute the columns of B for better visualization

#---Compute the latent representation and the correlation matrix:
Z = X * B;
absCor_Z = abs.(cor(Z, dims=1));
Z_perm = X * B_perm;
absCor_Z_perm = abs.(cor(Z_perm, dims=1));

#---Save the BAE encoder weight matrix:
#writedlm(datapath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs).txt", B);

#---Create and save plots:
vegaheatmap(B'; 
    path=figurespath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_withTitle.pdf", 
    xlabel="Gene", ylabel="Latent Dimension", Title="Epoch $(epochs)", legend_title="Weight value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
);
vegaheatmap(B'; 
    path=figurespath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs).pdf", 
    xlabel="Gene", ylabel="Latent Dimension", legend_title="Weight value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
);
vegaheatmap(Z; 
    path=figurespath * "BAE_latentRep_zdim$(zdim)_epochs$(epochs).pdf", 
    xlabel="Latent Dimension", ylabel="Observation", legend_title="Representation value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
);
vegaheatmap(absCor_Z; 
    path=figurespath * "BAE_latentDimsCor_zdim$(zdim)_epochs$(epochs).pdf.pdf", 
    xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
    color_field="value", scheme="reds", save_plot=true
);
vegaheatmap(B_perm'; 
    path=figurespath * "perm_BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs).pdf", 
    xlabel="Gene", ylabel="Latent Dimension", legend_title="Weight value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
);
vegaheatmap(Z_perm; 
    path=figurespath * "perm_BAE_latentRep_zdim$(zdim)_epochs$(epochs).pdf", 
    xlabel="Latent Dimension", ylabel="Observation", legend_title="Representation value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
);
vegaheatmap(absCor_Z_perm; 
    path=figurespath * "perm_BAE_latentDimsCor_zdim$(zdim)_epochs$(epochs).pdf.pdf", 
    xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
    color_field="value", scheme="reds", save_plot=true
);

#---Create and save plots (different decoder init.):
vegaheatmap(B'; 
    path=figurespath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_mseed$(modelseed).pdf", 
    xlabel="Gene", ylabel="Latent Dimension", legend_title="Weight value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
);