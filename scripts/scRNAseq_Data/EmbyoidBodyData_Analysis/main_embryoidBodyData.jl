#------------------------------
# timeBAE applied to the embryoid body scRNA-seq data: 
#------------------------------


#------------------------------
# Setup: 
#------------------------------
#---Activate the enviroment:
using Pkg;

# all paths are relative to the repository main folder

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
using StatsBase;
using Distances;
using CSV;
using ColorSchemes;
using ProgressMeter;



#------------------------------
# Define paths and include functions:
#------------------------------
#---Set paths:
projectpath = joinpath(@__DIR__, "../../../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/embryoidBodyData/";
figurespath = projectpath * "figures/embryoidBodyData/"; 
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
include(srcpath * "plotting.jl");



#------------------------------
# Load and format data:
#------------------------------
#---Load Data generated in the (python) preprocessing script for embryoidBodyData: 
X = readdlm(datapath * "EB_dataMat_DEGs.txt", Float32);
X_st = standardize(X);

DEGs = string.(vec(readdlm(datapath * "EB_data_DEGs.txt"))); 

timepoints = Int.(vec(readdlm(datapath * "EB_data_filt_timepoints.txt")));

clusters = Int.(vec(readdlm(datapath * "EB_data_filt_leiden_res0_025.txt")[2:end]));

EB_umap_coords_df = CSV.File(datapath * "EB_data_filt_umap_coords.csv") |> DataFrame;
EB_umap_coords = Matrix(EB_umap_coords_df[:, ["UMAP1", "UMAP2"]]);

EB_df = DataFrame(X, :auto);
EB_df[!, "timepoints"] = timepoints;

L = [];
L_st = [];
T = [];
for t in unique(timepoints)
    M= Matrix(EB_df[EB_df[:, "timepoints"].==t, 1:end-1])
    M_st = standardize(M)
    Y = EB_df[EB_df[:, "timepoints"].==t, end]
    push!(L, M)
    push!(L_st, M_st)
    push!(T, Y)
end

n, p = size(X);



#------------------------------
# Define and train timeBAE:
#------------------------------
#---Seeds for reproducibility:
modelseed = 1; 
batchseed = 5; 

#---Hyperparameters for training:
zdim = 4; 

batchsize = 1500; 

epochs = 15; 

ϵ = 0.01; 

mode = "alternating";



#---Build BAE:
encoder = LinearLayer(zeros(p, zdim));

Random.seed!(modelseed);
decoder = Chain(
            Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
            Dense(p, p, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Train timeBAE (jointLoss mode):
Random.seed!(batchseed);
B = trainBAE(L_st, BAE; mode=mode, time_series=true, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

B_perm = zeros(size(X_st, 2));
for dim in 1:zdim
    for t in 1:length(L)
        B_perm = hcat(B_perm, B[:, (t-1)*zdim+dim])
    end
end
B_perm = B_perm[:, 2:end];


#---Compute the latent representation and the correlation matrix:
Z_perm = X_st * B_perm;
absCor_Z_perm = abs.(cor(Z_perm, dims=1));

#---Save the timeBAE encoder weight matrix:
#writedlm(datapath * "timeBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize)_modelseed$(modelseed)_batchseed$(batchseed).txt", B);



#------------------------------
# Visualization:
#------------------------------
plotseed = 42;

#---re-name timepoint lables for plots: 
timepoints_plot = copy(timepoints);
label = 1
for t in unique(timepoints)
    timepoints_plot[timepoints_plot .== t] .= label
    label+=1
end

#---re-name cluster lables for plots:
clusters_plot = copy(clusters);
label = 3
for c in [2, 1, 0]
    clusters_plot[clusters_plot .== c] .= label
    label-=1
end

#---Create a UMAP plot of cells colored by measurement time points:
create_colored_umap_plot(X_st, timepoints_plot, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=true, path=figurespath * "/embryoidBodyData_umap_colBy_timepoints.pdf", 
                         colorlabel="Timepoint", legend_title="Time point", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of filtered EB data colored by timepoints", title_fontSize=15.0
);

#---Create a UMAP plot of cells colored by cluster membership:
create_colored_umap_plot(X_st, clusters_plot, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=true, path=figurespath * "/embryoidBodyData_umap_colBy_leidenCluster.pdf", 
                         colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of filtered EB data colored by Leiden clustering (res 0.025)", title_fontSize=15.0,
                         scheme="dark2"
);

#---Create a UMAP plot of cells colored by the representation value of each timeBAE latent dimension:
create_latent_umaps(X_st, plotseed, Z_perm, "timeBAE"; 
                    figurespath=figurespath * "/timeBAE_(pcaUMAP)",
                    precomputed=true, embedding=EB_umap_coords, save_plot=true, 
                    legend_title="Representation value", image_type=".pdf",  marker_size="10"
);


#---Creating Scatterplot showing top selected genes per latent dimension:
for l in 1:zdim*length(L)
    pl = normalized_scatter_top_values(B_perm[:, l], DEGs; top_n=15, dim=l)
    savefig(pl, figurespath * "scatterplot_genes_timeBAE_latdim$(l).pdf")
end


#---Heatmap of absolute Pearson correlation coefficients between timeBAE latent dimensions:
vegaheatmap(absCor_Z_perm; path=figurespath * "/abscor_latentrep_timeBAE.pdf", 
            ylabel="Latent dimensions", xlabel="Latent dimensions", legend_title="Correlation",
            scheme="reds", save_plot=true
);