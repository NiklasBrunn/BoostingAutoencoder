#------------------------------
# BAE applied to the embryoid body scRNA-seq data: 
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
figurespath = projectpath * "figures/embryoidBodyData_HVGs/";

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
#---EB markers (from the Phate Paper):
markerGenes = ["TAL1 ", "CD34", "PECAM1", "HOXD1", "HOXB4", "HOXD9", "HAND1", "GATA6", 
               "GATA5", "TNNT2", "TBX18", "TBX15", "PDGFRA", "SIX2", "TBX5", "WT1", 
               "MYC", "LEF1", "SOX10", "FOXD3", "PAX3", "SOX9", "HOXA2", "OLIG3", 
               "ONECUT2", "KLF7", "ONECUT1", "MAP2", "ISL1", "DLX1", "HOXB1", "NR2F1",
               "LMX1A", "DMRT3", "OLIG1", "PAX6", "NPAS1", "SOX1", "NKX2-8", "EN2", 
               "ZBTB16", "SOX17", "FOXA2", "EOMES", "T", "GATA4", "ASCL2", "CDX2", 
               "ARID3A", "KLF5", "RFX6", "NKX2-1", "SOX15", "TP63", "GATA3", "SATB1", 
               "CER1", "LHX5", "SIX3", "LHX2", "GLI3", "SIX6", "NES", "GBX2", 
               "ZIC2", "NANOG", "MIXL1", "OTX2", "POU5F1", "ZIC5"
]; 

#---Load Data generated in the (python) preprocessing scripts for embryoidBodyData: 
X_HVGs = Float32.(readdlm(datapath * "EB_dataMat_HVGs.txt")); #log1p-counts
X_HVGs_st = standardize(X_HVGs);

timepoints = Int.(vec(readdlm(datapath * "EB_data_timepoints.txt")));

HVGs = string.(vec(readdlm(datapath * "EB_data_HVGs.txt")));
@info "$(length(intersect(HVGs, markerGenes))) of the selected HVGs are included in the list of 70 marker genes"

EB_umap_coords = readdlm(datapath * "EB_data_UMAPcoords.txt"); 


EB_df = DataFrame(X_HVGs, :auto);
EB_df[!, "timepoints"] = timepoints;

L = [];
L_st = [];
T = [];
for t in unique(timepoints)
    X = Matrix(EB_df[EB_df[:, "timepoints"].==t, 1:end-1])
    X_st = standardize(X)
    Y = EB_df[EB_df[:, "timepoints"].==t, end]
    push!(L, X)
    push!(L_st, X_st)
    push!(T, Y)
end

n, p = size(X_HVGs);

#---Elbow plot:
pcs = prcomps(L[1]);
plot(1:size(pcs[:, 1:100], 2), vec(std(pcs[:, 1:100], dims=1)), linecolor = :blue, linewidth = 2)
vline!([10])



#------------------------------
# Define and train timeBAE:
#------------------------------
#---Seeds for reproducibility:
modelseed = 700; #700 (a), #404 (a), 700 (j), 700 (j)
batchseed = 17; #17 (a), #50 (a), 17 (j), 17 (j)

#---Hyperparameters for training:
mode = "jointLoss"; #options are "alternating" or "jointLoss" 

zdim = 10; #7 (a), 6 (a), 6 (j), 10 (j)

batchsize = 1500; #1500, (a), 1000 (a), 1500 (j), 1500 (j)

epochs = 25; #20 (a), 10 (a), 10 (j), 10 (j)

ϵ = 0.01; #0.01 (a), 0.01 (a), 0.01 (j), 0.01 (j)


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
B_perm = zeros(size(X_HVGs_st, 2));
for dim in 1:zdim
    for t in 1:length(L_st)
        B_perm = hcat(B_perm, B[:, (t-1)*zdim+dim])
    end
end
B_perm = B_perm[:, 2:end];

#---Compute the latent representation and the correlation matrix:
Z = X_HVGs_st * B;
absCor_Z = abs.(cor(Z, dims=1));
Z_perm = X_HVGs_st * B_perm;
absCor_Z_perm = abs.(cor(Z_perm, dims=1));

#---Save the timeBAE encoder weight matrix:
#writedlm(datapath * "timeBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize)_modelseed$(modelseed)_batchseed$(batchseed).txt", B);



#------------------------------
# Visualization:
#------------------------------
plotseed = 42;
create_colored_umap_plot(X_HVGs_st, timepoints, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=true, path=figurespath * "/embryoidBodyData_umap_colby_timepoints.pdf", 
                         colorlabel="Timepoint", legend_title="Time point", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of subset EB data colored by timepoints", title_fontSize=15.0
);
create_latent_umaps(X_HVGs_st, plotseed, Z_perm, "timeBAE"; 
                    figurespath=figurespath * "/timeBAE_umap",
                    precomputed=true, embedding=EB_umap_coords, save_plot=true, 
                    legend_title="Representation value"
);
vegaheatmap(absCor_Z_perm; path=figurespath * "/abscor_latentrep_timeBAE.pdf", 
            xlabel="Latent dimension", ylabel="Latent dimension", legend_title="Correlation", 
            scheme="reds", save_plot=true
);

#---Creating Scatterplots showing top selected genes per latent dimension:
for l in 1:zdim * length(L_st)
    pl = normalized_scatter_top_values(B_perm[:, l], HVGs; top_n=15, dim=l)
    savefig(pl, figurespath * "scatterplot_genes_timeBAE_latdim$(l).pdf")
end




##################
#Additional plots:
##################
dim = 32;
t = 5;
tp_inds = findall(x -> x==t, timepoints);


create_colored_umap_plot(X_HVGs_st, Z_perm[tp_inds, dim], plotseed; embedding=EB_umap_coords[tp_inds, :], 
                         precomputed=true, save_plot=false, path=figurespath * "/embryoidBodyData_umap_colby_timepoints.pdf", 
                         colorlabel="Timepoint", legend_title="Time point", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of subset EB data colored by timepoints", title_fontSize=15.0, value_type="continuouse",
                         color_field="values", scheme="blueorange"
);
create_colored_umap_plot(X_HVGs_st, timepoints[tp_inds], plotseed; embedding=EB_umap_coords[tp_inds, :], 
                         precomputed=true, save_plot=false, path=figurespath * "/embryoidBodyData_umap_colby_timepoints.pdf", 
                         colorlabel="Timepoint", legend_title="Time point", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of subset EB data colored by timepoints", title_fontSize=15.0
);