#------------------------------
#timeBAE applied to the embryoid body scRNA-seq data: 
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
using ProgressMeter;


#------------------------------
# Define paths and include functions:
#------------------------------
#---Set paths:
projectpath = joinpath(@__DIR__, "../../../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/embryoidBodyData/";
figurespath = projectpath * "figures/embryoidBodyData/";

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
]; #35 of the 70 genes are included in hvgs

#---Load Data generated in the (python) preprocessing scripts for embryoidBodyData: 
dataMat_hvgs = readdlm(datapath * "EB_data_HVGs.txt");
dataMat_DEGs = readdlm(datapath * "EB_data_DEGs.txt");
dataMat_clusterDEGs = readdlm(datapath * "EB_data_clusterDEGs.txt");
dataMat_timepointDEGs = readdlm(datapath * "EB_data_timepointDEGs.txt");
timepoints = Int.(vec(readdlm(datapath * "EB_data_timepoints.txt")));
leiden0_025 = vec(readdlm(datapath * "EB_data_leiden_res0_025.txt")[2:end]);
leiden0_25 = vec(readdlm(datapath * "EB_data_leiden_res0_25.txt")[2:end]);
leiden0_5 = vec(readdlm(datapath * "EB_data_leiden_res0_5.txt")[2:end]);
leiden0_75 = vec(readdlm(datapath * "EB_data_leiden_res0_75.txt")[2:end]);
leiden1_0 = vec(readdlm(datapath * "EB_data_leiden_res1.txt")[2:end]);
EB_umap_coords_df = CSV.File(datapath * "EB_data_umap_coordinates_noFirstTP.csv") |> DataFrame;
EB_umap_coords_df[!, "timepoints"] = timepoints;
EB_umap_coords = Matrix(EB_umap_coords_df[:, ["UMAP1", "UMAP2"]]);

p_hvg = size(dataMat_hvgs, 2);
EB_hvg_df = DataFrame(dataMat_hvgs, :auto);
EB_hvg_df[!, "timepoints"] = timepoints;
EB_hvg_df[!, "leiden0_025"] = leiden0_025;
EB_hvg_df[!, "leiden0_25"] = leiden0_25;
EB_hvg_df[!, "leiden0_5"] = leiden0_5;
EB_hvg_df[!, "leiden0_75"] = leiden0_75;
EB_hvg_df[!, "leiden1_0"] = leiden1_0;

p_deg = size(dataMat_DEGs, 2);
EB_deg_df = DataFrame(dataMat_DEGs, :auto);
EB_deg_df[!, "timepoints"] = timepoints;
EB_deg_df[!, "leiden0_025"] = leiden0_025;
EB_deg_df[!, "leiden0_25"] = leiden0_25;
EB_deg_df[!, "leiden0_5"] = leiden0_5;
EB_deg_df[!, "leiden0_75"] = leiden0_75;
EB_deg_df[!, "leiden1_0"] = leiden1_0;

p_cdeg = size(dataMat_clusterDEGs, 2);
EB_cdeg_df = DataFrame(dataMat_clusterDEGs, :auto);
EB_cdeg_df[!, "timepoints"] = timepoints;
EB_cdeg_df[!, "leiden0_025"] = leiden0_025;
EB_cdeg_df[!, "leiden0_25"] = leiden0_25;
EB_cdeg_df[!, "leiden0_5"] = leiden0_5;
EB_cdeg_df[!, "leiden0_75"] = leiden0_75;
EB_cdeg_df[!, "leiden1_0"] = leiden1_0;

p_tdeg = size(dataMat_timepointDEGs, 2);
EB_tdeg_df = DataFrame(dataMat_timepointDEGs, :auto);
EB_tdeg_df[!, "timepoints"] = timepoints;
EB_tdeg_df[!, "leiden0_025"] = leiden0_025;
EB_tdeg_df[!, "leiden0_25"] = leiden0_25;
EB_tdeg_df[!, "leiden0_5"] = leiden0_5;
EB_tdeg_df[!, "leiden0_75"] = leiden0_75;
EB_tdeg_df[!, "leiden1_0"] = leiden1_0;

X_log_hvg_tp2 = Matrix(filter(row -> row[:timepoints] == 2, EB_hvg_df))[:, 1:p_hvg];
X_log_deg_tp2 = Matrix(filter(row -> row[:timepoints] == 2, EB_deg_df))[:, 1:p_deg];
X_log_cdeg_tp2 = Matrix(filter(row -> row[:timepoints] == 2, EB_cdeg_df))[:, 1:p_cdeg];
X_log_tdeg_tp2 = Matrix(filter(row -> row[:timepoints] == 2, EB_tdeg_df))[:, 1:p_tdeg];
timepoints_t2 = filter(row -> row[:timepoints] == 2, EB_hvg_df)[!, "timepoints"];
leiden0_025_tp2 = filter(row -> row[:timepoints] == 2, EB_hvg_df)[!, "leiden0_025"];
leiden0_25_tp2 = filter(row -> row[:timepoints] == 2, EB_hvg_df)[!, "leiden0_25"];
leiden0_5_tp2 = filter(row -> row[:timepoints] == 2, EB_hvg_df)[!, "leiden0_5"];
leiden0_75_tp2 = filter(row -> row[:timepoints] == 2, EB_hvg_df)[!, "leiden0_75"];
leiden1_0_tp2 = filter(row -> row[:timepoints] == 2, EB_hvg_df)[!, "leiden1_0"];
EB_umap_coords_tp2 = Matrix(filter(row -> row[:timepoints] == 2, EB_umap_coords_df)[:, ["UMAP1", "UMAP2"]]);
X_log_hvg_tp34 = Matrix(filter(row -> row[:timepoints] == 34, EB_hvg_df))[:, 1:p_hvg];
X_log_deg_tp34 = Matrix(filter(row -> row[:timepoints] == 34, EB_deg_df))[:, 1:p_deg];
X_log_cdeg_tp34 = Matrix(filter(row -> row[:timepoints] == 34, EB_cdeg_df))[:, 1:p_cdeg];
X_log_tdeg_tp34 = Matrix(filter(row -> row[:timepoints] == 34, EB_tdeg_df))[:, 1:p_tdeg];
timepoints_t34 = filter(row -> row[:timepoints] == 34, EB_hvg_df)[!, "timepoints"];
leiden0_025_tp34 = filter(row -> row[:timepoints] == 34, EB_hvg_df)[!, "leiden0_025"];
leiden0_25_tp34 = filter(row -> row[:timepoints] == 34, EB_hvg_df)[!, "leiden0_25"];
leiden0_5_tp34 = filter(row -> row[:timepoints] == 34, EB_hvg_df)[!, "leiden0_5"];
leiden0_75_tp34 = filter(row -> row[:timepoints] == 34, EB_hvg_df)[!, "leiden0_75"];
leiden1_0_tp34 = filter(row -> row[:timepoints] == 34, EB_hvg_df)[!, "leiden1_0"];
EB_umap_coords_tp34 = Matrix(filter(row -> row[:timepoints] == 34, EB_umap_coords_df)[:, ["UMAP1", "UMAP2"]]);
X_log_hvg_tp5 = Matrix(filter(row -> row[:timepoints] == 5, EB_hvg_df))[:, 1:p_hvg];
X_log_deg_tp5 = Matrix(filter(row -> row[:timepoints] == 5, EB_deg_df))[:, 1:p_deg];
X_log_cdeg_tp5 = Matrix(filter(row -> row[:timepoints] == 5, EB_cdeg_df))[:, 1:p_cdeg];
X_log_tdeg_tp5 = Matrix(filter(row -> row[:timepoints] == 5, EB_tdeg_df))[:, 1:p_tdeg];
timepoints_t5 = filter(row -> row[:timepoints] == 5, EB_hvg_df)[!, "timepoints"];
leiden0_025_tp5 = filter(row -> row[:timepoints] == 5, EB_hvg_df)[!, "leiden0_025"];
leiden0_25_tp5 = filter(row -> row[:timepoints] == 5, EB_hvg_df)[!, "leiden0_25"];
leiden0_5_tp5 = filter(row -> row[:timepoints] == 5, EB_hvg_df)[!, "leiden0_5"];
leiden0_75_tp5 = filter(row -> row[:timepoints] == 5, EB_hvg_df)[!, "leiden0_75"];
leiden1_0_tp5 = filter(row -> row[:timepoints] == 5, EB_hvg_df)[!, "leiden1_0"];
EB_umap_coords_tp5 = Matrix(filter(row -> row[:timepoints] == 5, EB_umap_coords_df)[:, ["UMAP1", "UMAP2"]]);

X_log_hvg = vcat(X_log_hvg_tp2, X_log_hvg_tp34, X_log_hvg_tp5);
X_st_hvg = standardize(X_log_hvg);

X_log_deg = vcat(X_log_deg_tp2, X_log_deg_tp34, X_log_deg_tp5);
X_st_deg = standardize(X_log_deg);

X_log_cdeg = vcat(X_log_cdeg_tp2, X_log_cdeg_tp34, X_log_cdeg_tp5);
X_st_cdeg = standardize(X_log_cdeg);

X_log_tdeg = vcat(X_log_tdeg_tp2, X_log_tdeg_tp34, X_log_tdeg_tp5);
X_st_tdeg = standardize(X_log_tdeg);

X_st_hvg_tp2 = standardize(X_log_hvg_tp2);
X_st_hvg_tp34 = standardize(X_log_hvg_tp34);
X_st_hvg_tp5 = standardize(X_log_hvg_tp5);
L_hvg = [X_st_hvg_tp2, X_st_hvg_tp34, X_st_hvg_tp5];

X_st_deg_tp2 = standardize(X_log_deg_tp2);
X_st_deg_tp34 = standardize(X_log_deg_tp34);
X_st_deg_tp5 = standardize(X_log_deg_tp5);
L_deg = [X_st_deg_tp2, X_st_deg_tp34, X_st_deg_tp5];

X_st_cdeg_tp2 = standardize(X_log_cdeg_tp2);
X_st_cdeg_tp34 = standardize(X_log_cdeg_tp34);
X_st_cdeg_tp5 = standardize(X_log_cdeg_tp5);
L_cdeg = [X_st_cdeg_tp2, X_st_cdeg_tp34, X_st_cdeg_tp5];

X_st_tdeg_tp2 = standardize(X_log_tdeg_tp2);
X_st_tdeg_tp34 = standardize(X_log_tdeg_tp34);
X_st_tdeg_tp5 = standardize(X_log_tdeg_tp5);
L_tdeg = [X_st_tdeg_tp2, X_st_tdeg_tp34, X_st_tdeg_tp5];

#heatmap(isnan.(X_st_deg_tp2))
#size(X_st_tp5, 1)



#------------------------------
# Define and train timeBAE:
#------------------------------
#---Seeds for reproducibility:
modelseed = 14; #14
batchseed = 5; #5

#---Hyperparameters for training:
zdim = 4; #3 (nice results) #4 also works

batchsize = 600; #minimum([size(L_deg[1], 1), size(L_deg[2], 1), size(L_deg[3], 1)])-2648; #600 (nice results)

epochs = 10; #5 also ok #10 (nice res. in dim 4,5,6), #50 (also nice results in deim 4,5,6 and maybe also 7,8,9)

ϵ = 0.02; #0.02

p = p_deg;


#---Build BAE:
encoder = LinearLayer(zeros(p_deg, zdim));

Random.seed!(modelseed);
decoder = Chain(
            Dense(zdim, p_deg, tanh, initW=Flux.glorot_uniform),
            Dense(p_deg, p_deg, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Train timeBAE (jointLoss mode):
Random.seed!(batchseed);
B = trainBAE(L_deg, BAE; mode="jointLoss", time_series=true, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);
#B = trainBAE(L, BAE; mode="alternating", time_series=true, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);
B_perm = zeros(size(X_st_deg, 2));
for dim in 1:zdim
    for t in 1:length(L_deg)
        B_perm = hcat(B_perm, B[:, (t-1)*zdim+dim])
    end
end
B_perm = B_perm[:, 2:end];

#---Compute the latent representation and the correlation matrix:
Z = X_st_deg * B;
absCor_Z = abs.(cor(Z, dims=1));
Z_perm = X_st_deg * B_perm;
absCor_Z_perm = abs.(cor(Z_perm, dims=1));

#---Save the timeBAE encoder weight matrix:
#writedlm(datapath * "timeBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize)_modelseed$(modelseed)_batchseed$(batchseed).txt", B);



#------------------------------
# Visualization:
#------------------------------
plotseed = 42;
create_colored_umap_plot(dataMat_DEGs, timepoints, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=false, path=figurespath * "/embryoidBodyData_umap_colBy_timepoints.pdf", 
                         colorlabel="Timepoint", legend_title="Time point", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of subset EB data colored by timepoints", title_fontSize=15.0
);
create_colored_umap_plot(dataMat_DEGs, leiden0_025, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=false, path=figurespath * "/embryoidBodyData_umap_colBy_leidenCluster.pdf", 
                         colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
                         Title="EB data colored by leiden (res 0.75)", title_fontSize=15.0
);
create_latent_umaps(X_st_deg, plotseed, Z, "timeBAE"; 
                    figurespath=figurespath * "/timeBAE_(pcaUMAP)",
                    precomputed=true, embedding=EB_umap_coords, save_plot=false, 
                    legend_title="Representation value"
);

k = 12;
create_colored_umap_plot(X_st_deg, Z_perm[:, k], plotseed; 
    precomputed=true,
    Title = "UMAP of subset EB data colored by timeBAE latdim $(k)",
    path=figurespath * "/timeBAE_(pcaUMAP)_colBy_latdim$(k).pdf",
    legend_title="Representation value",
    color_field="values",
    scheme="blueorange",
    colorlabel="Representation value",
    save_plot=false,
    embedding=EB_umap_coords,
    value_type="continuous",
    marker_size="20",
    axis_labelFontSize=10.0,
    axis_titleFontSize=10.0,
    legend_labelFontSize=10.0,
    legend_titleFontSize=10.0,
    legend_symbolSize=10.0,
    title_fontSize=15.0
);



#------------------------------
# Define and train BAE (on a specific time point):
#------------------------------
#---Seeds for reproducibility:
modelseed = 7;
batchseed = 777; 
plotseed = 42;

#---Hyperparameters for training BAE:
t = 3;

zdim = 3; 

batchsize = 600; #larger?

epochs = 15; 

ϵ = 0.02; 

#---Build BAE:
encoder = LinearLayer(zeros(p_cdeg, zdim));

Random.seed!(modelseed);
decoder = Chain(
                Dense(zdim, p_cdeg, tanh, initW=Flux.glorot_uniform),
                Dense(p_cdeg, p_cdeg, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Training BAE:
Random.seed!(batchseed); 
B_BAE = trainBAE(L_cdeg[t], BAE; mode="jointLoss", zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);
#B_BAE = trainBAE(L[t], BAE; mode="alternating", zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

Z_BAE = L_cdeg[t] * B_BAE;

#---Save the BAE encoder weight matrix:
#writedlm(datapath * "BAE_encoderWeightMatrix_tp$(t+1)_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize)_modelseed$(modelseed)_batchseed$(batchseed).txt", B_BAE);


create_colored_umap_plot(L_cdeg[t], leiden0_025_tp5, plotseed; embedding=EB_umap_coords_tp5, 
                         precomputed=true, save_plot=false, path=figurespath * "/embryoidBodyData_umap_tp$(t+1)_colBy_leidenCluster.pdf", 
                         colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
                         Title="EB data colored by leiden (res 0.025)", title_fontSize=15.0
);

k = 1;
create_colored_umap_plot(L_cdeg[t], Z_BAE[:, k], plotseed; 
    precomputed=true,
    path=figurespath * "/BAE_(pcaUMAP)_colBy_latdim$(k).pdf",
    legend_title="Representation value",
    color_field="values",
    scheme="blueorange",
    colorlabel="Representation value",
    save_plot=false,
    embedding=EB_umap_coords_tp5,
    #embedding=pcs[:, 1:2],
    value_type="continuous",
    marker_size="20",
    axis_labelFontSize=10.0,
    axis_titleFontSize=10.0,
    legend_labelFontSize=10.0,
    legend_titleFontSize=10.0,
    legend_symbolSize=10.0,
    title_fontSize=10.0
);



#------------------------------
# Define and train BAE (on the whole data):
#------------------------------
#---Seeds for reproducibility:
modelseed = 7;
batchseed = 777; 
plotseed = 42;

#---Hyperparameters for training BAE:
zdim = 9; 

batchsize = 600; #size(X_st_tdeg, 1) - 8000; #larger?
 
epochs = 15; 

ϵ = 0.02; 

#---Build BAE:
encoder = LinearLayer(zeros(p_tdeg, zdim));

Random.seed!(modelseed);
decoder = Chain(
                Dense(zdim, p_tdeg, tanh, initW=Flux.glorot_uniform),
                Dense(p_tdeg, p_tdeg, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Training BAE:
Random.seed!(batchseed); 
B_BAE = trainBAE(X_st_tdeg, BAE; mode="jointLoss", zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);
#B_BAE = trainBAE(X_st, BAE; mode="alternating", zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

Z_BAE = X_st_tdeg * B_BAE;

#---Save the BAE encoder weight matrix:
#writedlm(datapath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize)_modelseed$(modelseed)_batchseed$(batchseed).txt", B_BAE);



create_colored_umap_plot(X_st_tdeg, timepoints, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=false, path=figurespath * "/embryoidBodyData_umap_colBy_timepoints.pdf", 
                         colorlabel="Timepoint", legend_title="Time point", legend_symbolSize=150.0, marker_size="10", 
                         Title="EB data colored by timepoints", title_fontSize=15.0
);
create_colored_umap_plot(X_st_tdeg, leiden0_025, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=false, path=figurespath * "/embryoidBodyData_umap_colBy_leidenCluster.pdf", 
                         colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
                         Title="EB data colored by leiden (res 0.75)", title_fontSize=15.0
);

k = 9
create_colored_umap_plot(X_st_tdeg, Z_BAE[:, k], plotseed; 
    precomputed=true,
    path=figurespath * "/BAE_(pcaUMAP)_colBy_latdim$(k).pdf",
    legend_title="Representation value",
    color_field="values",
    scheme="blueorange",
    colorlabel="Representation value",
    save_plot=false,
    embedding=EB_umap_coords,
    value_type="continuous",
    marker_size="20",
    axis_labelFontSize=10.0,
    axis_titleFontSize=10.0,
    legend_labelFontSize=10.0,
    legend_titleFontSize=10.0,
    legend_symbolSize=10.0,
    title_fontSize=10.0
);