#------------------------------
# timeBAE applied to the embryoid body scRNA-seq data: 
#------------------------------


#------------------------------
# Setup: 
#------------------------------
#---Activate the enviroment:
using Pkg;

Pkg.activate(".");
Pkg.instantiate();
Pkg.status()

#---Set paths:
# all paths are relative to the repository main folder
projectpath = joinpath(@__DIR__, "../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/embryoidBodyData/"; #"data/embryoidBodyData_old/embryoidBodyData/"
figurespath = projectpath * "figures/embryoidBodyData/";  #"figures/embryoidBodyData_old/"
if !isdir(figurespath)
    # Create the folder if it does not exist
    mkdir(figurespath)
end

#---Include functions:
include(projectpath * "/src/BAE.jl");
using .BoostingAutoEncoder;
using DelimitedFiles;
using CSV;
using Random;
using Flux;
using Statistics;
using DataFrames;
using Plots;
using MultivariateStats;
using LinearAlgebra;
using Distributions;
using XLSX;



#------------------------------
# Load and format data:
#------------------------------
#---Load Data generated in the (python) preprocessing script for embryoidBodyData: 
X = readdlm(datapath * "EB_dataMat_DEGs.txt", Float32);
X_st = standardize(X);

DEGs = string.(vec(readdlm(datapath * "EB_data_DEGs.txt"))); 

timepoints = Int.(vec(readdlm(datapath * "EB_data_filt_timepoints.txt")));
#timepoints = Int.(vec(readdlm(datapath * "EB_data_timepoints.txt")));

clusters = Int.(vec(readdlm(datapath * "EB_data_filt_leiden_res0_025.txt")[2:end]));
#clusters = Int.(vec(readdlm(datapath * "EB_data_leiden_res0_025.txt")[2:end]));

EB_umap_coords_df = CSV.File(datapath * "EB_data_filt_umap_coords.csv") |> DataFrame;
#EB_umap_coords_df = CSV.File(datapath * "EB_data_umap_coordinates_noFirstTP.csv") |> DataFrame;
EB_umap_coords = Matrix(EB_umap_coords_df[:, ["UMAP1", "UMAP2"]]);

EB_df = DataFrame(X, :auto);
EB_df[!, "timepoints"] = timepoints;
EB_df[!, "clusters"] = clusters;

L = [];
L_st = [];
Clusters_per_timepoint = [];
for t in unique(timepoints)
    M = Matrix(EB_df[EB_df[:, "timepoints"].==t, 1:end-2])
    M_st = standardize(M)
    Y = EB_df[EB_df[:, "timepoints"].==t, end]
    push!(L, M)
    push!(L_st, M_st)
    push!(Clusters_per_timepoint, Y)
end

n, p = size(X);



#------------------------------
# Define and train timeBAE:
#------------------------------
#---Seeds for reproducibility:
modelseed = 1; #14 #1
batchseed = 5; 

#---Hyperparameters for training:
zdim = 4; 

batchsize = 1500; #600 #1500

epochs = 15; #10 #15

ϵ = 0.01; #0.02 #0.01

mode = "alternating"; #"jointLoss" #"alternating"



#---Build BAE:
encoder = LinearLayer(zeros(p, zdim));

Random.seed!(modelseed);
decoder = Chain(
            Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
            Dense(p, p, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Train timeBAE:
Random.seed!(batchseed);
B = trainBAE(L_st, BAE; mode=mode, time_series=true, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

B_perm = zeros(size(B));
for dim in 1:zdim
    for t in 1:length(L)
        B_perm[:, length(L)*(dim-1)+t] = B[:, (t-1)*zdim+dim]
    end
end

#---Determine BAE top genes per latent dimension using the changepoint strategy:
selGenes_dict, selGenes_df = get_top_selected_genes(B_perm, DEGs; 
    data_path=datapath, 
    save_data=true
);
for i in 1:length(unique(timepoints))*zdim
    println("Top selected genes by BAE in zdim $(i): $(selGenes_dict[i])")
end

@info "Percentage of nonzero weights in the BAE encoder weight matrix: $(length(findall(x->x!=0, B_perm))/length(B_perm))"


#---Compute the latent representation and the correlation matrix:
Z_perm = X_st * B_perm;
absCor_Z_perm = abs.(cor(Z_perm, dims=1));

#---Save the timeBAE encoder weight matrix:
#writedlm(datapath * "timeBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize)_modelseed$(modelseed)_batchseed$(batchseed).txt", B);

#---Save the selected genes in a dictionary:
selGenes_dict_timeBAE = Dict()
for i in 1:size(B_perm, 2)
    vec = B_perm[:, i]
    non_zero_indices = findall(x -> x != 0, vec)
    non_zero_values = vec[non_zero_indices]
    selected_labels = DEGs[non_zero_indices]
    sorted_indices = sortperm(abs.(non_zero_values), rev=true)
    sorted_labels = selected_labels[sorted_indices]
    selGenes_dict_timeBAE["$(i)"] = sorted_labels
end

#---Save the selected genes as a CSV file:
max_length = maximum(length.(values(selGenes_dict_timeBAE)))
padded_geneLists = Dict(key => vcat(value, fill(missing, max_length - length(value))) for (key, value) in selGenes_dict_timeBAE)
df = DataFrame(padded_geneLists)
CSV.write(datapath * "timeBAE_GeneLists.csv", df)



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
                         colorlabel="Timepoint", legend_title="", show_axis=false,
                         Title="",  marker_size="20"
);

#---Create a UMAP plot of cells colored by cluster membership:
create_colored_umap_plot(X_st, clusters_plot, plotseed; embedding=EB_umap_coords, 
                         precomputed=true, save_plot=true, path=figurespath * "/embryoidBodyData_umap_colBy_leidenCluster.pdf", 
                         colorlabel="Cluster", legend_title="", show_axis=false,
                         Title="",  marker_size="20",
                         scheme="dark2"
);

#---Create a UMAP plot of cells colored by the representation value of each timeBAE latent dimension:
create_latent_umaps(X_st, plotseed, Z_perm; 
                    figurespath=figurespath * "/timeBAE_(pcaUMAP)",
                    precomputed=true, embedding=EB_umap_coords, save_plot=true, 
                    legend_title="", image_type=".pdf", show_axis=false, marker_size="20"
);


#---Creating Scatterplot showing top selected genes per latent dimension:
for l in 1:zdim*length(L)
    pl = normalized_scatter_top_values(B_perm[:, l], DEGs; top_n=10, dim=l)
    savefig(pl, figurespath * "scatterplot_genes_timeBAE_latdim$(l).pdf")
end


#---Heatmap of absolute Pearson correlation coefficients between timeBAE latent dimensions:
vegaheatmap(absCor_Z_perm; path=figurespath * "/abscor_latentrep_timeBAE.pdf", 
            ylabel="Latent dimension", xlabel="Latent dimension", legend_title="Correlation",
            scheme="reds", save_plot=true
);





#--------------------------------------
# Supervised linear regression analysis
#--------------------------------------
#---Setup the linear regression model:
B_LR = zeros(Float32, size(B_perm, 1), length(unique(timepoints))*length(unique(clusters)));
p_vals = zeros(Float32, size(B_perm, 1), length(unique(timepoints))*length(unique(clusters)));
adj_pvals = zeros(Float32, size(B_perm, 1), length(unique(timepoints))*length(unique(clusters)));

#---Fit the linear regression model:
counter = 0
for t in 1:length(unique(timepoints))
    Y = standardize(Matrix{Float32}(onehotcelltypes(Int32.(Clusters_per_timepoint[t]))))
    B = llsq(L_st[t], Y, dims=1, bias=false)
    B_LR[:, counter+1:counter+length(unique(clusters))] = B

    for i in 1:size(Y, 2)

        p_valuess = coefficients_tTest(B[:, i], L_st[t], Y[:, i]; 
            adjusted_pvals=true, 
            method="Bonferroni", 
            alpha=0.05
        )
        p_vals[:, counter+i] = p_valuess
    end

    counter+=length(unique(clusters))
end

#---Filter B_LR according to p-values:
B_LR_filtered = B_LR .* (p_vals.<0.05)

#---Permute the columns of B_LR_filtered such that the columns dimensions are grouped by the clusters and sorted by timepoints within the clusters:
B_LR_perm = zeros(size(B_LR));
for dim in 1:length(unique(clusters))
    for t in 1:length(L)
        B_LR_perm[:, length(L)*(dim-1)+t] = B_LR_filtered[:, (t-1)*length(unique(clusters))+dim]
        #B_LR_perm[:, length(L)*(dim-1)+t] = B_LR[:, (t-1)*length(unique(clusters))+dim]
    end
end

#---Compute the predictions and the correlation matrix:
Z_LR = X_st * B_LR_perm;
absCor_Z_LR = abs.(cor(Z_LR, dims=1));

heatmap(B_LR)
heatmap(B_LR_perm)

#---Get gene lists for each column of B_LR_filtered sorted by the absolute value of the weights:
selGenes_dict_LR = Dict()
for i in 1:size(B_LR_perm, 2)
    vec = B_LR_perm[:, i]
    non_zero_indices = findall(x -> x != 0, vec)
    non_zero_values = vec[non_zero_indices]
    selected_labels = DEGs[non_zero_indices]
    sorted_indices = sortperm(abs.(non_zero_values), rev=true)
    sorted_labels = selected_labels[sorted_indices]
    selGenes_dict_LR["$i"] = sorted_labels
end

#---Save the selected genes as a CSV file:
max_length = maximum(length.(values(selGenes_dict_LR)))
padded_geneLists = Dict(key => vcat(value, fill(missing, max_length - length(value))) for (key, value) in selGenes_dict_LR)
df = DataFrame(padded_geneLists)
CSV.write(datapath * "LR_GeneLists.csv", df)



#------------------------------
# Visualization:
#------------------------------
#---Create a UMAP plot of cells colored by the representation value of each LR latent dimension:
create_latent_umaps(X_st, plotseed, Z_LR; 
                    figurespath=figurespath * "/LR_(pcaUMAP)",
                    precomputed=true, embedding=EB_umap_coords, save_plot=true, 
                    legend_title="", image_type=".pdf", show_axis=false, marker_size="20"
);

#---Creating Scatterplot showing top selected genes per latent dimension:
for l in 1:length(unique(clusters))*length(L)
    pl = normalized_scatter_top_values(B_LR_perm[:, l], DEGs; top_n=10, dim=l)
    savefig(pl, figurespath * "scatterplot_genes_LR_latdim$(l).pdf")
end

#---Heatmap of absolute Pearson correlation coefficients between LR latent dimensions:
vegaheatmap(absCor_Z_LR; path=figurespath * "/abscor_latentrep_LR.pdf", 
            ylabel="Latent dimension", xlabel="Latent dimension", legend_title="Correlation",
            scheme="reds", save_plot=true
);





#---------------------------------------------
# Compare selected gene sets (timeBAE and LR):
#---------------------------------------------
#---Get the intersection of the selected genes by timeBAE and LR:
sel_genes_LR = vcat(values(selGenes_dict_LR)...) |> unique;
sel_genes_timeBAE = vcat(values(selGenes_dict_timeBAE)...) |> unique;

intersection_genes = intersect(sel_genes_LR, sel_genes_timeBAE);

pct_intersection = round(length(intersection_genes) / length(sel_genes_timeBAE) * 100, digits=2);

@info "Percentage of intersection between selected genes by timeBAE and LR: $pct_intersection%"


# Intersection of selected genes by timeBAE and LR in corresponding dimensions (Cluster 1):
gene_intersect_C1_T1 = intersect(selGenes_dict_timeBAE["4"], selGenes_dict_LR["1"]); #Cluster 1, timepoint 1
gene_intersect_C1_T2 = intersect(selGenes_dict_timeBAE["5"], selGenes_dict_LR["2"]); #Cluster 1, timepoint 2
gene_intersect_C1_T3 = intersect(selGenes_dict_timeBAE["6"], selGenes_dict_LR["3"]); #Cluster 1, timepoint 3

pct_intersection_C1_T1 = round(length(gene_intersect_C1_T1) / length(selGenes_dict_timeBAE["4"]) * 100, digits=2);
pct_intersection_C1_T2 = round(length(gene_intersect_C1_T2) / length(selGenes_dict_timeBAE["5"]) * 100, digits=2);
pct_intersection_C1_T3 = round(length(gene_intersect_C1_T3) / length(selGenes_dict_timeBAE["6"]) * 100, digits=2);

@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 1, Timepoint 1: $pct_intersection_C1_T1%"
@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 1, Timepoint 2: $pct_intersection_C1_T2%"
@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 1, Timepoint 3: $pct_intersection_C1_T3%"


# Intersection of selected genes by timeBAE and LR in corresponding dimensions (Cluster 2):
gene_intersect_C2_T1 = intersect(selGenes_dict_timeBAE["10"], selGenes_dict_LR["4"]); #Cluster 2, timepoint 1
gene_intersect_C2_T2 = intersect(selGenes_dict_timeBAE["11"], selGenes_dict_LR["5"]); #Cluster 2, timepoint 2
gene_intersect_C2_T3 = intersect(selGenes_dict_timeBAE["12"], selGenes_dict_LR["6"]); #Cluster 2, timepoint 3

pct_intersection_C2_T1 = round(length(gene_intersect_C2_T1) / length(selGenes_dict_timeBAE["10"]) * 100, digits=2);
pct_intersection_C2_T2 = round(length(gene_intersect_C2_T2) / length(selGenes_dict_timeBAE["11"]) * 100, digits=2);
pct_intersection_C2_T3 = round(length(gene_intersect_C2_T3) / length(selGenes_dict_timeBAE["12"]) * 100, digits=2);

@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 2, Timepoint 1: $pct_intersection_C2_T1%"
@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 2, Timepoint 2: $pct_intersection_C2_T2%"
@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 2, Timepoint 3: $pct_intersection_C2_T3%"


# Intersection of selected genes by timeBAE and LR in corresponding dimensions (Cluster 3):
gene_intersect_C3_T1 = intersect(selGenes_dict_timeBAE["1"], selGenes_dict_LR["7"]); #Cluster 3, timepoint 1
gene_intersect_C3_T2 = intersect(selGenes_dict_timeBAE["2"], selGenes_dict_LR["8"]); #Cluster 3, timepoint 2
gene_intersect_C3_T3 = intersect(selGenes_dict_timeBAE["3"], selGenes_dict_LR["9"]); #Cluster 3, timepoint 3

pct_intersection_C3_T1 = round(length(gene_intersect_C3_T1) / length(selGenes_dict_timeBAE["1"]) * 100, digits=2);
pct_intersection_C3_T2 = round(length(gene_intersect_C3_T2) / length(selGenes_dict_timeBAE["2"]) * 100, digits=2);
pct_intersection_C3_T3 = round(length(gene_intersect_C3_T3) / length(selGenes_dict_timeBAE["3"]) * 100, digits=2);

@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 3, Timepoint 1: $pct_intersection_C3_T1%"
@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 3, Timepoint 2: $pct_intersection_C3_T2%"
@info "Percentage of intersection between selected genes by timeBAE and LR in Cluster 3, Timepoint 3: $pct_intersection_C3_T3%"