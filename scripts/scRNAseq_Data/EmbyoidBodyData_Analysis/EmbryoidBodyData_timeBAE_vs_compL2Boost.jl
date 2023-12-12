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
using ColorSchemes; 


#------------------------------
# Define paths and include functions:
#------------------------------
#---Set paths:
projectpath = joinpath(@__DIR__, "../../../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/embryoidBodyData/";
figurespath = projectpath * "figures/embryoidBodyData_compL2Boost_comparison/";

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

X = readdlm(datapath * "EB_dataMat_DEGs.txt", Float32);
genenames = vec(readdlm(datapath * "EB_data_DEGs.txt", String));
timepoints = Int.(vec(readdlm(datapath * "EB_data_filt_timepoints.txt", Float32)));
clusters = vec(readdlm(datapath * "EB_data_filt_leiden_res0_025.txt", String))[2:end];

df = DataFrame(X, :auto);
rename!(df, Symbol.(genenames));
df[!, :timepoints] = timepoints;
df[!, :clusters] = clusters;



#------------------------------
# Apply compL2Boost:
#------------------------------
#---Hyperparameters for compL2Boost:
M = 200;

ϵ = 0.02; 

num_topgenes = 10;

B_Array = [];
zerogenes = [];
for c in unique(clusters)
    #filter df per cluster:
    df_c = df[df.clusters .== c, :]

    #get timepoints:
    Y_c = df_c[:, end-1]

    #onehot encoded response vector:
    Y = onehotcelltypes(Int32.(Y_c))

    #rearrange the one hot encoded matrix:
    Y = Y[:, [1, 3, 2]]

    #filter zero-columns (genes):
    zeroinds = findall(x->x==0, vec(sum(Matrix(df_c[:, 1:end-2]), dims=1)))
    push!(zerogenes, zeroinds)
    df_c = df_c[:, findall(x->x!=0, vec(sum(Matrix(df_c[:, 1:end-2]), dims=1)))]
    #zerogenes = findall(x->x==0, vec(sum(Matrix(df_c[:, 1:end-2]))))

    #standardize data:
    X_c = Float32.(standardize(Matrix(df_c)))

    #perform compL2Boost:
    B = Float32.(zeros(size(X_c, 2), size(Y, 2)));
    for l in 1:size(Y, 2)
        #B[:, l] = compL2Boost!(B[:, l], X_c, standardize(Y[:, l]), ϵ, M)
        B[:, l] = compL2Boost!(B[:, l], X_c, Y[:, l], ϵ, M)
    end
    push!(B_Array, B)

    #save gene lists with coefficients:
    TPs = [2, 34, 5];
    for l in 1:size(Y, 2)
        inds = findall(x->x!=0, B[:, l])
        df_gene = DataFrame(:coeffs => B[inds, l], :abscoeffs => abs.(B[inds, l]), :genes => names(df_c)[inds])
        sort!(df_gene, :abscoeffs, rev=true)
        CSV.write(datapath * "EB_data_compL2Boost_C$(c)_T$(TPs[l]).csv", df_gene)
    end

    #get top genes per latent dimension:
    for i in 1:size(Y, 2)
        println("Number of nonzero genes for cluster $(c) timepoint $(i): $(length(findall(x->x!=0, B[:, i])))")
        println("Top $(num_topgenes) selected genes by compL2Boost for cluster $(c) timepoint $(i): $(names(df_c)[sortperm(abs.(B[:, i]), rev=true)[1:num_topgenes]])")
        #println("Intersection of top selected genes by compL2Boost for cluster $(c) timepoint $(i): $(intersect(names(df_c)[sortperm(abs.(B[:, i]), rev=true)[1:num_topgenes]], markerGenes))")
        pl = normalized_scatter_top_values(B[:, i], genenames; top_n=15, dim=i)
        savefig(pl, figurespath * "scatterplot_genes_compL2Boost_cluster$(c)_timepoint$(i).pdf")
    end

end
zerogenes = zerogenes[[2, 1, 3]];
B_Array = B_Array[[2, 1, 3]];


#pca-UMAP
pcs = prcomps(X; standardizeinput=true);

embedding = generate_umap(pcs[:, 1:50], 42; n_neighbors=30, min_dist=0.4);

create_colored_umap_plot(X, clusters, 42; embedding=embedding, 
                         precomputed=true, save_plot=true, path=figurespath * "/embryoidBodyData_umap_colby_clusters.pdf", 
                         colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of subset EB data colored by clusters", title_fontSize=15.0
);

create_colored_umap_plot(X, timepoints, 42; embedding=embedding, 
                         precomputed=true, save_plot=true, path=figurespath * "/embryoidBodyData_umap_colby_timepoints.pdf", 
                         colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
                         Title="UMAP of subset EB data colored by timepoints", title_fontSize=15.0
);

for C in 1:3
    for T in 1:3
        inds = zerogenes[C]
        create_colored_umap_plot(X, standardize(X)[:, setdiff(1:size(X, 2), inds)]*B_Array[C][:,T], 42; embedding=embedding, 
                                precomputed=true, save_plot=true, path=figurespath * "/embryoidBodyData_umap_colby_CompL2Boost_latrep_C$(C-1)_T$(T).pdf", 
                                colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
                                Title="UMAP of subset EB data colored by clusters", title_fontSize=15.0, value_type="continuouse", color_field="values", scheme="blueorange"
        );
    end
end


#gene_selected = "FST";
#create_colored_umap_plot(X, vec(X[:, findall(x->x==gene_selected, genenames)]), 42; embedding=embedding, 
#                         precomputed=true, save_plot=true, path=figurespath * "/" * gene_selected * "embryoidBodyData_umap_colby_clusters.pdf", 
#                         colorlabel="Cluster", legend_title="Cluster", legend_symbolSize=150.0, marker_size="10", 
#                         Title="UMAP of subset EB data colored by clusters", title_fontSize=15.0, value_type="continuouse", color_field="values", scheme="reds"
#);