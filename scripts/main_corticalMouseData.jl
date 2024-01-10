#------------------------------
# BAE applied to the cortical mouse scRNA-seq data using 1500 pre-selected highly variable genes (HVGs): 
#------------------------------


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
using StatsBase;
using ColorSchemes; 
using ProgressMeter;



#------------------------------
# Define paths and include functions:
#------------------------------
#---Set paths:
projectpath = joinpath(@__DIR__, "../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/corticalMouseData/";
figurespath = projectpath * "figures/corticalMouseData/";
if !isdir(figurespath)
    # Create the folder if it does not exist
    mkdir(figurespath)
end

#---Include functions:
include(srcpath * "utils.jl");
include(srcpath * "model.jl");
include(srcpath * "losses.jl");
include(srcpath * "training.jl");
include(srcpath * "boosting.jl");
include(srcpath * "preprocessing.jl");
include(srcpath * "plotting.jl");



#------------------------------
# Generate data:
#------------------------------
#---Seed for reproducibility:
dataseed = 777; 

#---Load Data generated in the preprocessing script (get_corticalMouseData.jl):
st_dataMat = readdlm(datapath * "corticalMouseDataMat_HVGs_st.txt");
log1_dataMat = readdlm(datapath * "corticalMouseDataMat_HVGs_log1.txt");
celltype = vec(readdlm(datapath * "celltype.txt"));
genenames = vec(readdlm(datapath * "genenames_HVGs.txt"));


#---Create one-hot encoded labels:
sorted_st_datamat, sorted_numerical_labels = create_sorted_numlabels_and_datamat(st_dataMat, celltype);

Y = Int32.(Matrix(onehotcelltypes(sorted_numerical_labels)));

#---Generate train- and test data:
k = Int(round((1325))); 

(X_train_st, X_test_st, X_train_log1, X_test_log1, 
Y_train, Y_test, train_inds, test_inds) = split_traintestdata(log1_dataMat, Y; 
                                                                  dataseed=dataseed, k=k
);

n, p = Int32.(size(st_dataMat));



#------------------------------
# Apply compL2Boost:
#------------------------------
#---Hyperparameters for compL2Boost:
M_compL2Boost = 100;

ϵ_compL2Boost = 0.02; 

num_topgenes = 2;


#---compL2Boost:
B_compL2Boost = zeros(p, size(Y, 2));
for l in 1:size(Y, 2)
    B_compL2Boost[:, l] = compL2Boost!(B_compL2Boost[:, l], X_train_st, Y_train[:, l], ϵ_compL2Boost, M_compL2Boost);
end

Z_compL2Boost = st_dataMat * B_compL2Boost;
sort_Z_compL2Boost = sorted_st_datamat * B_compL2Boost;

#Determine compL2Boost top genes per latent dimension:
for i in 1:size(Y, 2)
    println("Top $(num_topgenes) selected genes by compL2Boost for cell type $(i): $(genenames[sortperm(abs.(B_compL2Boost[:, i]), rev=true)[1:num_topgenes]])")
end


#------------------------------
# Define and train BAE:
#------------------------------
#---Seeds for reproducibility:
modelseed = 7; 
batchseed = 777; 
plotseed = 1421;

#---Hyperparameters for training BAE:
mode = "jointLoss"; #options are: "alternating", "jointLoss"

zdim = 10; 

batchsize = 500; 

epochs = 25;  

ϵ = 0.01;

num_topgenes = 2;



#---Build BAE:
encoder = LinearLayer(zeros(p, zdim));

Random.seed!(modelseed);
decoder = Chain(
                Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
                Dense(p, p, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Training BAE:
Random.seed!(batchseed); 
B_BAE = trainBAE(X_train_st, BAE; mode=mode, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

Z_BAE = st_dataMat * B_BAE;
sort_Z_BAE = sorted_st_datamat * B_BAE;

#Determine BAE top genes per latent dimension:
for i in 1:zdim
    println("Top $(num_topgenes) selected genes by BAE in zdim $(i): $(genenames[sortperm(abs.(B_BAE[:, i]), rev=true)[1:num_topgenes]])")
end

#---Save the BAE encoder weight matrix:
#writedlm(datapath * "BAE_encoderWeightMatrix_mode$(mode)_zdim$(zdim)_batchs$(batchsize)_epochs$(epochs)_eps$(ϵ)_mseed$(modelseed)_bseed$(batchseed).txt", B);



#------------------------------
# Create and save Plots:
#------------------------------
#---Create a version of Y sorted by celltype and modify it for plotting:
df = DataFrame(Y, :auto);
df[!, :celltype] = sorted_numerical_labels;
sort!(df, order(:celltype));
sort_Y = Matrix(df)[:, 1:end-1];
sort_Y = hcat([sort_Y[:, l].*l for l in 1:size(sort_Y, 2)]...);
sort_Y[findall(x->x==0, sort_Y)] = sort_Y[findall(x->x==0, sort_Y)] .+ 20;

#---Sorted by cell type latent representation heatmaps:
vegaheatmap(sort_Y; path=figurespath * "sort_onehot_celltype_category20.svg", 
            ylabel="Cell", xlabel="Cell type", legend_title="Cell type",
            color_field="value:o", scheme="category20", save_plot=true
);
vegaheatmap(sort_Z_compL2Boost; path=figurespath * "/sort_latentRep_compL2Boost.svg", 
            legend_title="Representation value",
            scheme="blueorange", set_domain_mid=true, save_plot=true
);
vegaheatmap(sort_Z_BAE; path=figurespath * "sort_latentRep_BAE.svg", 
            legend_title="Representation value",
            scheme="blueorange", set_domain_mid=true, save_plot=true
);

#---Create colored UMAP plots:
num_pcs = zdim; 
pcs = prcomps(st_dataMat); 

embedding_pcaUMAP = generate_umap(pcs[:, 1:num_pcs], plotseed);
embedding_BAEUMAP = generate_umap(Z_BAE, plotseed);
embedding_compL2Boost = generate_umap(Z_compL2Boost, plotseed);


create_colored_umap_plot(st_dataMat, celltype, plotseed; embedding=embedding_BAEUMAP, 
                         precomputed=true, save_plot=true, path=figurespath * "/mousedata_(BAE)umap.svg", 
                         colorlabel="Celltype", legend_title="Cell type"
);
create_colored_umap_plot(st_dataMat, celltype, plotseed; embedding=embedding_compL2Boost, 
                         precomputed=true, save_plot=true, path=figurespath * "/mousedata_(compL2Boost)umap.svg", 
                         colorlabel="Celltype", legend_title="Cell type"
);
create_colored_umap_plot(st_dataMat, celltype, plotseed; embedding=embedding_pcaUMAP, 
                         precomputed=true, save_plot=true, path=figurespath * "/mousedata_(PCA)umap.svg", 
                         colorlabel="Celltype", legend_title="Cell type"
);

binary_obsvec = zeros(n);
binary_obsvec[test_inds].=1;

create_colored_umap_plot(st_dataMat, binary_obsvec, plotseed; embedding=embedding_pcaUMAP, 
                         precomputed=true, save_plot=true, path=figurespath * "/mousedata_(PCA)umap_traintest.svg", 
                         colorlabel="Celltype", legend_title="Train/Test data",
                         scheme="paired"
);
create_colored_umap_plot(st_dataMat, binary_obsvec, plotseed; embedding=embedding_BAEUMAP, 
                         precomputed=true, save_plot=true, path=figurespath * "/mousedata_(BAE)umap_traintest.svg", 
                         colorlabel="Celltype", legend_title="Train/Test data",
                         scheme="paired"
);


create_latent_umaps(st_dataMat, plotseed, Z_BAE; 
                    figurespath=figurespath * "/BAE_(BAEUMAP)",
                    precomputed=true, embedding=embedding_BAEUMAP, save_plot=true,
                    legend_title="", image_type=".svg"
);
create_latent_umaps(st_dataMat, plotseed, Z_BAE; 
                    figurespath=figurespath * "/BAE_(pcaUMAP)",
                    precomputed=true, embedding=embedding_pcaUMAP, save_plot=true, 
                    legend_title="", image_type=".svg"
);
create_latent_umaps(st_dataMat, plotseed, Z_compL2Boost; 
                    figurespath=figurespath * "/compL2Boost_(compL2BoostUMAP)",
                    precomputed=true, embedding=embedding_compL2Boost, save_plot=true,
                    legend_title="", image_type=".svg"
);



#---Creating Scatterplots showing top selected genes per latent dimension:
for l in 1:zdim
    pl = normalized_scatter_top_values(B_BAE[:, l], String.(genenames); top_n=15, dim=l)
    savefig(pl, figurespath * "scatterplot_genes_BAE_latdim$(l).svg")
end

for l in 1:size(Y, 2)
    pl = normalized_scatter_top_values(B_compL2Boost[:, l], String.(genenames); top_n=15, dim=l)
    savefig(pl, figurespath * "scatterplot_selGenes_compL2Boost_latdim$(l).svg")
end


celltype_test = celltype[test_inds];
embedding_test_BAE = embedding_BAEUMAP[test_inds, :];
Z_BAE_test = Z_BAE[test_inds, :];
create_colored_umap_plot(st_dataMat, celltype_test, plotseed; embedding=embedding_test_BAE, precomputed=true, 
                         save_plot=true, path=figurespath * "/mousedata_BAEumap_test.svg", 
                         colorlabel="Celltype", marker_size="40", legend_title="Cell type"
);
create_latent_umaps(st_dataMat, plotseed, Z_BAE_test; 
    figurespath=figurespath * "/test_BAE_(BAEUMAP)",
    precomputed=true, embedding=embedding_test_BAE, save_plot=true,
    legend_title="Representation value", image_type=".svg"
);



#---Create cor-heatmaps between compL2Boost and BAE latent representation:
abscor_latrep_BAE_compL2Boost = abs.(cor(sort_Z_BAE, sort_Z_compL2Boost));
abscor_latrep_BAE = abs.(cor(sort_Z_BAE, sort_Z_BAE));

vegaheatmap(abscor_latrep_BAE_compL2Boost; path=figurespath * "/abscor_latentrep_BAE_compL2Boost.svg", 
            xlabel="compL2Boost latent dimensions", ylabel="BAE latent dimensions", legend_title="Correlation", 
            scheme="reds", save_plot=true
);
vegaheatmap(abscor_latrep_BAE; path=figurespath * "/abscor_latentrep_BAE.svg", 
            xlabel="Latent dimension", ylabel="Latent dimension", legend_title="Correlation", 
            scheme="reds", save_plot=true
);