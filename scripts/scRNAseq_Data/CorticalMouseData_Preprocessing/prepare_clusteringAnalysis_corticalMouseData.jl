#------------------------------------------------------------------------------------------------------------------------
# Create 20 different BAE latent representations and a PCA representation of cortical mouse data for clustering analysis: 
#------------------------------------------------------------------------------------------------------------------------


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
using LinearAlgebra;
using DataFrames;
using UMAP;
using StatsBase;
using Distances;
using Distributions;
using CSV;
using ProgressMeter;



#------------------------------
# Define paths and include functions:
#------------------------------
#---Set paths:
projectpath = joinpath(@__DIR__, "../../../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/corticalMouseData/";

#---Include functions:
include(srcpath * "utility.jl");
include(srcpath * "model.jl");
include(srcpath * "losses.jl");
include(srcpath * "training.jl");
include(srcpath * "boosting.jl");
include(srcpath * "preprocessing.jl");



#------------------------------
# Generate data:
#------------------------------
#---Seed for reproducibility:
dataseed = 777; 

#---Load Data generated in the preprocessing script (get_corticalMouseData.jl):
#st_dataMat_allgenes = readdlm(datapath * "corticalMouseDataMat_allgenes_st.txt");
#log1_dataMat = readdlm(datapath * "corticalMouseDataMat_log1.txt");
#st_dataMat = readdlm(datapath * "corticalMouseDataMat_st.txt");
#selGenes = vec(readdlm(datapath * "selGenes.txt"));

st_dataMat = readdlm(datapath * "corticalMouseDataMat_HVGs_st.txt");
log1_dataMat = readdlm(datapath * "corticalMouseDataMat_HVGs_log1.txt");
celltype = vec(readdlm(datapath * "celltype.txt"));
#genenames = vec(readdlm(datapath * "genenames_HVGs.txt"));

#---Create one-hot encoded labels:
sorted_st_datamat, sorted_numerical_labels = create_sorted_numlabels_and_datamat(st_dataMat, celltype);
sorted_log1_datamat = create_sorted_numlabels_and_datamat(log1_dataMat, celltype)[1];

Y = Int32.(Matrix(onehotcelltypes(sorted_numerical_labels)));

#---Generate train- and test data:
k = Int(round((1325))); 

(X_train_st, X_test_st, X_train_log1, X_test_log1, 
Y_train, Y_test, train_inds, test_inds) = split_traintestdata(log1_dataMat, Y; 
                                                                  dataseed=dataseed, k=k
);

n, p = Int32.(size(st_dataMat));



#------------------------------
# Define and train BAE:
#------------------------------
#---Seeds for reproducibility:
Random.seed!(42)
modelseeds = rand(1:1000, 20);
batchseed = 777; 
plotseed = 1421;

#---Hyperparameters for training BAE:
mode="alternating"; #options are: "alternating", "jointLoss"

zdim = 10; 

batchsize = 500; 

epochs = 25; 

ϵ = 0.01; 

#---20-times Training BAE and storing results:
Z_BAE_array=[];
num = 1;

for modelseed in modelseeds
    #---Build BAE:
    encoder = LinearLayer(zeros(p, zdim));

    Random.seed!(modelseed);
    decoder = Chain(
                    Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
                    Dense(p, p, initW = Flux.glorot_uniform)         
    );

    BAE = Autoencoder(encoder, decoder);

    #---Training BAE:
    Random.seed!(batchseed) 
    B_BAE = trainBAE(st_dataMat, BAE; mode=mode, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

    Z_BAE = st_dataMat * B_BAE;
    writedlm(datapath * "Z_BAE_$(epochs)_modelseed$(num)_clustering.txt", Z_BAE);

    push!(Z_BAE_array, Z_BAE);

    num += 1;
end 


#------------------------------
# Clustering analysis:
#------------------------------
#---Create PCA representation of the data:
pcs = prcomps(st_dataMat);
Z_PCA = pcs[:, 1:zdim];

writedlm(datapath * "Z_PCA_clustering.txt", Z_PCA);