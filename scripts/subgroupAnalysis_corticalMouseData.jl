#------------------------------
# BAE applied for analyzing a subgroup of the cortical mouse scRNA-seq data: 
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
# All paths are relative to the repository main folder
projectpath = joinpath(@__DIR__, "../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/corticalMouseData/";
figurespath = projectpath * "figures/corticalMouseData_subgroupAnalysis/";
if !isdir(figurespath)
    # Create the folder if it does not exist
    mkdir(figurespath)
end

#---Include functions:
include(projectpath * "/src/BAE.jl");
using .BoostingAutoEncoder;
using DelimitedFiles;
using Random;
using Flux;
using Statistics;
using Plots;



#------------------------------
# Generate data:
#------------------------------
#---Load Data generated in the preprocessing script (get_corticalMouseData.jl):
ctype = "Sst"
st_dataMat = readdlm(datapath * "corticalMouseDataMat_" * ctype * "_HVGs_st.txt");
genenames = vec(readdlm(datapath * "genenames_" * ctype * "_HVGs.txt"));

n, p = Int32.(size(st_dataMat));



#------------------------------
# Define and train BAE:
#------------------------------
#---Seeds for reproducibility:
modelseed = 1984; 
batchseed = 777; 
plotseed = 1421;

#---Hyperparameters for training BAE:
mode = "jointLoss"; #options are: "alternating", "jointLoss"

zdim = 4; 

batchsize = 95; 

epochs = 20;  

ϵ = 0.01;

num_topgenes = 3;



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
B_BAE = trainBAE(st_dataMat, BAE; mode=mode, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);
Z_BAE = st_dataMat * B_BAE;

#---Determine BAE top genes per latent dimension using the changepoint strategy:
selGenes_dict, selGenes_df = get_top_selected_genes(B_BAE, genenames; 
    data_path=datapath, 
    save_data=true
);
for i in 1:zdim
    println("Top selected genes by BAE in zdim $(i): $(selGenes_dict[i])")
end

@info "Percentage of nonzero weights in the BAE encoder weight matrix: $(length(findall(x->x!=0, B_BAE))/length(B_BAE))"



#------------------------------
# Create and save Plots:
#------------------------------
#---Create colored UMAP plots:
num_pcs = 50; 
pcs = prcomps(st_dataMat); 

embedding_pcaUMAP = generate_umap(pcs[:, 1:num_pcs], plotseed);
embedding_BAEUMAP = generate_umap(Z_BAE, plotseed);


create_colored_umap_plot(st_dataMat, ones(size(st_dataMat, 1)), plotseed; embedding=embedding_BAEUMAP, 
                         precomputed=true, save_plot=true, path=figurespath * "/mousedata_(BAE)umap.pdf", 
                         colorlabel="Celltype", legend_title="", legend_symbolSize=0.0, legend_labelFontSize=0.0,
                         Title="", marker_size="150", show_axis=false
);
create_colored_umap_plot(st_dataMat, ones(size(st_dataMat, 1)), plotseed; embedding=embedding_pcaUMAP, 
                         precomputed=true, save_plot=true, path=figurespath * "/mousedata_(PCA)umap.pdf", 
                         colorlabel="Celltype", legend_title="", legend_symbolSize=0.0, legend_labelFontSize=0.0,
                         Title="", marker_size="150", show_axis=false
);

create_latent_umaps(st_dataMat, plotseed, Z_BAE; 
                    figurespath=figurespath * "/BAE_(BAEUMAP)",
                    precomputed=true, embedding=embedding_BAEUMAP, save_plot=true, 
                    legend_title="", image_type=".pdf", marker_size="150", show_axis=false
);


#---Creating Scatterplots showing top selected genes per latent dimension:
for l in 1:zdim
    pl = normalized_scatter_top_values(B_BAE[:, l], genenames; top_n=10, dim=l)
    savefig(pl, figurespath * "scatterplot_genes_BAE_latdim$(l).pdf")
end


#---Create cor-heatmap between BAE latent dimensions:
abscor_latrep_BAE = abs.(cor(Z_BAE, Z_BAE));

vegaheatmap(abscor_latrep_BAE; path=figurespath * "/abscor_latentrep_BAE.pdf", 
            xlabel="Latent dimension", ylabel="Latent dimension",  legend_title="Correlation", 
            scheme="reds", save_plot=true, legend_titleFontSize=28.0, axis_labelFontSize=24.0,
            axis_titleFontSize=28.0, legend_labelFontSize=24.0, legend_gradientThickness=25.0,
            legend_gradientLength=280.0
);