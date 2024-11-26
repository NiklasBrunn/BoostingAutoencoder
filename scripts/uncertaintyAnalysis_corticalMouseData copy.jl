#------------------------------
# BAE applied to the cortical mouse scRNA-seq data using 1500 pre-selected highly variable genes (HVGs): 
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
figurespath = projectpath * "figures/corticalMouseData_BAE/";
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
using DataFrames;
using Plots;
using CSV;
using StatsBase;
using XLSX;
using Distributions;



#------------------------------
# Load data:
#------------------------------
#---Load Data generated in the preprocessing script (get_corticalMouseData.jl):
st_dataMat = readdlm(datapath * "corticalMouseDataMat_HVGs_st.txt");
log1_dataMat = readdlm(datapath * "corticalMouseDataMat_HVGs_log1.txt");
celltype = vec(readdlm(datapath * "celltype.txt"));
genenames = vec(readdlm(datapath * "genenames_HVGs.txt"));

n, p = Int32.(size(st_dataMat));



#--------------------------------------------------------------------------------
# Define and train BAE 30 times with different decoder parameter initializations:
#--------------------------------------------------------------------------------
#---Seeds for reproducibility:
batchseed = 777; 
plotseed = 1421;

#---Hyperparameters for training BAE:
mode = "jointLoss"; #options are: "alternating", "jointLoss"

zdim = 10; 

top_n = 5;

n_genes = 5;

batchsize = 500; 

epochs = 25;  

ϵ = 0.01;

Seeds = range(1, 30, step=1);


#---Gene selection stability analysis:
#sel_genes = [];
selGenes_list = [];
for seed in Seeds

    #--- Define a new figures path:
    figurespath_seed = figurespath * "/seed$(seed)/";
    if !isdir(figurespath_seed)
        # Create the folder if it does not exist
        mkdir(figurespath_seed)
    end

    #--- Build BAE:
    encoder = LinearLayer(zeros(p, zdim));

    Random.seed!(seed);
    decoder = Chain(
                    Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
                    Dense(p, p, initW = Flux.glorot_uniform)         
    );

    BAE = Autoencoder(encoder, decoder);

    #--- Training BAE:
    Random.seed!(batchseed); 
    B_BAE = trainBAE(st_dataMat, BAE; mode=mode, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);
    Z_BAE = st_dataMat * B_BAE;

    #--- Save the BAE encoder weights as a .txt file
    writedlm(datapath * "B_BAE_Seed_$(seed).txt", B_BAE)

    #--- Get the indices of all BAE selected genes (rows in the BAE encoder weight matrix that have at least one nonzero coefficient):
    sel_BAEgene_inds = get_nonzero_rows(B_BAE)
    push!(selGenes_list, genenames[sel_BAEgene_inds])

    #--- Determine BAE top genes per latent dimension using the changepoint strategy:
    selGenes_dict, selGenes_df = get_top_selected_genes(B_BAE, genenames; 
        data_path=figurespath_seed, 
        save_data=true
    );
    for i in 1:zdim
        println("Top selected genes by BAE in zdim $(i): $(selGenes_dict[i])")
    end

    @info "Percentage of nonzero weights in the BAE encoder weight matrix: $(length(findall(x->x!=0, B_BAE))/length(B_BAE))"


    #--- Result visualization:
    embedding_BAEUMAP = generate_umap(Z_BAE, plotseed);

    #--- Generate UMAP plots:
    abscor_latrep_BAE = abs.(cor(Z_BAE, Z_BAE));
    vegaheatmap(abscor_latrep_BAE; path=figurespath_seed * "/abscor_latentrep_BAE.pdf", 
                xlabel="Latent dimension", ylabel="Latent dimension", legend_title="Correlation", 
                scheme="inferno", save_plot=true
    );

    create_colored_umap_plot(st_dataMat, celltype, plotseed; embedding=embedding_BAEUMAP, 
                         precomputed=true, save_plot=true, path=figurespath_seed * "/mousedata_(BAE)umap.pdf", 
                         colorlabel="Celltype", legend_title="Cell type", show_axis=false
    );

    create_latent_umaps(st_dataMat, plotseed, Z_BAE; 
                    figurespath=figurespath_seed * "/BAE_(BAEUMAP)",
                    precomputed=true, embedding=embedding_BAEUMAP, save_plot=true,
                    legend_title="", image_type=".pdf", show_axis=false
    );

    #--- Determine matching cell types for latent dimensions:
    df, df_filtered = find_matching_type_per_BAEdim(Z_BAE, String.(celltype); 
    upper=0.9, lower=0.1, threshold=0.5, 
    save_plot=true, figurespath=figurespath_seed
    );

    #push!(sel_genes, sel_genes_BAE)
end


#total_BAEselGenes = union(vcat(selGenes_list));


#---Save data (list of selected genes per latent dimension):
open(datapath * "sel_nonzeroBAEGenes.txt", "w") do f
    for vector in selGenes_list
        writedlm(f, [vector], " ")  # Write each vector as a row, separated by spaces
    end
end





#-------------------------------------------
# Gene selection stability results analysis:
#-------------------------------------------
#---Load data back in the script:
data = readdlm(datapath * "sel_nonzeroBAEGenes.txt", ' ')
selGenes_list = [filter!(x -> x != "", data[i, :]) for i in 1:size(data, 1)]

B = [];
for seed in Seeds
    push!(B, readdlm(datapath * "B_BAE_Seed_$(seed).txt"))
end

#---Generate a histogram of the selected genes across Seeds:
string_vector = vcat(selGenes_list ...)

n_sel_genes_per_run = [length(vec) for vec in selGenes_list]
@info "Number of selected genes per run: $(n_sel_genes_per_run)"
@info "Mean number of selected genes: $(mean(n_sel_genes_per_run))"
@info "Mean percentage of the number of selected genes: $(mean(n_sel_genes_per_run)/1500)"

# Create the histogram (countmap)
hist = countmap(string_vector)

# Display the histogram
println(hist)


# Extract labels and counts
labels = collect(keys(hist))
counts = collect(values(hist))

# Create a bar plot
pl = bar(labels, counts./length(Seeds), legend=false, xlabel="Genes", ylabel="Count", title="Gene Histogram", dpi=400)
savefig(pl, figurespath * "BAE_gene_selection_stability_distribution_genehistogram.png")



# Create a histogram
pl = Plots.histogram(counts, bins=30, title="BAE gene selection distribution", xlabel="Number of selections", ylabel="Number of genes", label="Number of genes", dpi=400)
pl = vline!([length(Seeds)*0.8], color=:orange, linestyle=:dash, linewidth=2, label="$(80)%")
savefig(pl, figurespath * "BAE_gene_selection_stability_distribution.png")

# Create a dataframe showing the top 20% selected genes:
inds = findall(x->x>=0.8, counts./length(Seeds))
#println(sort(labels[inds]))
df = DataFrame(Genes = labels[inds], Counts = counts[inds], Pct = counts[inds]./length(Seeds))
sort!(df, :Pct, rev=true)
# Save the df:
XLSX.writetable(datapath * "BAE_gene_selection_stability.xlsx", DataFrame(df), sheetname="Sheet1")



#Changepoint genes:
#changepoint_genes = []
#for seed in Seeds
#    changepoint_dict, changepoint_df = get_top_selected_genes(B[seed], genenames; method="Changepoint")
#    push!(changepoint_genes, vcat(collect(values(changepoint_dict)) ...))
#end

#changepoint_genes = vcat(changepoint_genes ...)
#hist = countmap(changepoint_genes)

#labels = collect(keys(hist))
#counts = collect(values(hist))

#changepoint_df = sort!(DataFrame(Genes=labels, Counts=counts), :Counts, rev=true)

# Create a histogram
#pl = Plots.histogram(counts, bins=30, title="BAE changepoint gene selection distribution", xlabel="Number of selections", ylabel="Number of genes", label="Number of genes", dpi=400)
#pl = vline!([length(Seeds)*0.25], color=:orange, linestyle=:dash, linewidth=2, label="$(25)%")
#savefig(pl, figurespath * "BAE_changepoint_gene_selection_stability_distribution.png")

# Create a dataframe showing the top 20% selected genes:
#changepoint_inds = findall(x->x>=0.25, counts./length(Seeds))
#println(sort(labels[changepoint_inds]))
#changepoint_df = DataFrame(Genes = labels[changepoint_inds], Counts = counts[changepoint_inds], Pct = counts[changepoint_inds]./length(Seeds))
#sort!(changepoint_df, :Pct, rev=true)
# Save the df:
#XLSX.writetable(datapath * "BAE_changepoint_gene_selection_stability.xlsx", DataFrame(df), sheetname="Sheet1")



#---PCTs of zero elements in the encoder weight matrix:
pcts_zeros = []
for seed in Seeds
    n_zeroels = length(findall(x -> x == 0, B[seed]))
    push!(pcts_zeros, n_zeroels / length(B[seed]))
end

# Calculate mean and standard deviation
mean_pcts_zeros = mean(pcts_zeros)
std_pcts_zeros = std(pcts_zeros)

# Number of samples
n = length(pcts_zeros)

# Critical value for 95% confidence interval (two-tailed, normal distribution)
critical_value = quantile(Normal(0, 1), 0.975)

# Compute the margin of error
margin_of_error = critical_value * (std_pcts_zeros / sqrt(n))

# Compute the confidence interval
ci_lower = mean_pcts_zeros - margin_of_error
ci_upper = mean_pcts_zeros + margin_of_error

println("Mean of the percentage of zero elements in the encoder weight matrices: ", mean_pcts_zeros)
println("Standard Deviation of the percentage of zero elements in the encoder weight matrices: ", std_pcts_zeros)
println("95% Confidence Interval for the percentage of zero elements in the encoder weight matrices: [", ci_lower, ", ", ci_upper, "]")





##########################################
#BAE application to the truncated dataset:
##########################################
#---Load data:
data = readdlm(datapath * "sel_nonzeroBAEGenes.txt", ' ')
selGenes_list = [filter!(x -> x != "", data[i, :]) for i in 1:size(data, 1)]

B = [];
for seed in Seeds
    push!(B, readdlm(datapath * "B_BAE_Seed_$(seed).txt"))
end

#---Generate a histogram of the selected genes across Seeds:
string_vector = vcat(selGenes_list ...)
# Create the histogram (countmap)
hist = countmap(string_vector)
labels = collect(keys(hist))


#---Define a new figures path:
figurespath = projectpath * "figures/truncated_corticalMouseData_BAE/";
if !isdir(figurespath)
    # Create the folder if it does not exist
    mkdir(figurespath)
end

#---Get the genes to remove from the data set:
#---Define a new figures path:
pcts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

for pct in pcts
    figurespath_sub = figurespath * "Top_$(Int(round((1.0-pct), digits=2)*100))%_genes_removed/";
    if !isdir(figurespath_sub)
        # Create the folder if it does not exist
        mkdir(figurespath_sub)
    end
    hist = countmap(string_vector);
    counts = collect(values(hist));
    inds = findall(x->x>=pct, counts./length(Seeds)); 
    df = DataFrame(Genes = labels[inds], Counts = counts[inds], Pct = counts[inds]./length(Seeds))
    sort!(df, :Pct, rev=true)

    #---Create a new truncated dataset without the top 20% BAE selected genes:
    sel_inds = findall(x->x in df.Genes, genenames);
    gene_inds_to_keep = setdiff(1:length(genenames), sel_inds);
    genenames_truncated = genenames[gene_inds_to_keep];
    writedlm(datapath * "genenames_truncated.txt", genenames_truncated);

    log1_dataMat = Float32.(readdlm(datapath * "corticalMouseDataMat_HVGs_log1.txt"));
    celltype = vec(readdlm(datapath * "celltype.txt"));
    X_st = BoostingAutoEncoder.standardize(log1_dataMat[:, gene_inds_to_keep]);
    n, p = Int32.(size(X_st));

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
    B_BAE = trainBAE(X_st, BAE; mode=mode, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

    #---Get BAE latent representation and PCA representation:
    Z_BAE = X_st * B_BAE;

    #---Determine BAE top genes per latent dimension:
    selGenes_dict, selGenes_df = get_top_selected_genes(B_BAE, genenames;
        save_data=true,
        data_path=figurespath_sub
    );
    for i in 1:zdim
        println("Top selected genes by BAE in zdim $(i): $(selGenes_dict[i])")
    end

    #---Result visualization:
    embedding_BAEUMAP = generate_umap(Z_BAE, plotseed);

    create_colored_umap_plot(st_dataMat, celltype, plotseed; embedding=embedding_BAEUMAP, 
                            precomputed=true, save_plot=true, path=figurespath_sub * "/mousedata_(BAE)umap.pdf", 
                            colorlabel="Celltype", legend_title="Cell type", show_axis=false
    );

    create_latent_umaps(X_st, plotseed, Z_BAE; 
                        figurespath=figurespath_sub * "/BAE_(BAEUMAP)",
                        precomputed=true, embedding=embedding_BAEUMAP, save_plot=true,
                        legend_title="", image_type=".pdf", show_axis=false
    );

    for l in 1:zdim
        pl = normalized_scatter_top_values(B_BAE[:, l], genenames; top_n=10, dim=l)
        savefig(pl, figurespath_sub * "scatterplot_genes_BAE_latdim$(l).pdf")
    end

    abscor_latrep_BAE = abs.(cor(Z_BAE, Z_BAE));
    vegaheatmap(abscor_latrep_BAE; path=figurespath_sub * "/abscor_latentrep_BAE.pdf", 
                xlabel="Latent dimension", ylabel="Latent dimension", legend_title="Correlation", 
                scheme="inferno", save_plot=true
    );

    df, df_filtered = find_matching_type_per_BAEdim(Z_BAE, String.(celltype); 
        upper=0.9, lower=0.1, threshold=0.5, 
        save_plot=true, figurespath=figurespath_sub
    );
end