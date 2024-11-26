import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import PCA
import scipy.stats as stats



# Load data: 
#---Setting paths: 
cur_dir = os.path.dirname(os.path.abspath(__file__))
projectpath = os.path.join(cur_dir, '../')
datapath = projectpath + 'data/corticalMouseData/';
figurespath = projectpath + 'figures/corticalMouseData_sparsePCA';
if not os.path.exists(figurespath):
    # Create the folder if it does not exist
    os.makedirs(figurespath)


X_log1 = np.loadtxt(datapath + 'corticalMouseDataMat_HVGs_log1.txt', delimiter='\t')
X_st = np.loadtxt(datapath + 'corticalMouseDataMat_HVGs_st.txt', delimiter='\t')
genenames = np.genfromtxt(datapath + 'genenames_HVGs.txt', dtype=str, delimiter='\n')
celltype = np.genfromtxt(datapath + 'celltype.txt', dtype=str, delimiter='\n')


#---Applying Sparse PCA
# HPS:
zdim = 10
top_n = 5
Seeds = list(range(1, 31))
n_alpha = 1 #default is 1
n_iter = 1000 #default is 1000

figurespath = figurespath + f'/Alpha_{n_alpha}_Iter_{n_iter}';
if not os.path.exists(figurespath):
    # Create the folder if it does not exist
    os.makedirs(figurespath)


loading_matrices = []
pct_zeroels = []
for seed in Seeds:
    print(f"Seed: {seed}")
    sparse_pca = SparsePCA(n_components=zdim, verbose=True, max_iter=n_iter, alpha=n_alpha)#, random_state=np.random.RandomState(seed)) 
    X_sparse_pca = sparse_pca.fit_transform(X_st)
    loadings = sparse_pca.components_.T 

    loading_matrices.append(loadings)

    pca = PCA(n_components=zdim) 
    X_pca = pca.fit_transform(X_st)
    pca_loadings = pca.components_.T 

    # Save the loading and the pc matrix:
    np.savetxt(f"{figurespath}/sparsePCA_loadings_Seed_{seed}.txt", loadings)
    np.savetxt(f"{figurespath}/PCA_loadings_Seed_{seed}.txt", pca_loadings)
    np.savetxt(f"{figurespath}/sparsePCA_pcs_Seed_{seed}.txt", X_sparse_pca)
    np.savetxt(f"{figurespath}/PCA_pcs_Seed_{seed}.txt", X_pca)


    #---Calculate the percentage of zero elements in the loading matrix:
    n_zeros = np.sum(loadings == 0)
    total_elements = loadings.size
    fraction_zeros = n_zeros / total_elements
    pct_zeroels.append(fraction_zeros)

    # Create an AnnData object
    adata = sc.AnnData(X=X_sparse_pca)

    # Add labels to metadata (obs)
    adata.obs['labels'] = celltype

    # Run UMAP
    sc.pp.neighbors(adata)  # Compute the neighborhood graph
    sc.tl.umap(adata)       # Apply UMAP

    # Plot UMAP, colored by labels
    sc.pl.umap(adata, color='labels', legend_loc='on data', show=False)
    plt.savefig(figurespath + f'/sparsePCA_umap_seed_{seed}.pdf', dpi=300)




    #---Corr plot: import numpy as np
    # Convert S to a pandas DataFrame for easier column correlation
    df = pd.DataFrame(adata.X.copy())

    # Compute Pearson correlation matrix between columns
    corr_matrix = np.abs(df.corr(method='pearson'))

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='inferno', linewidths=0.5)
    plt.savefig(figurespath + f'/correlation_heatmap_seed_{seed}.pdf', dpi=300)




    #---Create colored UMAP plots by activation of cells in sparsePCA components:
    topGenes_df = pd.DataFrame()
    for i in range(zdim):

        figurespath_sub = figurespath + '/Dim' + str(i+1);
        if not os.path.exists(figurespath_sub):
            # Create the folder if it does not exist
            os.makedirs(figurespath_sub)

        adata.obs[f'PC{i+1}'] = X_sparse_pca[:, i]
        sc.pl.umap(adata, color=f'PC{i+1}', legend_loc='on data', show=False, cmap='PuOr_r')
        plt.savefig(figurespath_sub + f'/sparsePCA_umap_seed_{seed}.pdf', dpi=300)


        #---Add top_n selected genes to the dataframe:
        top_gene_inds = np.argsort(np.abs(loadings[:, i]))[-top_n:]
        topGenes = genenames[top_gene_inds]
        topGenes_df[f'Dim{i+1}'] = topGenes

        topGenes_df.to_csv(figurespath + f'/topGenes_sparsePCA_seed_{seed}.csv', sep='\t', index=False)


        for k in range(3):

            adata.obs[f'TopGene{k+1}'] = X_log1[:, top_gene_inds[k]]
            sc.pl.umap(adata, color=f'TopGene{k+1}', legend_loc='on data', show=False, title=topGenes[k], cmap='inferno')

            plt.savefig(figurespath_sub + f'/topGene{k+1}_sparsePCA_umap_seed_{seed}.pdf', dpi=300)


#--- Create selected gene histogram:
# Loop through each matrix to find rows with nonzero elements
selGenes_list = []
for loadings in loading_matrices:
    nonzero_rows = np.any(loadings != 0, axis=1)  # Boolean array for nonzero rows
    row_indices = np.where(nonzero_rows)[0]  # Get indices of nonzero rows
    selGenes = genenames[row_indices]
    selGenes_list.append(selGenes)

from collections import Counter
all_selected_genes = np.concatenate(selGenes_list)
gene_counts = Counter(all_selected_genes)
#print(gene_counts.values())
#print(gene_counts.keys())

# Count how many genes have each specific frequency of appearance
#count_distribution = Counter(gene_counts.values())
#print(count_distribution)

# Sort by count for a clearer histogram
#sorted_counts = sorted(count_distribution.items())
#print(gene_counts.values())

# Separate the sorted data for plotting
#x_values, y_values = zip(*sorted_counts)

# Create histogram
plt.figure(figsize=(10, 6))
#plt.bar(x_values, y_values)
#plt.hist(x_values)
plt.hist(gene_counts.values(), bins=len(Seeds))
#plt.bar(gene_counts.keys(), gene_counts.values())
plt.ylabel("Number of genes")
plt.xlabel("Count of selections across runs")
plt.title("Histogram of selected genes across runs")
plt.xticks(rotation=45)
plt.savefig(figurespath + f'/sparsePCA_gene_histogram_alpha_{n_alpha}_iter_{n_iter}.pdf', dpi=300)



#---Create an excel table with the top x% of picked genes:
dict = {'Gene': gene_counts.keys(), 'Count': gene_counts.values(), 'Pct': np.array(list(gene_counts.values()))/len(Seeds)}
selGenes_df = pd.DataFrame(data=dict)
selGenes_df = selGenes_df.sort_values(by='Count', ascending=False).reset_index(drop=True)
selGenes_df.to_csv(figurespath + f'/sparsePCA_sel_genes_alpha_{n_alpha}_iter_{n_iter}.csv', sep='\t', index=False)



#---Mean std conf. interval for the pct of zero elements across runs: 
mean_pct = np.mean(pct_zeroels)
std_pct = np.std(pct_zeroels, ddof=1)  # Using ddof=1 for sample standard deviation

# Compute 95% confidence interval
confidence_level = 0.95
degrees_freedom = len(pct_zeroels) - 1
confidence_interval = stats.t.interval(
    confidence_level, degrees_freedom, loc=mean_pct, scale=std_pct / np.sqrt(len(pct_zeroels))
)

# Display results
print("Mean:", mean_pct)
print("Standard Deviation:", std_pct)
print("95% Confidence Interval:", confidence_interval)