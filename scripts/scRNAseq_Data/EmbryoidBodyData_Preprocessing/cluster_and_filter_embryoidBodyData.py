###########################
#This Script builds up on the preprocessing script 
#'get_and_preprocess_embryoidBodyData.py'
#and is part 2 of the embryoid body data preprocessing for the timeBAE analysis.
#
#Here:
#1) An anndata object is created
#2) Leidenalg clustering is performed
#3) Clusters that are not of interest are removed (clusters that do not consist of cells from all time points)
#4) DEGs are selected across two different conditions:
#    1. Per cluster
#    2. Between time points for each cluster 
#5) Data is stored
###########################

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata
import os
import logging



#-------------------------
#Define paths:
#-------------------------
#---Setting paths: 
cur_dir = os.path.dirname(os.path.abspath(__file__))
projectpath = os.path.join(cur_dir, '../../../')
datapath = projectpath + 'data/embryoidBodyData/';
figurespath = projectpath + 'figures/embryoidBodyData/';
if not os.path.exists(figurespath):
    # Create the folder if it does not exist
    os.makedirs(figurespath)



#-------------------------
#Settings:
#-------------------------
#---Configure logging to output messages with level INFO or higher
logging.basicConfig(level=logging.INFO)

#---Settings:
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=80, facecolor="white", frameon=False)



#-------------------------
#Functions:
#-------------------------
#---Defining leidenalg clustering function:
def cluster_data(adata, resolution=0.025, seed=1):
    
    sc.tl.leiden(adata, resolution=resolution, key_added='leiden_res'+str(resolution), random_state=seed)
    
    return adata



#-------------------------
#Loading data:
#-------------------------
#---Loading the preprocessed count data and meta data:
logging.info('Loading data ...')
data_norm = pd.read_pickle(datapath + 'embryoidbody_data_norm.pickle.gz')
metadata = pd.read_pickle(datapath + 'embryoidbody_metadata.pickle.gz')

#---Re-annotate time points:
np.unique(metadata['sample_labels'])
metadata['timepoint'] = metadata['sample_labels']
#metadata['timepoint'].replace("Day 00-03", 1, inplace=True) #TP 1
metadata['timepoint'].replace("Day 06-09", 2, inplace=True) #TP 2
metadata['timepoint'].replace("Day 12-15", 34, inplace=True) #TP 3
metadata['timepoint'].replace("Day 18-21", 34, inplace=True) #TP 4
metadata['timepoint'].replace("Day 24-27", 5, inplace=True) #TP 5



#-------------------------
#Create anndata object:
#-------------------------
#---Create an anndata object:
logging.info('Creating anndata object ...')
adata = anndata.AnnData(data_norm)

#---Add observation annotations:
adata.obs = metadata

#---Fix gene names: 
full_gene_names = adata.var.index
adata.var.index = full_gene_names.str.split().str[0]
adata.var['gene_codes'] = full_gene_names.str.split().str[1]
adata.var_names_make_unique

#---Add the normalized data and the filtered raw count data to adata.layers:
adata.layers['norm'] = data_norm

#---Convert the count matrix to a dense format:
adata.X.todense()

#---Normalize and log-transform (with a pseudo count of one) the countmatrix adata.X:
sc.pp.normalize_total(adata, target_sum=1e4) 
sc.pp.log1p(adata) 

#---Add a copy of the log-transformed count data to adata.layers:
adata.layers['log1'] = adata.X.copy()



#-------------------------
#Leidenalg clustering and UMAP:
#-------------------------
#---Standardize the data for computing PCA representation:
sc.pp.scale(adata) 

#---Compute PCA representation:
sc.tl.pca(adata, random_state=0, svd_solver='arpack') 

#---Compute the KNN graph:
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)

#---Perform Leidenalg clustering of the data from all time points for different resolution parameters:
logging.info('Performing leidenalg clustering ...')
for resolution in [0.025, 0.25]:
    cluster_data(adata, resolution=resolution, seed=31415926)

#---Adding string-valued labels for colormaps in plots:
adata.obs['leiden_res0.025_str'] = [str(num) for num in list(adata.obs['leiden_res0.025'])]
adata.obs['leiden_res0.25_str'] = [str(num) for num in list(adata.obs['leiden_res0.25'])]
adata.obs['timepoint_str'] = [str(num) for num in list(adata.obs['timepoint'])]

#---Plot the different clustering results in a UMAP plot:
logging.info('Computing 2D UMAP coordinats ...')
sc.tl.umap(adata)
sc.pl.umap(
                       adata,
                       color=["leiden_res0.025", "leiden_res0.25", "timepoint_str"],
                       legend_loc="on data", show=False
)
plt.savefig(figurespath + '/umap_EB_data.pdf', dpi=300)



#-------------------------
#Subset adata obj.:
#-------------------------
logging.info('Removing outlier clusters from the anndata object ...')
adata = adata[[cluster in ['0', '1', '3', '5', '6'] for cluster in adata.obs['leiden_res0.25']], :]
adata.var_names_make_unique()
adata.X = adata.layers['log1'].copy().toarray()
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.scale(adata) 
sc.tl.pca(adata, random_state=0, svd_solver='arpack') 
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
sc.pl.umap(adata,
           color=["leiden_res0.025", "leiden_res0.25", "timepoint_str"],
           legend_loc="on data", show=False
)
plt.savefig(figurespath + '/umap_EB_data_filt.pdf', dpi=300)



#-------------------------
#DEG selection (cluster):
#-------------------------
#---logp1 counts required for DEG selection with wilcoxon test:
adata.X = adata.layers['log1'].copy().toarray()

#---Set thresholds:
p_val_threshold = 1e-5 #threshold for keeping DEGs after DEG analysis (upper bound)
lfc_threshold = 2.0 #threshold for keeping DEGs after DEG analysis (lower bound)
cutoff = 50

cluster_DEGs_dict = {c: [] for c in np.unique(adata.obs['leiden_res0.025_str'])}
cluster_DEGs = []

adata.uns['log1p']["base"] = None
adata.var_names_make_unique()
sc.tl.rank_genes_groups(adata, groupby='leiden_res0.025_str', method='wilcoxon') 
DEG_results = adata.uns['rank_genes_groups']

for cluster in np.unique(adata.obs['leiden_res0.025_str']):
    gene_names = DEG_results['names'][cluster]
    logfoldchanges = DEG_results['logfoldchanges'][cluster]
    pvals_adj = DEG_results['pvals_adj'][cluster]

    df = pd.DataFrame({'gene': gene_names, 'logfoldchange': logfoldchanges, 'pval_adj': pvals_adj})
    df = df[(df['pval_adj'] < p_val_threshold) & (df['logfoldchange'] > lfc_threshold)]
    df.sort_values('logfoldchange', ascending=False, inplace = True)
    df = df.iloc[:cutoff] #cutoff all rows below the 50th row

    for ind in df.index:
        cluster_DEGs_dict[cluster].append(df['gene'][ind]) 
  
    cluster_DEGs = np.union1d(cluster_DEGs, cluster_DEGs_dict[cluster])



#-------------------------
#DEG selection (time points for each cluster):
#-------------------------
#---Set thresholds:
p_val_threshold = 1e-4 #threshold for keeping DEGs after DEG analysis (upper bound)
lfc_threshold = 1.0 #threshold for keeping DEGs after DEG analysis (lower bound)
cutoff = 20

timepoint_DEGs = []

for cluster in np.unique(adata.obs['leiden_res0.025_str']):
    adata_c = adata[adata.obs['leiden_res0.025_str']==cluster, :]

    adata_c.uns['log1p']["base"] = None
    adata_c.var_names_make_unique()
    sc.tl.rank_genes_groups(adata_c, groupby='timepoint_str', method='wilcoxon') 
    DEG_results = adata_c.uns['rank_genes_groups']

    timepoint_DEGs_dict = {t: [] for t in np.unique(adata.obs['timepoint_str'])}
    for t in np.unique(adata_c.obs['timepoint_str']):
        gene_names = DEG_results['names'][t]
        logfoldchanges = DEG_results['logfoldchanges'][t]
        pvals_adj = DEG_results['pvals_adj'][t]

        df = pd.DataFrame({'gene': gene_names, 'logfoldchange': logfoldchanges, 'pval_adj': pvals_adj})
        df = df[(df['pval_adj'] < p_val_threshold) & (df['logfoldchange'] > lfc_threshold)]
        df.sort_values('logfoldchange', ascending=False, inplace = True)
        df = df.iloc[:cutoff] #cutoff all rows below the 50th row

        for ind in df.index:
            timepoint_DEGs_dict[t].append(df['gene'][ind]) 
    
        timepoint_DEGs = np.union1d(timepoint_DEGs, timepoint_DEGs_dict[t])



#-------------------------
#Save data:
#-------------------------
logging.info('Saving data ...')

#---Save the DEGs:
DEGs = np.union1d(cluster_DEGs, timepoint_DEGs)
np.savetxt(datapath + 'EB_data_DEGs.txt', DEGs, fmt='%s', delimiter="\t")

#---Save log1p-counts of DEGs:
_, DEGs_inds, _ = np.intersect1d(list(adata.var_names), DEGs, return_indices=True)
adata_sub_DEGs = adata[:, DEGs_inds]
adata_sub_DEGs.X = adata_sub_DEGs.layers['log1'].copy().toarray()
np.savetxt(datapath + 'EB_dataMat_DEGs.txt', adata_sub_DEGs.X, delimiter="\t")

#---Save timepoint and cluster annotations:
np.savetxt(datapath + 'EB_data_filt_timepoints.txt', adata.obs['timepoint'].values, delimiter="\t")
adata.obs['leiden_res0.025'].to_csv(datapath + 'EB_data_filt_leiden_res0_025.txt', sep='\t', header=True, index=False)

#---Save UMAP coordinates:
umap_coords = adata.obsm['X_umap']
umap_df = pd.DataFrame(umap_coords, index=adata.obs_names, columns=['UMAP1', 'UMAP2'])
umap_df.to_csv(datapath + 'EB_data_filt_umap_coords.csv')