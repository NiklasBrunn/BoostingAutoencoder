###########################
#This Script builds up on the preprocessing script 
#'get_and_preprocess_embryoidBodyData.py'.
#
#Here:
#1) An anndata object is created
#2) Highly variable genes are selected
#3) Data is stored 
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



#-------------------------
#Settings:
#-------------------------
#---Configure logging to output messages with level INFO or higher
logging.basicConfig(level=logging.INFO)

#---Settings:
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=80, facecolor="white", frameon=False)





#-------------------------
#Loading data:
#-------------------------
#---Loading the preprocessed count data and meta data:
logging.info('Loading data ...')
data_filt = pd.read_pickle(datapath + 'embryoidbody_data_filt.pickle.gz')
data_norm = pd.read_pickle(datapath + 'embryoidbody_data_norm.pickle.gz')
metadata = pd.read_pickle(datapath + 'embryoidbody_metadata.pickle.gz')

#---Re-annotate time points:
np.unique(metadata['sample_labels'])
metadata['timepoint'] = metadata['sample_labels']
#metadata['timepoint'].replace("Day 00-03", 1, inplace=True) #TP 1
metadata['timepoint'].replace("Day 06-09", 2, inplace=True) #TP 2
metadata['timepoint'].replace("Day 12-15", 3, inplace=True) #TP 3
metadata['timepoint'].replace("Day 18-21", 4, inplace=True) #TP 4
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
adata.layers['filt'] = data_filt #raw counts
adata.layers['norm'] = data_norm #normalized counts (by library size and rescaled by factor 10^4)

#---Convert the count matrix to a dense format:
adata.X.todense()

#---Normalize and log-transform (with a pseudo count of one) the countmatrix adata.X:
sc.pp.normalize_total(adata, target_sum=1e4) 
sc.pp.log1p(adata) 

#---Add a copy of the log-transformed count data to adata.layers:
adata.layers['log1'] = adata.X.copy()



#-------------------------
#Batch correction (optional):
#-------------------------
#logging.info('Correcting for batch effects ...')
#---Performing batch correction with combat:
#sc.pp.combat(adata, key='timepoint')
#perform batch correction? -> results do not significantly change (only tried 2 different runs)



#-------------------------
#Highly variable gene selection:
#-------------------------
#logging.info('Selecting highly variable genes ...')
#---Compute highly variable genes (seurat_v3 requires raw counts!):
sc.pp.highly_variable_genes(adata, layer='filt', n_top_genes=2500, flavor='seurat_v3')
#more top genes (between 1000 and 5000) we tested 2500?



#-------------------------
#Save data:
#-------------------------
logging.info('Saving data ...')
#---Saving log-counts and metadata:
adata_sub = adata[:, adata.var['highly_variable']].copy()
adata_sub.var_names_make_unique()
adata_sub.X = adata_sub.layers['log1'].copy().toarray()
np.savetxt(datapath + 'EB_dataMat_HVGs.txt', adata_sub.X, delimiter="\t")
np.savetxt(datapath + 'EB_data_HVGs.txt', adata_sub.var_names, fmt='%s', delimiter="\t")
np.savetxt(datapath + 'EB_data_timepoints.txt', adata_sub.obs['timepoint'], fmt='%s', delimiter="\t")



#-------------------------
#UMAP:
#-------------------------
logging.info('Computing UMAP coordinates ...')
#---Computing UMAP coordinates:
sc.pp.scale(adata_sub)
sc.tl.pca(adata_sub)
sc.pp.neighbors(adata_sub, n_pcs=50)
sc.tl.umap(adata_sub)
#sc.pl.umap(adata_sub, color='timepoint')

#---Saving UMAP coordinates:
logging.info('Saving data ...')
np.savetxt(datapath + 'EB_data_UMAPcoords.txt', adata_sub.obsm['X_umap'], delimiter="\t")