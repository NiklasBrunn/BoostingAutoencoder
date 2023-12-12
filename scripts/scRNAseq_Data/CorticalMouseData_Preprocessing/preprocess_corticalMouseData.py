#-------------------------------------------------------------------------------------------------------------------
# Determine highly variable genes (HVGs) from the corticalMouseData dataset and for a specific cell type (e.g. Sst): 
#-------------------------------------------------------------------------------------------------------------------
#Part 2 of the preprocessing of cortical mouse data from 'Tasic et. al., Nat Neuroscience 2016'.
#This script builds up on the preprocessing julia script 'get_corticalMouseData.jl'.
#Here:
#1) Data is loaded
#2) Anndata object is created
#3) An anndata object is created for a specific cell type (e.g. Sst)
#4) Lowly expressed genes are filtered
#5) Highly variable genes (HVGs) are selected
#6) Data is stored



import scanpy as sc
import numpy as np
import anndata
import os
import logging


#---Configure logging to output messages with level INFO or higher
logging.basicConfig(level=logging.INFO)

#---Setting paths: 
cur_dir = os.path.dirname(os.path.abspath(__file__))
projectpath = os.path.join(cur_dir, '../../../')
datapath = projectpath + 'data/corticalMouseData/';


#---Load corticalMouseData:
logging.info('Create anndata oject ...')
data = np.loadtxt(datapath + 'corticalMouseDataMat_allgenes_log1.txt', delimiter="\t") #for seurat
genenames = np.genfromtxt(datapath + 'genenames.txt', dtype=str, delimiter="\n")
celltype = np.genfromtxt(datapath + 'celltype.txt', dtype=str, delimiter="\n")

adata = anndata.AnnData(data) 
adata.var_names = genenames
adata.obs['celltype'] = celltype



#---Subset the Anndata object based on the mask
logging.info('Subset anndata oject ...')
ctype = 'Sst'
cell_type_mask = (adata.obs['celltype'] == ctype)
adata_ctype = adata[cell_type_mask].copy()



#---Filter lowly expressed genes that are expressed in less than 10 cells:
logging.info('Filter lowly expressed genes ...')
sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_genes(adata_ctype, min_cells=10)



#---Select highly variable genes:
#Using flavor=seurat since counts are not integer values ...
logging.info('Selecting HVGs ...')
sc.pp.highly_variable_genes(adata, n_top_genes=1500, flavor='seurat')
sc.pp.highly_variable_genes(adata_ctype, n_top_genes=1500, flavor='seurat')



#---Save the count matrix (log1-transformed counts) consisting of only highly variable genes:
logging.info('Saving data ...')
adata_sub = adata[:, adata.var['highly_variable']]
adata_ctype_sub = adata_ctype[:, adata_ctype.var['highly_variable']]

np.savetxt(datapath + 'corticalMouseDataMat_HVGs_log1.txt', adata_sub.X, delimiter="\t")
np.savetxt(datapath + 'corticalMouseDataMat_' + ctype + '_HVGs_log1.txt', adata_ctype_sub.X, delimiter="\t")

sc.pp.scale(adata_sub)
sc.pp.scale(adata_ctype_sub)

np.savetxt(datapath + 'corticalMouseDataMat_HVGs_st.txt', adata_sub.X, delimiter="\t")
np.savetxt(datapath + 'corticalMouseDataMat_' + ctype + '_HVGs_st.txt', adata_ctype_sub.X, delimiter="\t")

np.savetxt(datapath + 'genenames_HVGs.txt', adata_sub.var_names, fmt='%s', delimiter="\n")
np.savetxt(datapath + 'genenames_' + ctype + '_HVGs.txt', adata_ctype_sub.var_names, fmt='%s', delimiter="\n")