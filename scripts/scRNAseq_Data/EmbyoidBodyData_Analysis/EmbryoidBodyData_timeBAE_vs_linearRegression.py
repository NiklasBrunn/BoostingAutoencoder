import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import pandas as pd
import anndata as ad
import os
import logging

#---Configure logging to output messages with level INFO or higher
logging.basicConfig(level=logging.INFO)

#---Setting paths: 
cur_dir = os.path.dirname(os.path.abspath(__file__))
projectpath = os.path.join(cur_dir, '../../../')
datapath = projectpath + 'data/embryoidBodyData/';
figurespath = projectpath + 'figures/embryoidBodyData_linearRegression_comparison/';

#---Load corticalMouseData:
data = np.loadtxt(datapath + 'EB_dataMat_DEGs.txt', delimiter="\t") #log1p transformed counts
genenames = np.genfromtxt(datapath + 'EB_data_DEGs.txt', dtype=str, delimiter="\n")
clusters = np.genfromtxt(datapath + 'EB_data_filt_leiden_res0_025.txt', dtype=str, delimiter="\n")
timepoints = np.genfromtxt(datapath + 'EB_data_filt_timepoints.txt', dtype=str, delimiter="\n")

#---Convert float values to desired format
timepoints = [str(int(float(tp))) if '.' not in tp else str(int(float(tp))) for tp in timepoints]

#---Create an AnnData object:
adata = ad.AnnData(data) 
adata.var_names = genenames
adata.obs['clusters'] = clusters[1:]
adata.obs['timepoints'] = timepoints



#---Linear Regression: 
for c in np.unique(adata.obs['clusters']):

    #subset anndata object:
    adata_c = adata[adata.obs['clusters'] == c]

    #filter genes that are not expressed at all:
    sc.pp.filter_genes(adata_c, min_cells=10)

    #standardize the data matrix:
    sc.pp.scale(adata_c)

    #define the data matrix X:
    X = adata_c.X

    #computing umap coords:
    sc.tl.pca(adata_c)
    sc.pp.neighbors(adata_c, n_pcs=50)
    sc.tl.umap(adata_c)
    sc.pl.umap(adata_c, color=['timepoints'], legend_loc="on data", show=False)
    plt.savefig(figurespath + '/C' + c + '.pdf', dpi=300)

    for t in np.unique(adata_c.obs['timepoints']):
        
        #Define binary response vector y:
        y = adata_c.obs['timepoints'] == t
        y = np.array(y.astype(int))

        #standardize y:
        y_st = (y - np.mean(y)) / np.std(y)

        #fit linear model:
        linear_model = sm.OLS(y_st, X).fit()
        #print(linear_model.summary())

        coeffs = linear_model.params
        pvals = linear_model.pvalues
        adjusted_pvals = multipletests(pvals, method='bonferroni')[1]  # Change 'bonferroni' to 'fdr_bh' for FDR control
        
        df = pd.DataFrame({'coeffs': coeffs, 'adjusted_pvals': adjusted_pvals, 'genenames' : adata_c.var_names})
        df.sort_values(by='adjusted_pvals', ascending=True).to_csv(datapath + 'EB_data_linear_model_C' + c + '_T' + t + '.csv')

        print(df.head())

        #umap:
        Z = np.matmul(X, coeffs)
        adata_c.obs['Z_c_t'] = Z
        sc.pl.umap(adata_c, color=['Z_c_t'], legend_loc="on data", show=False)
        save='_C' + c + '_T' + t + '.png'
        plt.savefig(figurespath + '/C' + c + '_T' + t + '.pdf', dpi=300)
