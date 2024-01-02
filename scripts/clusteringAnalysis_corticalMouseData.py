import os
import scanpy as sc
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import random
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score


#---Define clustering function:
def cluster_data(adata, resolution=0.75, seed=1, label=1):
    
    # Run the Leiden algorithm for clustering
    sc.tl.leiden(adata, resolution=resolution, key_added=f'leiden_res{resolution}_{label}', random_state=seed)
    
    return adata


#---Set paths:
sc.settings.verbosity = 0
sc.settings.set_figure_params(dpi=80, facecolor="white", frameon=False)

cur_dir = os.path.dirname(os.path.abspath(__file__))
projectpath = os.path.join(cur_dir, '../../../')
datapath = projectpath + 'data/corticalMouseData/';
figurespath = projectpath + 'figures/corticalMouseData_clusteringAnalysis/';
if not os.path.exists(figurespath):
    # Create the folder if it does not exist
    os.makedirs(figurespath)


#---Define random seeds for leidenalg clustering:
random.seed(5)
seeds = [random.randint(1, 1000) for _ in range(20)]



#---Set Hyperparameters for clustering analysis:
resolution_PCA = 0.5
resolution_BAE = 0.7
num_neighbors = 15


##################################
#---Clustering PCA representation:
##################################
Z_PCA = np.loadtxt(datapath + 'Z_PCA_clustering.txt', delimiter='\t')

#---Create an AnnData object:
adata_PCA = sc.AnnData(Z_PCA)

#---Compute the KNN graph:
sc.pp.neighbors(adata_PCA, n_neighbors=num_neighbors)

#---Compute the UMAP embedding:
sc.tl.umap(adata_PCA)

#---Compute the Leidenalg clusters for PCA latent representation:
clustering_results_df_PCA = pd.DataFrame()
PCA_sil_scores = []
label = 1
for seed in seeds:
    cluster_data(adata_PCA, resolution=resolution_PCA, seed=seed, label=label)
    clustering_results_df_PCA[f'leiden_res{resolution_PCA}_{label}'] = adata_PCA.obs[f'leiden_res{resolution_PCA}_{label}'].values
    PCA_sil_scores.append(silhouette_score(Z_PCA, clustering_results_df_PCA[f'leiden_res{resolution_PCA}_{label}'], metric='euclidean'))
    label+=1

PCA_sil_scores = np.array(PCA_sil_scores)
mean_PCA_sil_score = statistics.mean(PCA_sil_scores) 



##################################
#---Clustering BAE representation:
##################################
clustering_results_df_BAE = pd.DataFrame()
BAE_sil_scores = []
ARis = []
label = 1
for modelseed in range(1, 21):
    #---Load the BAE representations trained for 25 epochs in 'jointLoss' mode:
    Z_BAE = np.loadtxt(datapath + f'Z_BAE_25_modelseed{modelseed}_clustering.txt', delimiter='\t')

    adata_BAE = sc.AnnData(Z_BAE)

    sc.pp.neighbors(adata_BAE, n_neighbors=num_neighbors)

    sc.tl.umap(adata_BAE)

    #---Compute the Leidenalg clusters for BAE latent representation:
    seed = random.randint(1, 1000)
    cluster_data(adata_BAE, resolution=resolution_BAE, seed=seed, label=label)
    clustering_results_df_BAE[f'leiden_res{resolution_BAE}_{label}'] = adata_BAE.obs[f'leiden_res{resolution_BAE}_{label}'].values
    BAE_sil_scores.append(silhouette_score(Z_BAE, clustering_results_df_BAE[f'leiden_res{resolution_BAE}_{label}'], metric='euclidean'))
    ARis.append(adjusted_rand_score(clustering_results_df_BAE[f'leiden_res{resolution_BAE}_{label}'], clustering_results_df_PCA[f'leiden_res{resolution_PCA}_{label}']))

    label+=1


BAE_sil_scores = np.array(BAE_sil_scores)
mean_BAE_sil_score = statistics.mean(BAE_sil_scores)
ARis = np.array(ARis)

plt.figure(figsize=(6, 10))
plt.boxplot([PCA_sil_scores, BAE_sil_scores])
#---Adding title and labels
plt.title(f'Silhouette coefficient of PCA and BAE representation')
plt.ylabel('Silhouette coefficient')
plt.xticks(ticks=[1, 2], labels=['PCA', 'BAE'])
plt.scatter([1, 2], [mean_PCA_sil_score, mean_PCA_sil_score], marker='x', color='red', label='mean PCA sil-coef')
plt.scatter([1, 2], [mean_BAE_sil_score, mean_BAE_sil_score], marker='x', color='blue', label='mean BAE sil-coef')
    
plt.legend(loc='lower right', fontsize='small')
plt.ylim(0.2, 0.6)

plt.savefig(figurespath + f'20leidenalgClustering_res{resolution_BAE}_silScore_BAE_modelseed{modelseed}.pdf', dpi=300, bbox_inches='tight')
#plt.show()




plt.figure(figsize=(6, 10))
plt.boxplot(ARis)
#---Adding title and labels
plt.title('Adjusted Rand indices of 20 BAE and PCA representations')
plt.ylabel('Adjusted Rand index')
plt.xticks([1], ['BAE'])

plt.savefig(figurespath + f'20leidenalgClustering_res{resolution_BAE}_ARis_BAE_PCA.pdf', dpi=300, bbox_inches='tight')
#plt.show()


clustering_results_BAE_PCA_df = pd.DataFrame()
clustering_results_BAE_PCA_df['ARi'] = ARis
clustering_results_BAE_PCA_df['BAE_sil_score'] = BAE_sil_scores
clustering_results_BAE_PCA_df['PCA_sil_score'] = PCA_sil_scores
clustering_results_BAE_PCA_df.to_csv(datapath + f'clustering_results_BAE_PCA.csv', index=False)

Q1_BAE = np.percentile(BAE_sil_scores, 25)  
Q3_BAE = np.percentile(BAE_sil_scores, 75)  
IQR_BAE = Q3_BAE - Q1_BAE

print(f'Median Silhouette score BAE:{statistics.median(BAE_sil_scores)}') 
print(f'Median Silhouette score PCA:{statistics.median(PCA_sil_scores)}') 
print(f'Median adjusted Rand index BAE/PCA:{statistics.median(ARis)}') 
print(f'Interquartile range Silhouette score BAE:{IQR_BAE}') 