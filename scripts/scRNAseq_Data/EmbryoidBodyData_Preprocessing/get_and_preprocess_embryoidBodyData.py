###########################
#This Script is part 1 of the preperation of the embryoid body data investigated in
#'Visualizing structure and transitions in high-dimensional biological data'.
#The preprocessing in this script is oriented on the google colab notebook:
#https://colab.research.google.com/github/KrishnaswamyLab/SingleCellWorkshop/blob/master/exercises/Preprocessing/notebooks/00_Answers_Loading_and_preprocessing_scRNAseq_data.ipynb#scrollTo=ZxLVx9q1acXo
#
#Here:
#1) The embryoid body data is downloaded
#2) Cells from the first time interval are excluded
#3) Cells are getting filtered by library size
#4) Cells are getting filtered by mitochondrial gene counts
#5) Genes are getting filtered by low erpression across cells
#6) The normalized count matrix is stored 
#7) Information about cells is stored (measurement time points, library sizes)
###########################


import scprep
import os
import pandas as pd
import logging

#---Configure logging to output messages with level INFO or higher
logging.basicConfig(level=logging.INFO)

#---Setting paths: 
cur_dir = os.path.dirname(os.path.abspath(__file__))
projectpath = os.path.join(cur_dir, '../../../')
datapath = projectpath + 'data/embryoidBodyData/';
if not os.path.exists(datapath):
    # Create the folder if it does not exist
    os.makedirs(datapath)

#---Load data from 5 different timepoints 
#Note: We exclude data from time point one for the modeling task with the timeBAE:
logging.info('Loading data ...')
#T1:"Day 00-03", T2:"Day 06-09", T3:"Day 12-15", T4:"Day 18-21", T5:"Day 24-27":
if not os.path.isdir(os.path.join(datapath, "scRNAseq", "T0_1A")):
    scprep.io.download.download_and_extract_zip(
        url="https://data.mendeley.com/public-files/datasets/v6n743h5ng/files/b1865840-e8df-4381-8866-b04d57309e1d/file_downloaded",
        destination=datapath)

sparse=True
#data_time1 = scprep.io.load_10X(os.path.join(datapath, "scRNAseq", "T0_1A"), sparse=sparse, gene_labels='both')
data_time2 = scprep.io.load_10X(os.path.join(datapath, "scRNAseq", "T2_3B"), sparse=sparse, gene_labels='both')
data_time3 = scprep.io.load_10X(os.path.join(datapath, "scRNAseq", "T4_5C"), sparse=sparse, gene_labels='both')
data_time4 = scprep.io.load_10X(os.path.join(datapath, "scRNAseq", "T6_7D"), sparse=sparse, gene_labels='both')
data_time5 = scprep.io.load_10X(os.path.join(datapath, "scRNAseq", "T8_9E"), sparse=sparse, gene_labels='both')


#---Filter by library size to remove doublets and empty droplets
#   and filter (per time point) genes with low expression:
logging.info('Library size filtering ...')
filtered_batches = []
for batch in [#data_time1, 
    data_time2, data_time3, data_time4, data_time5]:

    #library size filtering of cells
    percentiles = (20, 80)
    batch = scprep.filter.filter_library_size(batch, percentile=percentiles)

    #lowly expressed gene filtering
    cutoff = 10
    batch = scprep.filter.filter_rare_genes(batch, min_cells=cutoff)

    filtered_batches.append(batch)

#del data_time1 
del data_time2, data_time3, data_time4, data_time5 


data, sample_labels = scprep.utils.combine_batches(
    filtered_batches, 
    [#"Day 00-03", 
     "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
)
del filtered_batches 


#---Filter by mitochondrial expression to remove dead cells:
logging.info('Mitochondrial gene expression filtering ...')
cutoff = 400
data_filt, sample_labels = scprep.filter.filter_gene_set_expression(
    data, sample_labels, starts_with="MT-",
    cutoff=cutoff, keep_cells='below', library_size_normalize=True
)
del data 


#---Library size normalization:
logging.info('Library size normalization ...')
data_norm, library_size = scprep.normalize.library_size_normalize(data_filt, return_library_size=True)

metadata = pd.concat([library_size, sample_labels], axis=1)


#---Save the data:
logging.info('Saving data ...')
data_filt.to_pickle(datapath + "embryoidbody_data_filt.pickle.gz")
data_norm.to_pickle(datapath + "embryoidbody_data_norm.pickle.gz")
metadata.to_pickle(datapath + "embryoidbody_metadata.pickle.gz")