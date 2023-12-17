#--------------------------------------
#Load and preprocess CorticalMouseData:
#--------------------------------------
#Part 1 of the preprocessing of cortical mouse data from 'Tasic et. al., Nat Neuroscience 2016'.
#
#Here:
#1) Data is downloaded
#2) Counts are normalized
#3) Nonneural cells are filtered
#4) Data is log1p transformed
#5) Data is standardized
#6) Data is stored
#
#Optional: the script contains code for loading the cortical mouse neurons 
#          with only the preselected marker and receptor genes 
#          (Tasic et. al., Nat Neuroscience 2016).


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
using GZip;
using DelimitedFiles;
using XLSX;
using DataFrames;
using Statistics;



#------------------------------
# Define paths and include functions:
#------------------------------
#---Set paths:
projectpath = joinpath(@__DIR__, "../../"); 
srcpath = projectpath * "src/";
datapath = projectpath * "data/corticalMouseData/";

#---Include functions:
include(srcpath * "/preprocessing.jl");



#------------------------------
# Download and preprocess CorticalMouseData:
#------------------------------
#---Download data [count matrix, gene names, sample info] (Tasic et. al., Nat Neuroscience 2016)::
x, genenames, sampleinfo = expressiondata(datapath);

#---Normalize count matrix (DESeq2):
xnorm = normalizecountdata(x);

#---(opt.) Load pre-selected marker-and receptor gene information (Tasic et. al., Nat Neuroscience 2016):
#markergenes = readdlm(datapath * "gene_subtypes.csv", ';')[2:end, 1];
#receptorgenes = [i  for i = readdlm(datapath * "S15.txt")[:, 2:end][:] if i!="" ];

#---Extract cell type information from sampleinfo:
celltype = [split(i)[1] for i = sampleinfo.PrimaryType];

#---Determine neural cells:
Nonneural = ["Astro Aqp4","OPC Pdgfra","Oligo 96*Rik","Oligo Opalin","Micro Ctss","Endo Xdh","SMC Myl9"];
neuralcells = [!(i in Nonneural) for i = sampleinfo.PrimaryType];
celltype = celltype[neuralcells];

#---(opt.) Select marker and receptor genes (Tasic et. al., Nat Neuroscience 2016):
#receptorandmarkers = [i in union(receptorgenes, markergenes) for i = genenames];

#---Exclude non-neural cells from the normalized count matrix:
xnorm_sel_allgenes = xnorm[:, neuralcells];

#---(opt.) Exclude non-receptor and non-marker genes the normalized count matrix consisting of only neural cells:
#xnorm_sel = xnorm_sel_allgenes[receptorandmarkers, :];

#---(opt.) Define a string vector of selected genes:
#genenames_sel = genenames[receptorandmarkers];

#---Log1 transform the comprised normalized count matrices:
x_all_log1 = log1transform(xnorm_sel_allgenes');
#x_log1 = log1transform(xnorm_sel');

#---Standardize the comprised normalized count matrices:
x_all_st = standardize(x_all_log1);
#x_st = standardize(x_log1);



#------------------------------
# Save data:
#------------------------------
writedlm(datapath * "corticalMouseDataMat_allgenes_st.txt", x_all_st);
writedlm(datapath * "corticalMouseDataMat_allgenes_log1.txt", x_all_log1);
writedlm(datapath * "celltype.txt", celltype);
writedlm(datapath * "genenames.txt", genenames);
#writedlm(datapath * "corticalMouseDataMat_allgenes.txt", transpose(x[:, neuralcells]));
#writedlm(datapath * "corticalMouseDataMat_norm.txt", xnorm_sel);
#writedlm(datapath * "corticalMouseDataMat_log1.txt", x_log1);
#writedlm(datapath * "corticalMouseDataMat_st.txt", x_st);
#writedlm(datapath * "selGenes.txt", genenames_sel);