#------------------------------
# BAE applied to the simulated 3 cell groups across 3 time points scRNA-seq data: 
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
datapath = projectpath * "data/sim3cellgroups3stagesScRNAseq/";
figurespath = projectpath * "figures/sim3cellgroups3stagesScRNAseq/";
if !isdir(figurespath)
    # Create the folder if it does not exist
    mkdir(figurespath)
end

#---Include functions:
include(projectpath * "/src/BAE.jl");
using .BoostingAutoEncoder;
using Random;
using Flux;
using Statistics;



#------------------------------
# Generate data:
#------------------------------
#---Seed for reproducibility:
dataseed = 5;  

#---Hyperparameter for data generation:
rescale_factor = 2.0; 

(L_dicho, L, X_dicho, X) = simulate_3cellgroups3stagesScRNAseq(dataseed; 
    n=[310, 298, 306], p=80, 
    p1=0.6, p2=0.1, rescale_factor=rescale_factor
);
L_st = simulate_3cellgroups3stagesScRNAseq(dataseed; 
    n=[310, 298, 306], p=80, 
    p1=0.6, p2=0.1, rescale_factor=1.0
)[2];


n, p = size(X);
timepoints = Int32.(size(L, 1));


#---Create and save data matrices and plots of the data matrices:
#writedlm(datapath * "sim3cellgroups3stagesScRNAseqData_binarized.txt", X_dicho);
vegaheatmap(X_dicho; 
    path=figurespath * "/sim3cellgroups3stagesScRNAseqData_binarized.pdf", 
    xlabel="Gene", ylabel="Cell", legend_title="Gene expression",
    color_field="value:o", scheme="paired", save_plot=true, 
    legend_symbolSize=300.0, legend_labelFontSize=24.0, legend_titleFontSize=24.0,
    axis_labelFontSize=18.0, axis_titleFontSize=22.0
);

for t in 1:length(L_dicho)
    #writedlm(datapath * "sim3cellgroups3stagesScRNAseqData_tp$(t)_binarized.txt", L_dicho[t]);
    vegaheatmap(L_dicho[t]; 
        path=figurespath * "/sim3cellgroups3stagesScRNAseqData_tp$(t)_binarized.pdf", 
        xlabel="Gene", ylabel="Cell", legend_title="Gene expression", 
        color_field="value:o", scheme="paired", save_plot=true
    );
    #writedlm(datapath * "sim3cellgroups3stagesScRNAseqData_tp$(t)_binarized.txt", L_dicho[t]);
    vegaheatmap(L_dicho[t]; 
        path=figurespath * "/sim3cellgroups3stagesScRNAseqData_tp$(t)_binarized_withTitle.pdf", 
        xlabel="Gene", ylabel="Cell", Title="Time point $(t)", legend_title="Gene expression",
        color_field="value:o", scheme="paired", save_plot=true, title_fontSize=28.0,
        legend_symbolSize=300.0, legend_labelFontSize=24.0, legend_titleFontSize=22.0,
        axis_labelFontSize=18.0, axis_titleFontSize=22.0
    );
    #writedlm(datapath * "sim3cellgroups3stagesScRNAseqData_tp$(t)_standardized.txt", L_st[t]);
    vegaheatmap(L_st[t]; 
        path=figurespath * "/sim3cellgroups3stagesScRNAseqData_tp$(t)_standardized.pdf", 
        xlabel="Gene", ylabel="Cell", legend_title="Gene expression", 
        color_field="value", scheme="inferno", save_plot=true
    );
    #writedlm(datapath * "sim3cellgroups3stagesScRNAseqData_tp$(t)_standardized_rescaleVal$(rescale_factor).txt", L_st[t]);
    vegaheatmap(L[t]; 
        path=figurespath * "/sim3cellgroups3stagesScRNAseqData_tp$(t)_standardized_rescaleVal$(rescale_factor).pdf", 
        xlabel="Gene", ylabel="Cell", legend_title="Gene expression",
        color_field="value", scheme="inferno", save_plot=true
    );
end



#------------------------------
# Define and train timeBAE:
#------------------------------
#---Seeds for reproducibility:
modelseed = 14; 
batchseed = 5; 

#---Hyperparameters for training:
mode = "jointLoss"; #options are: "alternating", "jointLoss"

zdim = 3;

batchsize = minimum([size(L[1], 1), size(L[2], 1), size(L[3], 1)])-100;

epochs = 9; 

ϵ = 0.01;


#---Build BAE:
encoder = LinearLayer(zeros(p, zdim));

Random.seed!(modelseed);
decoder = Chain(
            Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
            Dense(p, p, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);

#---Train timeBAE:
Random.seed!(batchseed);
B = trainBAE(L, BAE; mode=mode, time_series=true, zdim=zdim, ϵ=ϵ, batchsize=batchsize, epochs=epochs);

B_perm = zeros(size(B));
for dim in 1:zdim
    for t in 1:length(L)
        B_perm[:, zdim*(dim-1)+t] = B[:, (t-1)*zdim+dim]
    end
end


#---Compute the latent representation and the correlation matrix:
Z = X * B;
absCor_Z = abs.(cor(Z, dims=1));
Z_perm = X * B_perm;
absCor_Z_perm = abs.(cor(Z_perm, dims=1));

#---Save the BAE encoder weight matrix:
#writedlm(datapath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs).txt", B);

#---Create and save plots:
vegaheatmap(B'; 
    path=figurespath * "timeBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize).pdf", 
    xlabel="Gene", ylabel="Latent Dimension", legend_title="Weight value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true,
    legend_labelFontSize=24.0, legend_titleFontSize=22.0,
    axis_labelFontSize=18.0, axis_titleFontSize=22.0
);
vegaheatmap(Z; 
    path=figurespath * "timeBAE_latentRep_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize).pdf", 
    xlabel="Latent Dimension", ylabel="Cell", legend_title="Representation value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true,
    legend_labelFontSize=24.0, legend_titleFontSize=22.0,
    axis_labelFontSize=18.0, axis_titleFontSize=22.0
);
vegaheatmap(absCor_Z; 
    path=figurespath * "timeBAE_latentDimsCor_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize).pdf", 
    xlabel="Latent Dimension", ylabel="Latent Dimension",  legend_title="Correlation",
    color_field="value", scheme="reds", save_plot=true,
    legend_labelFontSize=24.0, legend_titleFontSize=22.0,
    axis_labelFontSize=18.0, axis_titleFontSize=22.0
);
vegaheatmap(B_perm'; 
    path=figurespath * "perm_timeBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize).pdf", 
    xlabel="Gene", ylabel="Latent Dimension", legend_title="Weight value",
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true,
    legend_labelFontSize=24.0, legend_titleFontSize=22.0,
    axis_labelFontSize=18.0, axis_titleFontSize=22.0
);
vegaheatmap(Z_perm; 
    path=figurespath * "perm_timeBAE_latentRep_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize).pdf", 
    xlabel="Latent Dimension", ylabel="Cell", legend_title="Representation value", Width=400,
    color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true,
    legend_labelFontSize=24.0, legend_titleFontSize=16.0,
    axis_labelFontSize=18.0, axis_titleFontSize=22.0
);
vegaheatmap(absCor_Z_perm; 
    path=figurespath * "perm_timeBAE_latentDimsCor_zdim$(zdim)_epochs$(epochs).pdf", 
    xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
    color_field="value", scheme="reds", save_plot=true,
    legend_labelFontSize=24.0, legend_titleFontSize=22.0,
    axis_labelFontSize=20.0, axis_titleFontSize=22.0
);

for t in 1:length(L)
    vegaheatmap(B[:,(t-1)*zdim+1:(t-1)*zdim+zdim]'; 
        path=figurespath * "timeBAE_encoderWeightMatrix_tp$(t)_zdim$(zdim)_epochs$(epochs)_batchsize$(batchsize).pdf", 
        xlabel="Gene", ylabel="Latent Dimension", legend_title="Weight",
        color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true,
        legend_labelFontSize=24.0, legend_titleFontSize=22.0,
        axis_labelFontSize=18.0, axis_titleFontSize=22.0
    );
end