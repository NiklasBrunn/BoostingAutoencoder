#------------------------------
# Model comparison on the simulated 10 stages scRNA-seq data (BAE, BAE without disentanglement criterion, AE, L1AE, corAE): 
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
figurespath = projectpath * "figures/sim10stagesScRNAseq_modelcomparison/";
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
dataseed = 1; 

#---Hyperparameter for data generation:
rescale_factor = 1.5; 

#---Data generation:
(X, X_dicho) = simulate_10StagesScRNAseq(dataseed; rescale_val=rescale_factor);
X_st = simulate_10StagesScRNAseq(dataseed; rescale_val=1)[1];

p = size(X, 2);

#---Save the data matrices):
#writedlm(datapath * "sim10stagesScRNAseqData_binarized.txt", X_dicho);
#writedlm(datapath * "sim10stagesScRNAseqData_standardized_rescaleVal$(rescale_factor).txt", X);
#writedlm(datapath * "sim10stagesScRNAseqData_standardized", X_st);

#---Create and save plots of the data matrices:
vegaheatmap(X_dicho; 
    path=figurespath * "sim10stagesScRNAseqData_binary.pdf", 
    xlabel="Gene", ylabel="Observation", legend_title="Gene expression",
    color_field="value:o", scheme="paired", save_plot=true
);
vegaheatmap(X; 
    path=figurespath * "sim10stagesScRNAseqData_st_rescaledNoiseFeatures.pdf", 
    xlabel="Gene", ylabel="Observation", legend_title="Gene expression",
    color_field="value", scheme="inferno", save_plot=true, set_domain_mid=true
);
vegaheatmap(X_st; 
    path=figurespath * "sim10stagesScRNAseqData_st.pdf", 
    xlabel="Gene", ylabel="Observation", legend_title="Gene expression",
    color_field="value", scheme="inferno", save_plot=true, set_domain_mid=true
);



#------------------------------
# Set hyperparameters:
#------------------------------
#---Seeds for reproducibility:
modelseed = 700; 
batchseed = 1; 

#---Hyperparameters for training:
zdim = 10;
M = 1;
batchsize = 800;

ϵ = 0.01;
ν = 0.01;

α_L1AE = 0.075;
α_corAE = 0.1;

num_epochs = 25000;
plot_epochs = [1, 15, 1000, 25000];
#plot_epochs = [1, 3, 5, 10, 15, 20, 25, 30, 50, 70, 100, 
#                  150, 200, 250, 500, 1000, 1500, 2000, 5000, 
#                  8000, 10000, 15000, 20000, 25000
#];



#------------------------------
# Training BAE (alternating version) and creating and saving plots:
#------------------------------
#---Build BAE:
encoder = LinearLayer(zeros(p, zdim));

Random.seed!(modelseed);
decoder = Chain(
                Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
                Dense(p, p, initW = Flux.glorot_uniform)         
);

BAE = Autoencoder(encoder, decoder);


#---Train BAE and generating and saving plots:
opt = ADAM(ν);
ps = Flux.params(BAE.decoder);
Random.seed!(batchseed);

@info "Training BAE in alternating mode for $(num_epochs) epochs ..."

epoch = 1;
while epoch <= num_epochs

    batch = Flux.Data.DataLoader(X', batchsize=batchsize, shuffle=true) 
            
    BAE.encoder.coeffs = seq_constr_compL2Boost(X, BAE, ϵ, zdim, M)
                            
    Flux.train!(loss_wrapper(BAE), ps, batch, opt) 
    B = BAE.encoder.coeffs;

    Z = X * B;
    absCor_Z = abs.(cor(Z, dims=1));
   

    if epoch in plot_epochs
        #writedlm(datapath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).txt", B);

        vegaheatmap(B; 
            path=figurespath * "BAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Gene", legend_title="Weight value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(Z; 
            path=figurespath * "BAE_latentRep_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Observation", legend_title="Representation value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(absCor_Z; 
            path=figurespath * "BAE_latentDimsCor_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
            color_field="value", scheme="reds", save_plot=true
        )
    end

    epoch+=1;

end



#------------------------------
# Training BAE (alternating version without disentanglement criterion) and creating and saving plots:
#------------------------------
#---Build nodisentBAE:
encoder = LinearLayer(zeros(p, zdim));

Random.seed!(modelseed);
decoder = Chain(
                Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
                Dense(p, p, initW = Flux.glorot_uniform)         
);

nodisentBAE = Autoencoder(encoder, decoder);


#---Train nodisentBAE and generating and saving plots:
opt = ADAM(ν);
ps = Flux.params(nodisentBAE.decoder);
Random.seed!(batchseed);

@info "Training nodisentBAE in alternating mode for $(num_epochs) epochs ..."

epoch = 1;
while epoch <= num_epochs

    batch = Flux.Data.DataLoader(X', batchsize=batchsize, shuffle=true) 
            
    BAE.encoder.coeffs = seq_compL2Boost(X, nodisentBAE, ϵ, zdim, M)
                            
    Flux.train!(loss_wrapper(nodisentBAE), ps, batch, opt) 
    B = nodisentBAE.encoder.coeffs;

    Z = X * B;
    absCor_Z = abs.(cor(Z, dims=1));
   

    if epoch in plot_epochs
        #writedlm(datapath * "nodisentBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).txt", B);

        vegaheatmap(B; 
            path=figurespath * "nodisentBAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Gene", legend_title="Weight value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(Z; 
            path=figurespath * "nodisentBAE_latentRep_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Observation", legend_title="Representation value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(absCor_Z; 
            path=figurespath * "nodisentBAE_latentDimsCor_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
            color_field="value", scheme="reds", save_plot=true
        )
    end

    epoch+=1;

end



#------------------------------
# Training AE and creating and saving plots:
#------------------------------
#---Build AE:
Random.seed!(modelseed); 

AE_encoder = Dense(p, zdim, initW=Flux.glorot_uniform); 

AE_decoder = Chain( 
                Dense(zdim, p, tanh, initW=Flux.glorot_uniform), 
                Dense(p, p, initW = Flux.glorot_uniform)    
); 

AE = Autoencoder(AE_encoder, AE_decoder); 


#---Train AE and generating and saving plots:
ps = Flux.params(AE.encoder, AE.decoder); 
opt = ADAM(ν); 
Random.seed!(batchseed);

@info "Training vanilla AE for $(num_epochs) epochs ..."

epoch = 1;
while epoch <= num_epochs
    data = Flux.Data.DataLoader(X', batchsize = batchsize, shuffle = true)        
    Flux.train!(loss_wrapper(AE), ps, data, opt) 
    B = AE.encoder.W; 
    Z = AE.encoder(X')';
    absCor_Z = abs.(cor(Z, dims=1));

    if epoch in plot_epochs
        #writedlm(datapath * "AE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).txt", B);

        vegaheatmap(B'; 
            path=figurespath * "AE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Gene", legend_title="Weight value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(Z; 
            path=figurespath * "AE_latentRep_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Observation", legend_title="Representation value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(absCor_Z; 
            path=figurespath * "AE_latentDimsCor_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
            color_field="value", scheme="reds", save_plot=true
        )
    end

    epoch+=1

end;



#------------------------------
# Training L1AE and creating and saving plots:
#------------------------------
#---Build L1AE:
Random.seed!(modelseed); 

L1AE_encoder = Dense(p, zdim, initW=Flux.glorot_uniform); 

L1AE_decoder = Chain( 
                Dense(zdim, p, tanh, initW=Flux.glorot_uniform), 
                Dense(p, p, initW = Flux.glorot_uniform)    
); 

L1AE = Autoencoder(L1AE_encoder, L1AE_decoder); 

#---Train L1AE and generating and saving plots:
ps = Flux.params(L1AE.encoder, L1AE.decoder); 
opt = ADAM(ν); 
Random.seed!(batchseed);

@info "Training L1AE for $(num_epochs) epochs ..."

epoch = 1;
while epoch <= num_epochs
    data = Flux.Data.DataLoader(X', batchsize = batchsize, shuffle = true)        
    Flux.train!(loss_wrapper_L1reg(L1AE, α_L1AE), ps, data, opt) 
    B = L1AE.encoder.W; 
   
    Z = L1AE.encoder(X')';
    absCor_Z = abs.(cor(Z, dims=1));

    if epoch in plot_epochs
        #writedlm(datapath * "L1AE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).txt", B);
        
        vegaheatmap(B'; 
            path=figurespath * "L1AE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Gene", legend_title="Weight value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(Z; 
            path=figurespath * "L1AE_latentRep_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Observation", legend_title="Representation value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true,
        )
        vegaheatmap(absCor_Z; 
            path=figurespath * "L1AE_latentDimsCor_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
            color_field="value", scheme="reds", save_plot=true
        )
    end

    epoch+=1

end;



#------------------------------
# Training corAE and creating and saving plots:
#------------------------------
#---Build corAE:
Random.seed!(modelseed); 

corAE_encoder = Dense(p, zdim, initW=Flux.glorot_uniform); 

corAE_decoder = Chain( 
                Dense(zdim, p, tanh, initW=Flux.glorot_uniform), 
                Dense(p, p, initW = Flux.glorot_uniform)    
); 

corAE = Autoencoder(corAE_encoder, corAE_decoder); 

#---Train corAE and generating and saving plots:
ps = Flux.params(corAE.encoder, corAE.decoder); 
opt = ADAM(ν); 
Random.seed!(batchseed);

@info "Training corAE for $(num_epochs) epochs ..."

epoch = 1;
while epoch <= num_epochs
    data = Flux.Data.DataLoader(X', batchsize = batchsize, shuffle = true)        
    Flux.train!(loss_wrapper_correg(corAE, α_corAE), ps, data, opt) 
    B = corAE.encoder.W; 
    Z = corAE.encoder(X')';
    absCor_Z = abs.(cor(Z, dims=1));

    if epoch in plot_epochs
        #writedlm(datapath * "corAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).txt", B);

        vegaheatmap(B'; 
            path=figurespath * "corAE_encoderWeightMatrix_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Gene", legend_title="Weight value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(Z; 
            path=figurespath * "corAE_latentRep_zdim$(zdim)_epochs$(epoch).pdf", 
            xlabel="Latent Dimension", ylabel="Observation", legend_title="Representation value",
            color_field="value", scheme="blueorange", save_plot=true, set_domain_mid=true
        )
        vegaheatmap(absCor_Z; 
            path=figurespath * "corAE_latentDimsCor_zdim$(zdim)_epochs$(epoch).pdf.pdf", 
            xlabel="Latent Dimension", ylabel="Latent Dimension", legend_title="Correlation",
            color_field="value", scheme="reds", save_plot=true
        )
    end

    epoch+=1

end;