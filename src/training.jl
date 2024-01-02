#------------------------------
# This file contains the training function for the BAE and timeBAE:
#------------------------------

"""
    trainBAE(X::AbstractArray, BAE::Autoencoder; mode::String="alternating", time_series::Bool=false, ϵ::Number=0.01, ν::Number=0.01, zdim::Int=10, m::Int=1, batchsize::Int=size(X, 1), epochs::Int=50)

Train a Boosting Autoencoder (BAE) or a time-series Boosting Autoencoder (timeBAE) on the given data. User can choose between the alternating and the jointLoss training mode. The alternating mode trains the encoder and decoder in an alternating fashion, while the jointLoss mode trains the encoder and decoder jointly. The user can also choose whether the input data is a time series or not. If the input data is a time series, the timeBAE is trained, otherwise the BAE is trained. 

# Arguments
- `X::AbstractArray`: The input data with size (n_samples, n_features).
- `BAE::Autoencoder`: The Boosting Autoencoder or time-series Boosting Autoencoder model to be trained.
- `mode::String="alternating"`: The training mode. Possible values are "alternating" and "jointLoss".
- `time_series::Bool=false`: Whether the input data is a time series or not.
- `ϵ::Number=0.01`: The step size parameter for the boosting component.
- `ν::Number=0.01`: The learning rate for the optimizer that updates the decoder parameters.
- `zdim::Int=10`: The dimension of the latent space.
- `m::Int=1`: The number of boosting iterations.
- `batchsize::Int=size(X, 1)`: The batch size for training.
- `epochs::Int=50`: The number of training epochs.

# Returns
- `BAE.encoder.coeffs`: The encoder weight matrix after trining. If the input data is a time series, the encoder weight matrix is a block matrix, where each block corresponds to the encoder weight matrix at a specific time point.
"""
function trainBAE(X::AbstractArray, BAE::Autoencoder; mode::String="alternating", time_series::Bool=false, ϵ::Number=0.01, ν::Number=0.01, zdim::Int=10, m::Int=1, batchsize::Int=size(X, 1), epochs::Int=50)

    if time_series == false
        #---BAE:
        opt = ADAM(ν)
        ps = Flux.params(BAE.decoder)

        if mode == "alternating"
            @info "Training BAE in alternating mode for $(epochs) epochs ..."
    
            @showprogress for iter in 1:epochs
                batch = Flux.Data.DataLoader(X', batchsize=batchsize, shuffle=true) 
                
                BAE.encoder.coeffs = seq_constr_compL2Boost(X, BAE, ϵ, zdim, m)
                                
                Flux.train!(loss_wrapper(BAE), ps, batch, opt) 
            end
    
        elseif mode == "jointLoss"
            @info "Training BAE in jointLoss mode for $(epochs) epochs ..."
    
            @showprogress for iter in 1:epochs
                batch = Flux.Data.DataLoader(X', batchsize=batchsize, shuffle=true) 
                                
                Flux.train!(jointLoss_wrapper(BAE, ϵ, m, zdim, iter), ps, batch, opt) 
            end
    
        end
    else
        
        #---timeBAE:
        opt = ADAM(ν)

        if mode == "alternating"
            @info "Training timeBAE in alternating mode for $(epochs) epochs per time point ..."

            timepoints = length(X)
            B = zeros(size(X[1], 2), zdim * (timepoints+1))

            for t in 1: timepoints
                @info "Training at timepoint $(t) ..."

                ps = Flux.params(BAE.decoder)

                @showprogress for iter in 1:epochs
                    batch = Flux.Data.DataLoader(X[t]', batchsize=batchsize, shuffle=true) 
                    
                    BAE.encoder.coeffs = seq_constr_compL2Boost(X[t], BAE, ϵ, zdim, m)
                                        
                    Flux.train!(loss_wrapper(BAE), ps, batch, opt) 
                end

                B[:, t*zdim+1: (t+1)*zdim] = BAE.encoder.coeffs - B[:, (t-1)*zdim+1: t*zdim]

                encoder = LinearLayer(B[:, t*zdim+1: (t+1)*zdim])
                decoder = Chain(
                                Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
                                Dense(p, p, initW = Flux.glorot_uniform)         
                )
                BAE = Autoencoder(encoder, decoder)


            end

            BAE.encoder.coeffs = B[:, zdim+1:zdim*(timepoints+1)]
        
        elseif mode == "jointLoss"
            @info "Training timeBAE in jointLoss mode for $(epochs) epochs per time point ..."

            timepoints = length(X)
            B = zeros(size(X[1], 2), zdim * (timepoints+1))
        
            for t in 1: timepoints
                @info "Training at timepoint $(t) ..."

                ps = Flux.params(BAE.decoder)

                @showprogress for iter in 1:epochs
                    batch = Flux.Data.DataLoader(X[t]', batchsize=batchsize, shuffle=true) 
                                        
                    Flux.train!(jointLoss_wrapper(BAE, ϵ, m, zdim, iter), ps, batch, opt) 
                end

                B[:, t*zdim+1: (t+1)*zdim] = BAE.encoder.coeffs - B[:, (t-1)*zdim+1: t*zdim]

                encoder = LinearLayer(B[:, t*zdim+1: (t+1)*zdim])
                decoder = Chain(
                                Dense(zdim, p, tanh, initW=Flux.glorot_uniform),
                                Dense(p, p, initW = Flux.glorot_uniform)         
                )
                BAE = Autoencoder(encoder, decoder)

            end

            BAE.encoder.coeffs = B[:, zdim+1:zdim*(timepoints+1)]
    
        end
    end

    return BAE.encoder.coeffs
end