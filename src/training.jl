#---training function for the BAE: #TODO! multiple dispatch #TODO! Add timeBAE
function trainBAE(X, BAE::Autoencoder; mode::String="alternating", time_series::Bool=false, ϵ::Number=0.02, ν::Number=0.01, zdim::Int=6, m::Int=1, batchsize::Int=size(X, 1), epochs::Int=50)

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
                                
                Flux.train!(combinedLoss_wrapper(BAE, ϵ, m, zdim, iter), ps, batch, opt) 
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
                                        
                    Flux.train!(combinedLoss_wrapper(BAE, ϵ, m, zdim, iter), ps, batch, opt) 
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