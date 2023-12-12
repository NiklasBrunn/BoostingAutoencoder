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




#---New training function for the BAE: 
#TODO! optimizer as function input?
function train_BAE(X::AbstractMatrix{<:AbstractFloat}, decoder::Union{Chain, Dense}, coeffs::AbstractMatrix{<:AbstractFloat}=zeros(Float32, p, 10); 
    #device = cpu,
    ϵ::AbstractFloat=0.01, 
    η::AbstractFloat=0.01, 
    λ::AbstractFloat=0.0,
    batchsize::Int=500, 
    epochs::Int=50, 
    M::Int=1#,
    #opt::AbstractOptimiser=AdamW(0.1, (0.9, 0.999), 0.1)
    )

    #if device = cpu
    #    @info "Training on CPU ..."
    #else
    #    @info "Training on GPU ..."
    #end

    @info "Training BAE for $(epochs) epochs ..."

    X = Float32.(X)
    coeffs = Float32.(coeffs)

    #decoder = decoder |> device

    encoder = LinearLayer(coeffs) #|> device

    BAE = Autoencoder(encoder, decoder) #|> device

    my_log = []
    
    Xt = X'

    #opt_state = Flux.setup(opt, BAE.decoder)
    opt_state = Flux.setup(AdamW(η, (0.9, 0.999), λ), BAE.decoder)

    zdim = size(BAE.encoder.coeffs, 2)
    
    @showprogress for iter in 1:epochs
        #loader = Flux.Data.DataLoader(Xt |> device, batchsize=batchsize, shuffle=true) 
        loader = Flux.Data.DataLoader(Xt, batchsize=batchsize, shuffle=true) 

        losses = Float32[]
        for batch in loader
            Z = BAE.encoder(batch)
             
            loss_val, grads = Flux.withgradient(BAE.decoder, Z) do m, z
               X̂ = m(z)
               Flux.mse(X̂, batch)
            end
            push!(losses, loss_val)

            BAE.encoder.coeffs = new_seq_constr_compL2Boost(batch, grads[2], BAE, ϵ, zdim, M)
            Flux.update!(opt_state, BAE.decoder, grads[1])
        end

        push!(my_log, mean(losses))
    end

    #decoder = decoder |> cpu

    return my_log, BAE.encoder.coeffs
end


#---New training function for the timeBAE: 
#TODO! optimizer as function input?
function train_timeBAE(L::AbstractVector, decoder::Union{Chain, Dense}, coeffs::AbstractMatrix{<:AbstractFloat}=zeros(Float32, p, 10); 
    #device = cpu,
    ϵ::AbstractFloat=0.01, 
    η::AbstractFloat=0.01, 
    λ::AbstractFloat=0.0,
    batchsize::Int=500, 
    epochs::Int=50, 
    M::Int=1#,
    #opt::AbstractOptimiser=AdamW(0.1, (0.9, 0.999), 0.1)
    )

    #if device = cpu
    #    @info "Training on CPU ..."
    #else
    #    @info "Training on GPU ..."
    #end

    @info "Training timeBAE for $(epochs) epochs per time point ..."


    timepoints = length(L)

    coeffs = Float32.(coeffs)

    zdim = size(coeffs, 2)

    #Define the encoder and the BAE for training at the first time point:
    #decoder = decoder |> device
    encoder = LinearLayer(coeffs) #|> device  #define encoder
    BAE = Autoencoder(encoder, decoder) #|> device

    #Create a container for storing the transformed weight matrices after training at each time point:
    B = zeros(size(L[1], 2), zdim * (timepoints+1))

    #Create a container for storing the mean-losses per training epoch for each time point:
    loss_list = []

    for t in 1: timepoints
        @info "Training at timepoint $(t) ..."

        #define training data for the current BAE at time point t:
        X = Float32.(L[t])
        Xt = X'

        #Setup the optimizer:
        #opt_state = Flux.setup(opt, BAE.decoder)
        opt_state = Flux.setup(AdamW(η, (0.9, 0.999), λ), BAE.decoder)

        #Train the current BAE with data at time point t for a fixed number of epochs:
        my_log = []
        @showprogress for iter in 1:epochs
            #loader = Flux.Data.DataLoader(Xt |> device, batchsize=batchsize, shuffle=true) 
            loader = Flux.Data.DataLoader(Xt, batchsize=batchsize, shuffle=true) 
    
            losses = Float32[]
            for batch in loader
                Z = BAE.encoder(batch)
                 
                loss_val, grads = Flux.withgradient(BAE.decoder, Z) do m, z
                   X̂ = m(z)
                   Flux.mse(X̂, batch)
                end
                push!(losses, loss_val)
    
                BAE.encoder.coeffs = new_seq_constr_compL2Boost(batch, grads[2], BAE, ϵ, zdim, M)
                Flux.update!(opt_state, BAE.decoder, grads[1])
            end
    
            push!(my_log, mean(losses))
        end

        #Save the difference in encoder weight matrices between the current and the previous time point:
        B[:, t*zdim+1: (t+1)*zdim] = BAE.encoder.coeffs - B[:, (t-1)*zdim+1: t*zdim]

        #Re-initialize BAE parameters (encoder based on prior-knowledge from training at the previous time point and decoder params randomly):
        encoder = LinearLayer(B[:, t*zdim+1: (t+1)*zdim])
        decoder = Chain(
                        Dense(zdim => p, tanh),
                        Dense(p => p)         
        )
        BAE = Autoencoder(encoder, decoder)

        push!(loss_list, my_log)

    end

    #return a list consisting the mean-losses per training epoch for each time point and the final weight matrix B: 
    return loss_list, B[:, zdim+1:zdim*(timepoints+1)]
end