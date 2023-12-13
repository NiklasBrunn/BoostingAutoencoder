#---boosting functions:
function get_latdim_grads(Xt::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder) 
    #compute the current latent representation:
    Z = BAE.encoder(Xt) 

    #compute the gradients of the reconstruction loss w.r.t. the current latent representation (matrix form)
    gs = transpose(gradient(arg -> loss_z(arg, Xt, BAE.decoder), Z)[1]) 
    return -gs 
end

function calcunibeta(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat}, n::Int, p::Int)
    unibeta = zeros(p)
    denom = zeros(p)

    #compute the univariate OLLS-estimator for each component 1:p:
    for j = 1:p

       for i=1:n
          unibeta[j] += X[i, j]*res[i]
          denom[j] += X[i, j]*X[i, j]
       end

       unibeta[j] /= denom[j] 

    end

    #return a vector unibeta consisting of the OLLS-estimators and another vector, 
    #consisting of the denominators (for later re-scaling)
    return unibeta, denom
end

function calcunibeta_combinedLoss(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat})
    #compute a vector unibeta consisting of the univariate OLLS-estimators:
    unibeta = X'* res ./ sum(X.^2, dims = 1)'
    return unibeta
end

function compL2Boost!(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:Number}, ϵ::Number, M::Int)
    #determine the number of observations and the number of features in the training data:
    n, p = size(X)

    for step in 1:M

        #compute the residual as the difference of the target vector and the current fit:
        curmodel = X * β
        res = y .- curmodel

        #determine the p unique univariate OLLS estimators for fitting the residual vector res:
        unibeta, denom = calcunibeta(X, res, n, p) 

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        optindex = findmax(collect(unibeta[j]^2 * denom[j] for j in 1:p))[2]

        #update β by adding a re-scaled version of the selected OLLS-estimator, by a scalar value ϵ ∈ (0,1):
        β[optindex] += unibeta[optindex] * ϵ 

    end

    return β
end

function compL2Boost_combinedLoss(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:AbstractFloat}, ϵ::Number, M::Int)
    #determine the number of features in the training data:
    p = size(X, 2)

    for step in 1:M

        #compute the residual as the difference of the target vector and the current fit:
        curmodel = X * β
        res = y .- curmodel

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        unibeta = calcunibeta_combinedLoss(X, res) 

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        optindex = findmin(collect(sum((res .- X[:, j]*unibeta[j]).^2) for j in 1:p))[2]

        #update β by adding a re-scaled version of the selected OLLS-estimator, by a scalar value ϵ ∈ (0,1):
        β = [i == optindex ? β[i] + unibeta[i] * ϵ : β[i] + 0 for i in 1:p]

    end

    return β
end

function seq_compL2Boost(X::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int)
    
    #compute neg. gradient - vectors w.r.t. latent dimensions:
    grads = get_latdim_grads(X', BAE) 
        
    for l in 1:zdim

        #the pseudo target is determined by the st. neg. grad.:
        y = standardize(grads[:, l])

        #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
        BAE.encoder.coeffs[:, l] = compL2Boost!(BAE.encoder.coeffs[:, l], X, y, ϵ, M)
        
    end
        
    return BAE.encoder.coeffs
end

function seq_constr_compL2Boost(X::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int)
    
    #compute neg. gradient - vectors w.r.t. latent dimensions:
    grads = get_latdim_grads(X', BAE) 
        
    for l in 1:zdim

        #determine the indices of latent dimensions excluded for determining the pseudo-target for boosting:
        Inds = union(find_zero_columns(BAE.encoder.coeffs), l)

        if length(Inds) == zdim
            #since all lat. dims. are excluded, the pseudo target is determined by the st. neg. grad.:
            y = standardize(grads[:, l])

            #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
            BAE.encoder.coeffs[:, l] = compL2Boost!(BAE.encoder.coeffs[:, l], X, y, ϵ, M)

        else
            #compute optimal current OLLS-fit of other latent repr. to the st. neg. grad:
            curdata = X * BAE.encoder.coeffs[:, Not(Inds)]
            curtarget = standardize(grads[:, l]) 
            curestimate = inv(curdata'curdata)*(curdata'curtarget) 
             #curestimate = inv(curdata'curdata + Float32.(1.0e-5 * Matrix(I, size(curdata, 2), size(curdata, 2))))*(curdata'curtarget)  #damped version for avoiding invertibility problems

            #compute the pseudo-target for the boosting [st. difference of st. neg. grad and optimal OLLS-fit]:
            y = standardize(curtarget .- (curdata * curestimate))

            #apply componentwise boosting to update one component of the β-vector (l-th col. of matrix B):
            BAE.encoder.coeffs[:, l] = compL2Boost!(BAE.encoder.coeffs[:, l], X, y, ϵ, M)
        end
        
    end
        
    return BAE.encoder.coeffs
end

function seq_constr_compL2Boost_combinedLoss(batch::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int, iter::Int)
    
    #compute neg. gradient - vectors w.r.t. latent dimensions:
    grads = get_latdim_grads(batch, BAE) 

    for l = 1:zdim 
        #determine the indices of latent dimensions excluded for determining the pseudo-target for boosting:
        if iter==1
            Inds = find_zero_columns(BAE.encoder.coeffs)
        else
            Inds = vcat(find_zero_columns(BAE.encoder.coeffs), l)
        end

        if length(Inds) == zdim
            #since all lat. dims. are excluded, the pseudo target is determined by the st. neg. grad.:
            y = standardize(grads[:, l])

            #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
            β = compL2Boost_combinedLoss(BAE.encoder.coeffs[:, l], batch', y, ϵ, M) 
            BAE.encoder.coeffs = hcat(BAE.encoder.coeffs[:, 1:l-1], β, BAE.encoder.coeffs[:,l+1:end])

        else
            #compute optimal current OLLS-fit of other latent repr. to the st. neg. grad:
            curdata = batch' * BAE.encoder.coeffs[:, Not(Inds)]
            curtarget = standardize(grads[:, l]) 
            curestimate = inv(curdata'curdata)*(curdata'curtarget) 
             #curestimate = inv(curdata'curdata + Float32.(1.0e-5 * Matrix(I, size(curdata, 2), size(curdata, 2))))*(curdata'curtarget)  #damped version for avoiding invertibility problems

            #compute the pseudo-target for the boosting [st. difference of st. neg. grad and optimal OLLS-fit]:
            y = standardize(curtarget .- (curdata * curestimate))

            #apply componentwise boosting to update one component of the β-vector (l-th col. of matrix B):
            β = compL2Boost_combinedLoss(BAE.encoder.coeffs[:, l], batch', y, ϵ, M) 
            BAE.encoder.coeffs = hcat(BAE.encoder.coeffs[:, 1:l-1], β, BAE.encoder.coeffs[:, l+1:end])
        end
    end
        
    return BAE.encoder.coeffs
end