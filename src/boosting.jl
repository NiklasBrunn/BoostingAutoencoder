#------------------------------
# This file contains the componentwise boosting functions for the BAE:
#------------------------------

"""
    get_latdim_grads(Xt::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder)

Compute the negative gradients of the reconstruction loss with respect to the current latent representation.

# Arguments
- `Xt::AbstractMatrix{<:AbstractFloat}`: The input data matrix.
- `BAE::Autoencoder`: The autoencoder model.

# Returns
- `gs::Matrix`: The negative gradients of the reconstruction loss with respect to the current latent representation.
"""
function get_latdim_grads(Xt::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder) 
    #compute the current latent representation:
    Z = BAE.encoder(Xt) 

    #compute the gradients of the reconstruction loss w.r.t. the current latent representation (matrix form)
    gs = -transpose(gradient(arg -> loss_z(arg, Xt, BAE.decoder), Z)[1]) 
    return gs 
end


"""
    calcunibeta(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat}, n::Int, p::Int)

Compute the univariate ordinary linear least squares estimator for each component j in 1:p, with responses `res` and observations `X[:, j]`.

# Arguments
- `X::AbstractMatrix{<:AbstractFloat}`: The input matrix of size (n, p) where n is the number of samples and p is the number of features.
- `res::AbstractVector{<:AbstractFloat}`: The response vector of size n.
- `n::Int`: The number of samples.
- `p::Int`: The number of features.

# Returns
- `unibeta::Vector`: A vector of size p containing the ordinary linear least squares estimators.
- `denom::Vector`: A vector of size p containing the denominators of the ordinary linear least squares estimators.
"""
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

"""
    calcunibeta_jointLoss(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat})

    Compute the univariate ordinary linear least squares estimator for each component j in 1:p, with responses `res` and observations `X[:, j]`.
    This version is similar to `calcunibeta` but computes the estimator such that differentiating through the boosting step does not produce a *Mutating arrays is not supported* error.

# Arguments
- `X`: An `AbstractMatrix` of input data, where each column represents a feature and each row represents an observation.
- `res`: An `AbstractVector` of response values corresponding to each observation.

# Returns
- `unibeta`: A vector of univariate ordinary linear least squares estimators.
"""
function calcunibeta_jointLoss(X::AbstractMatrix{<:AbstractFloat}, res::AbstractVector{<:AbstractFloat})
    #compute a vector unibeta consisting of the univariate OLLS-estimators:
    unibeta = X'* res ./ sum(X.^2, dims = 1)'
    return unibeta
end


"""
    compL2Boost!(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:Number}, ϵ::Number, M::Int)

Perform componentwise L2 boosting to fit a linear model.

# Arguments
- `β::AbstractVector{<:AbstractFloat}`: The initial coefficients of the linear model.
- `X::AbstractMatrix{<:AbstractFloat}`: The design matrix of the training data.
- `y::AbstractVector{<:Number}`: The response vector of the training data.
- `ϵ::Number`: The scaling factor for updating the coefficients.
- `M::Int`: The number of boosting steps.

# Returns
- `β`: The updated coefficients after M boosting steps.

# Description
This function implements the componentwise L2 boosting algorithm to fit a linear model. It iteratively updates the coefficients by adding a re-scaled version of the selected univariate ordinary least squares estimator. The algorithm aims to stepwise minimize the residual between the target vector and the current fit.

The algorithm proceeds as follows:
1. Compute the residual as the difference between the response vector and the current fit.
2. Determine the p unique univariate ordinary linear least squares estimators for fitting the residual vector.
3. Find the optimal index of the univariate estimators resulting in the currently optimal fit.
4. Update the coefficients by adding a re-scaled version of the selected estimator, by a scalar value ϵ ∈ (0,1).
5. Repeat steps 1-4 for M boosting steps.
"""
function compL2Boost!(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:Number}, ϵ::Number, M::Int)
    #determine the number of observations and the number of features in the training data:
    n, p = size(X)

    for step in 1:M

        #compute the residual as the difference of the response vector and the current fit:
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


"""
    compL2Boost_jointLoss(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:AbstractFloat}, ϵ::Number, M::Int)

The `compL2Boost_jointLoss` function implements the componentwise L2 boosting algorithm in a way such that differentiating through the boosting step does not produce a *Mutating arrays is not supported* error.
The function produces the same output as `compL2Boost!`.    

## Arguments
- `β::AbstractVector{<:AbstractFloat}`: The initial coefficients of the linear model.
- `X::AbstractMatrix{<:AbstractFloat}`: The design matrix of the training data.
- `y::AbstractVector{<:Number}`: The response vector of the training data.
- `ϵ::Number`: The scaling factor for updating the coefficients.
- `M::Int`: The number of boosting steps.

## Returns
- `β`: The updated coefficient vector after M boosting steps.
"""
function compL2Boost_jointLoss(β::AbstractVector{<:AbstractFloat}, X::AbstractMatrix{<:AbstractFloat}, y::AbstractVector{<:AbstractFloat}, ϵ::Number, M::Int)
    #determine the number of features in the training data:
    p = size(X, 2)

    for step in 1:M

        #compute the residual as the difference of the response vector and the current fit:
        curmodel = X * β
        res = y .- curmodel

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        unibeta = calcunibeta_jointLoss(X, res) 

        #determine the optimal index of the univariate estimators resulting in the currently optimal fit:
        optindex = findmin(collect(sum((res .- X[:, j]*unibeta[j]).^2) for j in 1:p))[2]

        #update β by adding a re-scaled version of the selected OLLS-estimator, by a scalar value ϵ ∈ (0,1):
        β = [i == optindex ? β[i] + unibeta[i] * ϵ : β[i] + 0 for i in 1:p]

    end

    return β
end


"""
    seq_compL2Boost(X::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int)

Sequential component-wise boosting algorithm for updating the weights of a boosting autoencoder's linear encoder.

# Arguments
- `X::AbstractMatrix{<:AbstractFloat}`: Input data matrix.
- `BAE::Autoencoder`: Autoencoder object.
- `ϵ::Number`: Boosting step size.
- `zdim::Int`: Number of latent dimensions.
- `M::Int`: Number of boosting iterations.

# Returns
- `BAE.encoder.coeffs`: Updated weights of the boosting autoencoder's linear encoder.
"""
function seq_compL2Boost(X::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int)
    
    #compute neg. gradient - vectors w.r.t. latent dimensions:
    grads = get_latdim_grads(X', BAE) 
        
    for l in 1:zdim

        #the pseudo response is determined by the st. neg. grad.:
        y = standardize(grads[:, l])

        #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
        BAE.encoder.coeffs[:, l] = compL2Boost!(BAE.encoder.coeffs[:, l], X, y, ϵ, M)
        
    end
        
    return BAE.encoder.coeffs
end


"""
    seq_constr_compL2Boost(X::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int)

The `seq_constr_compL2Boost` function updates the encoder weights of a boosting autoencoder by performing sequential componentwise boosting with addig a constraint to the response vectors.

## Arguments
- `X::AbstractMatrix{<:AbstractFloat}`: The input data matrix.
- `BAE::Autoencoder`: A boosting autoencoder.
- `ϵ::Number`: The boosting step size.
- `zdim::Int`: The number of latent dimensions.
- `M::Int`: The number of boosting iterations.

## Returns
- `BAE.encoder.coeffs`: The updated encoder weights.
"""
function seq_constr_compL2Boost(X::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int)
    
    #compute neg. gradient - vectors w.r.t. latent dimensions:
    grads = get_latdim_grads(X', BAE) 
        
    for l in 1:zdim

        #determine the indices of latent dimensions excluded for determining the pseudo response for boosting:
        Inds = union(find_zero_columns(BAE.encoder.coeffs), l)

        if length(Inds) == zdim
            #since all lat. dims. are excluded, the pseudo response is determined by the st. neg. grad.:
            y = standardize(grads[:, l])

            #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
            BAE.encoder.coeffs[:, l] = compL2Boost!(BAE.encoder.coeffs[:, l], X, y, ϵ, M)

        else
            #compute optimal current OLLS-fit of other latent repr. to the st. neg. grad:
            curdata = X * BAE.encoder.coeffs[:, Not(Inds)]
            curtarget = standardize(grads[:, l]) 
            curestimate = inv(curdata'curdata)*(curdata'curtarget) 
             #curestimate = inv(curdata'curdata + Float32.(1.0e-5 * Matrix(I, size(curdata, 2), size(curdata, 2))))*(curdata'curtarget)  #damped version for avoiding invertibility problems

            #compute the pseudo response for the boosting [st. difference of st. neg. grad and optimal OLLS-fit]:
            y = standardize(curtarget .- (curdata * curestimate))

            #apply componentwise boosting to update one component of the β-vector (l-th col. of matrix B):
            BAE.encoder.coeffs[:, l] = compL2Boost!(BAE.encoder.coeffs[:, l], X, y, ϵ, M)
        end
        
    end
        
    return BAE.encoder.coeffs
end


"""
    seq_constr_compL2Boost_jointLoss(batch::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int, iter::Int)

The `seq_constr_compL2Boost_jointLoss` function updates the encoder weights of a boosting autoencoder by performing sequential componentwise boosting with addig a constraint to the response vectors.
This version is similar to `seq_constr_compL2Boost` but computes the weights such that differentiating through the boosting step does not produce a *Mutating arrays is not supported* error.

## Arguments
- `batch::AbstractMatrix{<:AbstractFloat}`: The input batch of data.
- `BAE::Autoencoder`: The boosting autoencoder object.
- `ϵ::Number`: The regularization parameter.
- `zdim::Int`: The number of latent dimensions.
- `M::Int`: The number of boosting iterations.
- `iter::Int`: The current iteration.

## Returns
- `BAE.encoder.coeffs`: The updated weights of the boosting autoencoder's encoder.

"""
function seq_constr_compL2Boost_jointLoss(batch::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, zdim::Int, M::Int, iter::Int)
    
    #compute neg. gradient - vectors w.r.t. latent dimensions:
    grads = get_latdim_grads(batch, BAE) 

    for l = 1:zdim 
        #determine the indices of latent dimensions excluded for determining the pseudo-response for boosting:
        if iter==1
            Inds = find_zero_columns(BAE.encoder.coeffs)
        else
            Inds = vcat(find_zero_columns(BAE.encoder.coeffs), l)
        end

        if length(Inds) == zdim
            #since all lat. dims. are excluded, the pseudo-response is determined by the st. neg. grad.:
            y = standardize(grads[:, l])

            #apply componentwise boosting to update one component of the β-vector(l-th col. of matrix B):
            β = compL2Boost_jointLoss(BAE.encoder.coeffs[:, l], batch', y, ϵ, M) 
            BAE.encoder.coeffs = hcat(BAE.encoder.coeffs[:, 1:l-1], β, BAE.encoder.coeffs[:,l+1:end])

        else
            #compute optimal current OLLS-fit of other latent repr. to the st. neg. grad:
            curdata = batch' * BAE.encoder.coeffs[:, Not(Inds)]
            curtarget = standardize(grads[:, l]) 
            curestimate = inv(curdata'curdata)*(curdata'curtarget) 
             #curestimate = inv(curdata'curdata + Float32.(1.0e-5 * Matrix(I, size(curdata, 2), size(curdata, 2))))*(curdata'curtarget)  #damped version for avoiding invertibility problems

            #compute the pseudo-response for the boosting [st. difference of st. neg. grad and optimal OLLS-fit]:
            y = standardize(curtarget .- (curdata * curestimate))

            #apply componentwise boosting to update one component of the β-vector (l-th col. of matrix B):
            β = compL2Boost_jointLoss(BAE.encoder.coeffs[:, l], batch', y, ϵ, M) 
            BAE.encoder.coeffs = hcat(BAE.encoder.coeffs[:, 1:l-1], β, BAE.encoder.coeffs[:, l+1:end])
        end
    end
        
    return BAE.encoder.coeffs
end