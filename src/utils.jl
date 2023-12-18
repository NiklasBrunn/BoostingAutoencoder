#------------------------------
# This file contains utility functions:
#------------------------------

"""
    find_zero_columns(X::AbstractMatrix{<:Number})

Find the indices of columns in matrix `X` that contain only zeros.

# Arguments
- `X`: An abstract matrix of numbers.

# Returns
An array of indices representing the columns in `X` that contain only zeros.
"""
function find_zero_columns(X::AbstractMatrix{<:Number}) 
    v = vec(sum(abs.(X), dims=1))
    zero_cols = findall(x->x==0, v)
    return zero_cols
end


"""
    split_traintestdata(X::AbstractMatrix, Y::AbstractMatrix; dataseed::Int = 777, k::Int = 1325)

Split the input data `X` and `Y` into training and testing sets.

# Arguments
- `X::AbstractMatrix`: The input feature matrix.
- `Y::AbstractMatrix`: The input target matrix.
- `dataseed::Int`: (optional) The seed for random number generation. Default is 777.
- `k::Int`: (optional) The number of samples to be used for training. Default is 1325.

# Returns
- `X_train_st::AbstractMatrix`: The standardized training feature matrix.
- `X_test_st::AbstractMatrix`: The standardized testing feature matrix.
- `X_train::AbstractMatrix`: The training feature matrix before standardization.
- `X_test::AbstractMatrix`: The testing feature matrix before standardization.
- `Y_train::AbstractMatrix`: The training target matrix.
- `Y_test::AbstractMatrix`: The testing target matrix.
- `randindex_train::Vector{Int}`: The indices of the samples used for training.
- `randindex_test::Vector{Int}`: The indices of the samples used for testing.
"""
function split_traintestdata(X::AbstractMatrix, Y::AbstractMatrix; dataseed::Int = 777, k::Int = 1325)

    n = size(X, 1)

    Random.seed!(dataseed) 
    randindex = Random.randperm(n);

    randindex_train = sort(randindex[1:k]) 
    randindex_test = sort(randindex[k+1:end])

    X_train = X[randindex_train, :]
    X_test = X[randindex_test, :]
    X_train_st = standardize(X_train)
    X_test_st = standardize(X_test)

    Y_train = Y[randindex_train, :]; 
    Y_test = Y[randindex_test, :]; 

    return X_train_st, X_test_st, X_train, X_test, Y_train, Y_test, randindex_train, randindex_test
end  


"""
    create_sorted_numlabels_and_datamat(X, labels)

Create a sorted numerical label array and a data matrix based on the input data `X` and labels `labels`.

# Arguments
- `X`: The input data matrix.
- `labels`: The labels corresponding to each data point in `X`.

# Returns
- `M`: The data matrix with rows rearranged based on the sorted labels.
- `v`: The sorted numerical labels array.
"""
function create_sorted_numlabels_and_datamat(X, labels)
    M = zeros(size(X))
    v = Int32.(zeros(length(labels)))

    count = 1
    no_cells = 0

    c = sort(unique(labels))
    for i in c
        inds = findall(x->x==i, labels)
        v[inds] .= count 
        len = length(inds)
        M[no_cells+1:no_cells+len,:] = X[inds, :]
        no_cells+=len
        count+=1
    end
    return M, v
end


"""
    onehotcelltypes(y::AbstractVector{Int32})

Converts a vector of integer labels into a one-hot encoded matrix.

# Arguments
- `y`: A vector of integer labels.

# Returns
- `Y`: A matrix where each row represents a one-hot encoded label.
"""
function onehotcelltypes(y::AbstractVector{Int32})
    Y = indicatormat(vec(y))'
    return Y
end  


"""
    prcomps(mat; standardizeinput = false)

Compute the principal components of a matrix.

# Arguments
- `mat`: The input matrix.
- `standardizeinput`: A boolean indicating whether to standardize the input matrix. Default is `false`.

# Returns
- `prcomps`: The principal components of the matrix.
"""
function prcomps(mat; standardizeinput = false)
    if standardizeinput
        mat = standardize(mat)
    end
    u,s,v = svd(mat)
    prcomps = u * Diagonal(s)
    return prcomps
end


"""
    generate_umap(X, plotseed; n_neighbors::Int=30, min_dist::Float64=0.4)

Generate a 2D UMAP embedding for the given data `X`.

# Arguments
- `X`: The input data matrix of size `(n_samples, n_features)`.
- `plotseed`: The seed value for random number generation during plotting.
- `n_neighbors`: The number of nearest neighbors to consider during UMAP embedding. Default is 30.
- `min_dist`: The minimum distance between points in the UMAP embedding. Default is 0.4.

# Returns
- `embedding`: The 2D UMAP embedding of the input data `X`.
"""
function generate_umap(X, plotseed; n_neighbors::Int=30, min_dist::Float64=0.4)
    Random.seed!(plotseed)  
    embedding = umap(X', n_neighbors=n_neighbors, min_dist=min_dist)'
    return embedding
end