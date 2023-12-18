#------------------------------
# This file contains utility functions:
#------------------------------

function find_zero_columns(X::AbstractMatrix{<:Number}) 
    v = vec(sum(abs.(X), dims=1))
    zero_cols = findall(x->x==0, v)
    return zero_cols
end

function split_traintestdata(X::AbstractMatrix, Y::AbstractMatrix; dataseed::Int = 777, k::Int = 1325)

    n = size(X, 1)

    Random.seed!(dataseed) 
    randindex = Random.randperm(n);

    randindex_train = sort(randindex[1:k]) 
    randindex_test = sort(randindex[k+1:end])

    X_train_log1 = X[randindex_train, :]
    X_test_log1 = X[randindex_test, :]
    X_train_st = standardize(X_train_log1)
    X_test_st = standardize(X_test_log1)

    Y_train = Y[randindex_train, :]; 
    Y_test = Y[randindex_test, :]; 

    return X_train_st, X_test_st, X_train_log1, X_test_log1, Y_train, Y_test, randindex_train, randindex_test
end  

function create_sorted_numlabels_and_datamat(X, labels)
    Z = zeros(size(X))
    z = Int32.(zeros(length(labels)))

    count = 1
    no_cells = 0

    c = sort(unique(labels))
    for i in c
        inds = findall(x->x==i, labels)
        z[inds] .= count 
        len = length(inds)
        Z[no_cells+1:no_cells+len,:] = X[inds, :]
        no_cells+=len
        count+=1
    end
    return Z, z
end

function onehotcelltypes(y::AbstractVector{Int32})
    Y_onehot = indicatormat(vec(y))'
    return Y_onehot
end  

function prcomps(mat; standardizeinput = false)
    if standardizeinput
        mat = standardize(mat)
    end
    u,s,v = svd(mat)
    prcomps = u * Diagonal(s)
    return prcomps
end

function generate_umap(X, plotseed; n_neighbors::Int=30, min_dist::Float64=0.4)
    Random.seed!(plotseed)  
    embedding = umap(X', n_neighbors=n_neighbors, min_dist=min_dist)'
    return embedding
end

function replace_nan_columns_with_zeros(mat::Matrix{T}) where T
    for j in 1:size(mat, 2)  # Iterate over columns
        if all(isnan.(mat[:, j]))  # Check if the entire column is NaN
            mat[:, j] .= 0  # Replace with 0s
        end
    end
    return mat
end