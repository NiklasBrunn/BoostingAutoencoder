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

function prcomps_modified(X::AbstractMatrix; components::Int=10, num_entries::Int=3, standardizeinput::Bool=true)
    #standardize the input matrix if indicated
    if standardizeinput
        mat = standardize(mat)
    end

    # Compute the Singular Value Decomposition (SVD) of X
    U, S, Vt = svd(X)
    
    # The principal components are given by X * V
    W = Vt'
    
    # Create a modified W matrix, keeping only the top num_entries absolute values per column
    W_modified = zeros(size(W))
    for j = 1:size(W, 2)
        # Find the indices of the top k absolute values in column j
        sorted_indices = sortperm(abs.(W[:, j]), rev=true)
        top_k_indices = sorted_indices[1:num_entries]
        
        # Set the entries at these indices in the modified W to their original values
        W_modified[top_k_indices, j] = W[top_k_indices, j]
    end

    prcomps = X * W_modified[:, 1:components]
    
    return prcomps, W_modified[:, 1:components]
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