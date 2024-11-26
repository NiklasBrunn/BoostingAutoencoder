#------------------------------
# This file contains utility functions:
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




function get_top_selected_genes(B::AbstractMatrix{T}, genenames::AbstractVector; 
    method::String="Changepoint", 
    n_topGenes::Int=5, 
    data_path::Union{AbstractString, Nothing}=nothing, 
    save_data::Bool=false
    ) where T
    
    abs_B = abs.(B)

    dict = Dict{Int, Vector{String}}()

    df = DataFrame(Dim=1:size(B, 2))

    n_selgenes_vec = []
    n_topgenes_vec = []
    pct_vec_top = []
    pct_vec = []

    p = size(B, 1)

    if method == "Changepoint"

        @info "Finding top genes using Changepoint method ..."

        for l in 1:size(B, 2)

            inds = sortperm(abs_B[:, l]; rev=true)
            weights = abs_B[:, l][inds]
            sort_genenames = genenames[inds]

            nonzero_inds = findall(x->x!=0, weights)
            weights = weights[nonzero_inds]
            sort_genenames =  sort_genenames[nonzero_inds] 

            diffs = [weights[i] - weights[i+1] for i in 1:length(weights)-1]
            topGene_inds = findmax(diffs)[2]     
            dict[l] = sort_genenames[1:topGene_inds]

            n_selgenes = length(findall(x->x!=0, abs_B[:, l]))
            n_topgenes = length(weights[1:topGene_inds])
            pct_topGenes = round((n_topgenes / p) * 100, digits=2)

            pct = round((n_selgenes / p) * 100, digits=2)

            push!(n_selgenes_vec, n_selgenes)
            push!(n_topgenes_vec, n_topgenes)
            push!(pct_vec_top, pct_topGenes)
            push!(pct_vec, pct)
        end

    elseif method == "top"

        @info "Finding top genes using 'top' method ..."

        for l in 1:size(B, 2)
            inds = sortperm(abs_B[:, l]; rev=true)
            weights = abs_B[:, l][inds]

            dict[l] = genenames[inds][1:n_topGenes]

            n_selgenes = length(findall(x->x!=0, abs_B[:, l]))
            n_topgenes = length(weights[1:n_topGenes])
            pct_topGenes = round((n_topgenes / p) * 100, digits=2)

            pct = round((n_selgenes / p) * 100, digits=2)

            push!(n_selgenes_vec, n_selgenes)
            push!(n_topgenes_vec, n_topgenes)
            push!(pct_vec_top, pct_topGenes)
            push!(pct_vec, pct)
        end

    else
        @error "Method not recognized. Please use 'Changepoint' or 'top'."
    end

    df[:, "Nonzero genes"] = n_selgenes_vec
    df[:, "Percentage of nonzero genes"] = pct_vec
    df[:, "Top genes"] = n_topgenes_vec
    df[:, "Percentage of top genes"] = pct_vec_top


    if save_data
        if data_path == nothing
            @error "Please provide a valid path to save the data."
        else
            CSV.write(data_path * "BAE_latentDims_geneSelection_table.csv", df)
        end
    end

    return dict, df
end

function quantile_elements(v::Vector{T}; upper::Number=0.9, lower::Number=0.1) where T<:Number
    q_upper = quantile(v, upper)
    q_lower = quantile(v, lower)
    
    above_q_upper = findall(x->x >= q_upper, v)
    below_q_lower = findall(x->x <= q_lower, v)
    
    return above_q_upper, below_q_lower
end

function find_matching_type(inds::Vector{T}, celltype::AbstractVector) where T<:Number

    types = celltype[inds]
    n_cells = length(inds)

    other_types = setdiff(unique(celltype), types)
    freq = countmap(types)
    for type in other_types
        freq[type] = 0
    end

    sort_inds = sortperm(collect(keys(freq)))
    pcts = collect(values(freq))[sort_inds] ./ n_cells

    most_frequent_type = findmax(freq)[2]
    perc_most_frequent_type = findmax(freq)[1] / n_cells

    return most_frequent_type, perc_most_frequent_type, pcts
end

function find_matching_type_per_BAEdim(Z_BAE::AbstractMatrix{T}, celltype::AbstractVector; 
    upper::Float64=0.9, 
    lower::Float64=0.1, 
    threshold::Float64=0.5,
    save_plot::Bool=false,
    figurespath::Union{Nothing, String}=nothing
    ) where T

    if isnothing(figurespath) && save_plot == true
        @error "Please provide a path to save the plot"
    end

    unique_types = sort(unique(celltype))
    df_filtered = DataFrame(celltype=unique_types)
    df = DataFrame(celltype=unique_types)

    for l in 1:size(Z_BAE, 2)
        above_q_upper, below_q_lower = quantile_elements(Z_BAE[:, l]; upper=upper, lower=lower)
        most_frequent_type_upper, perc_most_frequent_type_upper, pcts_upper= find_matching_type(above_q_upper, celltype)
        df[:, "Dim $(l) pos"] = pcts_upper
        if perc_most_frequent_type_upper >= threshold
            v = zeros(length(unique_types))
            ind = findall(x->x==most_frequent_type_upper, unique_types)
            v[ind] .= perc_most_frequent_type_upper
            df_filtered[:, "Dim $(l) pos"] = v
        end
        most_frequent_type_lower, perc_most_frequent_type_lower, pcts_lower = find_matching_type(below_q_lower, celltype)
        df[:, "Dim $(l) neg"] = pcts_lower
        if perc_most_frequent_type_lower >= threshold
            v = zeros(length(unique_types))
            ind = findall(x->x==most_frequent_type_lower, unique_types)
            v[ind] .= perc_most_frequent_type_lower
            df_filtered[:, "Dim $(l) neg"] = v
        end
    end

    clusters = names(df)[2:end]
    num_clusters = [i for i in 1:length(clusters)]
    percentages = []
    types = []
    for group in names(df)[2:end]
        pct, matched_type = findmax(df[:, group])
        push!(percentages, pct)
        push!(types, df[:, "celltype"][matched_type])
    end

    if save_plot == true

        # Create the bar plot
        #pl=bar(num_clusters, percentages, legend=false, xlabel="BAE Cell Groups", ylabel="Matching Percentage", xticks=(clusters, clusters))
        pl=bar(percentages, legend=false, xlabel="BAE Cell Groups", ylabel="Matching Percentage", 
            xticks=(1:length(percentages), clusters), xrotation=50
        )

        pl=hline!([threshold], label="Threshold", color=:red, linestyle=:dash)

        # Add labels for each bar to show the "Most frequent type"
        for (i, num_clusters) in enumerate(num_clusters)
            annotate!(num_clusters, percentages[i] + 0.02, text(types[i], :black, 8, :center))
        end

        savefig(pl, figurespath * "/BAE_CellGroups_CellTypes_matches_barplot.pdf")
    end

    return df, df_filtered
end

#Optional?? -> NO! (uncertainty_Analysis_corticalMouseData.jl)
function get_nonzero_rows(X::AbstractMatrix{T}) where T
    nonzero_columns = findall(x -> x!=0, vec(any(X .!= 0, dims=2)))
    return nonzero_columns
end

#---VAE utility functions:
function divide_dimensions(x::AbstractArray{T}) where T
    if ndims(x) != 2
        throw(DimensionMismatch("Input array must be 2D"))
    end
    p = size(x, 1)
    mid = div(p, 2)
    first_half = x[1:mid, :]
    second_half = x[mid+1:end, :]
    return first_half, second_half
end

# Reparameterization trick
function reparameterize(mu::AbstractArray{T}, logvar::AbstractArray{T}) where T
    sigma = exp.(0.5f0 .* logvar)
    epsilon = randn(Float32, size(mu))
    return mu .+ sigma .* epsilon
end

# Latent representation (expected)
function get_VAE_latentRepresentation(encoder::Union{Chain, Dense}, X::AbstractArray{T}; sampling::Bool=false) where T
    μ, logvar, z = nothing, nothing, nothing

    if sampling     
        μ, logvar = divide_dimensions(encoder(X))
        z = reparameterize(μ, logvar)
    else
        μ, logvar = divide_dimensions(encoder(X))
    end

    return μ, logvar, z
end

function adjust_pvalues(pvals::AbstractVector{T}; method::Union{String, Nothing}="Bonferroni", alpha::F=0.05) where T where F<:AbstractFloat

    type = eltype(pvals)
    one_num = convert(type, 1.0)
    m = length(pvals)

    if method == "Bonferroni"
        return min.(1, m * pvals)

    elseif method == "Benjamini-Hochberg"
        # Sort p-values and keep track of their original indices
        sorted_p_values, sorted_indices = sort(pvals), sortperm(pvals)

        # Compute adjusted p-values
        adjusted_p_values = Vector{T}(undef, m)
        min_adjusted_p = one_num

        for i in reverse(1:m)
            rank = i
            adjusted_p = sorted_p_values[i] * m / rank
            min_adjusted_p = min(min_adjusted_p, adjusted_p)
            adjusted_p_values[i] = min(min_adjusted_p, one_num)
        end

        # Revert to original order
        adj_pvals = similar(adjusted_p_values)
        for i in 1:m
            adj_pvals[sorted_indices[i]] = min.(1, adjusted_p_values[i])
        end

        return adj_pvals

    elseif method == nothing
        return pvals

    else
        error("Method not recognized")

    end

end

function coefficients_tTest(coeffs::AbstractVector{T}, X::AbstractMatrix{T}, y::AbstractVector{T}; 
    adjusted_pvals::Bool=true, 
    method::Union{String, Nothing}="Bonferroni",
    alpha::F = 0.05
    ) where T where F<:AbstractFloat


    # Comp. Res.:
    Y_pred = X * coeffs
    res = y - Y_pred

    # Estimate the variance of the residuals:
    n, p = size(X) 
    var = sum(res.^2) / (n - p) # (no intercept term)

    # Compute the Variance-Covariance Matrix:
    XtX_inv = inv(X' * X)  # Inverse of X'X
    Var_B = var .* XtX_inv

    # Compute the t-Statistic:
    SE_B = sqrt.(diag(Var_B)) # Standard errors of the coefficients
    t_stats = coeffs ./ SE_B # t-statistics

    # Compute the (two-tailed) p-Values:
    df = n - p # Degrees of freedom (no intercept term)
    p_values = 2 * (1 .- cdf(TDist(df), abs.(t_stats)))

    # Adjust pvalues:
    if adjusted_pvals
        adj_pvalues = adjust_pvalues(p_values; method="Bonferroni", alpha=0.05) # Bonferroni #Benjamini-Hochberg
        return adj_pvalues
    else
        return p_values
    end

end

function predict_celllabels(Z_test::AbstractMatrix{T}, Z_train::AbstractMatrix{T}, train_labels::AbstractVector{Any}; 
    k::Int=10
    ) where T

    if size(Z_test, 2) != size(Z_train, 2)
        throw(DimensionMismatch("The feature dimensions of the test and training data do not match"))
    end

    if k > size(Z_train, 1)
        throw(DimensionMismatch("The number of nearest neighbors cannot exceed the number of training samples"))
    end

    if length(train_labels) != size(Z_train, 1)
        throw(DimensionMismatch("The number of training labels does not match the number of training samples"))
    end

    n = size(Z_test, 1);

    dist_metric = Euclidean(); #Distance metric to use for computing the nearest neighbours
    dist_matrix = pairwise(dist_metric, Z_test, Z_train, dims=1) ;
    nearest_neighbors = [partialsortperm(dist_matrix[i, :], 1:k) for i in 1:size(Z_test, 1)];

    pred_labels = []
    for i in 1:n
        types_of_closest_cells = train_labels[nearest_neighbors[i]]
        freq_map = countmap(types_of_closest_cells)
        most_frequent = argmax(freq_map)
        push!(pred_labels, most_frequent)
    end

    return pred_labels
end