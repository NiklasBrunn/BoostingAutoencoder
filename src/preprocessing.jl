#------------------------------
# This file contains preprocessing functions:
#------------------------------

"""
    standardize(X::AbstractArray; corrected_std::Bool=true, dims::Int=1)

Standardize the input array `X` along the first dimension of the array by subtracting 
the 2nd dimension-wise mean and dividing by the 2nd dimension-wise standard deviation. 
Optionally, it can use the corrected sample standard deviation when `corrected_std` 
is set to `true`.

# Arguments
- `X::AbstractArray`: The input array to be standardized.
- `corrected_std::Bool=true`: (Optional) A flag indicating whether to use the corrected
  sample standard deviation (default is `true`).

# Returns
A standardized array (along the first dimesnion) with the same dimensions as the input array.

# Example
```julia
data = [1.0 2.0; 3.0 4.0; 5.0 6.0]
standardized_data = standardize(data)
3×2 Matrix{Float64}:
 -1.0  -1.0
  0.0   0.0
  1.0   1.0
"""
function standardize(X::AbstractArray; corrected_std::Bool=true, dims::Int=1)
    return (X .- mean(X, dims=dims)) ./ std(X, corrected=corrected_std, dims=dims)
end


"""
    log1transform(X::AbstractArray)

Perform a log transformation with a pseudo-count of 1 on the elements of the input array `X`. 
The transformation applies a natural logarithm to each element after adding 1 to it. This is 
commonly used to handle data with small or zero values.

# Arguments
- `X::AbstractArray`: The input array to be log-transformed.

# Returns
An array with the same dimensions as the input array, where each element is the
natural logarithm of the corresponding element in `X` plus 1.

# Example
```julia
data = [1.0 2.0; 3.0 4.0; 5.0 100.0]
transformed_data = log1transform(data)
3×2 Matrix{Float64}:
 0.693147  1.09861
 1.38629   1.60944
 1.79176   4.61512
"""
function log1transform(X::AbstractArray)
    return log.(X .+ 1)
end


#-----------------
#CorticalMouseData:
#-----------------
"""
    downloadcountsandload(path)

Downloads a file from a specified URL and loads the count data from the downloaded file.

# Arguments
- `path::String`: The path where the downloaded file will be saved.

# Returns
- `countdata::Array{Any,2}`: The count data loaded from the downloaded file.
"""
function downloadcountsandload(path)
    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE71585&format=file&file=GSE71585%5FRefSeq%5Fcounts%2Ecsv%2Egz"
    outfile = path * "gse715858_counts.csv.gz"
    if !isfile(outfile)
        Base.download(url, outfile)
    end
    fh = GZip.open(outfile)
    countdata = readdlm(fh, ',')
    #rm(gse715858_counts.csv.gz)
    return countdata
end


"""
    phenodata(path)

Reads an Excel file containing information about the primary cell types located at `path` and returns a DataFrame containing the data.

# Arguments
- `path`: A string representing the path to the Excel file.

# Returns
A DataFrame containing the data from the Excel file.
"""
function phenodata(path)
    
    df = DataFrame(XLSX.readtable(path * "nn.4216-S5.xlsx", "Supplementary_Table_3"))
    rename!(df, [Symbol(replace(string(i), " " => "")) for i = names(df)])
    return df
end


"""
    expressiondata(path; countcutoff=10, samplecutoff=20)

This function preprocesses expression data by performing the following steps:
1. Downloads and loads count data from a specific URL.
2. Converts the data to a matrix of type Float64.
3. Extracts gene names and sample names from the data.
4. Modifies sample names by replacing "Nkx2-1" with "Nkx2.1".
5. Retrieves sample information about the primary cell types using `phenodata(path)`.
6. Filters genes based on count cutoff and sample cutoff.
7. Returns the filtered expression data, gene names, and sample information.

# Arguments
- `path`: The path where the sample information is located.
- `countcutoff`: The minimum count threshold for a gene to be considered.
- `samplecutoff`: The minimum number of samples a gene must have counts above `countcutoff` to be considered.

# Returns
- `xout`: The filtered expression data matrix.
- `genenames`: The gene names corresponding to the filtered expression data.
- `sampleinfo`: The sample information retrieved from `phenodata(path)`.
"""
function expressiondata(path; countcutoff=10, samplecutoff=20)
    data = downloadcountsandload(path)
    x = Float64.(data[2:end,2:end])
    genenames = string.(data[2:end,1])
    samplenames = string.(data[1,2:end])
    samplenames = [replace(i,"Nkx2-1"=>"Nkx2.1") for i = samplenames]
    sampleinfo = phenodata(path)
    annotatedcells = [findall(x->x==i,samplenames)[1] for i = sampleinfo.GEOSampleTitle]
    genebool = [sum(x[i,:] .> countcutoff)>samplecutoff for i = 1:size(x)[1]]
    xout = x[genebool,annotatedcells]
    genenames = genenames[genebool]
    return xout,genenames,sampleinfo
end


"""
    estimatesizefactorsformatrix(mat; locfunc=median)

Estimates size factors for a matrix of log-transformed counts. The function is a julia implementation of the function `estimateSizeFactors` from the R package DESeq2. 

# Arguments
- `mat`: A matrix of log-transformed counts.
- `locfunc`: The function used to calculate the location parameter for each sample. Default is `median`.

# Returns
- `size_factors`: An array of size factors.
"""
function estimatesizefactorsformatrix(mat; locfunc=median)
    logcounts = log.(mat)
    loggeomeans = vec(mean(logcounts, dims=2))
    finiteloggeomeans = isfinite.(loggeomeans)
    loggeomeans = loggeomeans[finiteloggeomeans]
    logcounts = logcounts[finiteloggeomeans,:]
    nsamples = size(logcounts, 2)
    size_factors = fill(0.0, nsamples)
    for i = 1:nsamples
        size_factors[i] = exp(locfunc(logcounts[:,i] .- loggeomeans))
    end
    return size_factors
end


"""
    normalizecountdata(mat)

Normalize the count data matrix `mat` by dividing each element by the corresponding size factor. Normalization is done using the DESeq2 normalization.

# Arguments
- `mat`: A count data matrix.

# Returns
- The normalized count data matrix.
"""
function normalizecountdata(mat)
    sizefactors = estimatesizefactorsformatrix(mat)
    return mat ./ sizefactors'
end