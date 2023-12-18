#------------------------------
# This file contains functions for generating simulated scRNA-seq-like datasets:
#------------------------------

"""
    addstages!(X, stageno; stagen=1, stagep=2, overlap=1, blockprob=1.0)

Add stages to a matrix `X` by modifying its values in-place.

# Arguments
- `X::Matrix`: The matrix to which stages will be added.
- `stageno::Int`: The number of stages to add.
- `stagen::Int=1`: The number of rows in each stage.
- `stagep::Int=2`: The number of columns in each stage.
- `overlap::Int=1`: The number of overlapping columns between stages.
- `blockprob::Float64=1.0`: The probability of a block being filled with ones.

# Returns
- `X::Matrix`: The modified matrix `X` with added stages.
"""
function addstages!(X, stageno; stagen=1, stagep=2, overlap=1, blockprob=1.0)
    curp = 1
    curn = 1
    for i = 1:stageno
        if blockprob < 1.0
            curblock = 1.0*(rand(stagen,stagep) .<= blockprob)
            X[curn:(curn+stagen-1),curp:(curp+stagep-1)] = curblock
        else
            X[curn:(curn+stagen-1),curp:(curp+stagep-1)] .= 1.0
        end
        curp += stagep - overlap
        curn += stagen
    end
    X
end


"""
    simulate_10StagesScRNAseq(dataseed = 1;
        rescale_val = 1.5,
        n = 1000, 
        num_genes = 50, 
        stageno = 10, 
        stagep = Int(50 / 10), 
        stagen = Int(1000 / 10), 
        stageoverlap = 2, 
        blockprob = 0.6, 
        noiseprob = 0.1)

Simulates a single-cell RNA sequencing (scRNA-seq) dataset consisting of cells in ten different stages of a developmental process.

# Arguments
- `dataseed::Int`: Seed for random number generation.
- `rescale_val::Float64`: Value used for rescaling the noise genes in the simulated data.
- `n::Int`: Number of cells in the dataset.
- `num_genes::Int`: Number of genes in the dataset.
- `stageno::Int`: Number of stages in the dataset.
- `stagep::Int`: Number of cells per stage.
- `stagen::Int`: Number of cells per stage group.
- `stageoverlap::Int`: Number of overlapping cells between stages.
- `blockprob::Float64`: Probability of a gene being blocked in a stage.
- `noiseprob::Float64`: Probability of a gene being noisy in a cell.

# Returns
- `X::Matrix{Float64}`: Simulated scRNA-seq dataset with standardized counts and re-scaled noise genes.
- `X_dicho::Matrix{Float64}`: Simulated scRNA-seq dataset with binary values.
"""
function simulate_10StagesScRNAseq(dataseed = 1;
    rescale_val = 1.5,
    n = 1000, 
    num_genes = 50, 
    stageno = 10, 
    stagep = Int(50 / 10), 
    stagen = Int(1000 / 10), 
    stageoverlap = 2, 
    blockprob = 0.6, 
    noiseprob = 0.1, 
    )

    Random.seed!(dataseed)
    X = 1.0*(rand(n, num_genes) .> (1 - noiseprob)) 
    X = addstages!(X, stageno, stagen=stagen, stagep=stagep, overlap=stageoverlap, blockprob=blockprob) 


    X_dicho = copy(X)
    X = standardize(X) 

    if rescale_val > 0
        X[:, 33:end] = X[:, 33:end] ./ rescale_val 
    end
     
    return (X, X_dicho)

end 


"""
    simulate_3cellgroups3stagesScRNAseq(dataseed; n=[310, 298, 306], p=80, p1=0.6, p2=0.1, rescale_factor=1.5)

Simulates a single-cell RNA sequencing dataset consisting of three count matrices modelling a developmental process of three cell groups across three time points.

# Arguments
- `dataseed`: The seed for the random number generator.
- `n`: An array specifying the number of cells at each time point. Default is `[310, 298, 306]`.
- `p`: The number of genes. Default is `80`.
- `p1`: The probability of getting the value one for a highly expressed gene. Default is `0.6`.
- `p2`: The probability of getting the value one for a lowly expressed gene. Default is `0.1`.
- `rescale_factor`: The rescaling factor for noise genes. Default is `1.5`.

# Returns
A tuple `(L_dicho, L, X_dicho, X)` containing the following matrices:
- `L_dicho`: An array containing the three binary count matrices containing the simulated gene expression data for cell groups at each time point.
- `L`: An array containing the three standardized count matrices containing the simulated gene expression data for cell groups at each time point with re-scaled noise genes.
- `X_dicho`: A matrix containing the simulated binary gene expression data for all cell groups from each timepoint.
- `X`: A matrix containing the simulated standardized gene expression data for all cell groups from each timepoint with re-scaled noise genes.
"""
function simulate_3cellgroups3stagesScRNAseq(dataseed; 
    n=[310, 298, 306], p=80, 
    p1=0.6, p2=0.1, rescale_factor=1.5)

    Random.seed!(dataseed)
    d1 = Binomial(1, p1) 
    d2 = Binomial(1, p2)

    X1_dicho=zeros(n[1], p)
    for i in 1:n[1]
        for j in 1:p
            if (i <= Int64(round(n[1]/3))) && (1 <= j <= 8)
                X1_dicho[i, j]=Float64(rand(d1, 1)[1])
            elseif (Int64(round(n[1]/3)) < i <= Int64(round(2*n[1]/3))) && (21 <= j <= 28)
                X1_dicho[i, j]=Float64(rand(d1, 1)[1])
            elseif (Int64(round(2*n[1]/3)) < i <= n[1]) && (41 <= j <= 48)
                X1_dicho[i ,j]=Float64(rand(d1, 1)[1])
            else
                X1_dicho[i, j]=Float64(rand(d2, 1)[1])
            end
        end
    end

    X2_dicho=zeros(n[2], p)
    for i in 1:n[2]
        for j in 1:p
            if (i <= Int64(round(n[2]/3))) && (6 <= j <= 13)
                X2_dicho[i, j]=Float64(rand(d1, 1)[1])
            elseif (Int64(round(n[2]/3)) < i <= Int64(round(2*n[2]/3))) && (26 <= j <= 33)
                X2_dicho[i, j]=Float64(rand(d1, 1)[1])
            elseif (Int64(round(2*n[2]/3)) < i <= n[2]) && (46 <= j <= 53)
                X2_dicho[i, j]=Float64(rand(d1, 1)[1])
            else
                X2_dicho[i, j]=Float64(rand(d2, 1)[1])
            end
        end
    end

    X3_dicho=zeros(n[3], p)
    for i in 1:n[3]
        for j in 1:p
            if (i <= Int64(round(n[3]/3))) && (11 <= j <= 18)
                X3_dicho[i, j]=Float64(rand(d1, 1)[1])
            elseif (Int64(round(n[3]/3)) < i <= Int64(round(2*n[3]/3))) && (31 <= j <= 38)
                X3_dicho[i, j]=Float64(rand(d1, 1)[1])
            elseif (Int64(round(2*n[3]/3)) < i <= n[3]) && (51 <= j <= 58)
                X3_dicho[i, j]=Float64(rand(d1, 1)[1])
            else
                X3_dicho[i, j]=Float64(rand(d2, 1)[1])
            end
        end
    end

    #collect timepoint matrices in a vcat matrix and additionally generate a standardized version:
    X_dicho = vcat(X1_dicho, X2_dicho, X3_dicho)

    X = standardize(X_dicho)
    for j in 1:p
        if 19 < j < 21  
            X[:, j] = X[:, j]./rescale_factor
        elseif 39 < j < 41
            X[:, j] = X[:, j]./rescale_factor
        elseif 59 < j <= p
            X[:, j] = X[:, j]./rescale_factor
        end
    end

    #standardized data per timepoint:
    X1=standardize(X1_dicho)
    for j in 1:p
        if 8 < j < 21  
            X1[:, j] = X1[:, j]./rescale_factor
        elseif 28 < j < 41
            X1[:, j] = X1[:, j]./rescale_factor
        elseif 48 < j <= p
            X1[:, j] = X1[:, j]./rescale_factor
        end
    end

    X2=standardize(X2_dicho)
    for j in 1:p
        if 1 <= j < 6   
            X2[:, j] = X2[:, j]./rescale_factor
        elseif 13 < j < 26 
            X2[:, j] = X2[:, j]./rescale_factor
        elseif 33 < j < 46
            X2[:, j] = X2[:, j]./rescale_factor
        elseif 53 < j <= p
            X2[:, j] = X2[:, j]./rescale_factor
        end
    end

    X3=standardize(X3_dicho)
    for j in 1:p
        if 1 <= j < 11   
            X3[:, j] = X3[:, j]./rescale_factor
        elseif 18 < j < 31 
            X3[:, j] = X3[:, j]./rescale_factor
        elseif 38 < j < 51
            X3[:, j] = X3[:, j]./rescale_factor
        elseif 58 < j <= p
            X3[:, j] = X3[:, j]./rescale_factor
        end
    end

    L_dicho = [X1_dicho, X2_dicho, X3_dicho]
    L = [X1, X2, X3]

    return (L_dicho, L, X_dicho, X)
end