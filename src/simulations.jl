function addstages!(X,stageno;stagen=1,stagep=2,overlap=1,blockprob=1.0)
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
Function for generating the cell-dataseet which is a n x num_genes - matrix containing 0/1 - values
depending on the block-and noise probability and the position of the value in the matrix.
The matrix gets standardized over the features (num_genes).

...

v is a parameter for changing the variance of the last 13 (noise) features for the standerdization.
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
    X = 1.0*(rand(n, num_genes) .> (1-noiseprob)) 
    X = addstages!(X,stageno,stagen=stagen,stagep=stagep,overlap=stageoverlap,blockprob=blockprob) 


    X_dicho = copy(X)
    X = standardize(X) 

    if rescale_val > 0
        X[:, 33:end] = X[:, 33:end] ./ rescale_val 
    end
     
    return (X, X_dicho)

end 


"""
Generating 3 sparse matrices containing only zeros and ones depending on specific areas in each matrix, 
with probabilities p1 and p2. Also we generate standardized versions of each timepoint matrix 
and versions of the standardized matrices where we inversly scale the noise features with the parameter v.
Output is the complete dichotomized matrix X_dicho, the complete standardized version X, for each timepoint 
from i=1,...,3 the dichotomized matrices Xi_dicho and standardized matrices Xi.
Further we collect all the standardized matrices for each timepoint in one matrix Xst.
Dataseed is a parameter to controle under which seed the data is generated (for reproducibility).
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

    #collect timepointmatrices in a vcat matrix and additionally generate a standardized version:
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