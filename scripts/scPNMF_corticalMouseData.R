# Load necessary libraries
library(umap)
library(ggplot2)
library(RColorBrewer)
library(reshape2)

# Get the current script's directory
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
parent_dir <- dirname(script_dir)
datapath <- file.path(parent_dir, "data/corticalMouseData")
figurespath <- file.path(parent_dir, "figures/corticalMouseData_scPNMF")
# Create the folder if it doesn't exist
if (!dir.exists(figurespath)) {
  dir.create(figurespath, recursive = TRUE)
}


X_log1 <- as.matrix(read.table(paste0(datapath, "/corticalMouseDataMat_HVGs_log1.txt"), header = FALSE, sep = "\t"))
data <- t(X_log1)
rm(X_log1)
gc()

genenames <- scan(paste0(datapath, "/genenames_HVGs.txt"), what = "", sep = "\n")
celltype <- scan(paste0(datapath, "/celltype.txt"), what = "", sep = "\n")

zdim <- 10
top_n <- 5

# Add row names
rownames(data) <- as.character(genenames)

# Add column names
n_rows <- ncol(data)
cellnames <- paste0("Cell", 1:n_rows)
colnames(data) <- as.character(cellnames)

Seeds <- 1:30

sel_genenames <- c()
pct_zeroels <- c()


#scPNMF-loop:
for (seed in Seeds) {
  
  print(paste0("Seed ", seed, ":"))
  
  #Run scPNMF
  res_pnmf <- scPNMF::PNMFfun(X = data, K = zdim, method="EucDist", verboseN = TRUE, seed=seed)
  
  W <- res_pnmf$Weight
  S <- res_pnmf$Score
  
  # Assuming `mat` is your matrix
  nonzero_rows <- which(apply(W, 1, function(row) any(row != 0)))
  sel_genenames <- c(sel_genenames, genenames[nonzero_rows])
  
  # Compute the fraction of nonzero elements in W:
  num_zeros <- sum(W == 0)
  total_elements <- length(W)
  fraction_zeros <- num_zeros / total_elements
  pct_zeroels <- c(pct_zeroels, fraction_zeros)
  
  
  
  # Compute Pearson correlation matrix between columns of S
  corr_matrix <- abs(cor(S, method = "pearson"))
  
  # Convert the correlation matrix to long format for ggplot2
  melted_corr <- melt(corr_matrix)
  
  # Create a heatmap using ggplot2
  heatmap_plot <- ggplot(data = melted_corr, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "black", high = "red", 
                         mid = "white", limit = c(0, 1), midpoint = 0, space = "Lab", 
                         name="Pearson\nCorrelation") +
    theme_minimal() + 
    theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
    coord_fixed()
  
  # Save the heatmap to a file
  file_path <- file.path(figurespath, paste0("seed_", paste0(seed, "correlation_heatmap_R.pdf")))
  ggsave(file_path, plot = heatmap_plot, width = 8, height = 6)
  
  # Display the plot (optional)
  print(heatmap_plot)
  
  # The heatmap is saved to the specified file path
  
  
  
  # Assume that S is the Score matrix from PNMF and celltype is a vector of labels
  
  # Step 1: Perform UMAP dimensionality reduction on S matrix
  umap_result <- umap(S)
  
  # Step 2: Extract the UMAP coordinates (2D)
  umap_coords <- as.data.frame(umap_result$layout)
  
  # Step 3: Add celltype as a column to the UMAP coordinates
  umap_coords$celltype <- celltype
  
  # Step 4: Plot the UMAP result, coloring points by celltype
  colors <- brewer.pal(n = 12, name = "Set3")  # Extendable up to 12 colors
  colors <- colorRampPalette(brewer.pal(9, "Set1"))(length(unique(celltype)))
  
  umap_plot <- ggplot(umap_coords, aes(x = V1, y = V2, color = celltype)) +
    geom_point(size = 1.5) +
    labs(title = "UMAP of scPNMF Scores", x = "UMAP1", y = "UMAP2") +
    theme_minimal() +
    scale_color_manual(values = colors) +
    guides(color = guide_legend(override.aes = list(size = 5, shape = 16)))
    #scale_color_discrete(name = "Cell Type")
  
  filepath <- file.path(figurespath, paste0(paste0("scPNMF_umap_celltype_seed", seed), ".pdf"))
  ggsave(filename = filepath, plot = umap_plot, width = 8, height = 6)
  
  
  
  #---Create UMAP plots colored by latent dim activations:
  genes_df <- data.frame(matrix(ncol=0, nrow=top_n))
  figurespath_sub <- paste0(figurespath, "/dim")
  
  for (l in 1:zdim){
    # Assuming you have the UMAP coordinates and a vector of continuous values
    # Let's assume umap_coords is your data and continuous_vector is your vector of continuous values
    figurespath_sub_dim <- paste0(figurespath_sub, l)
    if (!dir.exists(figurespath_sub_dim)) {
      dir.create(figurespath_sub_dim, recursive = TRUE)
    }
    
    # Add the continuous values to your UMAP data
    umap_coords$continuous_value <- S[, l]
    
    # Plot the UMAP result using a continuous color scale
    pl <- ggplot(umap_coords, aes(x = V1, y = V2, color = continuous_value)) +
      geom_point(size = 1.5) +
      labs(title = "UMAP Plot Colored by Continuous Values", x = "UMAP1", y = "UMAP2") +
      theme_minimal() +
      scale_color_gradient(low = "black", high = "red")  # Use a gradient from blue to red
    
    filepath <- file.path(figurespath_sub_dim, paste0(paste0("scPNMF_umap_seed", seed), ".pdf"))
    ggsave(filename = filepath, plot = pl, width = 8, height = 6)
    
    
    
      
    # Step 1: Find indices of nonzero elements
    nonzero_indices <- which(W[, l] != 0)
      
    # Step 2: Extract the absolute values of the nonzero elements
    nonzero_values <- abs(W[nonzero_indices, l])
      
    # Step 3: Use `order()` to get the permutation indices for sorting in descending order
    perm_indices <- order(nonzero_values, decreasing = TRUE)
      
    # Step 4: Use these indices to get the sorted indices of nonzero elements
    sorted_nonzero_indices <- nonzero_indices[perm_indices]
    sorted_nonzero_values <- nonzero_values[perm_indices]
    
    genes_df[[paste0("Dim_", l)]] <- genenames[sorted_nonzero_indices][1:top_n]

    filepath <- file.path(figurespath, paste0(paste0("selGenes_df_seed", seed), ".csv"))
    write.csv(genes_df, file = filepath, row.names = FALSE)
    
    
    
    
    for (i in 1:3){
      pl <- ggplot(umap_coords, aes(x = V1, y = V2, color = data[sorted_nonzero_indices[i],])) +
        geom_point(size = 1.5) +
        labs(title = genenames[sorted_nonzero_indices][i], x = "UMAP1", y = "UMAP2") +
        theme_minimal() +
        scale_color_gradient(low = "black", high = "red")  # Use a gradient from blue to red
      
      filepath <- file.path(figurespath_sub_dim, paste0(paste0(paste0("topGene_", i), "_scPNMF_umap_seed", seed), ".pdf"))
      ggsave(filename = filepath, plot = pl, width = 8, height = 6)
    }
    
  }
  
}


# Step 1: Create a table of element counts
element_counts <- table(sel_genenames)

as.vector(element_counts)
names(element_counts)

# Step 2: Create a table of the frequency of these counts
count_frequencies <- table(element_counts)

file_path <- file.path(figurespath, "scPNMF_gene_selection_stability_histogram.pdf")
pdf(file_path, width = 8, height = 6)

# Create the bar plot
barplot(count_frequencies, 
        main = "Frequency of gene selections", 
        xlab = "Number of gene selections across runs", 
        ylab = "Number of genes", 
        col = "blue")

# Close the PDF device
dev.off()

df <- data.frame(Genes=names(element_counts), Counts=as.vector(element_counts), Pct=as.vector(element_counts)/length(Seeds))
df <- df[order(-df$Counts), ]
csv_file_path <- paste0(datapath, "/scPNMF_gene_selection_stability.csv")
write.csv(df, file = csv_file_path, row.names = FALSE)



#---Mean std conf. interval for the pct of zero elements across runs: 
# Calculate mean and standard deviation
mean_pct <- mean(pct_zeroels)
std_dev_pct <- sd(pct_zeroels)

# Number of samples
n <- length(pct_zeroels)

# Critical value for 95% confidence interval (assuming normal distribution)
critical_value <- qt(0.975, df = n - 1)  # 0.975 for a two-tailed 95% CI

# Margin of error
margin_of_error <- critical_value * (std_dev_pct / sqrt(n))

# Compute the confidence interval
ci_lower <- mean_pct - margin_of_error
ci_upper <- mean_pct + margin_of_error

# Print results
cat("Mean:", mean_pct, "\n")
cat("Standard Deviation:", std_dev_pct, "\n")
cat("95% Confidence Interval:", "[", ci_lower, ",", ci_upper, "]\n")


