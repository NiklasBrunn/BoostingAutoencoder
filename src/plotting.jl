#------------------------------
# This file contains the functions for generating the plots of the results: 
#------------------------------

"""
    vegaheatmap(Z::AbstractMatrix; 
        path::String=joinpath(@__DIR__, "../") * "heatmap.pdf", 
        Title::String=" ",
        xlabel::String="Latent Dimension", 
        ylabel::String="Observation",
        legend_title::String="value",
        color_field::String="value",
        scheme::String="blueorange",
        sortx::String="ascending",
        sorty::String="descending",
        Width::Int=400, 
        Height::Int=400,
        save_plot::Bool=false,
        set_domain_mid::Bool=false,
        axis_labelFontSize::AbstractFloat=10.0,
        axis_titleFontSize::AbstractFloat=10.0,
        legend_labelFontSize::AbstractFloat=10.0,
        legend_titleFontSize::AbstractFloat=10.0,
        legend_symbolSize::AbstractFloat=180.0,
        title_fontSize::AbstractFloat=15.0
        )

This function generates a heatmap plot using the Vega-Lite library based on the input matrix `Z`.

## Arguments
- `Z::AbstractMatrix`: The input matrix for generating the heatmap plot.

## Optional Arguments
- `path::String`: The file path to save the generated plot. Default is `joinpath(@__DIR__, "../") * "heatmap.pdf"`.
- `Title::String`: The title of the plot. Default is an empty string.
- `xlabel::String`: The label for the x-axis. Default is "Latent Dimension".
- `ylabel::String`: The label for the y-axis. Default is "Observation".
- `legend_title::String`: The title for the legend. Default is "value".
- `color_field::String`: The field to use for coloring the heatmap. Default is "value".
- `scheme::String`: The color scheme to use for the heatmap. Default is "blueorange".
- `sortx::String`: The sorting order for the x-axis. Default is "ascending".
- `sorty::String`: The sorting order for the y-axis. Default is "descending".
- `Width::Int`: The width of the plot in pixels. Default is 400.
- `Height::Int`: The height of the plot in pixels. Default is 400.
- `save_plot::Bool`: Whether to save the plot. Default is false.
- `set_domain_mid::Bool`: Whether to set the domain midpoint for the color scale. Default is false.
- `axis_labelFontSize::AbstractFloat`: The font size for the axis labels. Default is 10.0.
- `axis_titleFontSize::AbstractFloat`: The font size for the axis titles. Default is 10.0.
- `legend_labelFontSize::AbstractFloat`: The font size for the legend labels. Default is 10.0.
- `legend_titleFontSize::AbstractFloat`: The font size for the legend title. Default is 10.0.
- `legend_symbolSize::AbstractFloat`: The size of the legend symbols. Default is 180.0.
- `title_fontSize::AbstractFloat`: The font size for the plot title. Default is 15.0.

## Returns
- `vega_hmap`: The generated heatmap plot as a Vega-Lite object.
"""
function vegaheatmap(Z::AbstractMatrix; 
    path::String=joinpath(@__DIR__, "../") * "heatmap.pdf", 
    Title::String=" ",
    xlabel::String="Latent Dimension", 
    ylabel::String="Observation",
    legend_title::String="value",
    color_field::String="value",
    scheme::String="blueorange",
    sortx::String="ascending",
    sorty::String="descending",
    Width::Int=400, 
    Height::Int=400,
    save_plot::Bool=false,
    set_domain_mid::Bool=false,
    axis_labelFontSize::AbstractFloat=10.0,
    axis_titleFontSize::AbstractFloat=10.0,
    legend_labelFontSize::AbstractFloat=10.0,
    legend_titleFontSize::AbstractFloat=10.0,
    legend_symbolSize::AbstractFloat=180.0,
    title_fontSize::AbstractFloat=15.0
    )

    n, p = size(Z)
    df = stack(DataFrame(Z', :auto), 1:n)
    df[!,:variable] = repeat(1:n, inner=p)
    df[!,:observation] = repeat(1:p, n)

    if set_domain_mid
        vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                            title={text=Title, fontSize=title_fontSize}, 
                            encoding={ 
                                x={"observation:o", sort=sortx, axis={title=xlabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}}, 
                                y={"variable:o", sort=sorty, axis={title=ylabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}},
                                color={color_field, scale={scheme=scheme, domainMid=0}, label="Value", legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title}}
                            } 
        ) 
    else
        vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                            title={text=Title, fontSize=title_fontSize}, 
                            encoding={ 
                                x={"observation:o", sort=sortx, axis={title=xlabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}}, 
                                y={"variable:o", sort=sorty, axis={title=ylabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}},
                                color={color_field, scale={scheme=scheme}, label="Value", 
                                       legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title}}
                            } 
        ) 
    end

    if save_plot == true
        vega_hmap |> VegaLite.save(path)
    end

    return vega_hmap
end


"""
    create_colored_umap_plot(X::AbstractMatrix, labels::AbstractVector, plotseed;
                            precomputed::Bool=false,
                            path::String=figurespath * "/umap_data_labels.pdf",
                            Title::String=" ",
                            legend_title::String="value",
                            n_neighbors::Int=30,
                            min_dist::Float64=0.4,
                            color_field::String="labels:o",
                            scheme::String="category20",
                            colorlabel::String="Cell Type",
                            save_plot::Bool=true,
                            embedding::AbstractMatrix=zeros(2,2),
                            value_type::String="discrete",
                            marker_size::String="20",
                            axis_labelFontSize::AbstractFloat=10.0,
                            axis_titleFontSize::AbstractFloat=10.0,
                            legend_labelFontSize::AbstractFloat=10.0,
                            legend_titleFontSize::AbstractFloat=10.0,
                            legend_symbolSize::AbstractFloat=10.0,
                            title_fontSize::AbstractFloat=10.0
                            )

Create a colored scatter plot of 2D UMAP based on the given data using the Vega-Lite library.

# Arguments
- `X::AbstractMatrix`: The input data matrix for generating the 2D UMAP coordinates.
- `labels::AbstractVector`: The labels or values for each data point used for the color coding.
- `plotseed`: The random seed for generating the plot.

# Optional Arguments
- `precomputed::Bool=false`: Whether the UMAP embedding is already precomputed.
- `path::String=figurespath * "/umap_data_labels.pdf"`: The path to save the plot.
- `Title::String=" "`: The title of the plot.
- `legend_title::String="value"`: The title of the legend.
- `n_neighbors::Int=30`: The number of neighbors to consider in the UMAP algorithm.
- `min_dist::Float64=0.4`: The minimum distance between points in the UMAP algorithm.
- `color_field::String="labels:o"`: The field to use for coloring the plot.
- `scheme::String="category20"`: The color scheme to use for generating the plot.
- `colorlabel::String="Cell Type"`: The label for the color field.
- `save_plot::Bool=true`: Whether to save the plot to a file.
- `embedding::AbstractMatrix=zeros(2,2)`: The precomputed UMAP embedding.
- `value_type::String="discrete"`: The type of values for the color field.
- `marker_size::String="20"`: The size of the markers in the plot.
- `axis_labelFontSize::AbstractFloat=10.0`: The font size of the axis labels.
- `axis_titleFontSize::AbstractFloat=10.0`: The font size of the axis titles.
- `legend_labelFontSize::AbstractFloat=10.0`: The font size of the legend labels.
- `legend_titleFontSize::AbstractFloat=10.0`: The font size of the legend title.
- `legend_symbolSize::AbstractFloat=10.0`: The size of the legend symbols.
- `title_fontSize::AbstractFloat=10.0`: The font size of the plot title.

# Returns
- `umap_plot`: The generated colored 2D UMAP plot.
"""
function create_colored_umap_plot(X::AbstractMatrix, labels::AbstractVector, plotseed; 
    precomputed::Bool=false,
    path::String=figurespath * "/umap_data_labels.pdf",
    Title::String=" ",
    legend_title::String="value",
    n_neighbors::Int=30,
    min_dist::Float64=0.4,
    color_field::String="labels:o",
    scheme::String="category20",
    colorlabel::String="Cell Type",
    save_plot::Bool=true,
    embedding::AbstractMatrix=zeros(2,2),
    value_type::String="discrete",
    marker_size::String="20",
    axis_labelFontSize::AbstractFloat=10.0,
    axis_titleFontSize::AbstractFloat=10.0,
    legend_labelFontSize::AbstractFloat=10.0,
    legend_titleFontSize::AbstractFloat=10.0,
    legend_symbolSize::AbstractFloat=10.0,
    title_fontSize::AbstractFloat=10.0
    )

    if precomputed == true
        if value_type == "discrete"
            df = DataFrame(UMAP1 = embedding[:,1], UMAP2 = embedding[:,2], labels = labels) 
        else
            df = DataFrame(UMAP1 = embedding[:,1], UMAP2 = embedding[:,2], values = labels) 
        end

        umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                                  title={text=Title, fontSize=title_fontSize},
                                  encoding={
                                      x={:UMAP1, type=:quantitative, axis={grid=false}, labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize},   
                                      y={:UMAP2, type=:quantitative, axis={grid=false}, labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize},    
                                      color={color_field, scale={scheme=scheme}, label=colorlabel, 
                                      legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title}}
                                  },
                                  width=600, height=600,
        ) 
    
        if save_plot == true
            umap_plot |> VegaLite.save(path)
        end

    else
        embedding = generate_umap(X, plotseed; n_neighbors=n_neighbors, min_dist=min_dist)
        df = DataFrame(UMAP1 = embedding[:,1], UMAP2 = embedding[:,2], labels = labels) 

        umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"}, 
                                  title={text=Title, fontSize=title_fontSize},
                                  encoding={
                                      x={:UMAP1, type=:quantitative, axis={grid=false}, labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize},   
                                      y={:UMAP2, type=:quantitative, axis={grid=false}, labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize},
                                      color={c, scale={scheme=scheme}, label=colorlabel, 
                                      legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title}},     
                                  },
                                  width=600, height=600,
        ) 
    
        if save_plot == true
            umap_plot |> VegaLite.save(path)
        end
    end

    return umap_plot
end

"""
    create_latent_umaps(X::AbstractMatrix, plotseed, Z::AbstractMatrix, model_name::String, ;
        figurespath::String=figurespath,
        image_type::String=".pdf",
        legend_title::String="value",
        precomputed::Bool=false,
        embedding::AbstractMatrix=zeros(2,2),
        save_plot::Bool=true,
        marker_size::String="20",
        axis_labelFontSize::AbstractFloat=10.0,
        axis_titleFontSize::AbstractFloat=10.0,
        legend_labelFontSize::AbstractFloat=10.0,
        legend_titleFontSize::AbstractFloat=10.0,
        legend_symbolSize::AbstractFloat=10.0,
        title_fontSize::AbstractFloat=15.0
        )

Given data X, this function creates 2D UMAP plots colored by each column of the matrix Z individually. Z can e.g. be a latent representation of the data X.

# Arguments
- `X::AbstractMatrix`: The input data matrix.
- `plotseed`: The random seed for plotting.
- `Z::AbstractMatrix`: A matrix used for coloring the generated plots.
- `model_name::String`: The name of the model that produced the matrix Z.
- `figurespath::String`: The path to save the figures. Default is `figurespath`.
- `image_type::String`: The image file type. Default is ".pdf".
- `legend_title::String`: The title for the legend. Default is "value".
- `precomputed::Bool`: Whether the UMAP coordinates are precomputed. Default is `false`.
- `embedding::AbstractMatrix`: The precomputed UMAP coordinates. If not specified by the user, default is a 2x2 matrix of zeros.
- `save_plot::Bool`: Whether to save the plots. Default is `true`.
- `marker_size::String`: The size of the markers in the plot. Default is "20".
- `axis_labelFontSize::AbstractFloat`: The font size for the axis labels. Default is 10.0.
- `axis_titleFontSize::AbstractFloat`: The font size for the axis titles. Default is 10.0.
- `legend_labelFontSize::AbstractFloat`: The font size for the legend labels. Default is 10.0.
- `legend_titleFontSize::AbstractFloat`: The font size for the legend title. Default is 10.0.
- `legend_symbolSize::AbstractFloat`: The size of the legend symbols. Default is 10.0.
- `title_fontSize::AbstractFloat`: The font size for the plot title. Default is 15.0.

# Note
- The function saves the generated UMAP plots in the specified `figurespath` directory.
"""
function create_latent_umaps(X::AbstractMatrix, plotseed, Z::AbstractMatrix, model_name::String, ;
    figurespath::String=figurespath,
    image_type::String=".pdf",
    legend_title::String="value",
    precomputed::Bool=false,
    embedding::AbstractMatrix=zeros(2,2),
    save_plot::Bool=true,
    marker_size::String="20",
    axis_labelFontSize::AbstractFloat=10.0,
    axis_titleFontSize::AbstractFloat=10.0,
    legend_labelFontSize::AbstractFloat=10.0,
    legend_titleFontSize::AbstractFloat=10.0,
    legend_symbolSize::AbstractFloat=10.0,
    title_fontSize::AbstractFloat=15.0
    )

    if precomputed == false
        embedding = generate_umap(X, plotseed)
    else
        @info "using precomputed coordinates for creating the UMAP plot ..."
    end

    len = size(Z, 2)

    for k in 1:len
        create_colored_umap_plot(X, Z[:, k], plotseed; 
                                 Title=model_name * " dimension $(k)", legend_title=legend_title,
                                 path=figurespath * "_dim$(k)_umap_data_labels" * image_type, 
                                 embedding=embedding, precomputed=true, color_field="values", scheme="blueorange", 
                                 colorlabel="Dim $(k)", value_type="continuous", save_plot=save_plot, marker_size=marker_size,
                                 axis_labelFontSize=axis_labelFontSize, axis_titleFontSize=axis_titleFontSize,
                                 legend_labelFontSize=legend_labelFontSize, legend_symbolSize=legend_symbolSize,
                                 legend_titleFontSize=legend_titleFontSize, title_fontSize=title_fontSize
        )
    end

end


"""
    normalized_scatter_top_values(vec, labels; top_n=15, dim=k)

This function plots a scatter plot of the top `top_n` values in the vector `vec`, along with their corresponding labels. The values are normalized by the maximum absolute value and sorted by absolute magnitude in descending order. The scatter plot is customized with x-axis labels, colors, and y-axis limits.

## Arguments
- `vec`: A vector of values.
- `labels`: An array of labels corresponding to the values in `vec`.
- `top_n`: The number of top values to plot. Default is 15.
- `dim`: The dimension of the data. Default is `k`.

## Returns
- `p`: A `Plots.Plot` object representing the scatter plot.

"""
function normalized_scatter_top_values(vec, labels; top_n=15, dim=k)
    # Filter out zeros and get indices of nonzero elements
    non_zero_indices = findall(x -> x != 0, vec)

    if length(non_zero_indices) < top_n
        top_n = length(non_zero_indices)
    end

    non_zero_values = vec[non_zero_indices]
    selected_labels = labels[non_zero_indices]

    # Normalize by the maximum absolute value
    normalized_values = non_zero_values / maximum(abs.(non_zero_values))

    # Get indices that would sort the normalized_values by absolute magnitude (in descending order)
    sorted_indices = sortperm(abs.(normalized_values), rev=true)
    
    # Select the top_n values and their corresponding labels
    top_values = normalized_values[sorted_indices[1:top_n]]
    top_labels = selected_labels[sorted_indices[1:top_n]]

    # Compute colors based on the blueorange scheme
    color_map(val) = val < 0 ? ColorSchemes.vik[0.5 * (1 + val)] : ColorSchemes.vik[0.5 + 0.5 * val]
    colors = color_map.(top_values)

    # Determine y-axis limits based on data range
    y_lower_limit = min(-1, minimum(top_values) - 0.1) # A bit below the smallest value for padding
    y_upper_limit = max(1, maximum(top_values) + 0.1)  # A bit above the largest value for padding

    
    # Plot scatter plot with custom x-axis labels and colors
    p = scatter(1:top_n, top_values, 
                size=(600, 500),
                xlabel="Gene", 
                ylabel="Maximum-normalized coefficient", 
                title="Top $(top_n) genes in dimension $(dim) ($(length(non_zero_indices)) nonzero genes)",
                legend=false, 
                markersize=6,
                xticks=(1:top_n, top_labels),
                ylims=(y_lower_limit, y_upper_limit),
                color=colors,
                grid=false,
                xrotation = 90)
    
    hline!(p, [0], color=:black, linewidth=1.5, linestyle=:dash) # Add horizontal line at y=0
    return p
end