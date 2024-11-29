#------------------------------
# This file contains the functions for generating the plots of the results: 
#------------------------------
"""
    vegaheatmap(Z::AbstractMatrix; 
        path::String=joinpath(@__DIR__, "../") * "heatmap.pdf", 
        Title::String=" ",
        xlabel::String="Latent dimension", 
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
        axis_labelFontSize::AbstractFloat=14.0,
        axis_titleFontSize::AbstractFloat=14.0,
        legend_labelFontSize::AbstractFloat=12.5,
        legend_titleFontSize::AbstractFloat=14.0,
        legend_symbolSize::AbstractFloat=180.0,
        title_fontSize::AbstractFloat=16.0,
        legend_gradientThickness::AbstractFloat=20.0,
        legend_gradientLength::AbstractFloat=200.0
        )

This function generates a heatmap plot using the Vega-Lite library based on the input matrix `Z`.

## Arguments
- `Z::AbstractMatrix`: The input matrix for generating the heatmap plot.

## Optional Arguments
- `path::String`: The file path to save the generated plot. Default is `joinpath(@__DIR__, "../") * "heatmap.pdf"`.
- `Title::String`: The title of the plot. Default is an empty string.
- `xlabel::String`: The label for the x-axis. Default is "Latent dimension".
- `ylabel::String`: The label for the y-axis. Default is "Observation".
- `legend_title::String`: The title for the legend. Default is "value".
- `color_field::String`: The field to use for coloring the heatmap. Default is "value".
- `scheme::String`: The color scheme to use for the heatmap. Default is "blueorange".
- `sortx::String`: The sorting order for the x-axis. Default is "ascending".
- `sorty::String`: The sorting order for the y-axis. Default is "descending".
- `Width::Int`: The width of the plot in pixels. Default is 400.
- `Height::Int`: The height of the plot in pixels. Default is 400.
- `save_plot::Bool`: Whether to save the plot as a PDF file. Default is `false`.
- `set_domain_mid::Bool`: Whether to set the domain midpoint for the color scale. Default is `false`.
- `axis_labelFontSize::AbstractFloat`: The font size for the axis labels. Default is 14.0.
- `axis_titleFontSize::AbstractFloat`: The font size for the axis titles. Default is 14.0.
- `legend_labelFontSize::AbstractFloat`: The font size for the legend labels. Default is 12.5.
- `legend_titleFontSize::AbstractFloat`: The font size for the legend title. Default is 14.0.
- `legend_symbolSize::AbstractFloat`: The size of the legend symbols. Default is 180.0.
- `title_fontSize::AbstractFloat`: The font size for the plot title. Default is 16.0.
- `legend_gradientThickness::AbstractFloat`: The thickness of the legend gradient. Default is 20.0.
- `legend_gradientLength::AbstractFloat`: The length of the legend gradient. Default is 200.0.

## Returns
- `vega_hmap`: The generated heatmap plot as a Vega-Lite object.
"""
function vegaheatmap(Z::AbstractMatrix; 
    path::String=joinpath(@__DIR__, "../") * "heatmap.pdf", 
    Title::String=" ",
    xlabel::String="Latent dimension", 
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
    axis_labelFontSize::AbstractFloat=14.0,
    axis_titleFontSize::AbstractFloat=14.0,
    legend_labelFontSize::AbstractFloat=12.5,
    legend_titleFontSize::AbstractFloat=14.0,
    legend_symbolSize::AbstractFloat=180.0,
    title_fontSize::AbstractFloat=16.0,
    legend_gradientThickness::AbstractFloat=20.0,
    legend_gradientLength::AbstractFloat=200.0
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
                                color={color_field, scale={scheme=scheme, domainMid=0}, label="Value", 
                                legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
                            } 
        ) 
    else
        vega_hmap = df |> @vlplot(mark=:rect, width=Width, height=Height, 
                            title={text=Title, fontSize=title_fontSize}, 
                            encoding={ 
                                x={"observation:o", sort=sortx, axis={title=xlabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}}, 
                                y={"variable:o", sort=sorty, axis={title=ylabel, labelOverlap="false", labelFontSize=axis_labelFontSize, titleFontSize=axis_titleFontSize}},
                                color={color_field, scale={scheme=scheme}, label="Value", 
                                legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}
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
                            Title::String="",
                            legend_title::String="value",
                            n_neighbors::Int=30,
                            min_dist::Float64=0.4,
                            color_field::String="labels:o",
                            scheme::String="category20",
                            colorlabel::String="Cell Type",
                            save_plot::Bool=true,
                            embedding::AbstractMatrix=zeros(2,2),
                            value_type::String="discrete",
                            marker_size::String="40",
                            axis_labelFontSize::AbstractFloat=0.0,
                            axis_titleFontSize::AbstractFloat=0.0,
                            legend_labelFontSize::AbstractFloat=24.0,
                            legend_titleFontSize::AbstractFloat=28.0,
                            legend_symbolSize::AbstractFloat=240.0,
                            title_fontSize::AbstractFloat=28.0,
                            axis_tickSize::AbstractFloat=0.0,
                            legend_gradientThickness::AbstractFloat=20.0,
                            legend_gradientLength::AbstractFloat=200.0,
                            show_axis::Bool=true
                            )

Create a colored two-dimensional UMAP scatter plot using the given data. The design of the plot can be adapted by changing the optional arguments.

# Arguments
- `X::AbstractMatrix`: The input data matrix.
- `labels::AbstractVector`: The labels for each data point.
- `plotseed`: The seed for generating the plot.

# Optional Arguments
- `precomputed::Bool=false`: Whether the UMAP embedding is precomputed.
- `path::String=figurespath * "/umap_data_labels.pdf"`: The path to save the plot.
- `Title::String=""`: The title of the plot.
- `legend_title::String="value"`: The title of the legend.
- `n_neighbors::Int=30`: The number of neighbors to consider in the UMAP algorithm.
- `min_dist::Float64=0.4`: The minimum distance between points in the UMAP algorithm.
- `color_field::String="labels:o"`: The field to use for coloring the plot.
- `scheme::String="category20"`: The color scheme to use.
- `colorlabel::String="Cell Type"`: The label for the color field.
- `save_plot::Bool=true`: Whether to save the plot.
- `embedding::AbstractMatrix=zeros(2,2)`: The precomputed UMAP embedding.
- `value_type::String="discrete"`: The type of values in the color field.
- `marker_size::String="40"`: The size of the markers in the plot.
- `axis_labelFontSize::AbstractFloat=0.0`: The font size of the axis labels.
- `axis_titleFontSize::AbstractFloat=0.0`: The font size of the axis titles.
- `legend_labelFontSize::AbstractFloat=24.0`: The font size of the legend labels.
- `legend_titleFontSize::AbstractFloat=28.0`: The font size of the legend title.
- `legend_symbolSize::AbstractFloat=240.0`: The size of the legend symbols.
- `title_fontSize::AbstractFloat=28.0`: The font size of the plot title.
- `axis_tickSize::AbstractFloat=0.0`: The size of the axis ticks.
- `legend_gradientThickness::AbstractFloat=20.0`: The thickness of the legend gradient.
- `legend_gradientLength::AbstractFloat=200.0`: The length of the legend gradient.
- `show_axis::Bool=true`: Whether to show the axis in the plot.

# Returns
- `umap_plot`: The generated UMAP plot.
"""
function create_colored_umap_plot(X::AbstractMatrix, labels::AbstractVector, plotseed; 
    precomputed::Bool=false,
    path::String=figurespath * "/umap_data_labels.pdf",
    Title::String="",
    legend_title::String="value",
    n_neighbors::Int=30,
    min_dist::Float64=0.4,
    color_field::String="labels:o",
    scheme::String="category20",
    colorlabel::String="Cell Type",
    save_plot::Bool=true,
    embedding::AbstractMatrix=zeros(2,2),
    value_type::String="discrete",
    marker_size::String="40",
    axis_labelFontSize::AbstractFloat=0.0,
    axis_titleFontSize::AbstractFloat=0.0,
    legend_labelFontSize::AbstractFloat=24.0,
    legend_titleFontSize::AbstractFloat=28.0,
    legend_symbolSize::AbstractFloat=240.0,
    title_fontSize::AbstractFloat=28.0,
    axis_tickSize::AbstractFloat=0.0,
    legend_gradientThickness::AbstractFloat=20.0,
    legend_gradientLength::AbstractFloat=200.0,
    show_axis::Bool=true
    )

    if precomputed == true
        if value_type == "discrete"
            df = DataFrame(UMAP1 = embedding[:,1], UMAP2 = embedding[:,2], labels = labels) 
        else
            df = DataFrame(UMAP1 = embedding[:,1], UMAP2 = embedding[:,2], values = labels) 
        end

        if show_axis==false
            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                                  title={text=Title, fontSize=title_fontSize},
                                  encoding={
                                      x={:UMAP1, type=:quantitative, axis=nothing},   
                                      y={:UMAP2, type=:quantitative, axis=nothing},    
                                      color={color_field, scale={scheme=scheme}, label=colorlabel, 
                                      legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                      gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}   
                                  },
                                  config={view={stroke=nothing}},
                                  width=600, height=600,
            ) 
        else
            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                                    title={text=Title, fontSize=title_fontSize},
                                    encoding={
                                        x={:UMAP1, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},   
                                        y={:UMAP2, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},    
                                        color={color_field, scale={scheme=scheme}, label=colorlabel, 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}     
                                    },
                                    width=600, height=600,
            ) 
        end
    else
        embedding = generate_umap(X, plotseed; n_neighbors=n_neighbors, min_dist=min_dist)

        if value_type == "discrete"
            df = DataFrame(UMAP1 = embedding[:,1], UMAP2 = embedding[:,2], labels = labels) 
        else
            df = DataFrame(UMAP1 = embedding[:,1], UMAP2 = embedding[:,2], values = labels) 
        end 

        if show_axis==false
            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"},
                                  title={text=Title, fontSize=title_fontSize},
                                  encoding={
                                      x={:UMAP1, type=:quantitative, axis=nothing},   
                                      y={:UMAP2, type=:quantitative, axis=nothing},    
                                      color={color_field, scale={scheme=scheme}, label=colorlabel, 
                                      legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                      gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}} 
                                  },
                                  config={view={stroke=nothing}},
                                  width=600, height=600,
            ) 
        else

            umap_plot = df |> @vlplot(mark={"point", size=marker_size, filled="true"}, 
                                    title={text=Title, fontSize=title_fontSize},
                                    encoding={
                                        x={:UMAP1, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},   
                                        y={:UMAP2, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},
                                        color={c, scale={scheme=scheme}, label=colorlabel, 
                                        legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                        gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}}     
                                    },
                                    width=600, height=600,
            ) 
        end
    end

    if save_plot == true
        umap_plot |> VegaLite.save(path)
    end

    return umap_plot
end


"""
    create_latent_umaps(X::AbstractMatrix, plotseed, Z::AbstractMatrix;
        figurespath::String=figurespath,
        image_type::String=".pdf",
        legend_title::String="value",
        precomputed::Bool=false,
        embedding::AbstractMatrix=zeros(2,2),
        save_plot::Bool=true,
        marker_size::String="40",
        axis_labelFontSize::AbstractFloat=0.0,
        axis_titleFontSize::AbstractFloat=0.0,
        legend_labelFontSize::AbstractFloat=24.0,
        legend_titleFontSize::AbstractFloat=28.0,
        legend_symbolSize::AbstractFloat=240.0,
        title_fontSize::AbstractFloat=28.0,
        axis_tickSize::AbstractFloat=0.0,
        legend_gradientThickness::AbstractFloat=25.0,
        legend_gradientLength::AbstractFloat=280.0,
        show_axis::Bool=true
        )

Create colored two-dimensional scatter plots given a pre-defined `embedding`, e.g., UMAP or a low-dimensional representation of a model. 
The data points in the `embedding` for each plot are colored by one of the column values of the matrix `Z`, respectively, where `size(Z, 2)` has to match the number of points in the `embedding`.
If no pre-defined embedding is passed to the function, i.e., `precomputed=false`, then a UMAP embedding of the data `X` is computed by calling the function. 
If `save_plot=true`, then the plots are saved in the directory `figurespath`. 

# Arguments
- `X::AbstractMatrix`: The input data matrix.
- `plotseed`: The random seed for plotting.
- `Z::AbstractMatrix`: The latent space matrix.
- `figurespath::String`: The path to save the figures. Default is `figurespath`.
- `image_type::String`: The image file type. Default is ".pdf".
- `legend_title::String`: The title for the legend. Default is "value".
- `precomputed::Bool`: Whether the UMAP coordinates are precomputed. Default is `false`.
- `embedding::AbstractMatrix`: The precomputed UMAP coordinates. Default is a 2x2 matrix of zeros.
- `save_plot::Bool`: Whether to save the plots. Default is `true`.
- `marker_size::String`: The size of the markers in the plot. Default is "40".
- `axis_labelFontSize::AbstractFloat`: The font size for axis labels. Default is 0.0.
- `axis_titleFontSize::AbstractFloat`: The font size for axis titles. Default is 0.0.
- `legend_labelFontSize::AbstractFloat`: The font size for legend labels. Default is 24.0.
- `legend_titleFontSize::AbstractFloat`: The font size for legend title. Default is 28.0.
- `legend_symbolSize::AbstractFloat`: The size of the legend symbols. Default is 240.0.
- `title_fontSize::AbstractFloat`: The font size for plot title. Default is 28.0.
- `axis_tickSize::AbstractFloat`: The size of the axis ticks. Default is 0.0.
- `legend_gradientThickness::AbstractFloat`: The thickness of the legend gradient. Default is 25.0.
- `legend_gradientLength::AbstractFloat`: The length of the legend gradient. Default is 280.0.
- `show_axis::Bool`: Whether to show the axis. Default is `true`.
"""
function create_latent_umaps(X::AbstractMatrix, plotseed, Z::AbstractMatrix;
    figurespath::String=figurespath,
    image_type::String=".pdf",
    legend_title::String="value",
    precomputed::Bool=false,
    embedding::AbstractMatrix=zeros(2,2),
    save_plot::Bool=true,
    marker_size::String="40",
    axis_labelFontSize::AbstractFloat=0.0,
    axis_titleFontSize::AbstractFloat=0.0,
    legend_labelFontSize::AbstractFloat=24.0,
    legend_titleFontSize::AbstractFloat=28.0,
    legend_symbolSize::AbstractFloat=240.0,
    title_fontSize::AbstractFloat=28.0,
    axis_tickSize::AbstractFloat=0.0,
    legend_gradientThickness::AbstractFloat=25.0,
    legend_gradientLength::AbstractFloat=280.0,
    show_axis::Bool=true
    )

    if precomputed == false
        embedding = generate_umap(X, plotseed)
    else
        @info "using precomputed coordinates for creating the UMAP plot ..."
    end

    len = size(Z, 2)

    for k in 1:len
        create_colored_umap_plot(X, Z[:, k], plotseed; 
                                 Title="Dimension $(k)", legend_title=legend_title,
                                 path=figurespath * "_dim$(k)_umap_data_labels" * image_type, 
                                 embedding=embedding, precomputed=true, color_field="values", scheme="blueorange", 
                                 colorlabel="Dim $(k)", value_type="continuous", save_plot=save_plot, marker_size=marker_size,
                                 axis_labelFontSize=axis_labelFontSize, axis_titleFontSize=axis_titleFontSize,
                                 legend_labelFontSize=legend_labelFontSize, legend_symbolSize=legend_symbolSize,
                                 legend_titleFontSize=legend_titleFontSize, title_fontSize=title_fontSize, axis_tickSize=axis_tickSize,
                                 legend_gradientThickness=legend_gradientThickness, legend_gradientLength=legend_gradientLength, show_axis=show_axis
        )
    end

end



"""
    normalized_scatter_top_values(vec, labels; top_n=15, dim=k)

This function plots a scatter plot of the top `top_n` values in the vector `vec`, along with their corresponding labels from the `labels` array. 
The values in `vec` are normalized by their maximum absolute value before plotting and sorted in descending order of magnitude from left to right. 
The scatter plot is colored based on a blue-orange color scheme, with negative values appearing blue and positive values appearing orange. 
The y-axis limits are determined based on the range of the top values, with a small padding added for better visualization. 
The function also adds a horizontal dashed line at y=0.

## Arguments
- `vec`: The vector of values to be plotted.
- `labels`: The array of labels corresponding to the values in `vec`.
- `top_n`: The number of top values to be plotted. Default is 15.
- `dim`: The dimension of the data. Default is `k`.

## Returns
- `p`: The scatter plot object.

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
    abs_max_val = maximum(abs.(top_values))
    y_lower_limit = min(-1, -abs_max_val - 0.1) # A bit below the smallest value for padding
    y_upper_limit = max(1, abs_max_val + 0.1)  # A bit above the largest value for padding

    
    # Plot scatter plot with custom x-axis labels and colors
    p = scatter(1:top_n, top_values, 
                size=(700, 500),
                #xlabel="Gene", 
                ylabel="Maximum-normalized coefficient", 
                #ylabel="", 
                guidefontsize=18,
                title="Top $(top_n) genes in dimension $(dim) \n ($(length(non_zero_indices)) nonzero genes)",
                titlefontsize=18,
                legend=false, 
                markersize=10,
                xticks=(1:top_n, top_labels),
                tickfontsize=18,
                ylims=(y_lower_limit, y_upper_limit),
                color=colors,
                grid=false,
                xrotation = 60)
    
    hline!(p, [0], color=:black, linewidth=1.5, linestyle=:dash) # Add horizontal line at y=0
    return p
end


function plot_coefficients_dynamics(coeffs, dim; 
    iters::Union{Int, Nothing}=nothing, 
    xscale::Symbol=:log10, 
    save_plot::Bool=false, 
    path::Union{String, Nothing}=nothing,
    title::String=""
    )

    # Number of iterations and number of coefficients
    num_iters = length(coeffs)
    num_coeffs = size(coeffs[1], 1)

    # Preallocate the coefficient matrix
    coeffs_dynamics = zeros(Float32, num_coeffs, num_iters)

    # Populate the coefficient matrix
    for iter in 1:num_iters
        coeffs_dynamics[:, iter] = coeffs[iter][:, dim]
    end

    # Handle the number of iterations to plot
    if typeof(iters) == Int && iters > num_iters
        @warn "Number of iterations to plot is greater than the number of iterations in the data. Plotting all iterations."
        iters = num_iters
    elseif isnothing(iters)
        iters = num_iters
    end

    # Prepare data for plotting
    x = 1:iters
    y = coeffs_dynamics[:, 1:iters]

    # Create the plot
    x_scale = string(xscale)
    pl = plot(x, y', xlabel="Iteration " * "(" * x_scale * "scale)", ylabel="Coefficient value", title=title, lw=2, legend=false, xscale=xscale)

    if save_plot
        savefig(pl, path)
    end

    return pl
end