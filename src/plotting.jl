#------------------------------
# This file contains the functions for generating the plots of the results: 
#------------------------------

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
    legend_gradientLength::AbstractFloat=200.0
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
                                      x={:UMAP1, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},   
                                      y={:UMAP2, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},    
                                      color={color_field, scale={scheme=scheme}, label=colorlabel, 
                                      legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                      gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}},     
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
                                      x={:UMAP1, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},   
                                      y={:UMAP2, type=:quantitative, axis={grid=false, tickSize=axis_tickSize, titleFontSize=axis_titleFontSize, labelFontSize=axis_labelFontSize}},
                                      color={c, scale={scheme=scheme}, label=colorlabel, 
                                      legend={labelFontSize=legend_labelFontSize, titleFontSize=legend_titleFontSize, symbolSize=legend_symbolSize, title=legend_title,
                                      gradientThickness=legend_gradientThickness, gradientLength=legend_gradientLength}},     
                                  },
                                  width=600, height=600,
        ) 
    
        if save_plot == true
            umap_plot |> VegaLite.save(path)
        end
    end

    return umap_plot
end


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
    legend_gradientLength::AbstractFloat=280.0
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
                                 legend_gradientThickness=legend_gradientThickness, legend_gradientLength=legend_gradientLength
        )
    end

end


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
                size=(600, 600),
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
