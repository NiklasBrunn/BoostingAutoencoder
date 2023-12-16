#------------------------------
# This file contains the loss functions used for training the different BAE and autoencoder models:
#------------------------------



#------------------------------
# Loss for computing BAE-pseudo-targets:
#------------------------------
"""
    loss_z(Z, Xt, decoder)

Description:

    Calculate the mean squared error (MSE) loss between the reconstructed data from a decoder and the original input data.

Arguments:

    - `Z::AbstractMatrix`: The encoded representation of input data.
    - `Xt::AbstractMatrix`: The original input data to reconstruct.
    - `decoder::Function`: The decoder function that reconstructs data from the encoded representation.

Returns:

    - `loss::Real`: The mean squared error loss between the reconstructed data and the original input data.

Example:

    ```julia
    # Calculate the reconstruction loss for a given encoded representation
    encoded_data = rand(64, 100)  # Example: 64-dimensional encoded data
    original_data = rand(784, 100)  # Example: 784-dimensional original data
    reconstruction_loss = loss_z(encoded_data, original_data, decoder_function)
    ```

Notes:

    - The `loss_z` function computes the mean squared error (MSE) loss, which is commonly used in autoencoder-based tasks to measure the difference between the original data and the reconstructed data.

    - The `Z` argument should be the encoded representation of the input data obtained from the autoencoder's encoder network.

    - The `Xt` argument represents the original input data that you want to compare with the reconstructed data.

    - The `decoder` argument should be a function that takes the encoded representation `Z` and produces the reconstructed data.

    - The MSE loss is suitable for tasks such as denoising autoencoders and reconstruction-based anomaly detection.

"""
loss_z(Z::AbstractMatrix, Xt::AbstractMatrix, decoder) = Flux.mse(decoder(Z), Xt) 



#------------------------------
# Loss functions for training BAE mode "alternating" or AE:
#------------------------------
"""
    loss(Xt::AbstractMatrix, BAE::Autoencoder)

Description:

    Calculate the mean squared error (MSE) loss between the reconstructed data from an autoencoder model and the original input data.

Arguments:

    - `Xt::AbstractMatrix`: The original input data for which you want to calculate the reconstruction loss.
    - `BAE::Autoencoder`: The autoencoder model, consisting of an encoder and a decoder, used for reconstruction.

Returns:

    - `loss::Real`: The mean squared error loss between the reconstructed data and the original input data.

Example:

    ```julia
    # Calculate the reconstruction loss for a given autoencoder model and input data
    input_data = rand(784, 100)  # Example: 784-dimensional input data
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    reconstruction_loss = loss(input_data, autoencoder_model)
    ```

Notes:

    - The `loss` function computes the mean squared error (MSE) loss, which quantifies the difference between the original input data and the data reconstructed by the autoencoder model.

    - The `Xt` argument should be the original input data you want to compare with the reconstructed data.

    - The `BAE` argument should be an instance of the `Autoencoder` struct, representing the autoencoder model. It consists of an encoder and a decoder network.

    - The MSE loss is commonly used in autoencoder-based tasks for measuring reconstruction accuracy and training the model.

"""
loss(Xt::AbstractMatrix, BAE::Autoencoder) = Flux.mse(BAE(Xt), Xt)


"""
    loss_wrapper(BAE::Autoencoder)

Description:

    Create a loss function wrapper for an autoencoder model. The wrapper function accepts input data and computes the mean squared error (MSE) loss between the reconstructed data from the autoencoder and the original input data.

Arguments:

    - `BAE::Autoencoder`: The autoencoder model, consisting of an encoder and a decoder, used for reconstruction.

Returns:

    - `loss_function::Function`: A loss function that can be used to calculate the reconstruction loss for a given input data.

Example:

    ```julia
    # Create a loss function wrapper for a given autoencoder model
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    loss_function = loss_wrapper(autoencoder_model)

    # Calculate the reconstruction loss for input data using the loss function
    input_data = rand(784, 100)  # Example: 784-dimensional input data
    reconstruction_loss = loss_function(input_data)
    ```

Notes:

    - The `loss_wrapper` function creates a closure that encapsulates the autoencoder model. It returns a loss function tailored to that specific autoencoder.

    - The returned loss function accepts input data and calculates the MSE loss between the original input data and the data reconstructed by the autoencoder model.

    - This wrapper is useful when you need to compute the reconstruction loss for multiple input samples efficiently.

"""
loss_wrapper(BAE::Autoencoder) = function(Xt::AbstractMatrix) loss(Xt, BAE) end



#------------------------------
# Loss functions for training BAE mode "jointLoss":
#------------------------------
"""
    combinedLoss(batch::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, m::Int, zdim::Int, iter::Int)

Description:

    Calculate a reconstruction loss (combinedLoss/jointLoss) for an autoencoder model, which consists of two components: A sequentially applied constraint componentwise boosting component (to each latent dimension in the latent space, defined by the output dimension of the encoder) and a reconstruction loss component. The sequentially applied boosting component modifies the autoencoder's encoder coefficients to encourage certain structural properties, i.e. disentanglement of latent dimensions, in the encoded representation. After modification of the encoder weights, the MSE of the autoencoder model with the updated encoder weights gets computed.

Arguments:

    - `batch::AbstractMatrix{<:AbstractFloat}`: A batch of input data samples for which to compute the combined loss.
    - `BAE::Autoencoder`: The autoencoder model, consisting of an encoder and a decoder.
    - `ϵ::Number`: A parameter controlling the step length of the updates computed via boosting for each latent dimension.
    - `m::Int`: The number of boosting steps to perform for each latent dimension.
    - `zdim::Int`: The desired dimensionality of the encoded representations (number of latent dimensions).
    - `iter::Int`: Specifies the number of the current training epoch in which the function combinedLoss() is applied.

Returns:

    - `rec_loss::Real`: The combined loss, which includes both the sequentially applied constraint boosting component and the reconstruction loss component.

Example:

    ```julia
    # Calculate the combined loss for a batch of input data using an autoencoder model
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    input_data_batch = rand(784, 100)  # Example: Batch of 100 784-dimensional input data samples
    boosting_step_length = 0.01
    num_boosting_steps = 10
    num_latent_dimensions = 32
    current_training_epoch = 100
    combined_loss_value = combinedLoss(input_data_batch, autoencoder_model, boosting_step_length, num_boosting_steps, num_latent_dimensions, current_training_epoch)
    ```

Notes:

    - The `combinedLoss` function calculates a combined loss that consists of two components:
        1. A sequentially applied constraint componentwise boosting component that encourages specific properties in the encoded representations. This component modifies the encoder coefficients of the autoencoder model.
        2. A reconstruction loss component (mean squared error) that measures the difference between the original input data and the data reconstructed by the autoencoder model.

    - The sequentially applied constraint componentwise boosting component is applied via an external function `seq_constr_compL2Boost_combinedLoss`, and the encoder weights of the autoencoder `BAE` are modified accordingly.

    - The combinedLoss/jointLoss may be used for training a BAE (autoencoder) with a linear encoder. By applying a gradient-based optimization scheme (e.g. stochastic gradient descent) for training the BAE, the encoder weights are getting updated in the forward pass of combinedLoss and the decoder parameters are updated using computed gradient information from the backward pass.

"""
function combinedLoss(batch::AbstractMatrix{<:AbstractFloat}, BAE::Autoencoder, ϵ::Number, m::Int, zdim::Int, iter::Int) 

    BAE.encoder.coeffs = seq_constr_compL2Boost_combinedLoss(batch, BAE, ϵ, zdim, m, iter)

    rec_loss = Flux.mse(BAE(batch), batch) 

    return rec_loss
end


"""
    combinedLoss_wrapper(BAE::Autoencoder, ϵ::Number, m::Int, zdim::Int, iter::Int)

Description:

    Create a batch-wise loss function wrapper for an autoencoder model with the combinedLoss function (for more details see documentation of combinedLoss). The wrapper function accepts a batch of input data samples and computes the combined loss for the entire batch.

Arguments:

    - `BAE::Autoencoder`: The autoencoder model, consisting of an encoder and a decoder.
    - `ϵ::Number`: A parameter controlling the step length of the updates computed via boosting for each latent dimension.
    - `m::Int`: The number of boosting steps to perform for each latent dimension.
    - `zdim::Int`: The desired dimensionality of the encoded representations (number of latent dimensions).
    - `iter::Int`: Specifies the number of the current training epoch in which the function combinedLoss() is applied.

Returns:

    - `loss_function::Function`: A loss function that can be applied to a batch of input data to calculate the combined loss.

Example:

    ```julia
    # Create a batch-wise loss function wrapper for a given autoencoder model
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    boosting_step_length = 0.01
    num_boosting_steps = 10
    num_latent_dimensions = 32
    current_training_epoch = 100
    batch_loss_function = combinedLoss_wrapper(autoencoder_model, boosting_step_length, num_boosting_steps, num_latent_dimensions, current_training_epoch)

    # Calculate the combined loss for a batch of input data using the batch loss function
    input_data_batch = rand(784, 100)  # Example: Batch of 100 784-dimensional input data samples
    combined_loss_value = batch_loss_function(input_data_batch)
    ```

Notes:

    - The `combinedLoss_wrapper` function creates a closure that encapsulates the autoencoder model and the parameters for the combined loss computation. It returns a batch-wise loss function that can be applied to a batch of input data.

    - The batch-wise loss function calculates the combined loss for the entire batch, including both a sequentially applied constraint componentwise boosting component and a reconstruction loss component. The boosting component is optimized using an external function `seq_constr_compL2Boost_combinedLoss`.

    - The combinedLoss/jointLoss may be used for training a BAE (autoencoder) with a linear encoder. By applying a gradient-based optimization scheme (e.g. stochastic gradient descent) for training the BAE, the encoder weights are getting updated in the forward pass of combinedLoss and the decoder parameters are updated using computed gradient information from the backward pass.

    - This wrapper is useful when you want to efficiently compute the combined loss for multiple input samples in a batch.

"""
combinedLoss_wrapper(BAE::Autoencoder, ϵ::Number, m::Int, zdim::Int, iter::Int) = function(batch) combinedLoss(batch, BAE, ϵ, m, zdim, iter) end 



#------------------------------
# Loss functions for training L1AE (MSE with L1-sparsity penalty, regularized by sparsity-parameter α):
#------------------------------
"""
    loss_L1reg(Xt::AbstractMatrix, AE::Autoencoder, α::AbstractFloat)

Description:

    Calculate a loss function with L1 regularization for an autoencoder model with a single-layer encoder. The loss function consists of the mean squared error (MSE) reconstruction loss and an additional L1 regularization term applied to the encoder's weights.

Arguments:

    - `Xt::AbstractMatrix`: The original input data for which you want to calculate the loss.
    - `AE::Autoencoder`: The autoencoder model, consisting of a single-layer encoder and a decoder.
    - `α::AbstractFloat`: The regularization parameter controlling the strength of the L1 regularization term.

Returns:

    - `loss::Real`: The combined loss, including the reconstruction loss and L1 regularization term.

Example:

    ```julia
    # Calculate the loss with L1 regularization for a given autoencoder model and input data
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    input_data = rand(784, 100)  # Example: 784-dimensional input data
    regularization_parameter = 0.001
    loss_value = loss_L1reg(input_data, autoencoder_model, regularization_parameter)
    ```

Notes:

    - The `loss_L1reg` function computes a loss function that consists of two components:
        1. A mean squared error (MSE) reconstruction loss that measures the difference between the original input data and the data reconstructed by the autoencoder model.
        2. An L1 regularization term applied to the single-layer encoder's weights (coefficients).

    - The regularization parameter `α` controls the strength of the L1 regularization. Higher values of `α` encourage sparsity in the encoder's weights.

    - L1 regularization encourages sparsity in the single-layer encoder's weights by shrinking many of the non-informative weights towards zero.

"""
loss_L1reg(Xt::AbstractMatrix, AE::Autoencoder, α::AbstractFloat) = Flux.mse(AE(Xt), Xt) + α * sum(abs.(AE.encoder.W)) 


"""
    loss_wrapper_L1reg(AE::Autoencoder, α::AbstractFloat)

Description:

    Create a loss function wrapper with L1 regularization for an autoencoder model witha single-layer encoder. The wrapper function accepts input data and computes the combined loss, including a mean squared error (MSE) reconstruction loss and an L1 regularization term applied to the single-layer encoder's weights.

Arguments:

    - `AE::Autoencoder`: The autoencoder model, consisting of a single-layer encoder and a decoder.
    - `α::AbstractFloat`: The regularization parameter controlling the strength of the L1 regularization term. It should be a floating-point number.

Returns:

    - `loss_function::Function`: A loss function that can be used to calculate the combined loss for a given input data.

Example:

    ```julia
    # Create a loss function wrapper with L1 regularization for a given autoencoder model
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    regularization_parameter = 0.001
    loss_function = loss_wrapper_L1reg(autoencoder_model, regularization_parameter)

    # Calculate the combined loss for input data using the loss function
    input_data = rand(784, 100)  # Example: 784-dimensional input data
    combined_loss_value = loss_function(input_data)
    ```

Notes:

    - The `loss_wrapper_L1reg` function creates a closure that encapsulates the autoencoder model and the L1 regularization parameter. It returns a loss function tailored to that specific autoencoder and regularization.

    - The returned loss function accepts input data and calculates the combined loss, which includes both a mean squared error (MSE) reconstruction loss and an L1 regularization term applied to the single-layer encoder's weights.

    - L1 regularization encourages sparsity in the single-layer encoder's weights by shrinking many of the non-informative weights towards zero.

"""
loss_wrapper_L1reg(AE::Autoencoder, α::AbstractFloat) = function(Xt::AbstractMatrix) loss_L1reg(Xt, AE, α) end



#------------------------------
# Loss functions for training corAE (MSE with correlation penalty on the latent dimensions, regularized by sparsity-parameter α):
#------------------------------
"""
    corpen(Z::AbstractMatrix)

Description:

    Calculate the sum of squared pairwise sample (Pearson) correlation coefficients (CORPEN) between columns of a given matrix `Z`. 

Arguments:

    - `Z::AbstractMatrix`: The input matrix for which to compute the CORPEN.

Returns:

    - `corpen_value::AbstractFloat`: The calculated CORPEN value (non-negative).

Example:

    ```julia
    # Calculate CORPEN for a given matrix Z
    Z_matrix = rand(10, 100)  # Example: 10x100 matrix
    corpen_result = corpen(Z_matrix) 
    ```

Notes:

    - CORPEN may be used as an additional loss term in the loss function for training an autoencoder model.

"""
function corpen(Z::AbstractMatrix)

    n, p = size(Z')
    w = 0.0
    u = 0.0
    for i in 1:p
        for j in 1:p
            num = sum((Z[i, :] .- mean(Z[i, :])) .* (Z[j, :] .- mean(Z[j, :])))
            denom = ((n-1) * std(Z[i, :]) * std(Z[j, :]))
            u = w + (num / denom)^2
            w = u
        end
    end

    return u
end


"""
    loss_correg(Xt::AbstractMatrix, AE::Autoencoder, α::AbstractFloat)

Description:

    Calculate a combined loss function for an autoencoder model with correlation-based regularization (CORREG). The loss function includes a mean squared error (MSE) reconstruction loss and a regularization term based on the sum of squared pairwise sample (Pearson) correlation coefficients (CORPEN) between columns of the encoder's output matrix.

Arguments:

    - `Xt::AbstractMatrix`: The original input data for which you want to calculate the loss.
    - `AE::Autoencoder`: The autoencoder model, consisting of an encoder and a decoder.
    - `α::AbstractFloat`: The regularization parameter controlling the strength of the CORREG term. It should be a floating-point number.

Returns:

    - `loss::AbstractFloat`: The combined loss, including the reconstruction loss and CORREG regularization term.

Example:

    ```julia
    # Calculate the combined loss with CORREG regularization for a given autoencoder model and input data
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    input_data = rand(784, 100)  # Example: 784-dimensional input data
    regularization_parameter = 0.001
    loss_value = loss_correg(input_data, autoencoder_model, regularization_parameter)
    ```

Notes:

    - The `loss_correg` function computes a loss function that consists of two components:
        1. A mean squared error (MSE) reconstruction loss that measures the difference between the original input data and the data reconstructed by the autoencoder model.
        2. A CORREG regularization term based on the CORPEN calculated on the representation computed by the encoder.

    - The regularization parameter `α` controls the strength of the CORREG term, influencing the trade-off between reconstruction accuracy and correlation-based regularization.

    - CORREG encourages the encoder's outputs to exhibit certain correlation properties, which can be useful for controlling the representations learned by the autoencoder.

"""
loss_correg(Xt::AbstractMatrix, AE::Autoencoder, α::AbstractFloat) = Flux.mse(AE(Xt), Xt) + α * corpen(AE.encoder(Xt)) 


"""
    loss_wrapper_correg(AE::Autoencoder, α)

Description:

    Create a loss function wrapper for an autoencoder model with correlation-based regularization (CORREG). The wrapper function accepts input data and computes the combined loss, including a mean squared error (MSE) reconstruction loss and a regularization term based on the sum of squared pairwise sample (Pearson) correlation coefficients (CORPEN) between columns of the encoder's output matrix.

Arguments:

    - `AE::Autoencoder`: The autoencoder model, consisting of an encoder and a decoder.
    - `α::AbstractFloat`: The regularization parameter controlling the strength of the CORREG term. It should be a floating-point number.

Returns:

    - `loss_function::Function`: A loss function that can be used to calculate the combined loss with CORREG regularization for a given input data.

Example:

    ```julia
    # Create a loss function wrapper with CORREG regularization for a given autoencoder model
    autoencoder_model = Autoencoder(encoder_network, decoder_network)  # Example autoencoder model
    regularization_parameter = 0.001
    loss_function = loss_wrapper_correg(autoencoder_model, regularization_parameter)

    # Calculate the combined loss with CORREG regularization for input data using the loss function
    input_data = rand(784, 100)  # Example: 784-dimensional input data
    combined_loss_value = loss_function(input_data)
    ```

Notes:

    - The `loss_wrapper_correg` function creates a closure that encapsulates the autoencoder model and the CORREG regularization parameter. It returns a loss function tailored to that specific autoencoder and regularization.

    - The returned loss function accepts input data and calculates the combined loss, including both a mean squared error (MSE) reconstruction loss and a correlation-based regularization term (CORREG) applied to the encoder's outputs.

    - CORREG encourages the encoder's outputs to exhibit certain correlation properties, which can be useful for controlling the representations learned by the autoencoder.

"""
loss_wrapper_correg(AE::Autoencoder, α) = function(Xt::AbstractMatrix) loss_correg(Xt, AE, α) end