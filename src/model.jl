#------------------------------
# This file contains the structures for defining the different autoencoder models:
#------------------------------

#---Linear layer:
"""
    LinearLayer(coeffs::AbstractMatrix)

Description:

    A custom linear layer implemented as a mutable struct. This layer represents a linear transformation that multiplies a matrix of coefficients with the input data.

Fields:

    - `coeffs::AbstractMatrix`: The coefficient matrix used for the linear transformation.

Constructor:

    - `LinearLayer(coeffs::AbstractMatrix)`: Creates a new `LinearLayer` instance with the specified coefficient matrix.

Example:

    ```julia
    # Create a LinearLayer with a coefficient matrix of 2 rows and 3 columns
    coeffs_matrix = rand(2, 3)  # Example: 2x3 coefficient matrix
    linear_layer = LinearLayer(coeffs_matrix)
    ```

Methods:

    - `bl(Xt::AbstractMatrix)`: Apply the linear transformation represented by this layer to the input matrix `Xt`.

Example:

    ```julia
    # Apply the linear layer to input data
    input_data = rand(3, 4)  # Example: 3x4 input data
    result = linear_layer(input_data)
    ```

Notes:

    - This custom linear layer is designed for use in neural networks and can be composed with other layers using Julia's Flux library.

    - The `coeffs` field contains the weight matrix of the linear transformation.

    - When using this layer as part of a neural network, you can update the `coeffs` matrix during training to learn the appropriate weights.

    - The `LinearLayer` struct is compatible with Flux's `@functor` macro for automatic differentiation and gradient-based optimization.

"""
mutable struct LinearLayer
    coeffs
end
(bl::LinearLayer)(Xt) = transpose(bl.coeffs) * Xt
Flux.@functor LinearLayer



#---AE structure:
"""
    struct Autoencoder(encoder, decoder)

Description:

    A custom autoencoder model implemented as a struct. An autoencoder is a type of neural network used for dimensionality reduction and feature learning (representation learning). It consists of an encoder and a decoder, both of which are neural networks.

Fields:

    - `encoder`: The encoder neural network that maps input data to a lower-dimensional representation.
    - `decoder`: The decoder neural network that reconstructs the input data from the lower-dimensional representation.

Constructor:

    - `Autoencoder(encoder, decoder)`: Creates a new `Autoencoder` instance with the specified encoder and decoder neural networks.

Example:

    ```julia
    # Create an Autoencoder with custom encoder and decoder networks
    encoder_network = Chain(Dense(784, 128, relu), Dense(128, 64, relu))
    decoder_network = Chain(Dense(64, 128, relu), Dense(128, 784, sigmoid))
    ae_model = Autoencoder(encoder_network, decoder_network)
    ```

Methods:

    - `(AE::Autoencoder)(Xt::AbstractMatrix)`: Apply the autoencoder model to input data `Xt`. It first encodes the input data using the encoder and then decodes the encoded representation using the decoder.

Example:

    ```julia
    # Apply the autoencoder to input data
    input_data = rand(784, 100)  # Example: 784-dimensional input data
    reconstructed_data = ae_model(input_data)
    ```

Notes:

    - Autoencoders are commonly used for tasks such as dimensionality reduction, feature learning, and data denoising.

    - The `encoder` and `decoder` fields should be constructed using neural network layers compatible with Julia's Flux library.

    - When using this autoencoder for specific tasks, you may need to fine-tune the encoder and decoder networks and train the model on your dataset.

"""
struct Autoencoder
    encoder
    decoder
end
(AE::Autoencoder)(Xt) = AE.decoder(AE.encoder(Xt))
Flux.@functor Autoencoder