
# Generative Models

This repo contains personal implementations of image generation models built with Tensorflow and Keras framework.
The models take images as input and produce generated images as output, this could be useful for conditional generation as well as image embedding.

The proposed models are:
- conditional U-Net ([cUnet](#cUnet));
- variational auto-encoder ([VAE](#VAE));

## cUnet

The `conditionalUnet` class is an implementation of a conditional U-Net model for image reconstruction using TensorFlow and Keras. This model takes a 1024x1024 pixel image as input and generates reconstructed images. It can condition the generation on a class label. The following sections describe the class and its functionalities.

### Class Overview

The `conditionalUnet` class is designed for image reconstruction tasks, with the ability to condition the reconstruction on a specified class label. It utilizes a U-Net architecture for image generation.

#### Class Constructor

```python
conditionalUnet(
    input_shape,
    batch_size=8,
    class_label=0,
    **kwargs
)
```

- input_shape: The shape of the input image.
- batch_size: The batch size for training (default is 8).
- class_label: The class label for conditioning the image generation (default is 0).
- **kwargs: Additional keyword arguments for configuring the U-Net model.

#### Main Methods

```python
generate_batch(images, class_labels)
```
Generates a **batch of images** from a batch of input images and class labels. The method returns the generated images.

```python
generate_single(image, class_label)
```
Generates an **image** from a single input image and a class label. The method returns the generated image.

```python
generate_embedding(image, class_label)
```
Generates an **embedding** from a given input image and class label. The method returns the generated embedding.

### Usage

To use the `conditionalUnet` class, you can create an instance of the class, compile it, and then train and evaluate it with your image data and labels. The class allows you to generate images, embeddings, and view model summaries.

**Example Usage:**
```python
# Create a conditional U-Net model
model = conditionalUnet(input_shape=(1024, 1024, 3), batch_size=8, class_label=0)

# Compile the model with optimizer and loss function
model.compile(optimizer="adam", loss="mse")

# Train the model using training data
model.fit(train_data, epochs=10)

# Evaluate the model using testing data
model.evaluate(test_data)

# Generate images
generated_image = model.generate_single(input_image, class_label)

# Generate embeddings
embedding = model.generate_embedding(input_image, class_label)

# View model summary
model.summary()
```

### Acknowledgements

For the Unet model behind the cUnet class, I thank [GianlucaCarlini](https://github.com/GianlucaCarlini). In particular check out its repo  [latent_diffusion_tf](https://github.com/GianlucaCarlini/latent_diffusion_tf) for model blocks and architectures.
## VAE


The `VAE` class is an implementation of a Variational Autoencoder model for image generation using TensorFlow and Keras. This model is designed for generating images and embedding from input images. The following sections describe the class and its functionalities.


### Class Overview

The `VAE` class is a Variational Autoencoder model with an encoder and decoder. It can be used for image reconstruction and generation.

#### Class Constructor

```python
VAE(
    image_shape,
    depths,
    initial_dim,
    latent_dim,
    **kwargs
)
```

- image_shape: The shape of the input image.
- depths: A list of integers representing the number of filters in each convolutional layer.
- initial_dim: The dimension of the first fully connected layer.
- latent_dim: The dimension of the latent space.
- **kwargs: Additional keyword arguments for configuring the U-Net model.

#### Main Methods

```python
generate_batch(images)
```
Generates a **batch of images** from a batch of input images. The method returns the generated images.

```python
generate_single(image)
```
Generates an **image** from a single input image. The method returns the generated image.

```python
generate_embedding(image)
```
Generates an **embedding** from a given input image. The method returns the generated embedding.

### Usage

To use the `VAE` class, you can create an instance of the class, compile it, and then train and evaluate it with your image data and labels. The class allows you to generate images, embeddings, and view model summaries.

**Example Usage:**
```python
# Create a VAE model
model = VAE(image_shape=(64, 64, 3), depths=[32, 64], initial_dim=128, latent_dim=16)

# Compile the model with optimizer and loss function
model.compile(optimizer="adam", loss="mse")

# Train the model using training data
model.fit(train_data, epochs=10)

# Evaluate the model using testing data
model.evaluate(test_data)

# Generate images
generated_image = model.generate_single(input_image)

# Generate embeddings
embedding = model.generate_embedding(input_image)

# View model summary
model.summary()

```
## Bugs and feature requests

Have a bug or a feature request? Please first read and search for existing and closed issues. If your problem or idea is not addressed yet, [please open a new issue](https://github.com/Torbidos7/PetWound/issues/new).
## Authors

- [@Torbidos7](https://github.com/Torbidos7)

## Thanks

Thank you for coming :stuck_out_tongue_closed_eyes:

## Copyright and license

Code and documentation copyright 2011-2018 the authors. Code released under the [MIT License](https://github.com/Torbidos7/PetWound//blob/master/LICENSE).

