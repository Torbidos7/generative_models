import keras
import tensorflow as tf 
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

class VAE(tf.keras.Model):
    """
    Variational Autoencoder (VAE) model for image generation.

    Args:
        image_shape (tuple): Shape of the input image.
        depths (list): List of integers representing the number of filters in each convolutional layer.
        initial_dim (int): Dimension of the first fully connected layer.
        latent_dim (int): Dimension of the latent space.

    Attributes:
        encoder (keras.Model): Encoder part of the VAE model.
        decoder (keras.Model): Decoder part of the VAE model.
        mse (keras.metrics.MeanSquaredError): Mean squared error metric.
        mae (keras.metrics.MeanAbsoluteError): Mean absolute error metric.
        kld (keras.metrics.KLDivergence): KL divergence metric.

    Methods:
        build_encoder(): Builds the encoder part of the VAE model.
        build_decoder(): Builds the decoder part of the VAE model.
        sampling(args): Samples from the latent space.
        call(inputs): Calls the VAE model.
        reconstruction_loss(inputs, reconstructed, z_mean, z_log_var): Calculates the reconstruction loss.
        generate_batch(images, class_labels): Generates a batch of images.
        generate_single(image, class_label): Generates a single image.
        generate_embedding(image, class_label): Generates an embedding for an image.
        train_step(data): Performs a training step.
        test_step(data): Performs a testing step.
        summary(): Prints a summary of the encoder and decoder models.
    """

    def __init__(self, image_shape, depths, initial_dim, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.depths = depths
        self.initial_dim = initial_dim
        self.latent_dim = latent_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.mse = tf.keras.metrics.MeanSquaredError(name='mse')
        self.mae = tf.keras.metrics.MeanAbsoluteError(name='mae')
        self.kld = tf.keras.metrics.KLDivergence(name='kld')

    def build_encoder(self):
        """
        Builds the encoder part of the VAE model.

        Returns:
            keras.Model: Encoder part of the VAE model.
        """
        inputs = Input(shape=self.image_shape)
        x = inputs
        for depth in self.depths:
            x = Conv2D(depth, 3, strides=2, padding='same', activation='relu')(x)
            x = Conv2D(depth, 3, strides=1, padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(self.initial_dim, activation='relu')(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Lambda(self.sampling)([z_mean, z_log_var])
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        return encoder

    def build_decoder(self):
        """
        Builds the decoder part of the VAE model.

        Returns:
            keras.Model: Decoder part of the VAE model.
        """
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(self.initial_dim, activation='relu')(latent_inputs)
        x = Dense(self.depths[-1] * self.image_shape[0] // (2 ** len(self.depths)) * self.image_shape[1] // (2 ** len(self.depths)), activation='relu')(x)
        x = Reshape((self.image_shape[0] // (2 ** len(self.depths)), self.image_shape[1] // (2 ** len(self.depths)), self.depths[-1]))(x)
        for depth in reversed(self.depths[:-1]):
            x = Conv2DTranspose(depth, 3, strides=2, padding='same', activation='relu')(x)
            x = Conv2DTranspose(depth, 3, strides=1, padding='same', activation='relu')(x)
        x = Conv2DTranspose(self.image_shape[-1], 3, strides=2, padding='same', activation='sigmoid')(x)
        decoder = Model(latent_inputs, x, name='decoder')
        return decoder

    def sampling(self, args):
        """
        Samples from the latent space.

        Args:
            args (list): List of tensors representing the mean and log variance of the latent space.

        Returns:
            tensor: Sampled tensor from the latent space.
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def call(self, inputs):
        """
        Calls the VAE model.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            tensor: Reconstructed tensor.
        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        self.add_loss(self.reconstruction_loss(inputs, reconstructed, z_mean, z_log_var), name='loss')
        self.add_metric(self.mse(inputs, reconstructed), name='mse')
        self.add_metric(self.mae(inputs, reconstructed), name='mae')
        self.add_metric(self.kld(z_mean, z_log_var), name='kld')
        return reconstructed

    def reconstruction_loss(self, inputs, reconstructed, z_mean, z_log_var):
        """
        Calculates the reconstruction loss.

        Args:
            inputs (tensor): Input tensor.
            reconstructed (tensor): Reconstructed tensor.
            z_mean (tensor): Mean tensor of the latent space.
            z_log_var (tensor): Log variance tensor of the latent space.

        Returns:
            tensor: Reconstruction loss tensor.
        """
        reconstruction_loss = self.mse(inputs, reconstructed)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    def generate_batch(self, images, class_labels):
        """
        Generates a batch of images.

        Args:
            images (tensor): Input tensor.
            class_labels (tensor): Class label tensor.

        Returns:
            tensor: Generated tensor.
        """
        z_mean, _, _ = self.encoder(images)
        generated = self.decoder(z_mean)
        return generated

    def generate_single(self, image, class_label):
        """
        Generates a single image.

        Args:
            image (tensor): Input tensor.
            class_label (tensor): Class label tensor.

        Returns:
            tensor: Generated tensor.
        """
        images = tf.expand_dims(image, axis=0)
        z_mean, _, _ = self.encoder(images)
        generated = self.decoder(z_mean)
        return generated

    def generate_embedding(self, image, class_label):
        """
        Generates an embedding for an image.

        Args:
            image (tensor): Input tensor.
            class_label (tensor): Class label tensor.

        Returns:
            tensor: Embedding tensor.
        """
        images = tf.expand_dims(image, axis=0)
        _, _, z = self.encoder(images)
        return z

    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data (tuple): Tuple of input tensor, target tensor, and class label tensor.

        Returns:
            dict: Dictionary of metric names and results.
        """
        images, _, labels = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(images)
            reconstructed = self.decoder(z)
            loss = self.reconstruction_loss(images, reconstructed, z_mean, z_log_var)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.compiled_metrics.update_state(labels, reconstructed)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Performs a testing step.

        Args:
            data (tuple): Tuple of input tensor, target tensor, and class label tensor.

        Returns:
            dict: Dictionary of metric names and results.
        """
        images, _, labels = data

        z_mean, z_log_var, z = self.encoder(images)
        reconstructed = self.decoder(z)
        loss = self.reconstruction_loss(images, reconstructed, z_mean, z_log_var)

        self.compiled_metrics.update_state(labels, reconstructed)
        return {m.name: m.result() for m in self.metrics}

    def summary(self):
        """
        Prints a summary of the encoder and decoder models.
        """
        self.encoder.summary()
        self.decoder.summary()