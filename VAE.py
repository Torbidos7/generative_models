import keras
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras import layers

class ResidualBlock(layers.Layer):
    def __init__(self, output_dim, activation="relu", **kwargs) -> None:
        """Standard residual block with pre-activation https://github.com/GianlucaCarlini/latent_diffusion_tf/blob/main/source/blocks.py

        Args:
            output_dim (int): The output channel dimension.
            norm_layer (tf.keras.layers.Layer, optional): The normalization layer
                to be applied befor the activation and the convolution. Defaults to None.
                If None, LayerNormalization is applied.
            activation (str, optional): The activation function for the residual.
                Defaults to "swish".
        """
        super().__init__(**kwargs)

        self.outpud_dim = output_dim

        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)

        self.act1 = tf.keras.activations.get(activation)
        self.conv1 = layers.Conv2D(output_dim, kernel_size=3, padding="same")
        self.act2 = layers.Activation(activation=activation)
        self.conv2 = layers.Conv2D(output_dim, kernel_size=3, padding="same")
        self.proj = layers.Conv2D(output_dim, kernel_size=1)

    def call(self, inputs):
        '''
        This is where the layer's logic lives.
        
        Args:
            inputs (tensor): Input tensor.
            
        Returns:
            tensor: Output tensor.
        '''

        x = self.norm1(inputs)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.outpud_dim != inputs.shape[-1]:
            inputs = self.proj(inputs)

        return x + inputs

    def get_config(self):
        '''
        Returns the config of the layer. This is used for saving and loading from a model
        '''
        config = super().get_config()
        config.update(
            {
                "output_dim": self.outpud_dim,
            }
        )
        return config
    
class VAE(tf.keras.Model):
    """
    Variational Autoencoder (VAE) model for image generation.

    Args:
        image_shape (tuple): Shape of the input image.
        depths (list): List of integers representing the number of filters in each convolutional layer.
        initial_dim (int): Dimension of the residual blocks.
        dense_dim (int): Dimension of the first fully connected layer.
        latent_dim (int): Dimension of the latent space.co

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
        generate_batch(images): Generates a batch of images.
        generate_single(image): Generates a single image.
        generate_embedding(image): Generates an embedding for an image.
        train_step(data): Performs a training step.
        test_step(data): Performs a testing step.
        summary(): Prints a summary of the encoder and decoder models.
    """

    def __init__(self, image_shape, depths, initial_dim, dense_dim, latent_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.depths = depths
        self.initial_dim = initial_dim
        self.dense_dim = dense_dim
        self.latent_dim = latent_dim
        self.embed_dims = [initial_dim * 2**i for i in range(len(depths))]


        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.total_loss_tracker = keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        # self.mse = tf.keras.metrics.MeanSquaredError(name='mse')
        # self.mae = tf.keras.metrics.MeanAbsoluteError(name='mae')
        # self.kld = tf.keras.metrics.KLDivergence(name='kld')

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            # self.mse,
            # self.mae,
            # self.kld
        ]

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

    def build_encoder(self):
        """
        Builds the encoder part of the VAE model.

        Returns:
            keras.Model: Encoder part of the VAE model.
        """
        inputs = Input(shape=self.image_shape)
        x = inputs
        
        for i, depth in enumerate(self.depths):
            for _ in range(depth):

                x = ResidualBlock(output_dim=self.embed_dims[i], activation="swish")(x)

            if i < len(self.depths) - 1:
                x = tf.keras.layers.Conv2D(filters=self.embed_dims[i], kernel_size=2, strides=2, padding="same")(x)
        x = Flatten()(x)
        x = Dense(self.dense_dim, activation='relu')(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        #reparameterization trick
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
        x = Dense(self.dense_dim, activation='relu')(latent_inputs)
        # dimensions must be reshapable to the initial image dimensions
        x = Dense(self.depths[-1] * self.image_shape[0] // (2 ** len(self.depths)) * self.image_shape[1] // (2 ** len(self.depths)), activation='relu')(x)
        x = Reshape((self.image_shape[0] // (2 ** len(self.depths)), self.image_shape[1] // (2 ** len(self.depths)), self.depths[-1]))(x)
        for i, depth in enumerate(self.depths[1:]):
            x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bicubic")(x)
            for _ in range(depth):
                x = ResidualBlock(output_dim=self.embed_dims[i + 1], activation="swish")(x)
        x = Conv2DTranspose(filters=self.image_shape[-1], kernel_size=3, strides=2, padding='same', activation='sigmoid')(x)
        decoder = Model(latent_inputs, x, name='decoder')
        return decoder

    # # #trick to pass los function to add_loss
    # # def reconstruction_loss_carrier():
        
    # def reconstruction_loss(self, inputs, reconstructed, z_mean, z_log_var):
    #     """
    #     Calculates the reconstruction loss.

    #     Args:
    #         inputs (tensor): Input tensor.
    #         reconstructed (tensor): Reconstructed tensor.
    #         z_mean (tensor): Mean tensor of the latent space.
    #         z_log_var (tensor): Log variance tensor of the latent space.

    #     Returns:
    #         tensor: Reconstruction loss tensor.
    #     """
    #     reconstruction_loss = self.mse(inputs, reconstructed)
    #     kl_loss = self.kld(z_mean, z_log_var)#-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    #     return tf.reduce_mean(reconstruction_loss, kl_loss)
    #     # return reconstruction_loss
    

    def call(self, inputs):
        """
        Calls the VAE model.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            tensor: Reconstructed tensor.
        """
        _, _, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # self.add_loss(self.kld(z_mean, z_log_var))
        # self.add_metric(self.mse(inputs, reconstructed), name='mse')
        # self.add_metric(self.mae(inputs, reconstructed), name='mae')
        # self.add_metric(self.kld(z_mean, z_log_var), name='kld')
        return reconstructed


    def train_step(self, data):
        """
        Performs a training step.

        Args:
            data (tensor): Input tensor.

        Returns:
            dict: Dictionary of metric names and results.
        """
        images = data

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(images, training=True)
            reconstructed = self.decoder(z, training=True)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(data, reconstructed))
                )
           
            kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.SUM)(z_mean, z_log_var)
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss))
            loss = reconstruction_loss + kl_loss
            # loss = self.reconstruction_loss_carrier()(images, reconstructed, z_mean, z_log_var)

        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            }

    def test_step(self, data):
        """
        Performs a testing step.

        Args:
            data (tensor): Input tensor.

        Returns:
            dict: Dictionary of metric names and results.
        """
        images = data

        z_mean, z_log_var , z = self.encoder(images, training=False)
        reconstructed = self.decoder(z, training=False)

        reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(data, reconstructed))
                )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss))
        loss = reconstruction_loss + kl_loss

        self.total_loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            }

    def summary(self):
        """
        Prints a summary of the encoder and decoder models.
        """
        self.encoder.summary()
        self.decoder.summary()
        encoder_trainable_count = np.sum([K.count_params(w) for w in self.encoder.trainable_weights], dtype=np.int32)
        decoder_trainable_count = np.sum([K.count_params(w) for w in self.decoder.trainable_weights], dtype=np.int32)
        print('#' * 80)
        print('VAE Summary' + '\n')
        print("Total Parameters:", "{:,}".format(self.encoder.count_params() + self.decoder.count_params()))
        print('Trainable Parameters:', "{:,}".format(encoder_trainable_count + decoder_trainable_count))
        print('Non-Trainable Parameters:', "{:,}".format(self.encoder.count_params() + self.decoder.count_params() - encoder_trainable_count - decoder_trainable_count))        
        print('#' * 80)
       
    def generate_batch(self, images):
        """
        Generates a batch of images.

        Args:
            images (tensor): Input tensor.
          

        Returns:
            tensor: Generated tensor.
        """
        z_mean, _, _ = self.encoder(images)
        generated = self.decoder(z_mean)
        return generated

    def generate_single(self, image):
        """
        Generates a single image.

        Args:
            image (tensor): Input tensor.
          

        Returns:
            tensor: Generated tensor.
        """
        images = tf.expand_dims(image, axis=0)
        z_mean, _, _ = self.encoder(images)
        generated = self.decoder(z_mean)
        return generated

    def generate_embedding(self, image):
        """
        Generates an embedding for an image.

        Args:
            image (tensor): Input tensor.
          

        Returns:
            tensor: Embedding tensor.
        """
        images = tf.expand_dims(image, axis=0)
        _, _, z = self.encoder(images)
        return z
        