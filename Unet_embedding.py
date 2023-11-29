import tensorflow as tf
from generative_models.latent_diffusion_tf.source.denoiser import get_unet_no_class
from generative_models.latent_diffusion_tf.source.losses import l1_loss
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model






class Unet(tf.keras.models.Model):
    '''
    A  U-Net model for image reconstruction.

        Args:
            input_shape (tuple): The shape of the input image.
            batch_size (int): The batch size for training.
            class_label (int): The class label for the image.
            **kwargs: Additional keyword arguments for the U-Net model.

        Attributes:
            input_shape (tuple): The shape of the input image.
            depths (list): The depths of the U-Net model.
            initial_dim (int): The initial dimension of the U-Net model.
            output_classes (int): The number of output classes.
            image_shape (tuple): The shape of the input image.
            batch_size (int): The batch size for training.
            class_label (int): The class label for the image.
            cUnet (tf.keras.models.Model): The U-Net model.

        Output:
            generated (tf.Tensor): The generated image.
                

    '''
    def __init__(
        self,
        input_shape,
        batch_size=8,
        # class_label=0,
        **kwargs
    ):
        super().__init__()

        self._input_shape = input_shape
        self.depths = kwargs.get("depths", [2, 8, 2])
        self.initial_dim = kwargs.get("initial_dim", 96)
        self.output_classes = kwargs.get("output_classes", 3)
        self.image_shape = kwargs.get("image_shape", input_shape)
        self.batch_size = int(batch_size)
        # self.class_label = class_label
        self.cUnet = get_unet_no_class(
            input_shape=self.image_shape,
            depths=self.depths,
            initial_dim=self.initial_dim,
            # noise=self.class_label,
            output_classes=self.output_classes,
        )

            
        # self.accuracy = tf.keras.metrics.Accuracy(name='accuracy')
        #self.f1score = tf.keras.metrics.F1Score(name='f1-score')
        self.mse = tf.keras.metrics.MeanSquaredError(name='mse')
        self.mae = tf.keras.metrics.MeanAbsoluteError(name='mae')
    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="loss"
        )

    @property
    def metrics(self):
        '''
        A list of metrics for the model.

        Returns:
            list: A list of metrics for the model.

        '''
        return [
            self.reconstruction_loss_tracker,
            # self.accuracy,
            self.mse,
            self.mae
            ]

    def generate_batch_image(self, images):
        '''
        Generates a batch of images from a given batch of images.

        Args:
            images (tf.Tensor): The input images.
           

        Returns:
            generated (tf.Tensor): The generated images.
        '''
        generated = self.cUnet([images], training=False)

        return generated

    def generate_single_image(self, image):
        '''
        Generates an image from a given image.

        Args:
            image (tf.Tensor): The input image.
            class_label (tf.Tensor): The class label.

        Returns:
            generated (tf.Tensor): The generated image.
        '''
        images = tf.expand_dims(image, axis=0)
        # classes = tf.expand_dims(class_label, axis=0)
        generated = self.cUnet([images], training=False)

        return generated
    
    
    def generate_embedding(self, image):
            '''
            Generates an embedding from a given image.

            Args:
                images (tf.Tensor): The input image.
                class_label (tf.Tensor): The class label.

            Returns:
                embedding (tf.Tensor): The generated embedding.
            '''
            # images = tf.expand_dims(image, axis=0)
            # classes = tf.expand_dims(class_label, axis=0)
            layer_name = 'residual_block_13'
            intermediate_layer_model = Model(inputs=self.cUnet.input,
                                                outputs=self.cUnet.get_layer(layer_name).output)
            embedding = intermediate_layer_model.predict([image], batch_size=1, steps=1)
            
            return embedding

    def  generate_batch_embedding(self, images):
            '''
            Generates an embedding from a given image.

            Args:
                images (tf.Tensor): The input image.
                class_label (tf.Tensor): The class label.

            Returns:
                embedding (tf.Tensor): The generated embedding.
            '''
            layer_name = 'residual_block_13'
            intermediate_layer_model = Model(inputs=self.cUnet.input,
                                                outputs=self.cUnet.get_layer(layer_name).output)
            embedding = intermediate_layer_model.predict([images], batch_size=1, steps=1)
            
            return embedding 
    
    def reconstruct_from_image_and_embedding(self, image, embedding):
         '''
         Reconstructs an image from a given image and embedding.
         Using the image for the skip connetions and the embedding for the residual blocks
         
         Args:
             image (tf.Tensor): The input image.
             embedding (tf.Tensor): The embedding.
             
         Returns:
             generated (tf.Tensor): The generated image.
         '''
         images = tf.expand_dims(image, axis=0)

         #create a new model with the same weights as the original model but with the embedding as input
         #and the image as skip connection
         

    def train_step(self, data):
        '''
        A single training step.
        
        Args:
            data (tuple): The training data.
            labels (tf.Tensor): The training labels.

        Returns:
            dict: A dictionary of metrics for the model.

        '''

        #unpack training data
        images, labels = data

        with tf.GradientTape() as tape:

            pred_images = self.cUnet(
                [images], training=True
            )

            image_loss = l1_loss(labels, pred_images)
            

        gradients = tape.gradient(image_loss, self.cUnet.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.cUnet.trainable_weights))

        self.reconstruction_loss_tracker.update_state(image_loss)
        # self.accuracy.update_state(labels, pred_images)
        self.mse.update_state(labels, pred_images)
        self.mae.update_state(labels, pred_images)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        '''
        A single testing step.

        Args:
            data (tuple): The testing data.
            labels (tf.Tensor): The testing labels.

        Returns:
            dict: A dictionary of metrics for the model.
        '''

        images, labels = data

        pred_images = self.cUnet(
            [images], training=False)

        image_loss = l1_loss(labels, pred_images)

        self.reconstruction_loss_tracker.update_state(image_loss)
        # self.accuracy.update_state(labels, pred_images)
        self.mse.update_state(labels, pred_images)
        self.mae.update_state(labels, pred_images)

        #plot image --> error Cannot convert a symbolic tf.Tensor (strided_slice:0) to a numpy array. This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.
        # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
        # ax1.imshow(np.array(images[0, ...])); ax1.axis('off')
        # ax1.set_title('Input Image')
        # ax2.imshow(np.array(labels[0, ...],)); ax2.axis('off')
        # ax2.set_title('Ground Truth')
        # ax3.imshow(np.array(pred_images[0, ...],)); ax3.axis('off')
        # ax3.set_title('Predicted Image')

        return {m.name: m.result() for m in self.metrics}


    def summary(self):
  
        self.cUnet.summary()
        
