import tensorflow as tf_keras
from tf_keras.datasets import fashion_mnist
from tf_keras.models import Sequential
from tf_keras import layers
from tf_keras.models import Model
from tf_keras.optimizers import Adam
import numpy as np
from matplotlib import pyplot as plt
(train_images, train_labels), (_, _) = fashion_mnist.load_data()

# plt.figure()
# plt.imshow(train_images[0], cmap='gray')
# plt.show()


# The data loaded is in the shape of (60000, 28, 28) since it’s grayscale. So we need to add the 4th dimension for the channel as 1,
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
# We normalize the input image to the range of [-1, 1] because the generator’s final layer activation uses tanh as mentioned earlier.
train_images = (train_images - 127.5) / 127.5

def build_generator():
    # latent dimension of the random noise
    LATENT_DIM = 100
    # weight initializer for G per DCGAN paper
    WEIGHT_INIT = tf_keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    # number of channels, 1 for gray scale and 3 for color images
    CHANNELS = 1
    model = Sequential(name='generator')
    model.add(layers.Dense(7 * 7 * 256, input_dim=LATENT_DIM))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((7, 7, 256)))
    # upsample to 14x14: apply a transposed CONV => BN => RELU
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization())
    model.add((layers.ReLU()))
    # upsample to 28x28: apply a transposed CONV => BN => RELU
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding="same", kernel_initializer=WEIGHT_INIT))
    model.add(layers.BatchNormalization())
    model.add((layers.ReLU()))
    model.add(layers.Conv2D(CHANNELS, (5, 5), padding="same", activation="tanh"))
    model.summary()

def normal_distribution():
    # Parameters for the normal distribution
    mean = 170  # Mean height in cm
    stddev = 10  # Standard deviation in cm

    # Generate data
    x = np.linspace(mean - 3 * stddev, mean + 3 * stddev, 1000)
    y = (1 / (stddev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / stddev) ** 2)

    # Plot
    plt.plot(x, y)
    plt.title('Normal Distribution of Heights')
    plt.xlabel('Height (cm)')
    plt.ylabel('Probability Density')
    plt.show()

def build_discriminator(width, height, depth, alpha=0.2):
    model = Sequential(name='discriminator')
    input_shape = (height, width, depth)
    # first set of CONV => BN => leaky ReLU layers
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same",input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))
    # second set of CONV => BN => leacy ReLU layers
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=alpha))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.summary()

class DCGAN():
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_metric = tf_keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf_keras.metrics.Mean(name="g_loss")

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def train_step(self, real_images):

        batch_size = tf_keras.shape(real_images)[0]
        noise = tf_keras.random.normal(shape=(batch_size, self.latent_dim))

        # Step 1. Train the discriminator with both real images (label as 1) and fake images (classified as label as 0)
        with tf_keras.GradientTape() as tape:
            # Compute discriminator loss on real images
            pred_real = self.discriminator(real_images, training=True)
            d_loss_real = self.loss_fn(tf_keras.ones((batch_size, 1)), pred_real)

            # Compute discriminator loss on fake images
            fake_images = self.generator(noise)
            pred_fake = self.discriminator(fake_images, training=True)
            d_loss_fake = self.loss_fn(tf_keras.zeros((batch_size, 1)), pred_fake)

            # total discriminator loss
            d_loss = (d_loss_real + d_loss_fake)/2
        # Compute discriminator gradients
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        # Update discriminator weights
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Step 2. Train the generator (do not update weights of the discriminator)
        # G wants D to think the fake images are real (label as 1)
        misleading_labels = tf_keras.ones((batch_size, 1))

        with tf_keras.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            pred_fake = self.discriminator(fake_images, training=True)
            g_loss = self.loss_fn(misleading_labels, pred_fake)
        # Compute generator gradients
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update generator weights
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {"d_loss": self.d_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

class GANMonitor(tf_keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=100):
        self.num_img = num_img
        self.latent_dim = latent_dim

        # Create random noise seed for visualization during training
        self.seed = tf.random.normal([16, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        # random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        # generated_images = self.model.generator(random_latent_vectors)
        generated_images = self.model.generator(self.seed)
        generated_images = (generated_images * 127.5) + 127.5
        generated_images.numpy()

        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_img):
            plt.subplot(4, 4, i+1)
            img = tf_keras.utils.array_to_img(generated_images[i])
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.savefig('epoch_{:03d}.png'.format(epoch))
        plt.show()

    def on_train_end(self, logs=None):
        self.model.generator.save('generator.h5')

dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

if __name__ == "__main__":
    build_generator()
