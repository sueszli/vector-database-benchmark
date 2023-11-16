"""
Title: WGAN-GP overriding `Model.train_step`
Author: [A_K_Nain](https://twitter.com/A_K_Nain)
Date created: 2020/05/9
Last modified: 2020/05/9
Description: Implementation of Wasserstein GAN with Gradient Penalty.
Accelerator: GPU
"""
'\n## Wasserstein GAN (WGAN) with Gradient Penalty (GP)\n\nThe original [Wasserstein GAN](https://arxiv.org/abs/1701.07875) leverages the\nWasserstein distance to produce a value function that has better theoretical\nproperties than the value function used in the original GAN paper. WGAN requires\nthat the discriminator (aka the critic) lie within the space of 1-Lipschitz\nfunctions. The authors proposed the idea of weight clipping to achieve this\nconstraint. Though weight clipping works, it can be a problematic way to enforce\n1-Lipschitz constraint and can cause undesirable behavior, e.g. a very deep WGAN\ndiscriminator (critic) often fails to converge.\n\nThe [WGAN-GP](https://arxiv.org/abs/1704.00028) method proposes an\nalternative to weight clipping to ensure smooth training. Instead of clipping\nthe weights, the authors proposed a "gradient penalty" by adding a loss term\nthat keeps the L2 norm of the discriminator gradients close to 1.\n'
'\n## Setup\n'
import tensorflow as tf
import keras
from keras import layers
'\n## Prepare the Fashion-MNIST data\n\nTo demonstrate how to train WGAN-GP, we will be using the\n[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Each\nsample in this dataset is a 28x28 grayscale image associated with a label from\n10 classes (e.g. trouser, pullover, sneaker, etc.)\n'
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 512
noise_dim = 128
fashion_mnist = keras.datasets.fashion_mnist
((train_images, train_labels), (test_images, test_labels)) = fashion_mnist.load_data()
print(f'Number of examples: {len(train_images)}')
print(f'Shape of the images in the dataset: {train_images.shape[1:]}')
train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE).astype('float32')
train_images = (train_images - 127.5) / 127.5
'\n## Create the discriminator (the critic in the original WGAN)\n\nThe samples in the dataset have a (28, 28, 1) shape. Because we will be\nusing strided convolutions, this can result in a shape with odd dimensions.\nFor example,\n`(28, 28) -> Conv_s2 -> (14, 14) -> Conv_s2 -> (7, 7) -> Conv_s2 ->(3, 3)`.\n\nWhile peforming upsampling in the generator part of the network, we won\'t get\nthe same input shape as the original images if we aren\'t careful. To avoid this,\nwe will do something much simpler:\n- In the discriminator: "zero pad" the input to change the shape to `(32, 32, 1)`\nfor each sample; and\n- Ihe generator: crop the final output to match the shape with input shape.\n'

def conv_block(x, filters, activation, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True, use_bn=False, use_dropout=False, drop_value=0.5):
    if False:
        for i in range(10):
            print('nop')
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

def get_discriminator_model():
    if False:
        print('Hello World!')
    img_input = layers.Input(shape=IMG_SHAPE)
    x = layers.ZeroPadding2D((2, 2))(img_input)
    x = conv_block(x, 64, kernel_size=(5, 5), strides=(2, 2), use_bn=False, use_bias=True, activation=layers.LeakyReLU(0.2), use_dropout=False, drop_value=0.3)
    x = conv_block(x, 128, kernel_size=(5, 5), strides=(2, 2), use_bn=False, activation=layers.LeakyReLU(0.2), use_bias=True, use_dropout=True, drop_value=0.3)
    x = conv_block(x, 256, kernel_size=(5, 5), strides=(2, 2), use_bn=False, activation=layers.LeakyReLU(0.2), use_bias=True, use_dropout=True, drop_value=0.3)
    x = conv_block(x, 512, kernel_size=(5, 5), strides=(2, 2), use_bn=False, activation=layers.LeakyReLU(0.2), use_bias=True, use_dropout=False, drop_value=0.3)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)
    d_model = keras.models.Model(img_input, x, name='discriminator')
    return d_model
d_model = get_discriminator_model()
d_model.summary()
'\n## Create the generator\n'

def upsample_block(x, filters, activation, kernel_size=(3, 3), strides=(1, 1), up_size=(2, 2), padding='same', use_bn=False, use_bias=True, use_dropout=False, drop_value=0.3):
    if False:
        while True:
            i = 10
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias)(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x

def get_generator_model():
    if False:
        for i in range(10):
            print('nop')
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(4 * 4 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((4, 4, 256))(x)
    x = upsample_block(x, 128, layers.LeakyReLU(0.2), strides=(1, 1), use_bias=False, use_bn=True, padding='same', use_dropout=False)
    x = upsample_block(x, 64, layers.LeakyReLU(0.2), strides=(1, 1), use_bias=False, use_bn=True, padding='same', use_dropout=False)
    x = upsample_block(x, 1, layers.Activation('tanh'), strides=(1, 1), use_bias=False, use_bn=True)
    x = layers.Cropping2D((2, 2))(x)
    g_model = keras.models.Model(noise, x, name='generator')
    return g_model
g_model = get_generator_model()
g_model.summary()
"\n## Create the WGAN-GP model\n\nNow that we have defined our generator and discriminator, it's time to implement\nthe WGAN-GP model. We will also override the `train_step` for training.\n"

class WGAN(keras.Model):

    def __init__(self, discriminator, generator, latent_dim, discriminator_extra_steps=3, gp_weight=10.0):
        if False:
            while True:
                i = 10
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        if False:
            for i in range(10):
                print('nop')
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        if False:
            for i in range(10):
                print('nop')
        'Calculates the gradient penalty.\n\n        This loss is calculated on an interpolated image\n        and added to the discriminator loss.\n        '
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training=True)
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if False:
            i = 10
            return i + 15
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        for i in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                d_loss = d_cost + gp * self.gp_weight
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.discriminator(generated_images, training=True)
            g_loss = self.g_loss_fn(gen_img_logits)
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        return {'d_loss': d_loss, 'g_loss': g_loss}
'\n## Create a Keras callback that periodically saves generated images\n'

class GANMonitor(keras.callbacks.Callback):

    def __init__(self, num_img=6, latent_dim=128):
        if False:
            return 10
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        if False:
            return 10
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = generated_images * 127.5 + 127.5
        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.utils.array_to_img(img)
            img.save('generated_img_{i}_{epoch}.png'.format(i=i, epoch=epoch))
'\n## Train the end-to-end model\n'
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

def discriminator_loss(real_img, fake_img):
    if False:
        return 10
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss

def generator_loss(fake_img):
    if False:
        i = 10
        return i + 15
    return -tf.reduce_mean(fake_img)
epochs = 20
cbk = GANMonitor(num_img=3, latent_dim=noise_dim)
wgan = WGAN(discriminator=d_model, generator=g_model, latent_dim=noise_dim, discriminator_extra_steps=3)
wgan.compile(d_optimizer=discriminator_optimizer, g_optimizer=generator_optimizer, g_loss_fn=generator_loss, d_loss_fn=discriminator_loss)
wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
'\nDisplay the last generated images:\n'
from IPython.display import Image, display
display(Image('generated_img_0_19.png'))
display(Image('generated_img_1_19.png'))
display(Image('generated_img_2_19.png'))
'\nExample available on HuggingFace.\n\n| Trained Model | Demo |\n| :--: | :--: |\n| [![Generic badge](https://img.shields.io/badge/🤗%20Model-WGAN%20GP-black.svg)](https://huggingface.co/keras-io/WGAN-GP) | [![Generic badge](https://img.shields.io/badge/🤗%20Spaces-WGAN%20GP-black.svg)](https://huggingface.co/spaces/keras-io/WGAN-GP) |\n'