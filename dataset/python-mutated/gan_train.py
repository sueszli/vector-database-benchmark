import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy.misc import toimage
import glob
from gan import Generator, Discriminator
from dataset import make_anime_dataset

def save_result(val_out, val_block_size, image_path, color_mode):
    if False:
        for i in range(10):
            print('nop')

    def preprocess(img):
        if False:
            while True:
                i = 10
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img
    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)
            single_row = np.array([])
    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    toimage(final_image).save(image_path)

def celoss_ones(logits):
    if False:
        return 10
    y = tf.ones_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def celoss_zeros(logits):
    if False:
        while True:
            i = 10
    y = tf.zeros_like(logits)
    loss = keras.losses.binary_crossentropy(y, logits, from_logits=True)
    return tf.reduce_mean(loss)

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    if False:
        while True:
            i = 10
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)
    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    loss = d_loss_fake + d_loss_real
    return loss

def g_loss_fn(generator, discriminator, batch_z, is_training):
    if False:
        for i in range(10):
            print('nop')
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)
    return loss

def main():
    if False:
        i = 10
        return i + 15
    tf.random.set_seed(3333)
    np.random.seed(3333)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')
    z_dim = 100
    epochs = 3000000
    batch_size = 64
    learning_rate = 0.0002
    is_training = True
    img_path = glob.glob('C:\\Users\\z390\\Downloads\\anime-faces\\*\\*.jpg') + glob.glob('C:\\Users\\z390\\Downloads\\anime-faces\\*\\*.png')
    print('images num:', len(img_path))
    (dataset, img_shape, _) = make_anime_dataset(img_path, batch_size, resize=64)
    print(dataset, img_shape)
    sample = next(iter(dataset))
    print(sample.shape, tf.reduce_max(sample).numpy(), tf.reduce_min(sample).numpy())
    dataset = dataset.repeat(100)
    db_iter = iter(dataset)
    generator = Generator()
    generator.build(input_shape=(4, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(4, 64, 64, 3))
    g_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    generator.load_weights('generator.ckpt')
    discriminator.load_weights('discriminator.ckpt')
    print('Loaded chpt!!')
    (d_losses, g_losses) = ([], [])
    for epoch in range(epochs):
        for _ in range(1):
            batch_z = tf.random.normal([batch_size, z_dim])
            batch_x = next(db_iter)
            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
            grads = tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))
        batch_z = tf.random.normal([batch_size, z_dim])
        batch_x = next(db_iter)
        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))
        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss:', float(g_loss))
            z = tf.random.normal([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('gan_images', 'gan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')
            d_losses.append(float(d_loss))
            g_losses.append(float(g_loss))
            if epoch % 10000 == 1:
                generator.save_weights('generator.ckpt')
                discriminator.save_weights('discriminator.ckpt')
if __name__ == '__main__':
    main()