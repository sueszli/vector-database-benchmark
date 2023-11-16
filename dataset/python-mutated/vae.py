import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from PIL import Image
from matplotlib import pyplot as plt
tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

def save_images(imgs, name):
    if False:
        for i in range(10):
            print('nop')
    new_im = Image.new('L', (280, 280))
    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1
    new_im.save(name)
h_dim = 20
batchsz = 512
lr = 0.001
((x_train, y_train), (x_test, y_test)) = keras.datasets.fashion_mnist.load_data()
(x_train, x_test) = (x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0)
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
z_dim = 10

class VAE(keras.Model):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(VAE, self).__init__()
        self.fc1 = layers.Dense(128)
        self.fc2 = layers.Dense(z_dim)
        self.fc3 = layers.Dense(z_dim)
        self.fc4 = layers.Dense(128)
        self.fc5 = layers.Dense(784)

    def encoder(self, x):
        if False:
            print('Hello World!')
        h = tf.nn.relu(self.fc1(x))
        mu = self.fc2(h)
        log_var = self.fc3(h)
        return (mu, log_var)

    def decoder(self, z):
        if False:
            return 10
        out = tf.nn.relu(self.fc4(z))
        out = self.fc5(out)
        return out

    def reparameterize(self, mu, log_var):
        if False:
            print('Hello World!')
        eps = tf.random.normal(log_var.shape)
        std = tf.exp(log_var * 0.5)
        z = mu + std * eps
        return z

    def call(self, inputs, training=None):
        if False:
            while True:
                i = 10
        (mu, log_var) = self.encoder(inputs)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return (x_hat, mu, log_var)
model = VAE()
model.build(input_shape=(4, 784))
optimizer = tf.optimizers.Adam(lr)
for epoch in range(1000):
    for (step, x) in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])
        with tf.GradientTape() as tape:
            (x_rec_logits, mu, log_var) = model(x)
            rec_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=x_rec_logits)
            rec_loss = tf.reduce_sum(rec_loss) / x.shape[0]
            kl_div = -0.5 * (log_var + 1 - mu ** 2 - tf.exp(log_var))
            kl_div = tf.reduce_sum(kl_div) / x.shape[0]
            loss = rec_loss + 1.0 * kl_div
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if step % 100 == 0:
            print(epoch, step, 'kl div:', float(kl_div), 'rec loss:', float(rec_loss))
    z = tf.random.normal((batchsz, z_dim))
    logits = model.decoder(z)
    x_hat = tf.sigmoid(logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.0
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/sampled_epoch%d.png' % epoch)
    x = next(iter(test_db))
    x = tf.reshape(x, [-1, 784])
    (x_hat_logits, _, _) = model(x)
    x_hat = tf.sigmoid(x_hat_logits)
    x_hat = tf.reshape(x_hat, [-1, 28, 28]).numpy() * 255.0
    x_hat = x_hat.astype(np.uint8)
    save_images(x_hat, 'vae_images/rec_epoch%d.png' % epoch)