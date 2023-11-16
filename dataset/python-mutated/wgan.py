import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Generator(keras.Model):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Generator, self).__init__()
        self.fc = layers.Dense(3 * 3 * 512)
        self.conv1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2DTranspose(3, 4, 3, 'valid')

    def call(self, inputs, training=None):
        if False:
            i = 10
            return i + 15
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x = tf.tanh(x)
        return x

class Discriminator(keras.Model):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Discriminator, self).__init__()
        self.conv1 = layers.Conv2D(64, 5, 3, 'valid')
        self.conv2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)

    def call(self, inputs, training=None):
        if False:
            for i in range(10):
                print('nop')
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

def main():
    if False:
        print('Hello World!')
    d = Discriminator()
    g = Generator()
    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])
    prob = d(x)
    print(prob)
    x_hat = g(z)
    print(x_hat.shape)
if __name__ == '__main__':
    main()