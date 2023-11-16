import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.datasets as datasets
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = ['STKaiti']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    if False:
        i = 10
        return i + 15
    ((x, y), (x_val, y_val)) = datasets.mnist.load_data()
    x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.0
    y = tf.convert_to_tensor(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    x = tf.reshape(x, (-1, 28 * 28))
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    train_dataset = train_dataset.batch(200)
    return train_dataset

def init_paramaters():
    if False:
        while True:
            i = 10
    w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
    b1 = tf.Variable(tf.zeros([256]))
    w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
    b2 = tf.Variable(tf.zeros([128]))
    w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
    b3 = tf.Variable(tf.zeros([10]))
    return (w1, b1, w2, b2, w3, b3)

def train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001):
    if False:
        i = 10
        return i + 15
    for (step, (x, y)) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            h1 = x @ w1 + tf.broadcast_to(b1, (x.shape[0], 256))
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            out = h2 @ w3 + b3
            loss = tf.square(y - out)
            loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        if step % 100 == 0:
            print(epoch, step, 'loss:', loss.numpy())
    return loss.numpy()

def train(epochs):
    if False:
        print('Hello World!')
    losses = []
    train_dataset = load_data()
    (w1, b1, w2, b2, w3, b3) = init_paramaters()
    for epoch in range(epochs):
        loss = train_epoch(epoch, train_dataset, w1, b1, w2, b2, w3, b3, lr=0.001)
        losses.append(loss)
    x = [i for i in range(0, epochs)]
    plt.plot(x, losses, color='blue', marker='s', label='训练')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('MNIST数据集的前向传播训练误差曲线.png')
    plt.close()
if __name__ == '__main__':
    train(epochs=20)