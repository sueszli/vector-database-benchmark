import tensorflow as tf
import numpy as np
((x_train, y_train), (x_test, y_test)) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
(dataType, dataShape) = (x_train.dtype, x_train.shape)
print(f'Data type and shape x_train: {dataType} {dataShape}')
(labelType, labelShape) = (y_train.dtype, y_train.shape)
print(f'Data type and shape y_train: {labelType} {labelShape}')
im_list = []
n_samples_to_show = 16
c = 0
for i in range(n_samples_to_show):
    im_list.append(x_train[i])
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
fig = plt.figure(figsize=(4.0, 4.0))
grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)
for (ax, im) in zip(grid, im_list):
    ax.imshow(im[:, :, 0], 'gray')
plt.show()
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)
NUM_CLASSES = 10
model = tf.keras.Sequential([tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)), tf.keras.layers.MaxPooling2D((2, 2)), tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(32, activation='relu'), tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')])
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
accuracy_metric = tf.keras.metrics.Accuracy()

def loss_fn(gt_label, pred):
    if False:
        print('Hello World!')
    return loss_object(y_true=gt_label, y_pred=pred)

def accuracy_fn(gt_label, output):
    if False:
        i = 10
        return i + 15
    pred = tf.argmax(output, axis=1, output_type=tf.int32)
    return accuracy_metric(pred, gt_label)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
NUM_EPOCHS = 5
EPOCH_PER_DISPLAY = 1
total_loss = []
for epoch in range(NUM_EPOCHS):
    running_loss = []
    running_accuracy = []
    for (input, target) in train_dataset:
        with tf.GradientTape() as tape:
            output = model(input, training=True)
            loss_ = loss_fn(target, output)
            accuracy_ = accuracy_fn(target, output)
            grads = tape.gradient(loss_, model.trainable_variables)
        running_loss.append(loss_)
        running_accuracy.append(accuracy_)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss = np.mean(running_loss)
    epoch_accuracy = np.mean(running_accuracy)
    if (epoch + 1) % EPOCH_PER_DISPLAY == 0:
        print('Epoch {}: Loss: {:.4f} Accuracy: {:.2f}%'.format(epoch + 1, epoch_loss, epoch_accuracy * 100))
running_accuracy = []
for (input, gt_label) in test_dataset:
    output = model(input, training=False)
    accuracy_ = accuracy_fn(gt_label, output)
    running_accuracy.append(accuracy_)
print('Test accuracy: {:.3%}'.format(np.mean(running_accuracy)))