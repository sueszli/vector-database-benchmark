import os
import time
import psutil
import tensorflow as tf
from tensorflow.python.keras.applications import VGG16
from exp_config import BATCH_SIZE, LERANING_RATE, MONITOR_INTERVAL, NUM_ITERS, random_input_generator
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
vgg = VGG16(weights=None)
info = psutil.virtual_memory()
monitor_interval = MONITOR_INTERVAL
avg_mem_usage = 0
max_mem_usage = 0
count = 0
total_time = 0
num_iter = NUM_ITERS
batch_size = BATCH_SIZE
train_weights = vgg.trainable_variables
optimizer = tf.optimizers.Adam(learning_rate=LERANING_RATE)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gen = random_input_generator(num_iter, batch_size)

def train_step(x_batch, y_batch):
    if False:
        return 10
    with tf.GradientTape() as tape:
        _logits = vgg(x_batch, training=True)
        _loss = loss_object(y_batch, _logits)
    grad = tape.gradient(_loss, train_weights)
    optimizer.apply_gradients(zip(grad, train_weights))
    return _loss
for (idx, data) in enumerate(gen):
    start_time = time.time()
    x_batch = tf.convert_to_tensor(data[0])
    y_batch = tf.convert_to_tensor(data[1])
    loss = train_step(x_batch, y_batch)
    end_time = time.time()
    consume_time = end_time - start_time
    total_time += consume_time
    if idx % monitor_interval == 0:
        cur_usage = psutil.Process(os.getpid()).memory_info().rss
        max_mem_usage = max(cur_usage, max_mem_usage)
        avg_mem_usage += cur_usage
        count += 1
        tf.print('[*] {} iteration: memory usage {:.2f}MB, consume time {:.4f}s, loss {:.4f}'.format(idx, cur_usage / (1024 * 1024), consume_time, loss))
print('consumed time:', total_time)
avg_mem_usage = avg_mem_usage / count / (1024 * 1024)
max_mem_usage = max_mem_usage / (1024 * 1024)
print('average memory usage: {:.2f}MB'.format(avg_mem_usage))
print('maximum memory usage: {:.2f}MB'.format(max_mem_usage))