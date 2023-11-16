import numpy as np
import tensorflow as tf

def mnist_dataset(batch_size):
    if False:
        for i in range(10):
            print('nop')
    ((x_train, y_train), _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / np.float32(255)
    y_train = y_train.astype(np.int64)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
    return train_dataset

def build_and_compile_cnn_model():
    if False:
        return 10
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(28, 28)), tf.keras.layers.Reshape(target_shape=(28, 28, 1)), tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dense(10)])
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=['accuracy'])
    return model

def train_func():
    if False:
        return 10
    batch_size = 64
    single_worker_dataset = mnist_dataset(batch_size)
    single_worker_model = build_and_compile_cnn_model()
    single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)
import json
import os

def train_func_distributed():
    if False:
        print('Hello World!')
    per_worker_batch_size = 64
    tf_config = json.loads(os.environ['TF_CONFIG'])
    num_workers = len(tf_config['cluster']['worker'])
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    global_batch_size = per_worker_batch_size * num_workers
    multi_worker_dataset = mnist_dataset(global_batch_size)
    with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()
    multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
if __name__ == '__main__':
    train_func()
    from ray.train.tensorflow import TensorflowTrainer
    from ray.train import ScalingConfig
    use_gpu = False
    trainer = TensorflowTrainer(train_func_distributed, scaling_config=ScalingConfig(num_workers=4, use_gpu=use_gpu))
    trainer.fit()