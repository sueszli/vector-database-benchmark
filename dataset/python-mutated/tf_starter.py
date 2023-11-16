import ray
import tensorflow as tf
from ray import train
from ray.train import ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
from ray.train.tensorflow.keras import ReportCheckpointCallback
use_gpu = False
a = 5
b = 10
size = 100

def build_model() -> tf.keras.Model:
    if False:
        return 10
    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=()), tf.keras.layers.Flatten(), tf.keras.layers.Dense(10), tf.keras.layers.Dense(1)])
    return model

def train_func(config: dict):
    if False:
        return 10
    batch_size = config.get('batch_size', 64)
    epochs = config.get('epochs', 3)
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        multi_worker_model = build_model()
        multi_worker_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=config.get('lr', 0.001)), loss=tf.keras.losses.mean_squared_error, metrics=[tf.keras.metrics.mean_squared_error])
    dataset = train.get_dataset_shard('train')
    results = []
    for _ in range(epochs):
        tf_dataset = dataset.to_tf(feature_columns='x', label_columns='y', batch_size=batch_size)
        history = multi_worker_model.fit(tf_dataset, callbacks=[ReportCheckpointCallback()])
        results.append(history.history)
    return results
config = {'lr': 0.001, 'batch_size': 32, 'epochs': 4}
train_dataset = ray.data.from_items([{'x': x / 200, 'y': 2 * x / 200} for x in range(200)])
scaling_config = ScalingConfig(num_workers=2, use_gpu=use_gpu)
trainer = TensorflowTrainer(train_loop_per_worker=train_func, train_loop_config=config, scaling_config=scaling_config, datasets={'train': train_dataset})
result = trainer.fit()
print(result.metrics)