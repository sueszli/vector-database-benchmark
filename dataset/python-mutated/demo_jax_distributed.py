import os, pprint, collections
os.environ['KERAS_BACKEND'] = 'jax'
pp = pprint.PrettyPrinter()
import jax
import jax.numpy as jnp
import tensorflow as tf
import keras
import numpy as np
from tqdm import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
' Dataset\nClassic MNIST, loaded using tf.data\n'
BATCH_SIZE = 192
((x_train, train_labels), (x_eval, eval_labels)) = keras.datasets.mnist.load_data()
x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
x_eval = np.expand_dims(x_eval, axis=-1).astype(np.float32)
train_data = tf.data.Dataset.from_tensor_slices((x_train, train_labels))
train_data = train_data.shuffle(5000, reshuffle_each_iteration=True)
train_data = train_data.batch(BATCH_SIZE, drop_remainder=True)
train_data = train_data.repeat()
eval_data = tf.data.Dataset.from_tensor_slices((x_eval, eval_labels))
eval_data = eval_data.batch(10000)
STEPS_PER_EPOCH = len(train_labels) // BATCH_SIZE
' Keras model\nSimple but non-trivial model with:\n* Batch Normalization (non-trainable state updated during trainig, different training-time and inference behavior)\n* Dropout (randomness, different training time and inference behavior)\n'

def make_backbone():
    if False:
        i = 10
        return i + 15
    return keras.Sequential([keras.layers.Rescaling(1.0 / 255.0), keras.layers.Conv2D(filters=12, kernel_size=3, padding='same', use_bias=False), keras.layers.BatchNormalization(scale=False, center=True), keras.layers.Activation('relu'), keras.layers.Conv2D(filters=24, kernel_size=6, padding='same', use_bias=False, strides=2), keras.layers.BatchNormalization(scale=False, center=True), keras.layers.Activation('relu'), keras.layers.Conv2D(filters=32, kernel_size=6, padding='same', use_bias=False, strides=2, name='large_k'), keras.layers.BatchNormalization(scale=False, center=True), keras.layers.Activation('relu')], name='backbone')

def make_model():
    if False:
        while True:
            i = 10
    input = keras.Input(shape=[28, 28, 1])
    y = make_backbone()(input)
    y = keras.layers.Flatten()(y)
    y = keras.layers.Dense(200, activation='relu')(y)
    y = keras.layers.Dropout(0.4)(y)
    y = keras.layers.Dense(10, activation='softmax')(y)
    model = keras.Model(inputs=input, outputs=y)
    return model
' JAX-native distribution with a Keras model\nFor now, you have to write a custom training loop for this\nNote: The features required by jax.sharding are not supported by the Colab TPU\nruntime at this time, but are available on Cloud TPU VMs and Kaggle TPU VMs.\n'
if len(jax.local_devices()) < 8:
    raise Exception('This part requires 8 devices to run')
else:
    print('\nIdentified local devices:')
    pp.pprint(jax.local_devices())
model = make_model()
lr = keras.optimizers.schedules.ExponentialDecay(0.01, STEPS_PER_EPOCH, 0.6)
optimizer = keras.optimizers.Adam(lr)
(one_batch, one_batch_labels) = next(iter(train_data))
model.build(one_batch)
optimizer.build(model.trainable_variables)
' Distribution settings\n\n* Sharding the data on the batch axis\n* Replicating all model variables\n\nNote: this implements standard "data parallel" distributed training\n\n* Just for show, sharding the largest convolutional kernel along the\n  "channels" axis 4-ways and replicating 2-ways\n\nNote: this does not reflect a best practice but is intended to show\n      that you can split a very large kernel across multiple devices\n      if you have to\n'
print('\nMostly data-parallel distribution. Data is sharded across devices while the model is replicated. For demo purposes, we split the largest kernel 4-ways (and replicate 2-ways since we have 8 devices).')
devices = mesh_utils.create_device_mesh((8,))
data_mesh = Mesh(devices, axis_names=('batch',))
data_sharding = NamedSharding(data_mesh, P('batch'))
var_mesh = Mesh(devices, axis_names='_')
var_replication = NamedSharding(var_mesh, P())
large_kernel_mesh = Mesh(devices.reshape((-1, 4)), axis_names=(None, 'out_chan'))
large_kernel_sharding = NamedSharding(large_kernel_mesh, P(None, None, None, 'out_chan'))
special_layer_var = model.get_layer('backbone').get_layer('large_k').kernel
non_trainable_variables = jax.device_put(model.non_trainable_variables, var_replication)
optimizer_variables = jax.device_put(optimizer.variables, var_replication)
print_once = True
trainable_variables = model.trainable_variables
for (i, v) in enumerate(trainable_variables):
    if v is special_layer_var:
        sharded_v = jax.device_put(v, large_kernel_sharding)
        trainable_variables[i] = sharded_v
        print('Sharding of convolutional', v.name, v.shape)
        jax.debug.visualize_array_sharding(jnp.reshape(sharded_v, [-1, v.shape[-1]]))
    else:
        replicated_v = jax.device_put(v, var_replication)
        trainable_variables[i] = replicated_v
        if print_once:
            print_once = False
            print('\nSharding of all other model variables (they are replicated)')
            jax.debug.visualize_array_sharding(jnp.reshape(replicated_v, [-1, v.shape[-1]]))
TrainingState = collections.namedtuple('TrainingState', ['trainable_variables', 'non_trainable_variables', 'optimizer_variables'])
device_train_state = TrainingState(trainable_variables=trainable_variables, non_trainable_variables=non_trainable_variables, optimizer_variables=optimizer_variables)
(x, y) = next(iter(train_data))
sharded_x = jax.device_put(x.numpy(), data_sharding)
print('Data sharding')
jax.debug.visualize_array_sharding(jnp.reshape(sharded_x, [-1, 28 * 28]))
loss = keras.losses.SparseCategoricalCrossentropy()

def compute_loss(trainable_variables, non_trainable_variables, x, y):
    if False:
        print('Hello World!')
    (y_pred, updated_non_trainable_variables) = model.stateless_call(trainable_variables, non_trainable_variables, x)
    loss_value = loss(y, y_pred)
    return (loss_value, updated_non_trainable_variables)
compute_gradients = jax.value_and_grad(compute_loss, has_aux=True)

@jax.jit
def train_step(train_state, x, y):
    if False:
        i = 10
        return i + 15
    ((loss_value, non_trainable_variables), grads) = compute_gradients(train_state.trainable_variables, train_state.non_trainable_variables, x, y)
    (trainable_variables, optimizer_variables) = optimizer.stateless_apply(train_state.optimizer_variables, grads, train_state.trainable_variables)
    return (loss_value, TrainingState(trainable_variables, non_trainable_variables, optimizer_variables))
EPOCHS = 5
print('\nTrainig:')
data_iter = iter(train_data)
for epoch in range(EPOCHS):
    for i in tqdm(range(STEPS_PER_EPOCH)):
        (x, y) = next(data_iter)
        sharded_x = jax.device_put(x.numpy(), data_sharding)
        (loss_value, device_train_state) = train_step(device_train_state, sharded_x, y.numpy())
    print('Epoch', epoch, 'loss:', loss_value)
(data, labels) = next(iter(eval_data))
sharded_data = jax.device_put(data.numpy(), data_sharding)

@jax.jit
def predict(data):
    if False:
        while True:
            i = 10
    (predictions, updated_non_trainable_variables) = model.stateless_call(device_train_state.trainable_variables, device_train_state.non_trainable_variables, data)
    return predictions
predictions = predict(sharded_data)
print('\nModel output sharding follows data sharding:')
jax.debug.visualize_array_sharding(predictions)
update = lambda variable, value: variable.assign(value)
jax.tree_map(update, model.trainable_variables, device_train_state.trainable_variables)
jax.tree_map(update, model.non_trainable_variables, device_train_state.non_trainable_variables)
jax.tree_map(update, optimizer.variables, device_train_state.optimizer_variables)
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), metrics=[keras.metrics.SparseCategoricalAccuracy()])
print('\nUpdating model and running an eval:')
(loss, accuracy) = model.evaluate(eval_data)
print('The model achieved an evaluation accuracy of:', accuracy)