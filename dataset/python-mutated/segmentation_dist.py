from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow_examples.models.pix2pix import pix2pix
import json
import os
import tensorflow_datasets as tfds
import tensorflow as tf
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
tf_config = json.loads(os.environ.get('TF_CONFIG'))
print('tf_config = ', tf_config)
print("I'm {}:{}".format(tf_config['task']['type'], tf_config['task']['index']))
(dataset, info) = tfds.load('oxford_iiit_pet:3.2.0', with_info=True)

def normalize(input_image, input_mask):
    if False:
        i = 10
        return i + 15
    input_image = tf.cast(input_image, tf.float32) / 128.0 - 1
    input_mask -= 1
    return (input_image, input_mask)

@tf.function
def load_image_train(datapoint):
    if False:
        print('Hello World!')
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    (input_image, input_mask) = normalize(input_image, input_mask)
    return (input_image, input_mask)

def load_image_test(datapoint):
    if False:
        i = 10
        return i + 15
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))
    (input_image, input_mask) = normalize(input_image, input_mask)
    return (input_image, input_mask)
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = dataset['test'].map(load_image_test)
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)
OUTPUT_CHANNELS = 3
with strategy.scope():
    base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)
    layer_names = ['block_1_expand_relu', 'block_3_expand_relu', 'block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']
    layers = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    down_stack.trainable = False
    up_stack = [pix2pix.upsample(512, 3), pix2pix.upsample(256, 3), pix2pix.upsample(128, 3), pix2pix.upsample(64, 3)]

    def unet_model(output_channels):
        if False:
            for i in range(10):
                print('nop')
        last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same', activation='softmax')
        inputs = tf.keras.layers.Input(shape=[128, 128, 3])
        x = inputs
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])
        for (up, skip) in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)
    model = unet_model(OUTPUT_CHANNELS)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
EPOCHS = 1
VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
model_history = model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH, validation_steps=VALIDATION_STEPS, validation_data=test_dataset)
if tf_config['task']['index'] == 0:
    model.save_weights('keras_weights', save_format='h5')