import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
import tensorflow_datasets as tfds
from bigdl.nano.tf.keras import Model

def create_datasets(img_size, batch_size):
    if False:
        for i in range(10):
            print('nop')
    ((train_ds, test_ds), info) = tfds.load('imagenette/320px-v2', data_dir='/tmp/data', split=['train', 'validation'], with_info=True, as_supervised=True)
    num_classes = info.features['label'].num_classes

    def preprocessing(img, label):
        if False:
            print('Hello World!')
        return (tf.image.resize(img, (img_size, img_size)), tf.one_hot(label, num_classes))
    train_ds = train_ds.repeat().map(preprocessing).batch(batch_size)
    test_ds = test_ds.map(preprocessing).batch(batch_size)
    return (train_ds, test_ds, info)

def create_model(num_classes, img_size):
    if False:
        for i in range(10):
            print('nop')
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    x = tf.cast(inputs, tf.float32)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    backbone = ResNet50(weights='imagenet')
    backbone.trainable = False
    x = backbone(x)
    x = layers.Dense(512, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
if __name__ == '__main__':
    img_size = 224
    batch_size = 32
    num_epochs = int(os.environ.get('NUM_EPOCHS', 10))
    (train_ds, test_ds, ds_info) = create_datasets(img_size, batch_size)
    num_classes = ds_info.features['label'].num_classes
    steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
    model = create_model(num_classes, img_size)
    model.fit(train_ds, epochs=num_epochs, steps_per_epoch=steps_per_epoch, num_processes=2)