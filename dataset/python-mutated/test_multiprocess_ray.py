import os
import pathlib
import tensorflow as tf
from tensorflow.keras import layers
from bigdl.nano.tf.keras import Sequential
if 'FTP_URI' in os.environ:
    URI = os.environ['FTP_URI']
    flower_dataset_url = URI + '/BigDL-data/flower_photos.tar.gz'
else:
    flower_dataset_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
(batch_size, img_height, img_width) = (32, 180, 180)

def dataset_generation():
    if False:
        for i in range(10):
            print('nop')
    data_dir = tf.keras.utils.get_file('flower_photos', origin=flower_dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', seed=123, image_size=(img_height, img_width), batch_size=batch_size)
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
    num_classes = len(class_names)
    return (num_classes, train_ds, val_ds)

def model_init(num_classes):
    if False:
        i = 10
        return i + 15
    model = Sequential([layers.Rescaling(1.0 / 255, input_shape=(img_height, img_width, 3)), layers.Conv2D(16, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(32, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Conv2D(64, 3, padding='same', activation='relu'), layers.MaxPooling2D(), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dense(num_classes)])
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    return model

def test_fit_function():
    if False:
        i = 10
        return i + 15
    (num_classes, train_ds, val_ds) = dataset_generation()
    model_multiprocess = model_init(num_classes)
    history_multiprocess = model_multiprocess.fit(train_ds, epochs=1, validation_data=val_ds, num_processes=1, backend='ray')