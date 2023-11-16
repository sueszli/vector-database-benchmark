import os
import tensorflow.compat.v1 as tf
from tensorflow import keras
zip_file = tf.keras.utils.get_file(origin='https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip', fname='cats_and_dogs_filtered.zip', extract=True)
(base_dir, _) = os.path.splitext(zip_file)
test_dir = os.path.join(base_dir, 'validation')
image_size = 160
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(image_size, image_size), batch_size=1, class_mode='binary')

def convert_to_ndarray(ImageGenerator):
    if False:
        return 10
    return ImageGenerator.next()[0]
model = tf.keras.models.load_model('path/to/model')
max_length = test_generator.__len__()
for i in range(max_length):
    test_input = convert_to_ndarray(test_generator)
    prediction = model.predict(test_input)