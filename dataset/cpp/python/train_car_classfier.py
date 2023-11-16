'''
This script trains a CNN classifier for predicting the viewpoint of a car.
'''
import numpy as np
import pandas
from keras.applications import InceptionV3
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing import image as image_utils
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

# Some image augmentation and normalization for the training images.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    '/home/ubuntu/csc420_data/segmented_cars',  # this is the target directory
    target_size=(128, 128),  # all images will be resized to 128x128
    batch_size=32,
    class_mode='categorical')

input_tensor = Input(shape=(128, 128, 3))

# We use the InceptionV3 /GoogLeNet model but retrain it to classify out datset. 
base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
# Add a global spatial average pooling layer.
x = base_model.output
x = GlobalAveragePooling2D()(x)
# Add a fully-connected layer.
x = Dense(1024, activation='relu')(x)
# Add a HEAVY dropout.
x = Dropout(0.7)(x)
predictions = Dense(12, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,
                    samples_per_epoch=20000, nb_epoch=10, callbacks=[TensorBoard(log_dir='/tmp/attempt1')])

# Save the top-only trained model.
model.save('object_classifierfit1.h5')

for i, layer in enumerate(base_model.layers):
    print(i, layer.name, "Input: ", layer.input_shape, "Output: ", layer.output_shape)


# Unlcok the "bottom" of the model and lock the "top"
for layer in model.layers[:172]:
    layer.trainable = False
for layer in model.layers[172:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit_generator(train_generator,
                    samples_per_epoch=20000, nb_epoch=50)
model.save('object_classifierfit2.h5')  # Save the final model.

