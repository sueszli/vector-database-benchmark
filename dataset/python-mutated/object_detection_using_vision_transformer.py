"""
Title: Object detection with Vision Transformers
Author: [Karan V. Dave](https://www.linkedin.com/in/karan-dave-811413164/)
Converted to Keras 3 by: [Gabriel Rasskin](https://github.com/grasskin), [Soumik Rakshit](http://github.com/soumik12345)
Date created: 2022/03/27
Last modified: 2022/03/27
Description: A simple Keras implementation of object detection using Vision Transformers.
Accelerator: GPU
"""
'\n## Introduction\n\nThe article\n[Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)\narchitecture by Alexey Dosovitskiy et al.\ndemonstrates that a pure transformer applied directly to sequences of image\npatches can perform well on object detection tasks.\n\nIn this Keras example, we implement an object detection ViT\nand we train it on the\n[Caltech 101 dataset](http://www.vision.caltech.edu/datasets/)\nto detect an airplane in the given image.\n\nThis example requires TensorFlow 2.4 or higher, and\n[TensorFlow Addons](https://www.tensorflow.org/addons/overview),\nfrom which we import the `AdamW` optimizer.\n\nTensorFlow Addons can be installed via the following command:\n\n```\npip install -U git+https://github.com/keras-team/keras\n```\n'
'\n## Imports and setup\n'
import os
os.environ['KERAS_BACKEND'] = 'jax'
import numpy as np
import keras
from keras import layers
from keras import ops
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import scipy.io
import shutil
'\n## Prepare dataset\n\nWe use the [Caltech 101 Dataset](https://data.caltech.edu/records/mzrjq-6wc02).\n'
path_images = '/101_ObjectCategories/airplanes/'
path_annot = '/Annotations/Airplanes_Side_2/'
path_to_downloaded_file = keras.utils.get_file(fname='caltech_101_zipped', origin='https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip', extract=True, archive_format='zip', cache_dir='/')
shutil.unpack_archive('/datasets/caltech-101/101_ObjectCategories.tar.gz', '/')
shutil.unpack_archive('/datasets/caltech-101/Annotations.tar', '/')
image_paths = [f for f in os.listdir(path_images) if os.path.isfile(os.path.join(path_images, f))]
annot_paths = [f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))]
image_paths.sort()
annot_paths.sort()
image_size = 224
(images, targets) = ([], [])
for i in range(0, len(annot_paths)):
    annot = scipy.io.loadmat(path_annot + annot_paths[i])['box_coord'][0]
    (top_left_x, top_left_y) = (annot[2], annot[0])
    (bottom_right_x, bottom_right_y) = (annot[3], annot[1])
    image = keras.utils.load_img(path_images + image_paths[i])
    (w, h) = image.size[:2]
    if i < int(len(annot_paths) * 0.8):
        image = image.resize((image_size, image_size))
    images.append(keras.utils.img_to_array(image))
    targets.append((float(top_left_x) / w, float(top_left_y) / h, float(bottom_right_x) / w, float(bottom_right_y) / h))
(x_train, y_train) = (np.asarray(images[:int(len(images) * 0.8)]), np.asarray(targets[:int(len(targets) * 0.8)]))
(x_test, y_test) = (np.asarray(images[int(len(images) * 0.8):]), np.asarray(targets[int(len(targets) * 0.8):]))
'\n## Implement multilayer-perceptron (MLP)\n\nWe use the code from the Keras example\n[Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/)\nas a reference.\n'

def mlp(x, hidden_units, dropout_rate):
    if False:
        return 10
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
'\n## Implement the patch creation layer\n'

class Patches(layers.Layer):

    def __init__(self, patch_size):
        if False:
            while True:
                i = 10
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        if False:
            print('Hello World!')
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(patches, (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))
        return patches

    def get_config(self):
        if False:
            i = 10
            return i + 15
        config = super().get_config()
        config.update({'patch_size': self.patch_size})
        return config
'\n## Display patches for an input image\n'
patch_size = 32
plt.figure(figsize=(4, 4))
plt.imshow(x_train[0].astype('uint8'))
plt.axis('off')
patches = Patches(patch_size)(np.expand_dims(x_train[0], axis=0))
print(f'Image size: {image_size} X {image_size}')
print(f'Patch size: {patch_size} X {patch_size}')
print(f'{patches.shape[1]} patches per image \n{patches.shape[-1]} elements per patch')
n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for (i, patch) in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = ops.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(ops.convert_to_numpy(patch_img).astype('uint8'))
    plt.axis('off')
'\n## Implement the patch encoding layer\n\nThe `PatchEncoder` layer linearly transforms a patch by projecting it into a\nvector of size `projection_dim`. It also adds a learnable position\nembedding to the projected vector.\n'

class PatchEncoder(layers.Layer):

    def __init__(self, num_patches, projection_dim):
        if False:
            print('Hello World!')
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        config = super().get_config().copy()
        config.update({'input_shape': input_shape, 'patch_size': patch_size, 'num_patches': num_patches, 'projection_dim': projection_dim, 'num_heads': num_heads, 'transformer_units': transformer_units, 'transformer_layers': transformer_layers, 'mlp_head_units': mlp_head_units})
        return config

    def call(self, patch):
        if False:
            print('Hello World!')
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
'\n## Build the ViT model\n\nThe ViT model has multiple Transformer blocks.\nThe `MultiHeadAttention` layer is used for self-attention,\napplied to the sequence of image patches. The encoded patches (skip connection)\nand self-attention layer outputs are normalized and fed into a\nmultilayer perceptron (MLP).\nThe model outputs four dimensions representing\nthe bounding box coordinates of an object.\n'

def create_vit_object_detector(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units):
    if False:
        for i in range(10):
            print('nop')
    inputs = keras.Input(shape=input_shape)
    patches = Patches(patch_size)(inputs)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-06)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-06)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-06)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.3)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.3)
    bounding_box = layers.Dense(4)(features)
    return keras.Model(inputs=inputs, outputs=bounding_box)
'\n## Run the experiment\n'

def run_experiment(model, learning_rate, weight_decay, batch_size, num_epochs):
    if False:
        i = 10
        return i + 15
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError())
    checkpoint_filepath = 'vit_object_detector.weights.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_loss', save_best_only=True, save_weights_only=True)
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_split=0.1, callbacks=[checkpoint_callback, keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])
    return history
input_shape = (image_size, image_size, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 15
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim]
transformer_layers = 4
mlp_head_units = [2048, 1024, 512, 64, 32]
history = []
num_patches = (image_size // patch_size) ** 2
vit_object_detector = create_vit_object_detector(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers, mlp_head_units)
history = run_experiment(vit_object_detector, learning_rate, weight_decay, batch_size, num_epochs)

def plot_history(item):
    if False:
        i = 10
        return i + 15
    plt.plot(history.history[item], label=item)
    plt.plot(history.history['val_' + item], label='val_' + item)
    plt.xlabel('Epochs')
    plt.ylabel(item)
    plt.title('Train and Validation {} Over Epochs'.format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()
plot_history('loss')
'\n## Evaluate the model\n'
import matplotlib.patches as patches
vit_object_detector.save('vit_object_detector.keras')

def bounding_box_intersection_over_union(box_predicted, box_truth):
    if False:
        print('Hello World!')
    top_x_intersect = max(box_predicted[0], box_truth[0])
    top_y_intersect = max(box_predicted[1], box_truth[1])
    bottom_x_intersect = min(box_predicted[2], box_truth[2])
    bottom_y_intersect = min(box_predicted[3], box_truth[3])
    intersection_area = max(0, bottom_x_intersect - top_x_intersect + 1) * max(0, bottom_y_intersect - top_y_intersect + 1)
    box_predicted_area = (box_predicted[2] - box_predicted[0] + 1) * (box_predicted[3] - box_predicted[1] + 1)
    box_truth_area = (box_truth[2] - box_truth[0] + 1) * (box_truth[3] - box_truth[1] + 1)
    return intersection_area / float(box_predicted_area + box_truth_area - intersection_area)
(i, mean_iou) = (0, 0)
for input_image in x_test[:10]:
    (fig, (ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 15))
    im = input_image
    ax1.imshow(im.astype('uint8'))
    ax2.imshow(im.astype('uint8'))
    input_image = cv2.resize(input_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    input_image = np.expand_dims(input_image, axis=0)
    preds = vit_object_detector.predict(input_image)[0]
    (h, w) = im.shape[0:2]
    (top_left_x, top_left_y) = (int(preds[0] * w), int(preds[1] * h))
    (bottom_right_x, bottom_right_y) = (int(preds[2] * w), int(preds[3] * h))
    box_predicted = [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
    rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y, facecolor='none', edgecolor='red', linewidth=1)
    ax1.add_patch(rect)
    ax1.set_xlabel('Predicted: ' + str(top_left_x) + ', ' + str(top_left_y) + ', ' + str(bottom_right_x) + ', ' + str(bottom_right_y))
    (top_left_x, top_left_y) = (int(y_test[i][0] * w), int(y_test[i][1] * h))
    (bottom_right_x, bottom_right_y) = (int(y_test[i][2] * w), int(y_test[i][3] * h))
    box_truth = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    mean_iou += bounding_box_intersection_over_union(box_predicted, box_truth)
    rect = patches.Rectangle((top_left_x, top_left_y), bottom_right_x - top_left_x, bottom_right_y - top_left_y, facecolor='none', edgecolor='red', linewidth=1)
    ax2.add_patch(rect)
    ax2.set_xlabel('Target: ' + str(top_left_x) + ', ' + str(top_left_y) + ', ' + str(bottom_right_x) + ', ' + str(bottom_right_y) + '\n' + 'IoU' + str(bounding_box_intersection_over_union(box_predicted, box_truth)))
    i = i + 1
print('mean_iou: ' + str(mean_iou / len(x_test[:10])))
plt.show()
'\nThis example demonstrates that a pure Transformer can be trained\nto predict the bounding boxes of an object in a given image,\nthus extending the use of Transformers to object detection tasks.\nThe model can be improved further by tuning hyper-parameters and pre-training.\n'