"""
Title: Keypoint Detection with Transfer Learning
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Converted to Keras 3 by: [Muhammad Anas Raza](https://anasrz.com)
Date created: 2021/05/02
Last modified: 2023/07/19
Description: Training a keypoint detector with data augmentation and transfer learning.
Accelerator: GPU
"""
'\nKeypoint detection consists of locating key object parts. For example, the key parts\nof our faces include nose tips, eyebrows, eye corners, and so on. These parts help to\nrepresent the underlying object in a feature-rich manner. Keypoint detection has\napplications that include pose estimation, face detection, etc.\n\nIn this example, we will build a keypoint detector using the\n[StanfordExtra dataset](https://github.com/benjiebob/StanfordExtra),\nusing transfer learning. This example requires TensorFlow 2.4 or higher,\nas well as [`imgaug`](https://imgaug.readthedocs.io/) library,\nwhich can be installed using the following command:\n'
'shell\npip install -q -U imgaug\n'
'\n## Data collection\n'
'\nThe StanfordExtra dataset contains 12,000 images of dogs together with keypoints and\nsegmentation maps. It is developed from the [Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).\nIt can be downloaded with the command below:\n'
'shell\nwget -q http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar\n'
'\nAnnotations are provided as a single JSON file in the StanfordExtra dataset and one needs\nto fill [this form](https://forms.gle/sRtbicgxsWvRtRmUA) to get access to it. The\nauthors explicitly instruct users not to share the JSON file, and this example respects this wish:\nyou should obtain the JSON file yourself.\n\nThe JSON file is expected to be locally available as `stanfordextra_v12.zip`.\n\nAfter the files are downloaded, we can extract the archives.\n'
'shell\ntar xf images.tar\nunzip -qq ~/stanfordextra_v12.zip\n'
'\n## Imports\n'
from keras import layers
import keras
from imgaug.augmentables.kps import KeypointsOnImage
from imgaug.augmentables.kps import Keypoint
import imgaug.augmenters as iaa
from PIL import Image
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import os
'\n## Define hyperparameters\n'
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5
NUM_KEYPOINTS = 24 * 2
'\n## Load data\n\nThe authors also provide a metadata file that specifies additional information about the\nkeypoints, like color information, animal pose name, etc. We will load this file in a `pandas`\ndataframe to extract information for visualization purposes.\n'
IMG_DIR = 'Images'
JSON = 'StanfordExtra_V12/StanfordExtra_v12.json'
KEYPOINT_DEF = 'https://github.com/benjiebob/StanfordExtra/raw/master/keypoint_definitions.csv'
with open(JSON) as infile:
    json_data = json.load(infile)
json_dict = {i['img_path']: i for i in json_data}
"\nA single entry of `json_dict` looks like the following:\n\n```\n'n02085782-Japanese_spaniel/n02085782_2886.jpg':\n{'img_bbox': [205, 20, 116, 201],\n 'img_height': 272,\n 'img_path': 'n02085782-Japanese_spaniel/n02085782_2886.jpg',\n 'img_width': 350,\n 'is_multiple_dogs': False,\n 'joints': [[108.66666666666667, 252.0, 1],\n            [147.66666666666666, 229.0, 1],\n            [163.5, 208.5, 1],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0],\n            [54.0, 244.0, 1],\n            [77.33333333333333, 225.33333333333334, 1],\n            [79.0, 196.5, 1],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0],\n            [150.66666666666666, 86.66666666666667, 1],\n            [88.66666666666667, 73.0, 1],\n            [116.0, 106.33333333333333, 1],\n            [109.0, 123.33333333333333, 1],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0],\n            [0, 0, 0]],\n 'seg': ...}\n```\n"
'\nIn this example, the keys we are interested in are:\n\n* `img_path`\n* `joints`\n\nThere are a total of 24 entries present inside `joints`. Each entry has 3 values:\n\n* x-coordinate\n* y-coordinate\n* visibility flag of the keypoints (1 indicates visibility and 0 indicates non-visibility)\n\nAs we can see `joints` contain multiple `[0, 0, 0]` entries which denote that those\nkeypoints were not labeled. In this example, we will consider both non-visible as well as\nunlabeled keypoints in order to allow mini-batch learning.\n'
keypoint_def = pd.read_csv(KEYPOINT_DEF)
keypoint_def.head()
colours = keypoint_def['Hex colour'].values.tolist()
colours = ['#' + colour for colour in colours]
labels = keypoint_def['Name'].values.tolist()

def get_dog(name):
    if False:
        for i in range(10):
            print('nop')
    data = json_dict[name]
    img_data = plt.imread(os.path.join(IMG_DIR, data['img_path']))
    if img_data.shape[-1] == 4:
        img_data = img_data.astype(np.uint8)
        img_data = Image.fromarray(img_data)
        img_data = np.array(img_data.convert('RGB'))
    data['img_data'] = img_data
    return data
'\n## Visualize data\n\nNow, we write a utility function to visualize the images and their keypoints.\n'

def visualize_keypoints(images, keypoints):
    if False:
        return 10
    (fig, axes) = plt.subplots(nrows=len(images), ncols=2, figsize=(16, 12))
    [ax.axis('off') for ax in np.ravel(axes)]
    for ((ax_orig, ax_all), image, current_keypoint) in zip(axes, images, keypoints):
        ax_orig.imshow(image)
        ax_all.imshow(image)
        if isinstance(current_keypoint, KeypointsOnImage):
            for (idx, kp) in enumerate(current_keypoint.keypoints):
                ax_all.scatter([kp.x], [kp.y], c=colours[idx], marker='x', s=50, linewidths=5)
        else:
            current_keypoint = np.array(current_keypoint)
            current_keypoint = current_keypoint[:, :2]
            for (idx, (x, y)) in enumerate(current_keypoint):
                ax_all.scatter([x], [y], c=colours[idx], marker='x', s=50, linewidths=5)
    plt.tight_layout(pad=2.0)
    plt.show()
samples = list(json_dict.keys())
num_samples = 4
selected_samples = np.random.choice(samples, num_samples, replace=False)
(images, keypoints) = ([], [])
for sample in selected_samples:
    data = get_dog(sample)
    image = data['img_data']
    keypoint = data['joints']
    images.append(image)
    keypoints.append(keypoint)
visualize_keypoints(images, keypoints)
'\nThe plots show that we have images of non-uniform sizes, which is expected in most\nreal-world scenarios. However, if we resize these images to have a uniform shape (for\ninstance (224 x 224)) their ground-truth annotations will also be affected. The same\napplies if we apply any geometric transformation (horizontal flip, for e.g.) to an image.\nFortunately, `imgaug` provides utilities that can handle this issue.\nIn the next section, we will write a data generator inheriting the\n[`keras.utils.Sequence`](https://keras.io/api/utils/python_utils/#sequence-class) class\nthat applies data augmentation on batches of data using `imgaug`.\n'
'\n## Prepare data generator\n'

class KeyPointsDataset(keras.utils.PyDataset):

    def __init__(self, image_keys, aug, batch_size=BATCH_SIZE, train=True, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.image_keys = image_keys
        self.aug = aug
        self.batch_size = batch_size
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.image_keys) // self.batch_size

    def on_epoch_end(self):
        if False:
            print('Hello World!')
        self.indexes = np.arange(len(self.image_keys))
        if self.train:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        image_keys_temp = [self.image_keys[k] for k in indexes]
        (images, keypoints) = self.__data_generation(image_keys_temp)
        return (images, keypoints)

    def __data_generation(self, image_keys_temp):
        if False:
            i = 10
            return i + 15
        batch_images = np.empty((self.batch_size, IMG_SIZE, IMG_SIZE, 3), dtype='int')
        batch_keypoints = np.empty((self.batch_size, 1, 1, NUM_KEYPOINTS), dtype='float32')
        for (i, key) in enumerate(image_keys_temp):
            data = get_dog(key)
            current_keypoint = np.array(data['joints'])[:, :2]
            kps = []
            for j in range(0, len(current_keypoint)):
                kps.append(Keypoint(x=current_keypoint[j][0], y=current_keypoint[j][1]))
            current_image = data['img_data']
            kps_obj = KeypointsOnImage(kps, shape=current_image.shape)
            (new_image, new_kps_obj) = self.aug(image=current_image, keypoints=kps_obj)
            batch_images[i,] = new_image
            kp_temp = []
            for keypoint in new_kps_obj:
                kp_temp.append(np.nan_to_num(keypoint.x))
                kp_temp.append(np.nan_to_num(keypoint.y))
            batch_keypoints[i,] = np.array(kp_temp).reshape(1, 1, 24 * 2)
        batch_keypoints = batch_keypoints / IMG_SIZE
        return (batch_images, batch_keypoints)
'\nTo know more about how to operate with keypoints in `imgaug` check out\n[this document](https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html).\n'
'\n## Define augmentation transforms\n'
train_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation='linear'), iaa.Fliplr(0.3), iaa.Sometimes(0.3, iaa.Affine(rotate=10, scale=(0.5, 0.7)))])
test_aug = iaa.Sequential([iaa.Resize(IMG_SIZE, interpolation='linear')])
'\n## Create training and validation splits\n'
np.random.shuffle(samples)
(train_keys, validation_keys) = (samples[int(len(samples) * 0.15):], samples[:int(len(samples) * 0.15)])
'\n## Data generator investigation\n'
train_dataset = KeyPointsDataset(train_keys, train_aug, workers=2, use_multiprocessing=True)
validation_dataset = KeyPointsDataset(validation_keys, test_aug, train=False, workers=2, use_multiprocessing=True)
print(f'Total batches in training set: {len(train_dataset)}')
print(f'Total batches in validation set: {len(validation_dataset)}')
(sample_images, sample_keypoints) = next(iter(train_dataset))
assert sample_keypoints.max() == 1.0
assert sample_keypoints.min() == 0.0
sample_keypoints = sample_keypoints[:4].reshape(-1, 24, 2) * IMG_SIZE
visualize_keypoints(sample_images[:4], sample_keypoints)
'\n## Model building\n\nThe [Stanford dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (on which\nthe StanfordExtra dataset is based) was built using the [ImageNet-1k dataset](http://image-net.org/).\nSo, it is likely that the models pretrained on the ImageNet-1k dataset would be useful\nfor this task. We will use a MobileNetV2 pre-trained on this dataset as a backbone to\nextract meaningful features from the images and then pass those to a custom regression\nhead for predicting coordinates.\n'

def get_model():
    if False:
        while True:
            i = 10
    backbone = keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone.trainable = False
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x)
    x = layers.Dropout(0.3)(x)
    x = layers.SeparableConv2D(NUM_KEYPOINTS, kernel_size=5, strides=1, activation='relu')(x)
    outputs = layers.SeparableConv2D(NUM_KEYPOINTS, kernel_size=3, strides=1, activation='sigmoid')(x)
    return keras.Model(inputs, outputs, name='keypoint_detector')
'\nOur custom network is fully-convolutional which makes it more parameter-friendly than the\nsame version of the network having fully-connected dense layers.\n'
get_model().summary()
'\nNotice the output shape of the network: `(None, 1, 1, 48)`. This is why we have reshaped\nthe coordinates as: `batch_keypoints[i, :] = np.array(kp_temp).reshape(1, 1, 24 * 2)`.\n'
'\n## Model compilation and training\n\nFor this example, we will train the network only for five epochs.\n'
model = get_model()
model.compile(loss='mse', optimizer=keras.optimizers.Adam(0.0001))
model.fit(train_dataset, validation_data=validation_dataset, epochs=EPOCHS)
'\n## Make predictions and visualize them\n'
(sample_val_images, sample_val_keypoints) = next(iter(validation_dataset))
sample_val_images = sample_val_images[:4]
sample_val_keypoints = sample_val_keypoints[:4].reshape(-1, 24, 2) * IMG_SIZE
predictions = model.predict(sample_val_images).reshape(-1, 24, 2) * IMG_SIZE
visualize_keypoints(sample_val_images, sample_val_keypoints)
visualize_keypoints(sample_val_images, predictions)
'\nPredictions will likely improve with more training.\n'
'\n## Going further\n\n* Try using other augmentation transforms from `imgaug` to investigate how that changes\nthe results.\n* Here, we transferred the features from the pre-trained network linearly that is we did\nnot [fine-tune](https://keras.io/guides/transfer_learning/) it. You are encouraged to fine-tune it on this task and see if that\nimproves the performance. You can also try different architectures and see how they\naffect the final performance.\n'