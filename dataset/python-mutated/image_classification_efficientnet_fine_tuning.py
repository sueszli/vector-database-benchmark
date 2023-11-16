"""
Title: Image classification via fine-tuning with EfficientNet
Author: [Yixing Fu](https://github.com/yixingfu)
Date created: 2020/06/30
Last modified: 2023/07/10
Description: Use EfficientNet with weights pre-trained on imagenet for Stanford Dogs classification.
Accelerator: GPU
"""
'\n\n## Introduction: what is EfficientNet\n\nEfficientNet, first introduced in [Tan and Le, 2019](https://arxiv.org/abs/1905.11946)\nis among the most efficient models (i.e. requiring least FLOPS for inference)\nthat reaches State-of-the-Art accuracy on both\nimagenet and common image classification transfer learning tasks.\n\nThe smallest base model is similar to [MnasNet](https://arxiv.org/abs/1807.11626), which\nreached near-SOTA with a significantly smaller model. By introducing a heuristic way to\nscale the model, EfficientNet provides a family of models (B0 to B7) that represents a\ngood combination of efficiency and accuracy on a variety of scales. Such a scaling\nheuristics (compound-scaling, details see\n[Tan and Le, 2019](https://arxiv.org/abs/1905.11946)) allows the\nefficiency-oriented base model (B0) to surpass models at every scale, while avoiding\nextensive grid-search of hyperparameters.\n\nA summary of the latest updates on the model is available at\n[here](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), where various\naugmentation schemes and semi-supervised learning approaches are applied to further\nimprove the imagenet performance of the models. These extensions of the model can be used\nby updating weights without changing model architecture.\n\n## B0 to B7 variants of EfficientNet\n\n*(This section provides some details on "compound scaling", and can be skipped\nif you\'re only interested in using the models)*\n\nBased on the [original paper](https://arxiv.org/abs/1905.11946) people may have the\nimpression that EfficientNet is a continuous family of models created by arbitrarily\nchoosing scaling factor in as Eq.(3) of the paper.  However, choice of resolution,\ndepth and width are also restricted by many factors:\n\n- Resolution: Resolutions not divisible by 8, 16, etc. cause zero-padding near boundaries\nof some layers which wastes computational resources. This especially applies to smaller\nvariants of the model, hence the input resolution for B0 and B1 are chosen as 224 and\n240.\n\n- Depth and width: The building blocks of EfficientNet demands channel size to be\nmultiples of 8.\n\n- Resource limit: Memory limitation may bottleneck resolution when depth\nand width can still increase. In such a situation, increasing depth and/or\nwidth but keep resolution can still improve performance.\n\nAs a result, the depth, width and resolution of each variant of the EfficientNet models\nare hand-picked and proven to produce good results, though they may be significantly\noff from the compound scaling formula.\nTherefore, the keras implementation (detailed below) only provide these 8 models, B0 to B7,\ninstead of allowing arbitray choice of width / depth / resolution parameters.\n\n## Keras implementation of EfficientNet\n\nAn implementation of EfficientNet B0 to B7 has been shipped with Keras since v2.3. To\nuse EfficientNetB0 for classifying 1000 classes of images from ImageNet, run:\n\n```python\nfrom tensorflow.keras.applications import EfficientNetB0\nmodel = EfficientNetB0(weights=\'imagenet\')\n```\n\nThis model takes input images of shape `(224, 224, 3)`, and the input data should be in the\nrange `[0, 255]`. Normalization is included as part of the model.\n\nBecause training EfficientNet on ImageNet takes a tremendous amount of resources and\nseveral techniques that are not a part of the model architecture itself. Hence the Keras\nimplementation by default loads pre-trained weights obtained via training with\n[AutoAugment](https://arxiv.org/abs/1805.09501).\n\nFor B0 to B7 base models, the input shapes are different. Here is a list of input shape\nexpected for each model:\n\n| Base model | resolution|\n|----------------|-----|\n| EfficientNetB0 | 224 |\n| EfficientNetB1 | 240 |\n| EfficientNetB2 | 260 |\n| EfficientNetB3 | 300 |\n| EfficientNetB4 | 380 |\n| EfficientNetB5 | 456 |\n| EfficientNetB6 | 528 |\n| EfficientNetB7 | 600 |\n\nWhen the model is intended for transfer learning, the Keras implementation\nprovides a option to remove the top layers:\n```\nmodel = EfficientNetB0(include_top=False, weights=\'imagenet\')\n```\nThis option excludes the final `Dense` layer that turns 1280 features on the penultimate\nlayer into prediction of the 1000 ImageNet classes. Replacing the top layer with custom\nlayers allows using EfficientNet as a feature extractor in a transfer learning workflow.\n\nAnother argument in the model constructor worth noticing is `drop_connect_rate` which controls\nthe dropout rate responsible for [stochastic depth](https://arxiv.org/abs/1603.09382).\nThis parameter serves as a toggle for extra regularization in finetuning, but does not\naffect loaded weights. For example, when stronger regularization is desired, try:\n\n```python\nmodel = EfficientNetB0(weights=\'imagenet\', drop_connect_rate=0.4)\n```\nThe default value is 0.2.\n\n## Example: EfficientNetB0 for Stanford Dogs.\n\nEfficientNet is capable of a wide range of image classification tasks.\nThis makes it a good model for transfer learning.\nAs an end-to-end example, we will show using pre-trained EfficientNetB0 on\n[Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html) dataset.\n\n'
'\n## Setup and data loading\n'
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.applications import EfficientNetB0
IMG_SIZE = 224
BATCH_SIZE = 64
'\n### Loading data\n\nHere we load data from [tensorflow_datasets](https://www.tensorflow.org/datasets)\n(hereafter TFDS).\nStanford Dogs dataset is provided in\nTFDS as [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs).\nIt features 20,580 images that belong to 120 classes of dog breeds\n(12,000 for training and 8,580 for testing).\n\nBy simply changing `dataset_name` below, you may also try this notebook for\nother datasets in TFDS such as\n[cifar10](https://www.tensorflow.org/datasets/catalog/cifar10),\n[cifar100](https://www.tensorflow.org/datasets/catalog/cifar100),\n[food101](https://www.tensorflow.org/datasets/catalog/food101),\netc. When the images are much smaller than the size of EfficientNet input,\nwe can simply upsample the input images. It has been shown in\n[Tan and Le, 2019](https://arxiv.org/abs/1905.11946) that transfer learning\nresult is better for increased resolution even if input images remain small.\n'
dataset_name = 'stanford_dogs'
((ds_train, ds_test), ds_info) = tfds.load(dataset_name, split=['train', 'test'], with_info=True, as_supervised=True)
NUM_CLASSES = ds_info.features['label'].num_classes
'\nWhen the dataset include images with various size, we need to resize them into a\nshared size. The Stanford Dogs dataset includes only images at least 200x200\npixels in size. Here we resize the images to the input size needed for EfficientNet.\n'
size = (IMG_SIZE, IMG_SIZE)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_test = ds_test.map(lambda image, label: (tf.image.resize(image, size), label))
'\n### Visualizing the data\n\nThe following code shows the first 9 images with their labels.\n'

def format_label(label):
    if False:
        for i in range(10):
            print('nop')
    string_label = label_info.int2str(label)
    return string_label.split('-')[1]
label_info = ds_info.features['label']
for (i, (image, label)) in enumerate(ds_train.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype('uint8'))
    plt.title('{}'.format(format_label(label)))
    plt.axis('off')
'\n### Data augmentation\n\nWe can use the preprocessing layers APIs for image augmentation.\n'
img_augmentation_layers = [layers.RandomRotation(factor=0.15), layers.RandomTranslation(height_factor=0.1, width_factor=0.1), layers.RandomFlip(), layers.RandomContrast(factor=0.1)]

def img_augmentation(images):
    if False:
        i = 10
        return i + 15
    for layer in img_augmentation_layers:
        images = layer(images)
    return images
'\nThis `Sequential` model object can be used both as a part of\nthe model we later build, and as a function to preprocess\ndata before feeding into the model. Using them as function makes\nit easy to visualize the augmented images. Here we plot 9 examples\nof augmentation result of a given figure.\n'
for (image, label) in ds_train.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        aug_img = img_augmentation(np.expand_dims(image.numpy(), axis=0))
        aug_img = np.array(aug_img)
        plt.imshow(aug_img[0].astype('uint8'))
        plt.title('{}'.format(format_label(label)))
        plt.axis('off')
'\n### Prepare inputs\n\nOnce we verify the input data and augmentation are working correctly,\nwe prepare dataset for training. The input data are resized to uniform\n`IMG_SIZE`. The labels are put into one-hot\n(a.k.a. categorical) encoding. The dataset is batched.\n\nNote: `prefetch` and `AUTOTUNE` may in some situation improve\nperformance, but depends on environment and the specific dataset used.\nSee this [guide](https://www.tensorflow.org/guide/data_performance)\nfor more information on data pipeline performance.\n'

def input_preprocess_train(image, label):
    if False:
        return 10
    image = img_augmentation(image)
    label = tf.one_hot(label, NUM_CLASSES)
    return (image, label)

def input_preprocess_test(image, label):
    if False:
        i = 10
        return i + 15
    label = tf.one_hot(label, NUM_CLASSES)
    return (image, label)
ds_train = ds_train.map(input_preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.batch(batch_size=BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(input_preprocess_test, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size=BATCH_SIZE, drop_remainder=True)
'\n## Training a model from scratch\n\nWe build an EfficientNetB0 with 120 output classes, that is initialized from scratch:\n\nNote: the accuracy will increase very slowly and may overfit.\n'
model = EfficientNetB0(include_top=True, weights=None, classes=NUM_CLASSES, input_shape=(IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
epochs = 40
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
'\nTraining the model is relatively fast. This might make it sounds easy to simply train EfficientNet on any\ndataset wanted from scratch. However, training EfficientNet on smaller datasets,\nespecially those with lower resolution like CIFAR-100, faces the significant challenge of\noverfitting.\n\nHence training from scratch requires very careful choice of hyperparameters and is\ndifficult to find suitable regularization. It would also be much more demanding in resources.\nPlotting the training and validation accuracy\nmakes it clear that validation accuracy stagnates at a low value.\n'
import matplotlib.pyplot as plt

def plot_hist(hist):
    if False:
        while True:
            i = 10
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
plot_hist(hist)
'\n## Transfer learning from pre-trained weights\n\nHere we initialize the model with pre-trained ImageNet weights,\nand we fine-tune it on our own dataset.\n'

def build_model(num_classes):
    if False:
        while True:
            i = 10
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')
    model.trainable = False
    x = layers.GlobalAveragePooling2D(name='avg_pool')(model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name='top_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='pred')(x)
    model = keras.Model(inputs, outputs, name='EfficientNet')
    optimizer = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
'\nThe first step to transfer learning is to freeze all layers and train only the top\nlayers. For this step, a relatively large learning rate (1e-2) can be used.\nNote that validation accuracy and loss will usually be better than training\naccuracy and loss. This is because the regularization is strong, which only\nsuppresses training-time metrics.\n\nNote that the convergence may take up to 50 epochs depending on choice of learning rate.\nIf image augmentation layers were not\napplied, the validation accuracy may only reach ~60%.\n'
model = build_model(num_classes=NUM_CLASSES)
epochs = 25
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
plot_hist(hist)
'\nThe second step is to unfreeze a number of layers and fit the model using smaller\nlearning rate. In this example we show unfreezing all layers, but depending on\nspecific dataset it may be desireble to only unfreeze a fraction of all layers.\n\nWhen the feature extraction with\npretrained model works good enough, this step would give a very limited gain on\nvalidation accuracy. In our case we only see a small improvement,\nas ImageNet pretraining already exposed the model to a good amount of dogs.\n\nOn the other hand, when we use pretrained weights on a dataset that is more different\nfrom ImageNet, this fine-tuning step can be crucial as the feature extractor also\nneeds to be adjusted by a considerable amount. Such a situation can be demonstrated\nif choosing CIFAR-100 dataset instead, where fine-tuning boosts validation accuracy\nby about 10% to pass 80% on `EfficientNetB0`.\nIn such a case the convergence may take more than 50 epochs.\n\nA side note on freezing/unfreezing models: setting `trainable` of a `Model` will\nsimultaneously set all layers belonging to the `Model` to the same `trainable`\nattribute. Each layer is trainable only if both the layer itself and the model\ncontaining it are trainable. Hence when we need to partially freeze/unfreeze\na model, we need to make sure the `trainable` attribute of the model is set\nto `True`.\n'

def unfreeze_model(model):
    if False:
        i = 10
        return i + 15
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
unfreeze_model(model)
epochs = 10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_test)
plot_hist(hist)
'\n### Tips for fine tuning EfficientNet\n\nOn unfreezing layers:\n\n- The `BatchNormalization` layers need to be kept frozen\n([more details](https://keras.io/guides/transfer_learning/)).\nIf they are also turned to trainable, the\nfirst epoch after unfreezing will significantly reduce accuracy.\n- In some cases it may be beneficial to open up only a portion of layers instead of\nunfreezing all. This will make fine tuning much faster when going to larger models like\nB7.\n- Each block needs to be all turned on or off. This is because the architecture includes\na shortcut from the first layer to the last layer for each block. Not respecting blocks\nalso significantly harms the final performance.\n\nSome other tips for utilizing EfficientNet:\n\n- Larger variants of EfficientNet do not guarantee improved performance, especially for\ntasks with less data or fewer classes. In such a case, the larger variant of EfficientNet\nchosen, the harder it is to tune hyperparameters.\n- EMA (Exponential Moving Average) is very helpful in training EfficientNet from scratch,\nbut not so much for transfer learning.\n- Do not use the RMSprop setup as in the original paper for transfer learning. The\nmomentum and learning rate are too high for transfer learning. It will easily corrupt the\npretrained weight and blow up the loss. A quick check is to see if loss (as categorical\ncross entropy) is getting significantly larger than log(NUM_CLASSES) after the same\nepoch. If so, the initial learning rate/momentum is too high.\n- Smaller batch size benefit validation accuracy, possibly due to effectively providing\nregularization.\n'