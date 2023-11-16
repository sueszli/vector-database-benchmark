"""
Title: Learning to Resize in Computer Vision
Author: [Sayak Paul](https://twitter.com/RisingSayak)
Date created: 2021/04/30
Last modified: 2023/07/26
Description: How to optimally learn representations of images for a given resolution.
Accelerator: GPU
"""
'\nIt is a common belief that if we constrain vision models to perceive things as humans do,\ntheir performance can be improved. For example, in [this work](https://arxiv.org/abs/1811.12231),\nGeirhos et al. showed that the vision models pre-trained on the ImageNet-1k dataset are\nbiased towards texture, whereas human beings mostly use the shape descriptor to develop a\ncommon perception. But does this belief always apply, especially when it comes to improving\nthe performance of vision models?\n\nIt turns out it may not always be the case. When training vision models, it is common to\nresize images to a lower dimension ((224 x 224), (299 x 299), etc.) to allow mini-batch\nlearning and also to keep up the compute limitations.  We generally make use of image\nresizing methods like **bilinear interpolation** for this step and the resized images do\nnot lose much of their perceptual character to the human eyes. In\n[Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/abs/2103.09950v1), Talebi et al. show\nthat if we try to optimize the perceptual quality of the images for the vision models\nrather than the human eyes, their performance can further be improved. They investigate\nthe following question:\n\n**For a given image resolution and a model, how to best resize the given images?**\n\nAs shown in the paper, this idea helps to consistently improve the performance of the\ncommon vision models (pre-trained on ImageNet-1k) like DenseNet-121, ResNet-50,\nMobileNetV2, and EfficientNets. In this example, we will implement the learnable image\nresizing module as proposed in the paper and demonstrate that on the\n[Cats and Dogs dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)\nusing the [DenseNet-121](https://arxiv.org/abs/1608.06993) architecture.\n\n'
'\n## Setup\n'
from keras import layers
import keras
from keras import ops
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import one_hot as tf_one_hot
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import matplotlib.pyplot as plt
import numpy as np
'\n## Define hyperparameters\n'
'\nIn order to facilitate mini-batch learning, we need to have a fixed shape for the images\ninside a given batch. This is why an initial resizing is required. We first resize all\nthe images to (300 x 300) shape and then learn their optimal representation for the\n(150 x 150) resolution.\n'
INP_SIZE = (300, 300)
TARGET_SIZE = (150, 150)
INTERPOLATION = 'bilinear'
AUTO = tf_data.AUTOTUNE
BATCH_SIZE = 50
EPOCHS = 5
'\nIn this example, we will use the bilinear interpolation but the learnable image resizer\nmodule is not dependent on any specific interpolation method. We can also use others,\nsuch as bicubic.\n'
'\n## Load and prepare the dataset\n\nFor this example, we will only use 40% of the total training dataset.\n'
(train_ds, validation_ds) = tfds.load('cats_vs_dogs', split=['train[:40%]', 'train[40%:50%]'], as_supervised=True)

def preprocess_dataset(image, label):
    if False:
        print('Hello World!')
    image = tf_image.resize(image, (INP_SIZE[0], INP_SIZE[1]))
    label = tf_one_hot(label, depth=2)
    return (image, label)
train_ds = train_ds.shuffle(BATCH_SIZE * 100).map(preprocess_dataset, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
validation_ds = validation_ds.map(preprocess_dataset, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
'\n## Define the learnable resizer utilities\n\nThe figure below (courtesy: [Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/abs/2103.09950v1))\npresents the structure of the learnable resizing module:\n\n![](https://i.ibb.co/gJYtSs0/image.png)\n'

def conv_block(x, filters, kernel_size, strides, activation=layers.LeakyReLU(0.2)):
    if False:
        return 10
    x = layers.Conv2D(filters, kernel_size, strides, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    return x

def res_block(x):
    if False:
        print('Hello World!')
    inputs = x
    x = conv_block(x, 16, 3, 1)
    x = conv_block(x, 16, 3, 1, activation=None)
    return layers.Add()([inputs, x])

def get_learnable_resizer(filters=16, num_res_blocks=1, interpolation=INTERPOLATION):
    if False:
        for i in range(10):
            print('nop')
    inputs = layers.Input(shape=[None, None, 3])
    naive_resize = layers.Resizing(*TARGET_SIZE, interpolation=interpolation)(inputs)
    x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)
    bottleneck = layers.Resizing(*TARGET_SIZE, interpolation=interpolation)(x)
    for _ in range(num_res_blocks):
        x = res_block(bottleneck)
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([bottleneck, x])
    x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding='same')(x)
    final_resize = layers.Add()([naive_resize, x])
    return keras.Model(inputs, final_resize, name='learnable_resizer')
learnable_resizer = get_learnable_resizer()
'\n## Visualize the outputs of the learnable resizing module\n\nHere, we visualize how the resized images would look like after being passed through the\nrandom weights of the resizer.\n'
(sample_images, _) = next(iter(train_ds))
get_np = lambda image: ops.convert_to_numpy(ops.squeeze(image))
plt.figure(figsize=(16, 10))
for (i, image) in enumerate(sample_images[:6]):
    image = image / 255
    ax = plt.subplot(3, 4, 2 * i + 1)
    plt.title('Input Image')
    plt.imshow(image.numpy().squeeze())
    plt.axis('off')
    ax = plt.subplot(3, 4, 2 * i + 2)
    resized_image = learnable_resizer(image[None, ...])
    plt.title('Resized Image')
    plt.imshow(get_np(resized_image))
    plt.axis('off')
'\n## Model building utility\n'

def get_model():
    if False:
        print('Hello World!')
    backbone = keras.applications.DenseNet121(weights=None, include_top=True, classes=2, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    backbone.trainable = True
    inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
    x = layers.Rescaling(scale=1.0 / 255)(inputs)
    x = learnable_resizer(x)
    outputs = backbone(x)
    return keras.Model(inputs, outputs)
'\nThe structure of the learnable image resizer module allows for flexible integrations with\ndifferent vision models.\n'
'\n## Compile and train our model with learnable resizer\n'
model = get_model()
model.compile(loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), optimizer='sgd', metrics=['accuracy'])
model.fit(train_ds, validation_data=validation_ds, epochs=EPOCHS)
'\n## Visualize the outputs of the trained visualizer\n'
plt.figure(figsize=(16, 10))
for (i, image) in enumerate(sample_images[:6]):
    image = image / 255
    ax = plt.subplot(3, 4, 2 * i + 1)
    plt.title('Input Image')
    plt.imshow(image.numpy().squeeze())
    plt.axis('off')
    ax = plt.subplot(3, 4, 2 * i + 2)
    resized_image = learnable_resizer(image[None, ...])
    plt.title('Resized Image')
    plt.imshow(get_np(resized_image) / 10)
    plt.axis('off')
"\nThe plot shows that the visuals of the images have improved with training. The following\ntable shows the benefits of using the resizing module in comparison to using the bilinear\ninterpolation:\n\n|           Model           \t| Number of  parameters (Million) \t| Top-1 accuracy \t|\n|:-------------------------:\t|:-------------------------------:\t|:--------------:\t|\n|   With the learnable resizer  \t|             7.051717            \t|      67.67%     \t|\n| Without the learnable resizer \t|             7.039554            \t|      60.19%      \t|\n\nFor more details, you can check out [this repository](https://github.com/sayakpaul/Learnable-Image-Resizing).\nNote the above-reported models were trained for 10 epochs on 90% of the training set of\nCats and Dogs unlike this example. Also, note that the increase in the number of\nparameters due to the resizing module is very negligible. To ensure that the improvement\nin the performance is not due to stochasticity, the models were trained using the same\ninitial random weights.\n\nNow, a question worth asking here is -  _isn't the improved accuracy simply a consequence\nof adding more layers (the resizer is a mini network after all) to the model, compared to\nthe baseline?_\n\nTo show that it is not the case, the authors conduct the following experiment:\n\n* Take a pre-trained model trained some size, say (224 x 224).\n\n* Now, first, use it to infer predictions on images resized to a lower resolution. Record\nthe performance.\n\n* For the second experiment, plug in the resizer module at the top of the pre-trained\nmodel and warm-start the training. Record the performance.\n\nNow, the authors argue that using the second option is better because it helps the model\nlearn how to adjust the representations better with respect to the given resolution.\nSince the results purely are empirical, a few more experiments such as analyzing the\ncross-channel interaction would have been even better. It is worth noting that elements\nlike [Squeeze and Excitation (SE) blocks](https://arxiv.org/abs/1709.01507), [Global Context (GC) blocks](https://arxiv.org/pdf/1904.11492) also add a few\nparameters to an existing network but they are known to help a network process\ninformation in systematic ways to improve the overall performance.\n"
'\n## Notes\n\n* To impose shape bias inside the vision models, Geirhos et al. trained them with a\ncombination of natural and stylized images. It might be interesting to investigate if\nthis learnable resizing module could achieve something similar as the outputs seem to\ndiscard the texture information.\n\n* The resizer module can handle arbitrary resolutions and aspect ratios which is very\nimportant for tasks like object detection and segmentation.\n\n* There is another closely related topic on ***adaptive image resizing*** that attempts\nto resize images/feature maps adaptively during training. [EfficientV2](https://arxiv.org/pdf/2104.00298)\nuses this idea.\n'