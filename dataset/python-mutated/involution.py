"""
Title: Involutional neural networks
Author: [Aritra Roy Gosthipaty](https://twitter.com/ariG23498)
Date created: 2021/07/25
Last modified: 2021/07/25
Description: Deep dive into location-specific and channel-agnostic "involution" kernels.
Accelerator: GPU
"""
'\n## Introduction\n\nConvolution has been the basis of most modern neural\nnetworks for computer vision. A convolution kernel is\nspatial-agnostic and channel-specific. Because of this, it isn\'t able\nto adapt to different visual patterns with respect to\ndifferent spatial locations. Along with location-related problems, the\nreceptive field of convolution creates challenges with regard to capturing\nlong-range spatial interactions.\n\nTo address the above issues, Li et. al. rethink the properties\nof convolution in\n[Involution: Inverting the Inherence of Convolution for VisualRecognition](https://arxiv.org/abs/2103.06255).\nThe authors propose the "involution kernel", that is location-specific and\nchannel-agnostic. Due to the location-specific nature of the operation,\nthe authors say that self-attention falls under the design paradigm of\ninvolution.\n\nThis example describes the involution kernel, compares two image\nclassification models, one with convolution and the other with\ninvolution, and also tries drawing a parallel with the self-attention\nlayer.\n'
'\n## Setup\n'
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
tf.random.set_seed(42)
'\n## Convolution\n\nConvolution remains the mainstay of deep neural networks for computer vision.\nTo understand Involution, it is necessary to talk about the\nconvolution operation.\n\n![Imgur](https://i.imgur.com/MSKLsm5.png)\n\nConsider an input tensor **X** with dimensions **H**, **W** and\n**C_in**. We take a collection of **C_out** convolution kernels each of\nshape **K**, **K**, **C_in**. With the multiply-add operation between\nthe input tensor and the kernels we obtain an output tensor **Y** with\ndimensions **H**, **W**, **C_out**.\n\nIn the diagram above `C_out=3`. This makes the output tensor of shape H,\nW and 3. One can notice that the convoltuion kernel does not depend on\nthe spatial position of the input tensor which makes it\n**location-agnostic**. On the other hand, each channel in the output\ntensor is based on a specific convolution filter which makes is\n**channel-specific**.\n'
'\n## Involution\n\nThe idea is to have an operation that is both **location-specific**\nand **channel-agnostic**. Trying to implement these specific properties poses\na challenge. With a fixed number of involution kernels (for each\nspatial position) we will **not** be able to process variable-resolution\ninput tensors.\n\nTo solve this problem, the authors have considered *generating* each\nkernel conditioned on specific spatial positions. With this method, we\nshould be able to process variable-resolution input tensors with ease.\nThe diagram below provides an intuition on this kernel generation\nmethod.\n\n![Imgur](https://i.imgur.com/jtrGGQg.png)\n'

class Involution(keras.layers.Layer):

    def __init__(self, channel, group_number, kernel_size, stride, reduction_ratio, name):
        if False:
            return 10
        super().__init__(name=name)
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        (_, height, width, num_channels) = input_shape
        height = height // self.stride
        width = width // self.stride
        self.stride_layer = keras.layers.AveragePooling2D(pool_size=self.stride, strides=self.stride, padding='same') if self.stride > 1 else tf.identity
        self.kernel_gen = keras.Sequential([keras.layers.Conv2D(filters=self.channel // self.reduction_ratio, kernel_size=1), keras.layers.BatchNormalization(), keras.layers.ReLU(), keras.layers.Conv2D(filters=self.kernel_size * self.kernel_size * self.group_number, kernel_size=1)])
        self.kernel_reshape = keras.layers.Reshape(target_shape=(height, width, self.kernel_size * self.kernel_size, 1, self.group_number))
        self.input_patches_reshape = keras.layers.Reshape(target_shape=(height, width, self.kernel_size * self.kernel_size, num_channels // self.group_number, self.group_number))
        self.output_reshape = keras.layers.Reshape(target_shape=(height, width, num_channels))

    def call(self, x):
        if False:
            i = 10
            return i + 15
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)
        kernel = self.kernel_reshape(kernel)
        input_patches = tf.image.extract_patches(images=x, sizes=[1, self.kernel_size, self.kernel_size, 1], strides=[1, self.stride, self.stride, 1], rates=[1, 1, 1, 1], padding='SAME')
        input_patches = self.input_patches_reshape(input_patches)
        output = tf.multiply(kernel, input_patches)
        output = tf.reduce_sum(output, axis=3)
        output = self.output_reshape(output)
        return (output, kernel)
'\n## Testing the Involution layer\n'
input_tensor = tf.random.normal((32, 256, 256, 3))
(output_tensor, _) = Involution(channel=3, group_number=1, kernel_size=5, stride=1, reduction_ratio=1, name='inv_1')(input_tensor)
print(f'with stride 1 ouput shape: {output_tensor.shape}')
(output_tensor, _) = Involution(channel=3, group_number=1, kernel_size=5, stride=2, reduction_ratio=1, name='inv_2')(input_tensor)
print(f'with stride 2 ouput shape: {output_tensor.shape}')
(output_tensor, _) = Involution(channel=16, group_number=1, kernel_size=5, stride=1, reduction_ratio=2, name='inv_3')(input_tensor)
print('with channel 16 and reduction ratio 2 ouput shape: {}'.format(output_tensor.shape))
'\n## Image Classification\n\nIn this section, we will build an image-classifier model. There will\nbe two models one with convolutions and the other with involutions.\n\nThe image-classification model is heavily inspired by this\n[Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn)\ntutorial from Google.\n'
'\n## Get the CIFAR10 Dataset\n'
print('loading the CIFAR10 dataset...')
((train_images, train_labels), (test_images, test_labels)) = keras.datasets.cifar10.load_data()
(train_images, test_images) = (train_images / 255.0, test_images / 255.0)
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(256).batch(256)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(256)
'\n## Visualise the data\n'
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
'\n## Convolutional Neural Network\n'
print('building the convolution model...')
conv_model = keras.Sequential([keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same'), keras.layers.ReLU(name='relu1'), keras.layers.MaxPooling2D((2, 2)), keras.layers.Conv2D(64, (3, 3), padding='same'), keras.layers.ReLU(name='relu2'), keras.layers.MaxPooling2D((2, 2)), keras.layers.Conv2D(64, (3, 3), padding='same'), keras.layers.ReLU(name='relu3'), keras.layers.Flatten(), keras.layers.Dense(64, activation='relu'), keras.layers.Dense(10)])
print('compiling the convolution model...')
conv_model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'], jit_compile=False)
print('conv model training...')
conv_hist = conv_model.fit(train_ds, epochs=20, validation_data=test_ds)
'\n## Involutional Neural Network\n'
print('building the involution model...')
inputs = keras.Input(shape=(32, 32, 3))
(x, _) = Involution(channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name='inv_1')(inputs)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
(x, _) = Involution(channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name='inv_2')(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D((2, 2))(x)
(x, _) = Involution(channel=3, group_number=1, kernel_size=3, stride=1, reduction_ratio=2, name='inv_3')(x)
x = keras.layers.ReLU()(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(10)(x)
inv_model = keras.Model(inputs=[inputs], outputs=[outputs], name='inv_model')
print('compiling the involution model...')
inv_model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'], jit_compile=False)
print('inv model training...')
inv_hist = inv_model.fit(train_ds, epochs=20, validation_data=test_ds)
'\n## Comparisons\n\nIn this section, we will be looking at both the models and compare a\nfew pointers.\n'
'\n### Parameters\n\nOne can see that with a similar architecture the parameters in a CNN\nis much larger than that of an INN (Involutional Neural Network).\n'
conv_model.summary()
inv_model.summary()
'\n### Loss and Accuracy Plots\n\nHere, the loss and the accuracy plots demonstrate that INNs are slow\nlearners (with lower parameters).\n'
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Convolution Loss')
plt.plot(conv_hist.history['loss'], label='loss')
plt.plot(conv_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Involution Loss')
plt.plot(inv_hist.history['loss'], label='loss')
plt.plot(inv_hist.history['val_loss'], label='val_loss')
plt.legend()
plt.show()
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.title('Convolution Accuracy')
plt.plot(conv_hist.history['accuracy'], label='accuracy')
plt.plot(conv_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('Involution Accuracy')
plt.plot(inv_hist.history['accuracy'], label='accuracy')
plt.plot(inv_hist.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()
'\n## Visualizing Involution Kernels\n\nTo visualize the kernels, we take the sum of **K×K** values from each\ninvolution kernel. **All the representatives at different spatial\nlocations frame the corresponding heat map.**\n\nThe authors mention:\n\n"Our proposed involution is reminiscent of self-attention and\nessentially could become a generalized version of it."\n\nWith the visualization of the kernel we can indeed obtain an attention\nmap of the image. The learned involution kernels provides attention to\nindividual spatial positions of the input tensor. The\n**location-specific** property makes involution a generic space of models\nin which self-attention belongs.\n'
layer_names = ['inv_1', 'inv_2', 'inv_3']
outputs = [inv_model.get_layer(name).output[1] for name in layer_names]
vis_model = keras.Model(inv_model.input, outputs)
(fig, axes) = plt.subplots(nrows=10, ncols=4, figsize=(10, 30))
for (ax, test_image) in zip(axes, test_images[:10]):
    inv_out = vis_model.predict(test_image[None, ...])
    (inv1_kernel, inv2_kernel, inv3_kernel) = inv_out
    inv1_kernel = tf.reduce_sum(inv1_kernel, axis=[-1, -2, -3])
    inv2_kernel = tf.reduce_sum(inv2_kernel, axis=[-1, -2, -3])
    inv3_kernel = tf.reduce_sum(inv3_kernel, axis=[-1, -2, -3])
    ax[0].imshow(keras.utils.array_to_img(test_image))
    ax[0].set_title('Input Image')
    ax[1].imshow(keras.utils.array_to_img(inv1_kernel[0, ..., None]))
    ax[1].set_title('Involution Kernel 1')
    ax[2].imshow(keras.utils.array_to_img(inv2_kernel[0, ..., None]))
    ax[2].set_title('Involution Kernel 2')
    ax[3].imshow(keras.utils.array_to_img(inv3_kernel[0, ..., None]))
    ax[3].set_title('Involution Kernel 3')
"\n## Conclusions\n\nIn this example, the main focus was to build an `Involution` layer which\ncan be easily reused. While our comparisons were based on a specific\ntask, feel free to use the layer for different tasks and report your\nresults.\n\nAccording to me, the key take-away of involution is its\nrelationship with self-attention. The intuition behind location-specific\nand channel-spefic processing makes sense in a lot of tasks.\n\nMoving forward one can:\n\n- Look at [Yannick's video](https://youtu.be/pH2jZun8MoY) on\n    involution for a better understanding.\n- Experiment with the various hyperparameters of the involution layer.\n- Build different models with the involution layer.\n- Try building a different kernel generation method altogether.\n\nYou can use the trained model hosted on [Hugging Face Hub](https://huggingface.co/keras-io/involution)\nand try the demo on [Hugging Face Spaces](https://huggingface.co/spaces/keras-io/involution).\n"