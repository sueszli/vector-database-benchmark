"""
Title: The Functional API
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2019/03/01
Last modified: 2020/04/12
Description: Complete guide to the functional API.
Accelerator: GPU
"""
'\n## Setup\n'
import numpy as np
import keras
from keras import layers
from keras import ops
'\n## Introduction\n\nThe Keras *functional API* is a way to create models that are more flexible\nthan the `keras.Sequential` API. The functional API can handle models\nwith non-linear topology, shared layers, and even multiple inputs or outputs.\n\nThe main idea is that a deep learning model is usually\na directed acyclic graph (DAG) of layers.\nSo the functional API is a way to build *graphs of layers*.\n\nConsider the following model:\n\n<div class="k-default-codeblock">\n```\n(input: 784-dimensional vectors)\n       ↧\n[Dense (64 units, relu activation)]\n       ↧\n[Dense (64 units, relu activation)]\n       ↧\n[Dense (10 units, softmax activation)]\n       ↧\n(output: logits of a probability distribution over 10 classes)\n```\n</div>\n\nThis is a basic graph with three layers.\nTo build this model using the functional API, start by creating an input node:\n'
inputs = keras.Input(shape=(784,))
'\nThe shape of the data is set as a 784-dimensional vector.\nThe batch size is always omitted since only the shape of each sample is specified.\n\nIf, for example, you have an image input with a shape of `(32, 32, 3)`,\nyou would use:\n'
img_inputs = keras.Input(shape=(32, 32, 3))
"\nThe `inputs` that is returned contains information about the shape and `dtype`\nof the input data that you feed to your model.\nHere's the shape:\n"
inputs.shape
"\nHere's the dtype:\n"
inputs.dtype
'\nYou create a new node in the graph of layers by calling a layer on this `inputs`\nobject:\n'
dense = layers.Dense(64, activation='relu')
x = dense(inputs)
'\nThe "layer call" action is like drawing an arrow from "inputs" to this layer\nyou created.\nYou\'re "passing" the inputs to the `dense` layer, and you get `x` as the output.\n\nLet\'s add a few more layers to the graph of layers:\n'
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)
'\nAt this point, you can create a `Model` by specifying its inputs and outputs\nin the graph of layers:\n'
model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
"\nLet's check out what the model summary looks like:\n"
model.summary()
'\nYou can also plot the model as a graph:\n'
keras.utils.plot_model(model, 'my_first_model.png')
'\nAnd, optionally, display the input and output shapes of each layer\nin the plotted graph:\n'
keras.utils.plot_model(model, 'my_first_model_with_shape_info.png', show_shapes=True)
'\nThis figure and the code are almost identical. In the code version,\nthe connection arrows are replaced by the call operation.\n\nA "graph of layers" is an intuitive mental image for a deep learning model,\nand the functional API is a way to create models that closely mirrors this.\n'
'\n## Training, evaluation, and inference\n\nTraining, evaluation, and inference work exactly in the same way for models\nbuilt using the functional API as for `Sequential` models.\n\nThe `Model` class offers a built-in training loop (the `fit()` method)\nand a built-in evaluation loop (the `evaluate()` method). Note\nthat you can easily [customize these loops](/guides/customizing_what_happens_in_fit/)\nto implement training routines beyond supervised learning\n(e.g. [GANs](https://keras.io/examples/generative/dcgan_overriding_train_step/)).\n\nHere, load the MNIST image data, reshape it into vectors,\nfit the model on the data (while monitoring performance on a validation split),\nthen evaluate the model on the test data:\n'
((x_train, y_train), (x_test, y_test)) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
'\nFor further reading, see the [training and evaluation](/guides/training_with_built_in_methods/) guide.\n'
'\n## Save and serialize\n\nSaving the model and serialization work the same way for models built using\nthe functional API as they do for `Sequential` models. The standard way\nto save a functional model is to call `model.save()`\nto save the entire model as a single file. You can later recreate the same model\nfrom this file, even if the code that built the model is no longer available.\n\nThis saved file includes the:\n- model architecture\n- model weight values (that were learned during training)\n- model training config, if any (as passed to `compile()`)\n- optimizer and its state, if any (to restart training where you left off)\n'
model.save('my_model.keras')
del model
model = keras.models.load_model('my_model.keras')
'\nFor details, read the model [serialization & saving](\n    /guides/serialization_and_saving/) guide.\n'
'\n## Use the same graph of layers to define multiple models\n\nIn the functional API, models are created by specifying their inputs\nand outputs in a graph of layers. That means that a single\ngraph of layers can be used to generate multiple models.\n\nIn the example below, you use the same stack of layers to instantiate two models:\nan `encoder` model that turns image inputs into 16-dimensional vectors,\nand an end-to-end `autoencoder` model for training.\n'
encoder_input = keras.Input(shape=(28, 28, 1), name='img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()
x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)
autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()
'\nHere, the decoding architecture is strictly symmetrical\nto the encoding architecture, so the output shape is the same as\nthe input shape `(28, 28, 1)`.\n\nThe reverse of a `Conv2D` layer is a `Conv2DTranspose` layer,\nand the reverse of a `MaxPooling2D` layer is an `UpSampling2D` layer.\n'
"\n## All models are callable, just like layers\n\nYou can treat any model as if it were a layer by invoking it on an `Input` or\non the output of another layer. By calling a model you aren't just reusing\nthe architecture of the model, you're also reusing its weights.\n\nTo see this in action, here's a different take on the autoencoder example that\ncreates an encoder model, a decoder model, and chains them in two calls\nto obtain the autoencoder model:\n"
encoder_input = keras.Input(shape=(28, 28, 1), name='original_img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()
decoder_input = keras.Input(shape=(16,), name='encoded_img')
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)
decoder = keras.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()
autoencoder_input = keras.Input(shape=(28, 28, 1), name='img')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
autoencoder.summary()
"\nAs you can see, the model can be nested: a model can contain sub-models\n(since a model is just like a layer).\nA common use case for model nesting is *ensembling*.\nFor example, here's how to ensemble a set of models into a single model\nthat averages their predictions:\n"

def get_model():
    if False:
        return 10
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return keras.Model(inputs, outputs)
model1 = get_model()
model2 = get_model()
model3 = get_model()
inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
"\n## Manipulate complex graph topologies\n\n### Models with multiple inputs and outputs\n\nThe functional API makes it easy to manipulate multiple inputs and outputs.\nThis cannot be handled with the `Sequential` API.\n\nFor example, if you're building a system for ranking customer issue tickets by\npriority and routing them to the correct department,\nthen the model will have three inputs:\n\n- the title of the ticket (text input),\n- the text body of the ticket (text input), and\n- any tags added by the user (categorical input)\n\nThis model will have two outputs:\n\n- the priority score between 0 and 1 (scalar sigmoid output), and\n- the department that should handle the ticket (softmax output\nover the set of departments).\n\nYou can build this model in a few lines with the functional API:\n"
num_tags = 12
num_words = 10000
num_departments = 4
title_input = keras.Input(shape=(None,), name='title')
body_input = keras.Input(shape=(None,), name='body')
tags_input = keras.Input(shape=(num_tags,), name='tags')
title_features = layers.Embedding(num_words, 64)(title_input)
body_features = layers.Embedding(num_words, 64)(body_input)
title_features = layers.LSTM(128)(title_features)
body_features = layers.LSTM(32)(body_features)
x = layers.concatenate([title_features, body_features, tags_input])
priority_pred = layers.Dense(1, name='priority')(x)
department_pred = layers.Dense(num_departments, name='department')(x)
model = keras.Model(inputs=[title_input, body_input, tags_input], outputs={'priority': priority_pred, 'department': department_pred})
'\nNow plot the model:\n'
keras.utils.plot_model(model, 'multi_input_and_output_model.png', show_shapes=True)
'\nWhen compiling this model, you can assign different losses to each output.\nYou can even assign different weights to each loss -- to modulate\ntheir contribution to the total training loss.\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss=[keras.losses.BinaryCrossentropy(from_logits=True), keras.losses.CategoricalCrossentropy(from_logits=True)], loss_weights=[1.0, 0.2])
'\nSince the output layers have different names, you could also specify\nthe losses and loss weights with the corresponding layer names:\n'
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss={'priority': keras.losses.BinaryCrossentropy(from_logits=True), 'department': keras.losses.CategoricalCrossentropy(from_logits=True)}, loss_weights={'priority': 1.0, 'department': 0.2})
'\nTrain the model by passing lists of NumPy arrays of inputs and targets:\n'
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tags_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
priority_targets = np.random.random(size=(1280, 1))
dept_targets = np.random.randint(2, size=(1280, num_departments))
model.fit({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets}, epochs=2, batch_size=32)
"\nWhen calling fit with a `Dataset` object, it should yield either a\ntuple of lists like `([title_data, body_data, tags_data], [priority_targets, dept_targets])`\nor a tuple of dictionaries like\n`({'title': title_data, 'body': body_data, 'tags': tags_data}, {'priority': priority_targets, 'department': dept_targets})`.\n\nFor more detailed explanation, refer to the [training and evaluation](/guides/training_with_built_in_methods/) guide.\n"
"\n### A toy ResNet model\n\nIn addition to models with multiple inputs and outputs,\nthe functional API makes it easy to manipulate non-linear connectivity\ntopologies -- these are models with layers that are not connected sequentially,\nwhich the `Sequential` API cannot handle.\n\nA common use case for this is residual connections.\nLet's build a toy ResNet model for CIFAR10 to demonstrate this:\n"
inputs = keras.Input(shape=(32, 32, 3), name='img')
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.Conv2D(64, 3, activation='relu')(x)
block_1_output = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_2_output = layers.add([x, block_1_output])
x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
block_3_output = layers.add([x, block_2_output])
x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs, outputs, name='toy_resnet')
model.summary()
'\nPlot the model:\n'
keras.utils.plot_model(model, 'mini_resnet.png', show_shapes=True)
'\nNow train the model:\n'
((x_train, y_train), (x_test, y_test)) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
model.compile(optimizer=keras.optimizers.RMSprop(0.001), loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['acc'])
model.fit(x_train[:1000], y_train[:1000], batch_size=64, epochs=1, validation_split=0.2)
"\n## Shared layers\n\nAnother good use for the functional API are models that use *shared layers*.\nShared layers are layer instances that are reused multiple times in the same model --\nthey learn features that correspond to multiple paths in the graph-of-layers.\n\nShared layers are often used to encode inputs from similar spaces\n(say, two different pieces of text that feature similar vocabulary).\nThey enable sharing of information across these different inputs,\nand they make it possible to train such a model on less data.\nIf a given word is seen in one of the inputs,\nthat will benefit the processing of all inputs that pass through the shared layer.\n\nTo share a layer in the functional API, call the same layer instance multiple times.\nFor instance, here's an `Embedding` layer shared across two different text inputs:\n"
shared_embedding = layers.Embedding(1000, 128)
text_input_a = keras.Input(shape=(None,), dtype='int32')
text_input_b = keras.Input(shape=(None,), dtype='int32')
encoded_input_a = shared_embedding(text_input_a)
encoded_input_b = shared_embedding(text_input_b)
'\n## Extract and reuse nodes in the graph of layers\n\nBecause the graph of layers you are manipulating is a static data structure,\nit can be accessed and inspected. And this is how you are able to plot\nfunctional models as images.\n\nThis also means that you can access the activations of intermediate layers\n("nodes" in the graph) and reuse them elsewhere --\nwhich is very useful for something like feature extraction.\n\nLet\'s look at an example. This is a VGG19 model with weights pretrained on ImageNet:\n'
vgg19 = keras.applications.VGG19()
'\nAnd these are the intermediate activations of the model,\nobtained by querying the graph data structure:\n'
features_list = [layer.output for layer in vgg19.layers]
'\nUse these features to create a new feature-extraction model that returns\nthe values of the intermediate layer activations:\n'
feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
img = np.random.random((1, 224, 224, 3)).astype('float32')
extracted_features = feat_extraction_model(img)
'\nThis comes in handy for tasks like\n[neural style transfer](https://keras.io/examples/generative/neural_style_transfer/),\namong other things.\n'
"\n## Extend the API using custom layers\n\n`keras` includes a wide range of built-in layers, for example:\n\n- Convolutional layers: `Conv1D`, `Conv2D`, `Conv3D`, `Conv2DTranspose`\n- Pooling layers: `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `AveragePooling1D`\n- RNN layers: `GRU`, `LSTM`, `ConvLSTM2D`\n- `BatchNormalization`, `Dropout`, `Embedding`, etc.\n\nBut if you don't find what you need, it's easy to extend the API by creating\nyour own layers. All layers subclass the `Layer` class and implement:\n\n- `call` method, that specifies the computation done by the layer.\n- `build` method, that creates the weights of the layer (this is just a style\nconvention since you can create weights in `__init__`, as well).\n\nTo learn more about creating layers from scratch, read\n[custom layers and models](/guides/making_new_layers_and_models_via_subclassing) guide.\n\nThe following is a basic implementation of `keras.layers.Dense`:\n"

class CustomDense(layers.Layer):

    def __init__(self, units=32):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.units = units

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        if False:
            print('Hello World!')
        return ops.matmul(inputs, self.w) + self.b
inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)
'\nFor serialization support in your custom layer, define a `get_config()`\nmethod that returns the constructor arguments of the layer instance:\n'

class CustomDense(layers.Layer):

    def __init__(self, units=32):
        if False:
            return 10
        super().__init__()
        self.units = units

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        return ops.matmul(inputs, self.w) + self.b

    def get_config(self):
        if False:
            print('Hello World!')
        return {'units': self.units}
inputs = keras.Input((4,))
outputs = CustomDense(10)(inputs)
model = keras.Model(inputs, outputs)
config = model.get_config()
new_model = keras.Model.from_config(config, custom_objects={'CustomDense': CustomDense})
'\nOptionally, implement the class method `from_config(cls, config)` which is used\nwhen recreating a layer instance given its config dictionary.\nThe default implementation of `from_config` is:\n\n```python\ndef from_config(cls, config):\n  return cls(**config)\n```\n'
"\n## When to use the functional API\n\nShould you use the Keras functional API to create a new model,\nor just subclass the `Model` class directly? In general, the functional API\nis higher-level, easier and safer, and has a number of\nfeatures that subclassed models do not support.\n\nHowever, model subclassing provides greater flexibility when building models\nthat are not easily expressible as directed acyclic graphs of layers.\nFor example, you could not implement a Tree-RNN with the functional API\nand would have to subclass `Model` directly.\n\nFor an in-depth look at the differences between the functional API and\nmodel subclassing, read\n[What are Symbolic and Imperative APIs in TensorFlow 2.0?](https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html).\n\n### Functional API strengths:\n\nThe following properties are also true for Sequential models\n(which are also data structures), but are not true for subclassed models\n(which are Python bytecode, not data structures).\n\n#### Less verbose\n\nThere is no `super().__init__(...)`, no `def call(self, ...):`, etc.\n\nCompare:\n\n```python\ninputs = keras.Input(shape=(32,))\nx = layers.Dense(64, activation='relu')(inputs)\noutputs = layers.Dense(10)(x)\nmlp = keras.Model(inputs, outputs)\n```\n\nWith the subclassed version:\n\n```python\nclass MLP(keras.Model):\n\n  def __init__(self, **kwargs):\n    super().__init__(**kwargs)\n    self.dense_1 = layers.Dense(64, activation='relu')\n    self.dense_2 = layers.Dense(10)\n\n  def call(self, inputs):\n    x = self.dense_1(inputs)\n    return self.dense_2(x)\n\n# Instantiate the model.\nmlp = MLP()\n# Necessary to create the model's state.\n# The model doesn't have a state until it's called at least once.\n_ = mlp(ops.zeros((1, 32)))\n```\n\n#### Model validation while defining its connectivity graph\n\nIn the functional API, the input specification (shape and dtype) is created\nin advance (using `Input`). Every time you call a layer,\nthe layer checks that the specification passed to it matches its assumptions,\nand it will raise a helpful error message if not.\n\nThis guarantees that any model you can build with the functional API will run.\nAll debugging -- other than convergence-related debugging --\nhappens statically during the model construction and not at execution time.\nThis is similar to type checking in a compiler.\n\n#### A functional model is plottable and inspectable\n\nYou can plot the model as a graph, and you can easily access intermediate nodes\nin this graph. For example, to extract and reuse the activations of intermediate\nlayers (as seen in a previous example):\n\n```python\nfeatures_list = [layer.output for layer in vgg19.layers]\nfeat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)\n```\n\n#### A functional model can be serialized or cloned\n\nBecause a functional model is a data structure rather than a piece of code,\nit is safely serializable and can be saved as a single file\nthat allows you to recreate the exact same model\nwithout having access to any of the original code.\nSee the [serialization & saving guide](/guides/serialization_and_saving/).\n\nTo serialize a subclassed model, it is necessary for the implementer\nto specify a `get_config()`\nand `from_config()` method at the model level.\n\n\n### Functional API weakness:\n\n#### It does not support dynamic architectures\n\nThe functional API treats models as DAGs of layers.\nThis is true for most deep learning architectures, but not all -- for example,\nrecursive networks or Tree RNNs do not follow this assumption and cannot\nbe implemented in the functional API.\n"
"\n## Mix-and-match API styles\n\nChoosing between the functional API or Model subclassing isn't a\nbinary decision that restricts you into one category of models.\nAll models in the `keras` API can interact with each other, whether they're\n`Sequential` models, functional models, or subclassed models that are written\nfrom scratch.\n\nYou can always use a functional model or `Sequential` model\nas part of a subclassed model or layer:\n"
units = 32
timesteps = 10
input_dim = 5
inputs = keras.Input((None, units))
x = layers.GlobalAveragePooling1D()(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

class CustomRNN(layers.Layer):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation='tanh')
        self.projection_2 = layers.Dense(units=units, activation='tanh')
        self.classifier = model

    def call(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        outputs = []
        state = ops.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = ops.stack(outputs, axis=1)
        print(features.shape)
        return self.classifier(features)
rnn_model = CustomRNN()
_ = rnn_model(ops.zeros((1, timesteps, input_dim)))
"\nYou can use any subclassed layer or model in the functional API\nas long as it implements a `call` method that follows one of the following patterns:\n\n- `call(self, inputs, **kwargs)` --\nWhere `inputs` is a tensor or a nested structure of tensors (e.g. a list of tensors),\nand where `**kwargs` are non-tensor arguments (non-inputs).\n- `call(self, inputs, training=None, **kwargs)` --\nWhere `training` is a boolean indicating whether the layer should behave\nin training mode and inference mode.\n- `call(self, inputs, mask=None, **kwargs)` --\nWhere `mask` is a boolean mask tensor (useful for RNNs, for instance).\n- `call(self, inputs, training=None, mask=None, **kwargs)` --\nOf course, you can have both masking and training-specific behavior at the same time.\n\nAdditionally, if you implement the `get_config` method on your custom Layer or model,\nthe functional models you create will still be serializable and cloneable.\n\nHere's a quick example of a custom RNN, written from scratch,\nbeing used in a functional model:\n"
units = 32
timesteps = 10
input_dim = 5
batch_size = 16

class CustomRNN(layers.Layer):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.units = units
        self.projection_1 = layers.Dense(units=units, activation='tanh')
        self.projection_2 = layers.Dense(units=units, activation='tanh')
        self.classifier = layers.Dense(1)

    def call(self, inputs):
        if False:
            print('Hello World!')
        outputs = []
        state = ops.zeros(shape=(inputs.shape[0], self.units))
        for t in range(inputs.shape[1]):
            x = inputs[:, t, :]
            h = self.projection_1(x)
            y = h + self.projection_2(state)
            state = y
            outputs.append(y)
        features = ops.stack(outputs, axis=1)
        return self.classifier(features)
inputs = keras.Input(batch_shape=(batch_size, timesteps, input_dim))
x = layers.Conv1D(32, 3)(inputs)
outputs = CustomRNN()(x)
model = keras.Model(inputs, outputs)
rnn_model = CustomRNN()
_ = rnn_model(ops.zeros((1, 10, 5)))