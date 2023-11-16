"""shell
pip install autokeras
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import autokeras as ak
'\nIn this tutorial, we show how to customize your search space with\n[AutoModel](/auto_model/#automodel-class) and how to implement your own block\nas search space.  This API is mainly for advanced users who already know what\ntheir model should look like.\n\n## Customized Search Space\nFirst, let us see how we can build the following neural network using the\nbuilding blocks in AutoKeras.\n\n<div class="mermaid">\ngraph LR\n    id1(ImageInput) --> id2(Normalization)\n    id2 --> id3(Image Augmentation)\n    id3 --> id4(Convolutional)\n    id3 --> id5(ResNet V2)\n    id4 --> id6(Merge)\n    id5 --> id6\n    id6 --> id7(Classification Head)\n</div>\n\nWe can make use of the [AutoModel](/auto_model/#automodel-class) API in\nAutoKeras to implemented as follows.\nThe usage is the same as the [Keras functional\nAPI](https://www.tensorflow.org/guide/keras/functional).\nSince this is just a demo, we use small amount of `max_trials` and `epochs`.\n'
input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node1 = ak.ConvBlock()(output_node)
output_node2 = ak.ResNetBlock(version='v2')(output_node)
output_node = ak.Merge()([output_node1, output_node2])
output_node = ak.ClassificationHead()(output_node)
auto_model = ak.AutoModel(inputs=input_node, outputs=output_node, overwrite=True, max_trials=1)
"\nWhild building the model, the blocks used need to follow this topology:\n`Preprocessor` -> `Block` -> `Head`. `Normalization` and `ImageAugmentation`\nare `Preprocessor`s.\n`ClassificationHead` is `Head`. The rest are `Block`s.\n\nIn the code above, we use `ak.ResNetBlock(version='v2')` to specify the version\nof ResNet to use.  There are many other arguments to specify for each building\nblock.  For most of the arguments, if not specified, they would be tuned\nautomatically.  Please refer to the documentation links at the bottom of the\npage for more details.\n\nThen, we prepare some data to run the model.\n"
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(y_train[:3])
auto_model.fit(x_train[:100], y_train[:100], epochs=1)
predicted_y = auto_model.predict(x_test)
print(auto_model.evaluate(x_test, y_test))
'\nFor multiple input nodes and multiple heads search space, you can refer to\n[this section](/tutorial/multi/#customized-search-space).\n\n## Validation Data\nIf you would like to provide your own validation data or change the ratio of\nthe validation data, please refer to the Validation Data section of the\ntutorials of [Image\nClassification](/tutorial/image_classification/#validation-data), [Text\nClassification](/tutorial/text_classification/#validation-data), [Structured\nData\nClassification](/tutorial/structured_data_classification/#validation-data),\n[Multi-task and Multiple Validation](/tutorial/multi/#validation-data).\n\n## Data Format\nYou can refer to the documentation of\n[ImageInput](/node/#imageinput-class),\n[StructuredDataInput](/node/#structureddatainput-class),\n[TextInput](/node/#textinput-class),\n[RegressionHead](/block/#regressionhead-class),\n[ClassificationHead](/block/#classificationhead-class),\nfor the format of different types of data.\nYou can also refer to the Data Format section of the tutorials of\n[Image Classification](/tutorial/image_classification/#data-format),\n[Text Classification](/tutorial/text_classification/#data-format),\n[Structured Data\nClassification](/tutorial/structured_data_classification/#data-format).\n\n## Implement New Block\n\nYou can extend the [Block](/base/#block-class)\nclass to implement your own building blocks and use it with\n[AutoModel](/auto_model/#automodel-class).\n\nThe first step is to learn how to write a build function for\n[KerasTuner](https://keras-team.github.io/keras-tuner/#usage-the-basics).  You\nneed to override the [build function](/base/#build-method) of the block.  The\nfollowing example shows how to implement a single Dense layer block whose\nnumber of neurons is tunable.\n'

class SingleDenseLayerBlock(ak.Block):

    def build(self, hp, inputs=None):
        if False:
            print('Hello World!')
        input_node = tf.nest.flatten(inputs)[0]
        layer = tf.keras.layers.Dense(hp.Int('num_units', min_value=32, max_value=512, step=32))
        output_node = layer(input_node)
        return output_node
'\nYou can connect it with other blocks and build it into an\n[AutoModel](/auto_model/#automodel-class).\n'
input_node = ak.Input()
output_node = SingleDenseLayerBlock()(input_node)
output_node = ak.RegressionHead()(output_node)
auto_model = ak.AutoModel(input_node, output_node, overwrite=True, max_trials=1)
num_instances = 100
x_train = np.random.rand(num_instances, 20).astype(np.float32)
y_train = np.random.rand(num_instances, 1).astype(np.float32)
x_test = np.random.rand(num_instances, 20).astype(np.float32)
y_test = np.random.rand(num_instances, 1).astype(np.float32)
auto_model.fit(x_train, y_train, epochs=1)
print(auto_model.evaluate(x_test, y_test))
'\n## Reference\n\n[AutoModel](/auto_model/#automodel-class)\n\n**Nodes**:\n[ImageInput](/node/#imageinput-class),\n[Input](/node/#input-class),\n[StructuredDataInput](/node/#structureddatainput-class),\n[TextInput](/node/#textinput-class).\n\n**Preprocessors**:\n[FeatureEngineering](/block/#featureengineering-class),\n[ImageAugmentation](/block/#imageaugmentation-class),\n[LightGBM](/block/#lightgbm-class),\n[Normalization](/block/#normalization-class),\n[TextToIntSequence](/block/#texttointsequence-class),\n[TextToNgramVector](/block/#texttongramvector-class).\n\n**Blocks**:\n[ConvBlock](/block/#convblock-class),\n[DenseBlock](/block/#denseblock-class),\n[Embedding](/block/#embedding-class),\n[Merge](/block/#merge-class),\n[ResNetBlock](/block/#resnetblock-class),\n[RNNBlock](/block/#rnnblock-class),\n[SpatialReduction](/block/#spatialreduction-class),\n[TemporalReduction](/block/#temporalreduction-class),\n[XceptionBlock](/block/#xceptionblock-class),\n[ImageBlock](/block/#imageblock-class),\n[StructuredDataBlock](/block/#structureddatablock-class),\n[TextBlock](/block/#textblock-class).\n'