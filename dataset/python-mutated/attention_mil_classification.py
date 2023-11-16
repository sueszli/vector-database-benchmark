"""
Title: Classification using Attention-based Deep Multiple Instance Learning (MIL).
Author: [Mohamad Jaber](https://www.linkedin.com/in/mohamadjaber1/)
Date created: 2021/08/16
Last modified: 2021/11/25
Description: MIL approach to classify bags of instances and get their individual instance score.
Accelerator: GPU
"""
'\n## Introduction\n\n### What is Multiple Instance Learning (MIL)?\n\nUsually, with supervised learning algorithms, the learner receives labels for a set of\ninstances. In the case of MIL, the learner receives labels for a set of bags, each of which\ncontains a set of instances. The bag is labeled positive if it contains at least\none positive instance, and negative if it does not contain any.\n\n### Motivation\n\nIt is often assumed in image classification tasks that each image clearly represents a\nclass label. In medical imaging (e.g. computational pathology, etc.) an *entire image*\nis represented by a single class label (cancerous/non-cancerous) or a region of interest\ncould be given. However, one will be interested in knowing which patterns in the image\nis actually causing it to belong to that class. In this context, the image(s) will be\ndivided and the subimages will form the bag of instances.\n\nTherefore, the goals are to:\n\n1. Learn a model to predict a class label for a bag of instances.\n2. Find out which instances within the bag caused a position class label\nprediction.\n\n### Implementation\n\nThe following steps describe how the model works:\n\n1. The feature extractor layers extract feature embeddings.\n2. The embeddings are fed into the MIL attention layer to get\nthe attention scores. The layer is designed as permutation-invariant.\n3. Input features and their corresponding attention scores are multiplied together.\n4. The resulting output is passed to a softmax function for classification.\n\n### References\n\n- [Attention-based Deep Multiple Instance Learning](https://arxiv.org/abs/1802.04712).\n- Some of the attention operator code implementation was inspired from https://github.com/utayao/Atten_Deep_MIL.\n- Imbalanced data [tutorial](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)\nby TensorFlow.\n\n'
'\n## Setup\n'
import numpy as np
import keras
from keras import layers
from keras import ops
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.style.use('ggplot')
'\n## Create dataset\n\nWe will create a set of bags and assign their labels according to their contents.\nIf at least one positive instance\nis available in a bag, the bag is considered as a positive bag. If it does not contain any\npositive instance, the bag will be considered as negative.\n\n### Configuration parameters\n\n- `POSITIVE_CLASS`: The desired class to be kept in the positive bag.\n- `BAG_COUNT`: The number of training bags.\n- `VAL_BAG_COUNT`: The number of validation bags.\n- `BAG_SIZE`: The number of instances in a bag.\n- `PLOT_SIZE`: The number of bags to plot.\n- `ENSEMBLE_AVG_COUNT`: The number of models to create and average together. (Optional:\noften results in better performance - set to 1 for single model)\n'
POSITIVE_CLASS = 1
BAG_COUNT = 1000
VAL_BAG_COUNT = 300
BAG_SIZE = 3
PLOT_SIZE = 3
ENSEMBLE_AVG_COUNT = 1
'\n### Prepare bags\n\nSince the attention operator is a permutation-invariant operator, an instance with a\npositive class label is randomly placed among the instances in the positive bag.\n'

def create_bags(input_data, input_labels, positive_class, bag_count, instance_count):
    if False:
        while True:
            i = 10
    bags = []
    bag_labels = []
    input_data = np.divide(input_data, 255.0)
    count = 0
    for _ in range(bag_count):
        index = np.random.choice(input_data.shape[0], instance_count, replace=False)
        instances_data = input_data[index]
        instances_labels = input_labels[index]
        bag_label = 0
        if positive_class in instances_labels:
            bag_label = 1
            count += 1
        bags.append(instances_data)
        bag_labels.append(np.array([bag_label]))
    print(f'Positive bags: {count}')
    print(f'Negative bags: {bag_count - count}')
    return (list(np.swapaxes(bags, 0, 1)), np.array(bag_labels))
((x_train, y_train), (x_val, y_val)) = keras.datasets.mnist.load_data()
(train_data, train_labels) = create_bags(x_train, y_train, POSITIVE_CLASS, BAG_COUNT, BAG_SIZE)
(val_data, val_labels) = create_bags(x_val, y_val, POSITIVE_CLASS, VAL_BAG_COUNT, BAG_SIZE)
'\n## Create the model\n\nWe will now build the attention layer, prepare some utilities, then build and train the\nentire model.\n\n### Attention operator implementation\n\nThe output size of this layer is decided by the size of a single bag.\n\nThe attention mechanism uses a weighted average of instances in a bag, in which the sum\nof the weights must equal to 1 (invariant of the bag size).\n\nThe weight matrices (parameters) are **w** and **v**. To include positive and negative\nvalues, hyperbolic tangent element-wise non-linearity is utilized.\n\nA **Gated attention mechanism** can be used to deal with complex relations. Another weight\nmatrix, **u**, is added to the computation.\nA sigmoid non-linearity is used to overcome approximately linear behavior for *x* ∈ [−1, 1]\nby hyperbolic tangent non-linearity.\n'

class MILAttentionLayer(layers.Layer):
    """Implementation of the attention-based Deep MIL layer.

    Args:
      weight_params_dim: Positive Integer. Dimension of the weight matrix.
      kernel_initializer: Initializer for the `kernel` matrix.
      kernel_regularizer: Regularizer function applied to the `kernel` matrix.
      use_gated: Boolean, whether or not to use the gated mechanism.

    Returns:
      List of 2D tensors with BAG_SIZE length.
      The tensors are the attention scores after softmax with shape `(batch_size, 1)`.
    """

    def __init__(self, weight_params_dim, kernel_initializer='glorot_uniform', kernel_regularizer=None, use_gated=False, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.weight_params_dim = weight_params_dim
        self.use_gated = use_gated
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.v_init = self.kernel_initializer
        self.w_init = self.kernel_initializer
        self.u_init = self.kernel_initializer
        self.v_regularizer = self.kernel_regularizer
        self.w_regularizer = self.kernel_regularizer
        self.u_regularizer = self.kernel_regularizer

    def build(self, input_shape):
        if False:
            while True:
                i = 10
        input_dim = input_shape[0][1]
        self.v_weight_params = self.add_weight(shape=(input_dim, self.weight_params_dim), initializer=self.v_init, name='v', regularizer=self.v_regularizer, trainable=True)
        self.w_weight_params = self.add_weight(shape=(self.weight_params_dim, 1), initializer=self.w_init, name='w', regularizer=self.w_regularizer, trainable=True)
        if self.use_gated:
            self.u_weight_params = self.add_weight(shape=(input_dim, self.weight_params_dim), initializer=self.u_init, name='u', regularizer=self.u_regularizer, trainable=True)
        else:
            self.u_weight_params = None
        self.input_built = True

    def call(self, inputs):
        if False:
            print('Hello World!')
        instances = [self.compute_attention_scores(instance) for instance in inputs]
        instances = ops.stack(instances)
        alpha = ops.softmax(instances, axis=0)
        return [alpha[i] for i in range(alpha.shape[0])]

    def compute_attention_scores(self, instance):
        if False:
            for i in range(10):
                print('nop')
        original_instance = instance
        instance = ops.tanh(ops.tensordot(instance, self.v_weight_params, axes=1))
        if self.use_gated:
            instance = instance * ops.sigmoid(ops.tensordot(original_instance, self.u_weight_params, axes=1))
        return ops.tensordot(instance, self.w_weight_params, axes=1)
'\n## Visualizer tool\n\nPlot the number of bags (given by `PLOT_SIZE`) with respect to the class.\n\nMoreover, if activated, the class label prediction with its associated instance score\nfor each bag (after the model has been trained) can be seen.\n'

def plot(data, labels, bag_class, predictions=None, attention_weights=None):
    if False:
        while True:
            i = 10
    ' "Utility for plotting bags and attention weights.\n\n    Args:\n      data: Input data that contains the bags of instances.\n      labels: The associated bag labels of the input data.\n      bag_class: String name of the desired bag class.\n        The options are: "positive" or "negative".\n      predictions: Class labels model predictions.\n      If you don\'t specify anything, ground truth labels will be used.\n      attention_weights: Attention weights for each instance within the input data.\n      If you don\'t specify anything, the values won\'t be displayed.\n    '
    return
    labels = np.array(labels).reshape(-1)
    if bag_class == 'positive':
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        else:
            labels = np.where(labels == 1)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
    elif bag_class == 'negative':
        if predictions is not None:
            labels = np.where(predictions.argmax(1) == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
        else:
            labels = np.where(labels == 0)[0]
            bags = np.array(data)[:, labels[0:PLOT_SIZE]]
    else:
        print(f'There is no class {bag_class}')
        return
    print(f'The bag class label is {bag_class}')
    for i in range(PLOT_SIZE):
        figure = plt.figure(figsize=(8, 8))
        print(f'Bag number: {labels[i]}')
        for j in range(BAG_SIZE):
            image = bags[j][i]
            figure.add_subplot(1, BAG_SIZE, j + 1)
            plt.grid(False)
            if attention_weights is not None:
                plt.title(np.around(attention_weights[labels[i]][j], 2))
            plt.imshow(image)
        plt.show()
plot(val_data, val_labels, 'positive')
plot(val_data, val_labels, 'negative')
'\n## Create model\n\nFirst we will create some embeddings per instance, invoke the attention operator and then\nuse the softmax function to output the class probabilities.\n'

def create_model(instance_shape):
    if False:
        print('Hello World!')
    (inputs, embeddings) = ([], [])
    shared_dense_layer_1 = layers.Dense(128, activation='relu')
    shared_dense_layer_2 = layers.Dense(64, activation='relu')
    for _ in range(BAG_SIZE):
        inp = layers.Input(instance_shape)
        flatten = layers.Flatten()(inp)
        dense_1 = shared_dense_layer_1(flatten)
        dense_2 = shared_dense_layer_2(dense_1)
        inputs.append(inp)
        embeddings.append(dense_2)
    alpha = MILAttentionLayer(weight_params_dim=256, kernel_regularizer=keras.regularizers.L2(0.01), use_gated=True, name='alpha')(embeddings)
    multiply_layers = [layers.multiply([alpha[i], embeddings[i]]) for i in range(len(alpha))]
    concat = layers.concatenate(multiply_layers, axis=1)
    output = layers.Dense(2, activation='softmax')(concat)
    return keras.Model(inputs, output)
"\n## Class weights\n\nSince this kind of problem could simply turn into imbalanced data classification problem,\nclass weighting should be considered.\n\nLet's say there are 1000 bags. There often could be cases were ~90 % of the bags do not\ncontain any positive label and ~10 % do.\nSuch data can be referred to as **Imbalanced data**.\n\nUsing class weights, the model will tend to give a higher weight to the rare class.\n"

def compute_class_weights(labels):
    if False:
        i = 10
        return i + 15
    negative_count = len(np.where(labels == 0)[0])
    positive_count = len(np.where(labels == 1)[0])
    total_count = negative_count + positive_count
    return {0: 1 / negative_count * (total_count / 2), 1: 1 / positive_count * (total_count / 2)}
'\n## Build and train model\n\nThe model is built and trained in this section.\n'

def train(train_data, train_labels, val_data, val_labels, model):
    if False:
        return 10
    file_path = '/tmp/best_model.weights.h5'
    model_checkpoint = keras.callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=0, mode='min', save_best_only=True, save_weights_only=True)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=20, class_weight=compute_class_weights(train_labels), batch_size=1, callbacks=[early_stopping, model_checkpoint], verbose=0)
    model.load_weights(file_path)
    return model
instance_shape = train_data[0][0].shape
models = [create_model(instance_shape) for _ in range(ENSEMBLE_AVG_COUNT)]
print(models[0].summary())
trained_models = [train(train_data, train_labels, val_data, val_labels, model) for model in tqdm(models)]
'\n## Model evaluation\n\nThe models are now ready for evaluation.\nWith each model we also create an associated intermediate model to get the\nweights from the attention layer.\n\nWe will compute a prediction for each of our `ENSEMBLE_AVG_COUNT` models, and\naverage them together for our final prediction.\n'

def predict(data, labels, trained_models):
    if False:
        print('Hello World!')
    models_predictions = []
    models_attention_weights = []
    models_losses = []
    models_accuracies = []
    for model in trained_models:
        predictions = model.predict(data)
        models_predictions.append(predictions)
        intermediate_model = keras.Model(model.input, model.get_layer('alpha').output)
        intermediate_predictions = intermediate_model.predict(data)
        attention_weights = np.squeeze(np.swapaxes(intermediate_predictions, 1, 0))
        models_attention_weights.append(attention_weights)
        (loss, accuracy) = model.evaluate(data, labels, verbose=0)
        models_losses.append(loss)
        models_accuracies.append(accuracy)
    print(f'The average loss and accuracy are {np.sum(models_losses, axis=0) / ENSEMBLE_AVG_COUNT:.2f} and {100 * np.sum(models_accuracies, axis=0) / ENSEMBLE_AVG_COUNT:.2f} % resp.')
    return (np.sum(models_predictions, axis=0) / ENSEMBLE_AVG_COUNT, np.sum(models_attention_weights, axis=0) / ENSEMBLE_AVG_COUNT)
(class_predictions, attention_params) = predict(val_data, val_labels, trained_models)
plot(val_data, val_labels, 'positive', predictions=class_predictions, attention_weights=attention_params)
plot(val_data, val_labels, 'negative', predictions=class_predictions, attention_weights=attention_params)
'\n## Conclusion\n\nFrom the above plot, you can notice that the weights always sum to 1. In a\npositively predict bag, the instance which resulted in the positive labeling will have\na substantially higher attention score than the rest of the bag. However, in a negatively\npredicted bag, there are two cases:\n\n* All instances will have approximately similar scores.\n* An instance will have relatively higher score (but not as high as of a positive instance).\nThis is because the feature space of this instance is close to that of the positive instance.\n\n## Remarks\n\n- If the model is overfit, the weights will be equally distributed for all bags. Hence,\nthe regularization techniques are necessary.\n- In the paper, the bag sizes can differ from one bag to another. For simplicity, the\nbag sizes are fixed here.\n- In order not to rely on the random initial weights of a single model, averaging ensemble\nmethods should be considered.\n'