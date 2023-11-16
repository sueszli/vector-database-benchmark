"""Neural Network Module"""
import numpy as np
from ..utils.features import prepare_for_training
from ..utils.hypothesis import sigmoid, sigmoid_gradient

class MultilayerPerceptron:
    """Multilayer Perceptron Class"""

    def __init__(self, data, labels, layers, epsilon, normalize_data=False):
        if False:
            while True:
                i = 10
        'Multilayer perceptron constructor.\n\n        :param data: training set.\n        :param labels: training set outputs (correct values).\n        :param layers: network layers configuration.\n        :param epsilon: Defines the range for initial theta values.\n        :param normalize_data: flag that indicates that features should be normalized.\n        '
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed
        self.labels = labels
        self.layers = layers
        self.epsilon = epsilon
        self.normalize_data = normalize_data
        self.thetas = MultilayerPerceptron.thetas_init(layers, epsilon)

    def train(self, regularization_param=0, max_iterations=1000, alpha=1):
        if False:
            for i in range(10):
                print('nop')
        'Train the model'
        unrolled_thetas = MultilayerPerceptron.thetas_unroll(self.thetas)
        (optimized_thetas, cost_history) = MultilayerPerceptron.gradient_descent(self.data, self.labels, unrolled_thetas, self.layers, regularization_param, max_iterations, alpha)
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_thetas, self.layers)
        return (self.thetas, cost_history)

    def predict(self, data):
        if False:
            return 10
        'Predictions function that does classification using trained model'
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        num_examples = data_processed.shape[0]
        predictions = MultilayerPerceptron.feedforward_propagation(data_processed, self.thetas, self.layers)
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    @staticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, regularization_param, max_iteration, alpha):
        if False:
            i = 10
            return i + 15
        'Gradient descent function.\n\n        Iteratively optimizes theta model parameters.\n\n        :param data: the set of training or test data.\n        :param labels: training set outputs (0 or 1 that defines the class of an example).\n        :param unrolled_theta: initial model parameters.\n        :param layers: model layers configuration.\n        :param regularization_param: regularization parameter.\n        :param max_iteration: maximum number of gradient descent steps.\n        :param alpha: gradient descent step size.\n        '
        optimized_theta = unrolled_theta
        cost_history = []
        for _ in range(max_iteration):
            cost = MultilayerPerceptron.cost_function(data, labels, MultilayerPerceptron.thetas_roll(optimized_theta, layers), layers, regularization_param)
            cost_history.append(cost)
            theta_gradient = MultilayerPerceptron.gradient_step(data, labels, optimized_theta, layers, regularization_param)
            optimized_theta = optimized_theta - alpha * theta_gradient
        return (optimized_theta, cost_history)

    @staticmethod
    def gradient_step(data, labels, unrolled_thetas, layers, regularization_param):
        if False:
            while True:
                i = 10
        'Gradient step function.\n\n        Computes the cost and gradient of the neural network for unrolled theta parameters.\n\n        :param data: training set.\n        :param labels: training set labels.\n        :param unrolled_thetas: model parameters.\n        :param layers: model layers configuration.\n        :param regularization_param: parameters that fights with model over-fitting.\n        '
        thetas = MultilayerPerceptron.thetas_roll(unrolled_thetas, layers)
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data, labels, thetas, layers, regularization_param)
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients

    @staticmethod
    def cost_function(data, labels, thetas, layers, regularization_param):
        if False:
            for i in range(10):
                print('nop')
        'Cost function.\n\n        It shows how accurate our model is based on current model parameters.\n\n        :param data: the set of training or test data.\n        :param labels: training set outputs (0 or 1 that defines the class of an example).\n        :param thetas: model parameters.\n        :param layers: layers configuration.\n        :param regularization_param: regularization parameter.\n        '
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]
        predictions = MultilayerPerceptron.feedforward_propagation(data, thetas, layers)
        bitwise_labels = np.zeros((num_examples, num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1
        theta_square_sum = 0
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            theta_square_sum = theta_square_sum + np.sum(theta[:, 1:] ** 2)
        regularization = regularization_param / (2 * num_examples) * theta_square_sum
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        cost = -1 / num_examples * (bit_set_cost + bit_not_set_cost) + regularization
        return cost

    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        if False:
            while True:
                i = 10
        'Feedforward propagation function'
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(in_layer_activation @ theta.T)
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation
        return in_layer_activation[:, 1:]

    @staticmethod
    def back_propagation(data, labels, thetas, layers, regularization_param):
        if False:
            return 10
        'Backpropagation function'
        num_layers = len(layers)
        (num_examples, num_features) = data.shape
        num_label_types = layers[-1]
        deltas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count + 1))
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layer_activation = data[example_index, :].reshape((num_features, 1))
            layers_activations[0] = layer_activation
            for layer_index in range(num_layers - 1):
                layer_theta = thetas[layer_index]
                layer_input = layer_theta @ layer_activation
                layer_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))
                layers_inputs[layer_index + 1] = layer_input
                layers_activations[layer_index + 1] = layer_activation
            output_layer_activation = layer_activation[1:, :]
            delta = {}
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1
            delta[num_layers - 1] = output_layer_activation - bitwise_label
            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index + 1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array([[1]]), layer_input))
                delta[layer_index] = layer_theta.T @ next_delta * sigmoid_gradient(layer_input)
                delta[layer_index] = delta[layer_index][1:, :]
            for layer_index in range(num_layers - 1):
                layer_delta = delta[layer_index + 1] @ layers_activations[layer_index].T
                deltas[layer_index] = deltas[layer_index] + layer_delta
        for layer_index in range(num_layers - 1):
            current_delta = deltas[layer_index]
            current_delta = np.hstack((np.zeros((current_delta.shape[0], 1)), current_delta[:, 1:]))
            regularization = regularization_param / num_examples * current_delta
            deltas[layer_index] = 1 / num_examples * deltas[layer_index] + regularization
        return deltas

    @staticmethod
    def thetas_init(layers, epsilon):
        if False:
            return 10
        'Randomly initialize the weights for each neural network layer\n\n        Each layer will have its own theta matrix W with L_in incoming connections and L_out\n        outgoing connections. Note that W will be set to a matrix of size(L_out, 1 + L_in) as the\n        first column of W handles the "bias" terms.\n\n        :param layers:\n        :param epsilon:\n        :return:\n        '
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 2 * epsilon - epsilon
        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        if False:
            while True:
                i = 10
        'Unrolls cells of theta matrices into one long vector.'
        unrolled_thetas = np.array([])
        num_theta_layers = len(thetas)
        for theta_layer_index in range(num_theta_layers):
            unrolled_thetas = np.hstack((unrolled_thetas, thetas[theta_layer_index].flatten()))
        return unrolled_thetas

    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        if False:
            return 10
        'Rolls NN params vector into the matrix'
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            thetas_width = in_count + 1
            thetas_height = out_count
            thetas_volume = thetas_width * thetas_height
            start_index = unrolled_shift
            end_index = unrolled_shift + thetas_volume
            layer_thetas_unrolled = unrolled_thetas[start_index:end_index]
            thetas[layer_index] = layer_thetas_unrolled.reshape((thetas_height, thetas_width))
            unrolled_shift = unrolled_shift + thetas_volume
        return thetas