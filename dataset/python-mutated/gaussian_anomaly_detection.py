"""Anomaly Detection Module"""
import math
import numpy as np

class GaussianAnomalyDetection:
    """GaussianAnomalyDetection Class"""

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        'GaussianAnomalyDetection constructor'
        (self.mu_param, self.sigma_squared) = GaussianAnomalyDetection.estimate_gaussian(data)
        self.data = data

    def multivariate_gaussian(self, data):
        if False:
            while True:
                i = 10
        'Computes the probability density function of the multivariate gaussian distribution'
        mu_param = self.mu_param
        sigma_squared = self.sigma_squared
        (num_examples, num_features) = data.shape
        probabilities = np.ones((num_examples, 1))
        for example_index in range(num_examples):
            for feature_index in range(num_features):
                power_dividend = (data[example_index, feature_index] - mu_param[feature_index]) ** 2
                power_divider = 2 * sigma_squared[feature_index]
                e_power = -1 * power_dividend / power_divider
                probability_prefix = 1 / math.sqrt(2 * math.pi * sigma_squared[feature_index])
                probability = probability_prefix * math.e ** e_power
                probabilities[example_index] *= probability
        return probabilities

    @staticmethod
    def estimate_gaussian(data):
        if False:
            return 10
        'This function estimates the parameters of a Gaussian distribution using the data in X.'
        num_examples = data.shape[0]
        mu_param = 1 / num_examples * np.sum(data, axis=0)
        sigma_squared = 1 / num_examples * np.sum((data - mu_param) ** 2, axis=0)
        return (mu_param, sigma_squared)

    @staticmethod
    def select_threshold(labels, probabilities):
        if False:
            i = 10
            return i + 15
        'Finds the best threshold (epsilon) to use for selecting outliers'
        best_epsilon = 0
        best_f1 = 0
        precision_history = []
        recall_history = []
        f1_history = []
        min_probability = np.min(probabilities)
        max_probability = np.max(probabilities)
        step_size = (max_probability - min_probability) / 1000
        for epsilon in np.arange(min_probability, max_probability, step_size):
            predictions = probabilities < epsilon
            false_positives = np.sum((predictions == 1) & (labels == 0))
            false_negatives = np.sum((predictions == 0) & (labels == 1))
            true_positives = np.sum((predictions == 1) & (labels == 1))
            if true_positives + false_positives == 0 or true_positives + false_negatives == 0:
                continue
            precision = true_positives / (true_positives + false_positives)
            recall = true_positives / (true_positives + false_negatives)
            f1_score = 2 * precision * recall / (precision + recall)
            precision_history.append(precision)
            recall_history.append(recall)
            f1_history.append(f1_score)
            if f1_score > best_f1:
                best_epsilon = epsilon
                best_f1 = f1_score
        return (best_epsilon, best_f1, precision_history, recall_history, f1_history)