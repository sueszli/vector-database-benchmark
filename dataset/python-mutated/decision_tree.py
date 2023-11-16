import numpy
from sklearn.externals import joblib

class DecisionTree:

    def __init__(self, classifier):
        if False:
            print('Hello World!')
        self.classifier = classifier

    @staticmethod
    def load(file):
        if False:
            return 10
        data = joblib.load(file)
        tree = DecisionTree(data[0])
        return (tree, data[1])

    def classify_with_feature_vector(self, feature_vector, labels):
        if False:
            return 10
        numpy_format = []
        for label in labels:
            numpy_format.append((label, numpy.float64))
        converted_features = numpy.zeros(1, dtype=numpy_format)
        for name in converted_features.dtype.names:
            converted_features[name] = feature_vector[name]
        samples = converted_features.view(numpy.float64).reshape(converted_features.shape + (-1,))
        results = self.classifier.predict(samples)
        return numpy.array(results)