"""
Reducer module
"""
import pickle
try:
    from sklearn.decomposition import TruncatedSVD
    REDUCER = True
except ImportError:
    REDUCER = False
from ...version import __pickle__

class Reducer:
    """
    LSA dimensionality reduction model
    """

    def __init__(self, embeddings=None, components=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a dimensionality reduction model.\n\n        Args:\n            embeddings: input embeddings matrix\n            components: number of model components\n        '
        if not REDUCER:
            raise ImportError('Dimensionality reduction is not available - install "similarity" extra to enable')
        self.model = self.build(embeddings, components) if embeddings is not None and components else None

    def __call__(self, embeddings):
        if False:
            while True:
                i = 10
        '\n        Applies a dimensionality reduction model to embeddings, removed the top n principal components. Operation applied\n        directly on array.\n\n        Args:\n            embeddings: input embeddings matrix\n        '
        pc = self.model.components_
        factor = embeddings.dot(pc.transpose())
        if pc.shape[0] == 1:
            embeddings -= factor * pc
        elif len(embeddings.shape) > 1:
            for x in range(embeddings.shape[0]):
                embeddings[x] -= factor[x].dot(pc)
        else:
            embeddings -= factor.dot(pc)

    def build(self, embeddings, components):
        if False:
            print('Hello World!')
        '\n        Builds a LSA model. This model is used to remove the principal component within embeddings. This helps to\n        smooth out noisy embeddings (common words with less value).\n\n        Args:\n            embeddings: input embeddings matrix\n            components: number of model components\n\n        Returns:\n            LSA model\n        '
        model = TruncatedSVD(n_components=components, random_state=0)
        model.fit(embeddings)
        return model

    def load(self, path):
        if False:
            return 10
        '\n        Loads a Reducer object from path.\n\n        Args:\n            path: directory path to load model\n        '
        with open(path, 'rb') as handle:
            self.model = pickle.load(handle)

    def save(self, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves a Reducer object to path.\n\n        Args:\n            path: directory path to save model\n        '
        with open(path, 'wb') as handle:
            pickle.dump(self.model, handle, protocol=__pickle__)