import numpy as np

def Jaccard(a, b):
    if False:
        print('Hello World!')
    return 1.0 * (a * b).sum() / (a + b - a * b).sum()

class Recommender:
    sim = None

    def similarity(self, x, distance):
        if False:
            print('Hello World!')
        y = np.ones((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                y[i, j] = distance(x[i], x[j])
        return y

    def fit(self, x, distance=Jaccard):
        if False:
            i = 10
            return i + 15
        self.sim = self.similarity(x, distance)

    def recommend(self, a):
        if False:
            while True:
                i = 10
        return np.dot(self.sim, a) * (1 - a)