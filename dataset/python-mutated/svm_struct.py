import dlib

def main():
    if False:
        i = 10
        return i + 15
    samples = [[0, 2, 0], [1, 0, 0], [0, 4, 0], [0, 0, 3]]
    labels = [1, 0, 1, 2]
    problem = ThreeClassClassifierProblem(samples, labels)
    weights = dlib.solve_structural_svm_problem(problem)
    print(weights)
    for (k, s) in enumerate(samples):
        print('Predicted label for sample[{0}]: {1}'.format(k, predict_label(weights, s)))

def predict_label(weights, sample):
    if False:
        while True:
            i = 10
    'Given the 9-dimensional weight vector which defines a 3 class classifier,\n    predict the class of the given 3-dimensional sample vector.   Therefore, the\n    output of this function is either 0, 1, or 2 (i.e. one of the three possible\n    labels).'
    w0 = weights[0:3]
    w1 = weights[3:6]
    w2 = weights[6:9]
    scores = [dot(w0, sample), dot(w1, sample), dot(w2, sample)]
    max_scoring_label = scores.index(max(scores))
    return max_scoring_label

def dot(a, b):
    if False:
        print('Hello World!')
    'Compute the dot product between the two vectors a and b.'
    return sum((i * j for (i, j) in zip(a, b)))

class ThreeClassClassifierProblem:
    C = 1

    def __init__(self, samples, labels):
        if False:
            return 10
        self.num_samples = len(samples)
        self.num_dimensions = len(samples[0]) * 3
        self.samples = samples
        self.labels = labels

    def make_psi(self, x, label):
        if False:
            return 10
        'Compute PSI(x,label).'
        psi = dlib.vector()
        psi.resize(self.num_dimensions)
        dims = len(x)
        if label == 0:
            for i in range(0, dims):
                psi[i] = x[i]
        elif label == 1:
            for i in range(dims, 2 * dims):
                psi[i] = x[i - dims]
        else:
            for i in range(2 * dims, 3 * dims):
                psi[i] = x[i - 2 * dims]
        return psi

    def get_truth_joint_feature_vector(self, idx):
        if False:
            return 10
        return self.make_psi(self.samples[idx], self.labels[idx])

    def separation_oracle(self, idx, current_solution):
        if False:
            i = 10
            return i + 15
        samp = self.samples[idx]
        dims = len(samp)
        scores = [0, 0, 0]
        scores[0] = dot(current_solution[0:dims], samp)
        scores[1] = dot(current_solution[dims:2 * dims], samp)
        scores[2] = dot(current_solution[2 * dims:3 * dims], samp)
        if self.labels[idx] != 0:
            scores[0] += 1
        if self.labels[idx] != 1:
            scores[1] += 1
        if self.labels[idx] != 2:
            scores[2] += 1
        max_scoring_label = scores.index(max(scores))
        if max_scoring_label == self.labels[idx]:
            loss = 0
        else:
            loss = 1
        psi = self.make_psi(samp, max_scoring_label)
        return (loss, psi)
if __name__ == '__main__':
    main()