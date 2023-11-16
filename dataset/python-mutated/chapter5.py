import numpy as np
import cvxopt
import math

class NaiveBayes:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.parameters = []
        self.y = None
        self.classes = None

    def fit(self, X, y):
        if False:
            while True:
                i = 10
        self.y = y
        self.classes = np.unique(y)
        for (i, c) in enumerate(self.classes):
            X_where_c = X[np.where(self.y == c)]
            self.parameters.append([])
            for col in X_where_c.T:
                parameters = {'mean': col.mean(), 'var': col.var()}
                self.parameters[i].append(parameters)

    def _calculate_prior(self, c):
        if False:
            for i in range(10):
                print('nop')
        '\n        先验函数。\n        '
        frequency = np.mean(self.y == c)
        return frequency

    def _calculate_likelihood(self, mean, var, X):
        if False:
            return 10
        '\n        似然函数。\n        '
        eps = 0.0001
        coeff = 1.0 / math.sqrt(2.0 * math.pi * var + eps)
        exponent = math.exp(-(math.pow(X - mean, 2) / (2 * var + eps)))
        return coeff * exponent

    def _calculate_probabilities(self, X):
        if False:
            i = 10
            return i + 15
        posteriors = []
        for (i, c) in enumerate(self.classes):
            posterior = self._calculate_prior(c)
            for (feature_value, params) in zip(X, self.parameters[i]):
                likelihood = self._calculate_likelihood(params['mean'], params['var'], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        y_pred = [self._calculate_probabilities(sample) for sample in X]
        return y_pred

    def score(self, X, y):
        if False:
            while True:
                i = 10
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

def Sigmoid(x):
    if False:
        print('Hello World!')
    return 1 / (1 + np.exp(-x))

class LogisticRegression:

    def __init__(self, learning_rate=0.1):
        if False:
            while True:
                i = 10
        self.param = None
        self.learning_rate = learning_rate
        self.sigmoid = Sigmoid

    def _initialize_parameters(self, X):
        if False:
            i = 10
            return i + 15
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        if False:
            return 10
        self._initialize_parameters(X)
        for i in range(n_iterations):
            y_pred = self.sigmoid(X.dot(self.param))
            self.param -= self.learning_rate * -(y - y_pred).dot(X)

    def predict(self, X):
        if False:
            print('Hello World!')
        y_pred = self.sigmoid(X.dot(self.param))
        return y_pred

    def score(self, X, y):
        if False:
            return 10
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy
cvxopt.solvers.options['show_progress'] = False

def linear_kernel(**kwargs):
    if False:
        print('Hello World!')
    '\n    线性核\n    '

    def f(x1, x2):
        if False:
            return 10
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power, coef, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    多项式核\n    '

    def f(x1, x2):
        if False:
            while True:
                i = 10
        return (np.inner(x1, x2) + coef) ** power
    return f

def rbf_kernel(gamma, **kwargs):
    if False:
        return 10
    '\n    高斯核\n    '

    def f(x1, x2):
        if False:
            print('Hello World!')
        distance = np.linalg.norm(x1 - x2) ** 2
        return np.exp(-gamma * distance)
    return f

class SupportVectorMachine:

    def __init__(self, kernel=linear_kernel, power=4, gamma=None, coef=4):
        if False:
            return 10
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        (n_samples, n_features) = np.shape(X)
        if not self.gamma:
            self.gamma = 1 / n_features
        self.kernel = self.kernel(power=self.power, gamma=self.gamma, coef=self.coef)
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')
        G = cvxopt.matrix(np.identity(n_samples) * -1)
        h = cvxopt.matrix(np.zeros(n_samples))
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        lagr_mult = np.ravel(minimization['x'])
        idx = lagr_mult > 1e-07
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        y_pred = []
        for sample in X:
            prediction = 0
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    def score(self, X, y):
        if False:
            while True:
                i = 10
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

class KNN:

    def __init__(self, k=10):
        if False:
            i = 10
            return i + 15
        self._k = k

    def fit(self, X, y):
        if False:
            print('Hello World!')
        self._unique_labels = np.unique(y)
        self._class_num = len(self._unique_labels)
        self._datas = X
        self._labels = y.astype(np.int32)

    def predict(self, X):
        if False:
            while True:
                i = 10
        dist = np.sum(np.square(X), axis=1, keepdims=True) - 2 * np.dot(X, self._datas.T)
        dist = dist + np.sum(np.square(self._datas), axis=1, keepdims=True).T
        dist = np.argsort(dist)[:, :self._k]
        return np.array([np.argmax(np.bincount(self._labels[dist][i])) for i in range(len(X))])
        idx = lagr_mult > 1e-07
        self.lagr_multipliers = lagr_mult[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        if False:
            print('Hello World!')
        y_pred = []
        for sample in X:
            prediction = 0
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    def score(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

class DecisionNode:

    def __init__(self, feature_i=None, threshold=None, value=None, true_branch=None, false_branch=None):
        if False:
            i = 10
            return i + 15
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.true_branch = true_branch
        self.false_branch = false_branch

def divide_on_feature(X, feature_i, threshold):
    if False:
        while True:
            i = 10
    '\n    依据切分变量和切分点，将数据集分为两个子区域\n    '
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])
    return np.array([X_1, X_2])

class DecisionTree(object):

    def __init__(self, min_samples_split=2, min_impurity=1e-07, max_depth=float('inf'), loss=None):
        if False:
            for i in range(10):
                print('nop')
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self._impurity_calculation = None
        self._leaf_value_calculation = None
        self.one_dim = None

    def fit(self, X, y):
        if False:
            while True:
                i = 10
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        if False:
            i = 10
            return i + 15
        '\n        递归方法建立决策树\n        '
        largest_impurity = 0
        best_criteria = None
        best_sets = None
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        Xy = np.concatenate((X, y), axis=1)
        (n_samples, n_features) = np.shape(X)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    (Xy1, Xy2) = divide_on_feature(Xy, feature_i, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        impurity = self._impurity_calculation(y, y1, y2)
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {'feature_i': feature_i, 'threshold': threshold}
                            best_sets = {'leftX': Xy1[:, :n_features], 'lefty': Xy1[:, n_features:], 'rightX': Xy2[:, :n_features], 'righty': Xy2[:, n_features:]}
        if largest_impurity > self.min_impurity:
            true_branch = self._build_tree(best_sets['leftX'], best_sets['lefty'], current_depth + 1)
            false_branch = self._build_tree(best_sets['rightX'], best_sets['righty'], current_depth + 1)
            return DecisionNode(feature_i=best_criteria['feature_i'], threshold=best_criteria['threshold'], true_branch=true_branch, false_branch=false_branch)
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        if False:
            print('Hello World!')
        '\n        预测样本，沿着树递归搜索\n        '
        if tree is None:
            tree = self.root
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_i]
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch
        return self.predict_value(x, branch)

    def predict(self, X):
        if False:
            print('Hello World!')
        y_pred = [self.predict_value(sample) for sample in X]
        return y_pred

    def score(self, X, y):
        if False:
            return 10
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

    def print_tree(self, tree=None, indent=' '):
        if False:
            print('Hello World!')
        '\n        输出树\n        '
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print('feature|threshold -> %s | %s' % (tree.feature_i, tree.threshold))
            print('%sT->' % indent, end='')
            self.print_tree(tree.true_branch, indent + indent)
            print('%sF->' % indent, end='')
            self.print_tree(tree.false_branch, indent + indent)

def calculate_entropy(y):
    if False:
        print('Hello World!')
    log2 = lambda x: math.log(x) / math.log(2)
    unique_labels = np.unique(y)
    entropy = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        entropy += -p * log2(p)
    return entropy

def calculate_gini(y):
    if False:
        return 10
    unique_labels = np.unique(y)
    var = 0
    for label in unique_labels:
        count = len(y[y == label])
        p = count / len(y)
        var += p ** 2
    return 1 - var

class ClassificationTree(DecisionTree):
    """
    分类树，在决策书节点选择计算信息增益/基尼指数，在叶子节点选择多数表决。
    """

    def _calculate_gini_index(self, y, y1, y2):
        if False:
            print('Hello World!')
        '\n        计算基尼指数\n        '
        p = len(y1) / len(y)
        gini = calculate_gini(y)
        gini_index = gini - p * calculate_gini(y1) - (1 - p) * calculate_gini(y2)
        return gini_index

    def _calculate_information_gain(self, y, y1, y2):
        if False:
            print('Hello World!')
        '\n        计算信息增益\n        '
        p = len(y1) / len(y)
        entropy = calculate_entropy(y)
        info_gain = entropy - p * calculate_entropy(y1) - (1 - p) * calculate_entropy(y2)
        return info_gain

    def _majority_vote(self, y):
        if False:
            print('Hello World!')
        '\n        多数表决\n        '
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    def fit(self, X, y):
        if False:
            print('Hello World!')
        self._impurity_calculation = self._calculate_gini_index
        self._leaf_value_calculation = self._majority_vote
        super(ClassificationTree, self).fit(X, y)

def calculate_mse(y):
    if False:
        print('Hello World!')
    return np.mean((y - np.mean(y)) ** 2)

def calculate_variance(y):
    if False:
        for i in range(10):
            print('nop')
    n_samples = np.shape(y)[0]
    variance = 1 / n_samples * np.diag((y - np.mean(y)).T.dot(y - np.mean(y)))
    return variance

class RegressionTree(DecisionTree):
    """
    回归树，在决策书节点选择计算MSE/方差降低，在叶子节点选择均值。
    """

    def _calculate_mse(self, y, y1, y2):
        if False:
            print('Hello World!')
        '\n        计算MSE降低\n        '
        mse_tot = calculate_mse(y)
        mse_1 = calculate_mse(y1)
        mse_2 = calculate_mse(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        mse_reduction = mse_tot - (frac_1 * mse_1 + frac_2 * mse_2)
        return mse_reduction

    def _calculate_variance_reduction(self, y, y1, y2):
        if False:
            print('Hello World!')
        '\n        计算方差降低\n        '
        var_tot = calculate_variance(y)
        var_1 = calculate_variance(y1)
        var_2 = calculate_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _mean_of_y(self, y):
        if False:
            return 10
        '\n        计算均值\n        '
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    def fit(self, X, y):
        if False:
            return 10
        self._impurity_calculation = self._calculate_mse
        self._leaf_value_calculation = self._mean_of_y
        super(RegressionTree, self).fit(X, y)

class PCA:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def fit(self, X, n_components):
        if False:
            while True:
                i = 10
        n_samples = np.shape(X)[0]
        covariance_matrix = 1 / (n_samples - 1) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))
        (eigenvalues, eigenvectors) = np.linalg.eig(covariance_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :n_components]
        X_transformed = X.dot(eigenvectors)

def distEclud(x, y):
    if False:
        return 10
    '\n    计算欧氏距离\n    '
    return np.sqrt(np.sum((x - y) ** 2))

def randomCent(dataSet, k):
    if False:
        print('Hello World!')
    '\n    为数据集构建一个包含 K 个随机质心的集合\n    '
    (m, n) = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))
        centroids[i, :] = dataSet[index, :]
    return centroids

class KMeans:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.dataSet = None
        self.k = None

    def fit(self, dataSet, k):
        if False:
            print('Hello World!')
        self.dataSet = dataSet
        self.k = k
        m = np.shape(dataSet)[0]
        clusterAssment = np.mat(np.zeros((m, 2)))
        clusterChange = True
        centroids = randomCent(self.dataSet, k)
        while clusterChange:
            clusterChange = False
            for i in range(m):
                minDist = 1000000.0
                minIndex = -1
                for j in range(k):
                    distance = distEclud(centroids[j, :], self.dataSet[i, :])
                    if distance < minDist:
                        minDist = distance
                        minIndex = j
                if clusterAssment[i, 0] != minIndex:
                    clusterChange = True
                    clusterAssment[i, :] = (minIndex, minDist ** 2)
            for j in range(k):
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]
                centroids[j, :] = np.mean(pointsInCluster, axis=0)
        return (centroids, clusterAssment)
        return X_transformed