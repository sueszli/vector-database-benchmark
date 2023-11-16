from abc import ABC, abstractmethod
import numpy as np
import math
import re
import progressbar
from chapter5 import RegressionTree, DecisionTree, ClassificationTree

class RegularizerBase(ABC):

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @abstractmethod
    def loss(self, **kwargs):
        if False:
            return 10
        raise NotImplementedError

    @abstractmethod
    def grad(self, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class L1Regularizer(RegularizerBase):

    def __init__(self, lambd=0.001):
        if False:
            print('Hello World!')
        super().__init__()
        self.lambd = lambd

    def loss(self, params):
        if False:
            return 10
        loss = 0
        pattern = re.compile('^W\\d+')
        for (key, val) in params.items():
            if pattern.match(key):
                loss += 0.5 * np.sum(np.abs(val)) * self.lambd
        return loss

    def grad(self, params):
        if False:
            return 10
        for (key, val) in params.items():
            grad = self.lambd * np.sign(val)
        return grad

class L2Regularizer(RegularizerBase):

    def __init__(self, lambd=0.001):
        if False:
            while True:
                i = 10
        super().__init__()
        self.lambd = lambd

    def loss(self, params):
        if False:
            print('Hello World!')
        loss = 0
        for (key, val) in params.items():
            loss += 0.5 * np.sum(np.square(val)) * self.lambd
        return loss

    def grad(self, params):
        if False:
            print('Hello World!')
        for (key, val) in params.items():
            grad = self.lambd * val
        return grad

class RegularizerInitializer(object):

    def __init__(self, regular_name='l2'):
        if False:
            while True:
                i = 10
        self.regular_name = regular_name

    def __call__(self):
        if False:
            return 10
        r = '([a-zA-Z]*)=([^,)]*)'
        regular_str = self.regular_name.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, regular_str)])
        if 'l1' in regular_str.lower():
            regular = L1Regularizer(**kwargs)
        elif 'l2' in regular_str.lower():
            regular = L2Regularizer(**kwargs)
        else:
            raise ValueError('Unrecognized regular: {}'.format(regular_str))
        return regular

class Image(object):

    def __init__(self, image):
        if False:
            print('Hello World!')
        self._set_params(image)

    def _set_params(self, image):
        if False:
            while True:
                i = 10
        self.img = image
        self.row = image.shape[0]
        self.col = image.shape[1]
        self.transform = None

    def Translation(self, delta_x, delta_y):
        if False:
            print('Hello World!')
        '\n        平移。\n        \n        参数说明：\n        delta_x：控制左右平移，若大于0左移，小于0右移\n        delta_y：控制上下平移，若大于0上移，小于0下移\n        '
        self.transform = np.array([[1, 0, delta_x], [0, 1, delta_y], [0, 0, 1]])

    def Resize(self, alpha):
        if False:
            for i in range(10):
                print('nop')
        '\n        缩放。\n        \n        参数说明：\n        alpha：缩放因子，不进行缩放设置为1\n        '
        self.transform = np.array([[alpha, 0, 0], [0, alpha, 0], [0, 0, 1]])

    def HorMirror(self):
        if False:
            while True:
                i = 10
        '\n        水平镜像。\n        '
        self.transform = np.array([[1, 0, 0], [0, -1, self.col - 1], [0, 0, 1]])

    def VerMirror(self):
        if False:
            print('Hello World!')
        '\n        垂直镜像。\n        '
        self.transform = np.array([[-1, 0, self.row - 1], [0, 1, 0], [0, 0, 1]])

    def Rotate(self, angle):
        if False:
            i = 10
            return i + 15
        '\n        旋转。\n        \n        参数说明：\n        angle：旋转角度\n        '
        self.transform = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])

    def operate(self):
        if False:
            print('Hello World!')
        temp = np.zeros(self.img.shape, dtype=self.img.dtype)
        for i in range(self.row):
            for j in range(self.col):
                temp_pos = np.array([i, j, 1])
                [x, y, z] = np.dot(self.transform, temp_pos)
                x = int(x)
                y = int(y)
                if x >= self.row or y >= self.col or x < 0 or (y < 0):
                    temp[i, j, :] = 0
                else:
                    temp[i, j, :] = self.img[x, y]
        return temp

    def __call__(self, act):
        if False:
            for i in range(10):
                print('nop')
        r = '([a-zA-Z]*)=([^,)]*)'
        act_str = act.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, act_str)])
        if 'translation' in act_str:
            self.Translation(**kwargs)
        elif 'resize' in act_str:
            self.Resize(**kwargs)
        elif 'hormirror' in act_str:
            self.HorMirror(**kwargs)
        elif 'vermirror' in act_str:
            self.VerMirror(**kwargs)
        elif 'rotate' in act_str:
            self.Rotate(**kwargs)
        return self.operate()

def early_stopping(valid):
    if False:
        for i in range(10):
            print('nop')
    '\n    参数说明：\n    valid：验证集正确率列表\n    '
    if len(valid) > 5:
        if valid[-1] < valid[-5] and valid[-2] < valid[-5] and (valid[-3] < valid[-5]) and (valid[-4] < valid[-5]):
            return True
    return False

def bootstrap_sample(X, Y):
    if False:
        for i in range(10):
            print('nop')
    (N, M) = X.shape
    idxs = np.random.choice(N, N, replace=True)
    return (X[idxs], Y[idxs])

class BaggingModel(object):

    def __init__(self, n_models):
        if False:
            print('Hello World!')
        '\n        参数说明：\n        n_models：网络模型数目\n        '
        self.models = []
        self.n_models = n_models

    def fit(self, X, Y):
        if False:
            i = 10
            return i + 15
        self.models = []
        for i in range(self.n_models):
            print('training {} base model:'.format(i))
            (X_samp, Y_samp) = bootstrap_sample(X, Y)
            model = DFN(hidden_dims_1=200, hidden_dims_2=10)
            model.fit(X_samp, Y_samp)
            self.models.append(model)

    def predict(self, X):
        if False:
            while True:
                i = 10
        model_preds = np.array([[np.argmax(t.forward(x)[0]) for x in X] for t in self.models])
        return self._vote(model_preds)

    def _vote(self, predictions):
        if False:
            while True:
                i = 10
        out = [np.bincount(x).argmax() for x in predictions.T]
        return np.array(out)

    def evaluate(self, X_test, y_test):
        if False:
            print('Hello World!')
        acc = 0.0
        y_pred = self.predict(X_test)
        y_true = np.argmax(y_test, axis=1)
        acc += np.sum(y_pred == y_true)
        return acc / X_test.shape[0]

class Dropout(ABC):

    def __init__(self, wrapped_layer, p):
        if False:
            for i in range(10):
                print('nop')
        '\n        参数说明：\n        wrapped_layer：被 dropout 的层\n        p：神经元保留率\n        '
        super().__init__()
        self._base_layer = wrapped_layer
        self.p = p
        self._init_wrapper_params()

    def _init_wrapper_params(self):
        if False:
            return 10
        self._wrapper_derived_variables = {'dropout_mask': None}
        self._wrapper_hyperparams = {'wrapper': 'Dropout', 'p': self.p}

    def flush_gradients(self):
        if False:
            return 10
        '\n        函数作用：调用 base layer 重置更新参数列表\n        '
        self._base_layer.flush_gradients()

    def update(self):
        if False:
            while True:
                i = 10
        '\n        函数作用：调用 base layer 更新参数\n        '
        self._base_layer.update()

    def forward(self, X, is_train=True):
        if False:
            print('Hello World!')
        '\n        参数说明：\n        X：输入数组；\n        is_train：是否为训练阶段，bool型；\n        '
        mask = np.ones(X.shape).astype(bool)
        if is_train:
            mask = (np.random.rand(*X.shape) < self.p) / self.p
            X = mask * X
        self._wrapper_derived_variables['dropout_mask'] = mask
        return self._base_layer.forward(X)

    def backward(self, dLda):
        if False:
            while True:
                i = 10
        return self._base_layer.backward(dLda)

    @property
    def hyperparams(self):
        if False:
            i = 10
            return i + 15
        hp = self._base_layer.hyperparams
        hpw = self._wrapper_hyperparams
        if 'wrappers' in hp:
            hp['wrappers'].append(hpw)
        else:
            hp['wrappers'] = [hpw]
        return hp
bar_widgets = ['Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='-', left='[', right=']'), ' ', progressbar.ETA()]

def get_random_subsets(X, y, n_subsets, replacements=True):
    if False:
        while True:
            i = 10
    '从训练数据中抽取数据子集 (默认可重复抽样)'
    n_samples = np.shape(X)[0]
    Xy = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
    np.random.shuffle(Xy)
    subsets = []
    subsample_size = int(n_samples // 2)
    if replacements:
        subsample_size = n_samples
    for _ in range(n_subsets):
        idx = np.random.choice(range(n_samples), size=np.shape(range(subsample_size)), replace=replacements)
        X = Xy[idx][:, :-1]
        y = Xy[idx][:, -1]
        subsets.append([X, y])
    return subsets

class Bagging:
    """
    Bagging分类器。使用一组分类树，这些分类树使用特征训练数据的随机子集。
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, min_gain=0, max_depth=float('inf')):
        if False:
            for i in range(10):
                print('nop')
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=min_gain, max_depth=self.max_depth))

    def fit(self, X, y):
        if False:
            return 10
        subsets = get_random_subsets(X, y, self.n_estimators)
        for i in self.progressbar(range(self.n_estimators)):
            (X_subset, y_subset) = subsets[i]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        y_preds = np.empty((X.shape[0], len(self.trees)))
        for (i, tree) in enumerate(self.trees):
            prediction = tree.predict(X)
            y_preds[:, i] = prediction
        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred

    def score(self, X, y):
        if False:
            for i in range(10):
                print('nop')
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

class RandomForest:
    """
    随机森林分类器。使用一组分类树，这些分类树使用特征的随机子集训练数据的随机子集。
    """

    def __init__(self, n_estimators=100, max_features=None, min_samples_split=2, min_gain=0, max_depth=float('inf')):
        if False:
            while True:
                i = 10
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=min_gain, max_depth=self.max_depth))

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        n_features = np.shape(X)[1]
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))
        subsets = get_random_subsets(X, y, self.n_estimators)
        for i in self.progressbar(range(self.n_estimators)):
            (X_subset, y_subset) = subsets[i]
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            self.trees[i].feature_indices = idx
            X_subset = X_subset[:, idx]
            self.trees[i].fit(X_subset, y_subset)

    def predict(self, X):
        if False:
            for i in range(10):
                print('nop')
        y_preds = np.empty((X.shape[0], len(self.trees)))
        for (i, tree) in enumerate(self.trees):
            idx = tree.feature_indices
            prediction = tree.predict(X[:, idx])
            y_preds[:, i] = prediction
        y_pred = []
        for sample_predictions in y_preds:
            y_pred.append(np.bincount(sample_predictions.astype('int')).argmax())
        return y_pred

    def score(self, X, y):
        if False:
            i = 10
            return i + 15
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

class DecisionStump:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaboost:
    """
    Adaboost 算法。
    """

    def __init__(self, n_estimators=5):
        if False:
            while True:
                i = 10
        self.n_estimators = n_estimators
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def fit(self, X, y):
        if False:
            print('Hello World!')
        (n_samples, n_features) = np.shape(X)
        w = np.full(n_samples, 1 / n_samples)
        self.trees = []
        for _ in self.progressbar(range(self.n_estimators)):
            tree = DecisionStump()
            min_error = float('inf')
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[X[:, feature_i] < threshold] = -1
                    error = sum(w[y != prediction])
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        tree.polarity = p
                        tree.threshold = threshold
                        tree.feature_index = feature_i
                        min_error = error
            tree.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            predictions = np.ones(np.shape(y))
            negative_idx = tree.polarity * X[:, tree.feature_index] < tree.polarity * tree.threshold
            predictions[negative_idx] = -1
            w *= np.exp(-tree.alpha * y * predictions)
            w /= np.sum(w)
            self.trees.append(tree)

    def predict(self, X):
        if False:
            while True:
                i = 10
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        for tree in self.trees:
            predictions = np.ones(np.shape(y_pred))
            negative_idx = tree.polarity * X[:, tree.feature_index] < tree.polarity * tree.threshold
            predictions[negative_idx] = -1
            y_pred += tree.alpha * predictions
        y_pred = np.sign(y_pred).flatten()
        return y_pred

    def score(self, X, y):
        if False:
            return 10
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

class Loss(ABC):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        if False:
            for i in range(10):
                print('nop')
        return NotImplementedError()

    @abstractmethod
    def grad(self, y, y_pred):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

class SquareLoss(Loss):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def loss(self, y, y_pred):
        if False:
            for i in range(10):
                print('nop')
        pass

    def grad(self, y, y_pred):
        if False:
            while True:
                i = 10
        return -(y - y_pred)

    def hess(self, y, y_pred):
        if False:
            return 10
        return 1

class CrossEntropyLoss(Loss):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def loss(self, y, y_pred):
        if False:
            while True:
                i = 10
        pass

    def grad(self, y, y_pred):
        if False:
            while True:
                i = 10
        return -(y - y_pred)

    def hess(self, y, y_pred):
        if False:
            for i in range(10):
                print('nop')
        return y_pred * (1 - y_pred)

def softmax(x):
    if False:
        print('Hello World!')
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def line_search(self, y, y_pred, h_pred):
    if False:
        print('Hello World!')
    Lp = 2 * np.sum((y - y_pred) * h_pred)
    Lpp = np.sum(h_pred * h_pred)
    return 1 if np.sum(Lpp) == 0 else Lp / Lpp

def to_categorical(x, n_classes=None):
    if False:
        return 10
    '\n    One-hot编码\n    '
    if not n_classes:
        n_classes = np.amax(x) + 1
    one_hot = np.zeros((x.shape[0], n_classes))
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot

class GradientBoostingDecisionTree(object):
    """
    GBDT 算法。用一组基学习器 (回归树) 学习损失函数的梯度。
    """

    def __init__(self, n_estimators, learning_rate=1, min_samples_split=2, min_impurity=1e-07, max_depth=float('inf'), is_regression=False, line_search=False):
        if False:
            print('Hello World!')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.is_regression = is_regression
        self.line_search = line_search
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        self.loss = SquareLoss()
        if not self.is_regression:
            self.loss = CrossEntropyLoss()

    def fit(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_regression:
            Y = to_categorical(Y.flatten())
        else:
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
        self.out_dims = Y.shape[1]
        self.trees = np.empty((self.n_estimators, self.out_dims), dtype=object)
        Y_pred = np.full(np.shape(Y), np.mean(Y, axis=0))
        self.weights = np.ones((self.n_estimators, self.out_dims))
        self.weights[1:, :] *= self.learning_rate
        for i in self.progressbar(range(self.n_estimators)):
            for c in range(self.out_dims):
                tree = RegressionTree(min_samples_split=self.min_samples_split, min_impurity=self.min_impurity, max_depth=self.max_depth)
                if not self.is_regression:
                    Y_hat = softmax(Y_pred)
                    (y, y_pred) = (Y[:, c], Y_hat[:, c])
                else:
                    (y, y_pred) = (Y[:, c], Y_pred[:, c])
                neg_grad = -1 * self.loss.grad(y, y_pred)
                tree.fit(X, neg_grad)
                h_pred = tree.predict(X)
                if self.line_search == True:
                    self.weights[i, c] *= line_search(y, y_pred, h_pred)
                Y_pred[:, c] += np.multiply(self.weights[i, c], h_pred)
                self.trees[i, c] = tree

    def predict(self, X):
        if False:
            return 10
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        for c in range(self.out_dims):
            y_pred = np.array([])
            for i in range(self.n_estimators):
                update = np.multiply(self.weights[i, c], self.trees[i, c].predict(X))
                y_pred = update if not y_pred.any() else y_pred + update
            Y_pred[:, c] = y_pred
        if not self.is_regression:
            Y_pred = Y_pred.argmax(axis=1)
        return Y_pred

    def score(self, X, y):
        if False:
            print('Hello World!')
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

class GradientBoostingRegressor(GradientBoostingDecisionTree):

    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2, min_impurity=1e-07, max_depth=float('inf'), is_regression=True, line_search=False):
        if False:
            for i in range(10):
                print('nop')
        super(GradientBoostingRegressor, self).__init__(n_estimators=n_estimators, learning_rate=learning_rate, min_samples_split=min_samples_split, min_impurity=min_impurity, max_depth=max_depth, is_regression=is_regression, line_search=line_search)

class GradientBoostingClassifier(GradientBoostingDecisionTree):

    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2, min_impurity=1e-07, max_depth=float('inf'), is_regression=False, line_search=False):
        if False:
            for i in range(10):
                print('nop')
        super(GradientBoostingClassifier, self).__init__(n_estimators=n_estimators, learning_rate=learning_rate, min_samples_split=min_samples_split, min_impurity=min_impurity, max_depth=max_depth, is_regression=is_regression, line_search=line_search)

class XGBoostRegressionTree(DecisionTree):
    """
    XGBoost 回归树。此处基于第五章介绍的决策树，故采用贪心算法找到特征上分裂点 (枚举特征上所有可能的分裂点)。
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-07, max_depth=float('inf'), loss=None, gamma=0.0, lambd=0.0):
        if False:
            return 10
        super(XGBoostRegressionTree, self).__init__(min_impurity=min_impurity, min_samples_split=min_samples_split, max_depth=max_depth)
        self.gamma = gamma
        self.lambd = lambd
        self.loss = loss

    def _split(self, y):
        if False:
            for i in range(10):
                print('nop')
        col = int(np.shape(y)[1] / 2)
        (y, y_pred) = (y[:, :col], y[:, col:])
        return (y, y_pred)

    def _gain(self, y, y_pred):
        if False:
            return 10
        nominator = np.power((y * self.loss.grad(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return nominator / (denominator + self.lambd)

    def _gain_by_taylor(self, y, y1, y2):
        if False:
            return 10
        (y, y_pred) = self._split(y)
        (y1, y1_pred) = self._split(y1)
        (y2, y2_pred) = self._split(y2)
        true_gain = self._gain(y1, y1_pred)
        false_gain = self._gain(y2, y2_pred)
        gain = self._gain(y, y_pred)
        return 0.5 * (true_gain + false_gain - gain) - self.gamma

    def _approximate_update(self, y):
        if False:
            i = 10
            return i + 15
        (y, y_pred) = self._split(y)
        gradient = self.loss.grad(y, y_pred).sum()
        hessian = self.loss.hess(y, y_pred).sum()
        leaf_approximation = -gradient / (hessian + self.lambd)
        return leaf_approximation

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        self._impurity_calculation = self._gain_by_taylor
        self._leaf_value_calculation = self._approximate_update
        super(XGBoostRegressionTree, self).fit(X, y)

class XGBoost(object):
    """
    XGBoost学习器。
    """

    def __init__(self, n_estimators=200, learning_rate=0.001, min_samples_split=2, min_impurity=1e-07, max_depth=2, is_regression=False, gamma=0.0, lambd=0.0):
        if False:
            while True:
                i = 10
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.gamma = gamma
        self.lambd = lambd
        self.is_regression = is_regression
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)
        self.loss = SquareLoss()
        if not self.is_regression:
            self.loss = CrossEntropyLoss()

    def fit(self, X, Y):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_regression:
            Y = to_categorical(Y.flatten())
        else:
            Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
        self.out_dims = Y.shape[1]
        self.trees = np.empty((self.n_estimators, self.out_dims), dtype=object)
        Y_pred = np.zeros(np.shape(Y))
        self.weights = np.ones((self.n_estimators, self.out_dims))
        self.weights[1:, :] *= self.learning_rate
        for i in self.progressbar(range(self.n_estimators)):
            for c in range(self.out_dims):
                tree = XGBoostRegressionTree(min_samples_split=self.min_samples_split, min_impurity=self.min_impurity, max_depth=self.max_depth, loss=self.loss, gamma=self.gamma, lambd=self.lambd)
                if not self.is_regression:
                    Y_hat = softmax(Y_pred)
                    (y, y_pred) = (Y[:, c], Y_hat[:, c])
                else:
                    (y, y_pred) = (Y[:, c], Y_pred[:, c])
                (y, y_pred) = (y.reshape(-1, 1), y_pred.reshape(-1, 1))
                y_and_ypred = np.concatenate((y, y_pred), axis=1)
                tree.fit(X, y_and_ypred)
                h_pred = tree.predict(X)
                Y_pred[:, c] += np.multiply(self.weights[i, c], h_pred)
                self.trees[i, c] = tree

    def predict(self, X):
        if False:
            i = 10
            return i + 15
        Y_pred = np.zeros((X.shape[0], self.out_dims))
        for c in range(self.out_dims):
            y_pred = np.array([])
            for i in range(self.n_estimators):
                update = np.multiply(self.weights[i, c], self.trees[i, c].predict(X))
                y_pred = update if not y_pred.any() else y_pred + update
            Y_pred[:, c] = y_pred
        if not self.is_regression:
            Y_pred = Y_pred.argmax(axis=1)
        return Y_pred

    def score(self, X, y):
        if False:
            i = 10
            return i + 15
        y_pred = self.predict(X)
        accuracy = np.sum(y == y_pred, axis=0) / len(y)
        return accuracy

class XGBRegressor(XGBoost):

    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2, min_impurity=1e-07, max_depth=float('inf'), is_regression=True, gamma=0.0, lambd=0.0):
        if False:
            i = 10
            return i + 15
        super(XGBRegressor, self).__init__(n_estimators=n_estimators, learning_rate=learning_rate, min_samples_split=min_samples_split, min_impurity=min_impurity, max_depth=max_depth, is_regression=is_regression, gamma=gamma, lambd=lambd)

class XGBClassifier(XGBoost):

    def __init__(self, n_estimators=200, learning_rate=1, min_samples_split=2, min_impurity=1e-07, max_depth=float('inf'), is_regression=False, gamma=0.0, lambd=0.0):
        if False:
            while True:
                i = 10
        super(XGBClassifier, self).__init__(n_estimators=n_estimators, learning_rate=learning_rate, min_samples_split=min_samples_split, min_impurity=min_impurity, max_depth=max_depth, is_regression=is_regression, gamma=gamma, lambd=lambd)