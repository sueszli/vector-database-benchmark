import pandas as pd
import numpy as np
import itertools
import time
import re
from scipy.stats import norm
import matplotlib.pyplot as plt

def cal_conf_matrix(labels, preds):
    if False:
        i = 10
        return i + 15
    '\n    计算混淆矩阵。\n    \n    参数说明：\n    labels：样本标签 (真实结果)\n    preds：预测结果\n    '
    n_sample = len(labels)
    result = pd.DataFrame(index=range(0, n_sample), columns=('probability', 'label'))
    result['label'] = np.array(labels)
    result['probability'] = np.array(preds)
    cm = np.arange(4).reshape(2, 2)
    cm[0, 0] = len(result[result['label'] == 1][result['probability'] >= 0.5])
    cm[0, 1] = len(result[result['label'] == 1][result['probability'] < 0.5])
    cm[1, 0] = len(result[result['label'] == 0][result['probability'] >= 0.5])
    cm[1, 1] = len(result[result['label'] == 0][result['probability'] < 0.5])
    return cm

def cal_PRF1(labels, preds):
    if False:
        return 10
    '\n    计算查准率P，查全率R，F1值。\n    '
    cm = cal_conf_matrix(labels, preds)
    P = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    R = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    F1 = 2 * P * R / (P + R)
    return (P, R, F1)

def cal_PRcurve(labels, preds):
    if False:
        while True:
            i = 10
    '\n    计算PR曲线上的值。\n    '
    n_sample = len(labels)
    result = pd.DataFrame(index=range(0, n_sample), columns=('probability', 'label'))
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    result['label'] = np.array(labels)
    result['probability'] = np.array(preds)
    result.sort_values('probability', inplace=True, ascending=False)
    PandR = pd.DataFrame(index=range(len(labels)), columns=('P', 'R'))
    for j in range(len(result)):
        result_j = result.head(n=j + 1)
        P = len(result_j[result_j['label'] == 1]) / float(len(result_j))
        R = len(result_j[result_j['label'] == 1]) / float(len(result[result['label'] == 1]))
        PandR.iloc[j] = [P, R]
    return PandR

def cal_ROCcurve(labels, preds):
    if False:
        i = 10
        return i + 15
    '\n    计算ROC曲线上的值。\n    '
    n_sample = len(labels)
    result = pd.DataFrame(index=range(0, n_sample), columns=('probability', 'label'))
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    result['label'] = np.array(labels)
    result['probability'] = np.array(preds)
    result.sort_values('probability', inplace=True, ascending=False)
    TPRandFPR = pd.DataFrame(index=range(len(result)), columns=('TPR', 'FPR'))
    for j in range(len(result)):
        result_j = result.head(n=j + 1)
        TPR = len(result_j[result_j['label'] == 1]) / float(len(result[result['label'] == 1]))
        FPR = len(result_j[result_j['label'] == 0]) / float(len(result[result['label'] == 0]))
        TPRandFPR.iloc[j] = [TPR, FPR]
    return TPRandFPR

def timeit(func):
    if False:
        print('Hello World!')
    '\n    装饰器，计算函数执行时间\n    '

    def wrapper(*args, **kwargs):
        if False:
            print('Hello World!')
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        exec_time = time_end - time_start
        print('{function} exec time: {time}s'.format(function=func.__name__, time=exec_time))
        return result
    return wrapper

@timeit
def area_auc(labels, preds):
    if False:
        for i in range(10):
            print('nop')
    '\n    AUC值的梯度法计算\n    '
    TPRandFPR = cal_ROCcurve(labels, preds)
    auc = 0.0
    prev_x = 0
    for (x, y) in zip(TPRandFPR.FPR, TPRandFPR.TPR):
        if x != prev_x:
            auc += (x - prev_x) * y
            prev_x = x
    return auc

@timeit
def naive_auc(labels, preds):
    if False:
        i = 10
        return i + 15
    '\n    AUC值的概率法计算\n    '
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    total_pair = n_pos * n_neg
    labels_preds = zip(labels, preds)
    labels_preds = sorted(labels_preds, key=lambda x: x[1])
    count_neg = 0
    satisfied_pair = 0
    for i in range(len(labels_preds)):
        if labels_preds[i][0] == 1:
            satisfied_pair += count_neg
        else:
            count_neg += 1
    return satisfied_pair / float(total_pair)

class KernelBase(ABC):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.params = {}
        self.hyperparams = {}

    @abstractmethod
    def _kernel(self, X, Y):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __call__(self, X, Y=None):
        if False:
            while True:
                i = 10
        return self._kernel(X, Y)

    def __str__(self):
        if False:
            while True:
                i = 10
        (P, H) = (self.params, self.hyperparams)
        p_str = ', '.join(['{}={}'.format(k, v) for (k, v) in P.items()])
        return '{}({})'.format(H['op'], p_str)

    def summary(self):
        if False:
            for i in range(10):
                print('nop')
        return {'op': self.hyperparams['op'], 'params': self.params, 'hyperparams': self.hyperparams}

class RBFKernel(KernelBase):

    def __init__(self, sigma=None):
        if False:
            return 10
        '\n        RBF 核。\n        '
        super().__init__()
        self.hyperparams = {'op': 'RBFKernel'}
        self.params = {'sigma': sigma}

    def _kernel(self, X, Y=None):
        if False:
            while True:
                i = 10
        '\n        对 X 和 Y 的行的每一对计算 RBF 核。如果 Y 为空，则 Y=X。\n\n        参数说明：\n        X：输入数组，为 (n_samples, n_features)\n        Y：输入数组，为 (m_samples, n_features)\n        '
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        Y = X if Y is None else Y
        Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
        assert X.ndim == 2 and Y.ndim == 2, 'X and Y must have 2 dimensions'
        sigma = np.sqrt(X.shape[1] / 2) if self.params['sigma'] is None else self.params['sigma']
        (X, Y) = (X / sigma, Y / sigma)
        D = -2 * X @ Y.T + np.sum(Y ** 2, axis=1) + np.sum(X ** 2, axis=1)[:, np.newaxis]
        D[D < 0] = 0
        return np.exp(-0.5 * D)

class KernelInitializer(object):

    def __init__(self, param=None):
        if False:
            while True:
                i = 10
        self.param = param

    def __call__(self):
        if False:
            return 10
        r = '([a-zA-Z0-9]*)=([^,)]*)'
        kr_str = self.param.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, self.param)])
        if 'rbf' in kr_str:
            kernel = RBFKernel(**kwargs)
        else:
            raise NotImplementedError('{}'.format(kr_str))
        return kernel

class GPRegression:
    """
    高斯过程回归
    """

    def __init__(self, kernel='RBFKernel', sigma=1e-10):
        if False:
            i = 10
            return i + 15
        self.kernel = KernelInitializer(kernel)()
        self.params = {'GP_mean': None, 'GP_cov': None, 'X': None}
        self.hyperparams = {'kernel': str(self.kernel), 'sigma': sigma}

    def fit(self, X, y):
        if False:
            i = 10
            return i + 15
        '\n        用已有的样本集合得到 GP 先验。\n\n        参数说明：\n        X：输入数组，为 (n_samples, n_features)\n        y：输入数组 X 的目标值，为 (n_samples)\n        '
        mu = np.zeros(X.shape[0])
        Cov = self.kernel(X, X)
        self.params['X'] = X
        self.params['y'] = y
        self.params['GP_cov'] = Cov
        self.params['GP_mean'] = mu

    def predict(self, X_star, conf_interval=0.95):
        if False:
            for i in range(10):
                print('nop')
        '\n        对新的样本 X 进行预测。\n\n        参数说明：\n        X_star：输入数组，为 (n_samples, n_features)\n        conf_interval：置信区间，浮点型 (0, 1)，default=0.95\n        '
        X = self.params['X']
        y = self.params['y']
        K = self.params['GP_cov']
        sigma = self.hyperparams['sigma']
        K_star = self.kernel(X_star, X)
        K_star_star = self.kernel(X_star, X_star)
        sig = np.eye(K.shape[0]) * sigma
        K_y_inv = np.linalg.pinv(K + sig)
        mean = K_star @ K_y_inv @ y
        cov = K_star_star - K_star @ K_y_inv @ K_star.T
        percentile = norm.ppf(conf_interval)
        conf = percentile * np.sqrt(np.diag(cov))
        return (mean, conf, cov)

class BayesianOptimization:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.model = GPRegression()

    def acquisition_function(self, Xsamples):
        if False:
            i = 10
            return i + 15
        (mu, _, cov) = self.model.predict(Xsamples)
        mu = mu if mu.ndim == 1 else mu.T[0]
        ysample = np.random.multivariate_normal(mu, cov)
        return ysample

    def opt_acquisition(self, X, n_samples=20):
        if False:
            for i in range(10):
                print('nop')
        Xsamples = np.random.randint(low=1, high=50, size=n_samples * X.shape[1])
        Xsamples = Xsamples.reshape(n_samples, X.shape[1])
        scores = self.acquisition_function(Xsamples)
        ix = np.argmax(scores)
        return Xsamples[ix, 0]

    def fit(self, f, X, y):
        if False:
            i = 10
            return i + 15
        self.model.fit(X, y)
        for i in range(15):
            x_star = self.opt_acquisition(X)
            y_star = f(x_star)
            (mean, conf, cov) = self.model.predict(np.array([[x_star]]))
            X = np.vstack((X, [[x_star]]))
            y = np.vstack((y, [[y_star]]))
            self.model.fit(X, y)
        ix = np.argmax(y)
        print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))
        return (X[ix], y[ix])