import warnings
import numpy as np
from scipy.optimize import minimize

def create_counterfactual(x_reference, y_desired, model, X_dataset, y_desired_proba=None, lammbda=0.1, random_seed=None):
    if False:
        return 10
    '\n    Implementation of the counterfactual method by Wachter et al. 2017\n\n    References:\n\n    - Wachter, S., Mittelstadt, B., & Russell, C. (2017).\n    Counterfactual explanations without opening the black box:\n     Automated decisions and the GDPR. Harv. JL & Tech., 31, 841.,\n     https://arxiv.org/abs/1711.00399\n\n    Parameters\n    ----------\n\n    x_reference : array-like, shape=[m_features]\n        The data instance (training example) to be explained.\n\n    y_desired : int\n        The desired class label for `x_reference`.\n\n    model : estimator\n        A (scikit-learn) estimator implementing `.predict()` and/or\n        `predict_proba()`.\n        - If `model` supports `predict_proba()`, then this is used by\n        default for the first loss term,\n        `(lambda * model.predict[_proba](x_counterfact) - y_desired[_proba])^2`\n        - Otherwise, method will fall back to `predict`.\n\n    X_dataset : array-like, shape=[n_examples, m_features]\n        A (training) dataset for picking the initial counterfactual\n        as initial value for starting the optimization procedure.\n\n    y_desired_proba : float (default: None)\n        A float within the range [0, 1] designating the desired\n        class probability for `y_desired`.\n        - If `y_desired_proba=None` (default), the first loss term\n        is `(lambda * model(x_counterfact) - y_desired)^2` where `y_desired`\n        is a class label\n        - If `y_desired_proba` is not None, the first loss term\n        is `(lambda * model(x_counterfact) - y_desired_proba)^2`\n\n    lammbda : Weighting parameter for the first loss term,\n        `(lambda * model(x_counterfact) - y_desired[_proba])^2`\n\n    random_seed : int (default=None)\n        If int, random_seed is the seed used by\n        the random number generator for selecting the inital counterfactual\n        from `X_dataset`.\n\n    '
    if y_desired_proba is not None:
        use_proba = True
        if not hasattr(model, 'predict_proba'):
            raise AttributeError('Your `model` does not support `predict_proba`. Set `y_desired_proba`  to `None` to use `predict`instead.')
    else:
        use_proba = False
    if y_desired_proba is None:
        y_to_be_annealed_to = y_desired
    else:
        y_to_be_annealed_to = y_desired_proba
    rng = np.random.RandomState(random_seed)
    x_counterfact = X_dataset[rng.randint(X_dataset.shape[0])]
    mad = np.abs(np.median(X_dataset, axis=0) - x_reference)

    def dist(x_reference, x_counterfact):
        if False:
            i = 10
            return i + 15
        numerator = np.abs(x_reference - x_counterfact)
        return np.sum(numerator / mad)

    def loss(x_counterfact, lammbda):
        if False:
            for i in range(10):
                print('nop')
        if use_proba:
            y_predict = model.predict_proba(x_counterfact.reshape(1, -1)).flatten()[y_desired]
        else:
            y_predict = model.predict(x_counterfact.reshape(1, -1))
        diff = lammbda * (y_predict - y_to_be_annealed_to) ** 2
        return diff + dist(x_reference, x_counterfact)
    res = minimize(loss, x_counterfact, args=lammbda, method='Nelder-Mead')
    if not res['success']:
        warnings.warn(res['message'])
    x_counterfact = res['x']
    return x_counterfact