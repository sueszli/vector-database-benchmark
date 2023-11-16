import numpy as np

def lift_score(y_target, y_predicted, binary=True, positive_label=1):
    if False:
        print('Hello World!')
    'Lift measures the degree to which the predictions of a\n    classification model are better than randomly-generated predictions.\n\n    The in terms of True Positives (TP), True Negatives (TN),\n    False Positives (FP), and False Negatives (FN), the lift score is\n    computed as:\n    [ TP / (TP+FP) ] / [ (TP+FN) / (TP+TN+FP+FN) ]\n\n\n    Parameters\n    -----------\n    y_target : array-like, shape=[n_samples]\n        True class labels.\n    y_predicted : array-like, shape=[n_samples]\n        Predicted class labels.\n    binary : bool (default: True)\n        Maps a multi-class problem onto a\n        binary, where\n        the positive class is 1 and\n        all other classes are 0.\n    positive_label : int (default: 0)\n        Class label of the positive class.\n\n    Returns\n    ----------\n    score : float\n        Lift score in the range [0, infinity]\n\n    Examples\n    -----------\n    For usage examples, please see\n    https://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/\n    '
    if not isinstance(y_target, np.ndarray):
        targ_tmp = np.asarray(y_target)
    else:
        targ_tmp = y_target
    if not isinstance(y_predicted, np.ndarray):
        pred_tmp = np.asarray(y_predicted)
    else:
        pred_tmp = y_predicted
    pred_tmp = pred_tmp.T
    targ_tmp = targ_tmp.T
    if len(pred_tmp) != len(targ_tmp):
        raise AttributeError("`y_target` and `y_predicted`don't have the same number of elements.")
    if binary:
        targ_tmp = np.where(targ_tmp != positive_label, 0, 1)
        pred_tmp = np.where(pred_tmp != positive_label, 0, 1)
    binary_check_targ_tmp = np.extract(targ_tmp > 1, targ_tmp)
    binary_check_pred_tmp = np.extract(pred_tmp > 1, pred_tmp)
    if len(binary_check_targ_tmp) or len(binary_check_pred_tmp):
        raise AttributeError('`y_target` and `y_predicted` have different elements from 0 and 1.')
    return support(targ_tmp, pred_tmp) / (support(targ_tmp) * support(pred_tmp))

def support(y_target, y_predicted=None):
    if False:
        i = 10
        return i + 15
    'Support is the fraction of the true value\n        in predictions and target values.\n\n    Parameters\n    -----------\n    y_target : array-like, shape=[n_samples]\n        True class labels.\n    y_predicted : array-like, shape=[n_samples]\n        Predicted class labels.\n\n    Returns\n    ----------\n    score : float\n        Support score in the range [0, 1]\n\n    '
    if y_predicted is None:
        if y_target.ndim == 1:
            return (y_target == 1).sum() / float(y_target.shape[0])
        return (y_target == 1).all(axis=1).sum() / float(y_target.shape[0])
    else:
        all_prod = np.column_stack([y_target, y_predicted])
        return (all_prod == 1).all(axis=1).sum() / float(all_prod.shape[0])