import ivy
from ivy.functional.frontends.xgboost.linear.coordinate_common import get_bias_gradient, coordinate_delta_bias, update_bias_residual, coordinate_delta

def coordinate_updater(gpair, data, lr, weight, n_feat, n_iter, reg_alpha, reg_lambda):
    if False:
        return 10
    "\n    Implements one step of coordinate descent. The original optimizer implements\n    parallel calculations. The below code is an approximation of the original one, but\n    rather than computing the update direction for a single parameter at a time using a\n    for loop and cumulative gradients, it does the update in parallel by means of\n    matrix-vector multiplications. Given that xgboost's updater is non-deterministic,\n    the approximated and original implementations converge to pretty the same optima,\n    resulting in metrics' values(accuracy, f1-score) differing at a level of 0.001(for\n    separate runs metrics may end up being the same).\n\n    Parameters\n    ----------\n    gpair\n        Array of shape (n_samples, 2) holding gradient-hessian pairs.\n    data\n        Training data of shape (n_samples, n_features).\n    lr\n        Learning rate.\n    weight\n        Array of shape (n_features+1, n_output_group) holding feature weights\n        and biases.\n    n_feat\n        Number of features in the training data.\n    n_iter\n        Number of current iteration.\n    reg_alpha\n        Denormalized regularization parameter alpha.\n    reg_lambda\n        Denormalized regularization parameter lambda.\n\n    Returns\n    -------\n        Updated weights of shape (n_features+1, n_output_group).\n    "
    bias_grad = get_bias_gradient(gpair)
    dbias = lr * coordinate_delta_bias(bias_grad[0], bias_grad[1])
    bias_weights = weight[-1] + dbias
    grad = update_bias_residual(dbias, gpair)
    hess = ivy.expand_dims(gpair[:, 1], axis=1)
    mask = ivy.where(hess < 0.0, 0.0, 1.0)
    sum_hess = ivy.sum(ivy.square(data) * hess * mask, axis=0, keepdims=True)
    sum_grad = ivy.sum(data * grad * mask, axis=0, keepdims=True)
    dw = lr * coordinate_delta(sum_grad.T, sum_hess.T, weight[:-1, :], reg_alpha, reg_lambda)
    feature_weights = weight[:-1] + dw
    return ivy.vstack([feature_weights, bias_weights])