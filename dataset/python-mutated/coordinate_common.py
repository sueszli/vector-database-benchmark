import ivy

def coordinate_delta(sum_grad, sum_hess, w, reg_alpha, reg_lambda):
    if False:
        return 10
    mask = ivy.where(sum_hess < 1e-05, 0.0, 1.0)
    sum_grad_l2 = sum_grad + reg_lambda * w
    sum_hess_l2 = sum_hess + reg_lambda
    tmp = w - sum_grad_l2 / sum_hess_l2
    return ivy.where(tmp >= 0, ivy.fmax(-(sum_grad_l2 + reg_alpha) / sum_hess_l2, -w) * mask, ivy.fmin(-(sum_grad_l2 - reg_alpha) / sum_hess_l2, -w) * mask)

def coordinate_delta_bias(sum_grad, sum_hess):
    if False:
        for i in range(10):
            print('nop')
    return -sum_grad / sum_hess

def get_bias_gradient(gpair):
    if False:
        while True:
            i = 10
    mask = ivy.where(gpair[:, 1] < 0.0, 0.0, 1.0)
    sum_grad = ivy.sum(gpair[:, 0] * mask)
    sum_hess = ivy.sum(gpair[:, 1] * mask)
    return (sum_grad, sum_hess)

def update_bias_residual(dbias, gpair):
    if False:
        return 10
    mask = ivy.where(gpair[:, 1] < 0.0, 0.0, 1.0)
    return ivy.expand_dims(gpair[:, 0] + gpair[:, 1] * mask * dbias, axis=1)