import numpy as np
import paddle
from .line_search import strong_wolfe
from .utils import _value_and_gradient, check_initial_inverse_hessian_estimate, check_input_type

def minimize_bfgs(objective_func, initial_position, max_iters=50, tolerance_grad=1e-07, tolerance_change=1e-09, initial_inverse_hessian_estimate=None, line_search_fn='strong_wolfe', max_line_search_iters=50, initial_step_length=1.0, dtype='float32', name=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Minimizes a differentiable function `func` using the BFGS method.\n    The BFGS is a quasi-Newton method for solving an unconstrained optimization problem over a differentiable function.\n    Closely related is the Newton method for minimization. Consider the iterate update formula:\n\n    .. math::\n        x_{k+1} = x_{k} + H_k \\nabla{f_k}\n\n    If :math:`H_k` is the inverse Hessian of :math:`f` at :math:`x_k`, then it's the Newton method.\n    If :math:`H_k` is symmetric and positive definite, used as an approximation of the inverse Hessian, then\n    it's a quasi-Newton. In practice, the approximated Hessians are obtained\n    by only using the gradients, over either whole or part of the search\n    history, the former is BFGS, the latter is L-BFGS.\n\n    Reference:\n        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006. pp140: Algorithm 6.1 (BFGS Method).\n\n    Args:\n        objective_func: the objective function to minimize. ``objective_func`` accepts a 1D Tensor and returns a scalar.\n        initial_position (Tensor): the starting point of the iterates, has the same shape with the input of ``objective_func`` .\n        max_iters (int, optional): the maximum number of minimization iterations. Default value: 50.\n        tolerance_grad (float, optional): terminates if the gradient norm is smaller than this. Currently gradient norm uses inf norm. Default value: 1e-7.\n        tolerance_change (float, optional): terminates if the change of function value/position/parameter between two iterations is smaller than this value. Default value: 1e-9.\n        initial_inverse_hessian_estimate (Tensor, optional): the initial inverse hessian approximation at initial_position. It must be symmetric and positive definite. If not given, will use an identity matrix of order N, which is size of ``initial_position`` . Default value: None.\n        line_search_fn (str, optional): indicate which line search method to use, only support 'strong wolfe' right now. May support 'Hager Zhang' in the futrue. Default value: 'strong wolfe'.\n        max_line_search_iters (int, optional): the maximum number of line search iterations. Default value: 50.\n        initial_step_length (float, optional): step length used in first iteration of line search. different initial_step_length may cause different optimal result. For methods like Newton and quasi-Newton the initial trial step length should always be 1.0. Default value: 1.0.\n        dtype ('float32' | 'float64', optional): data type used in the algorithm, the data type of the input parameter must be consistent with the dtype. Default value: 'float32'.\n        name (str, optional): Name for the operation. For more information, please refer to :ref:`api_guide_Name`. Default value: None.\n\n    Returns:\n        output(tuple):\n\n            - is_converge (bool): Indicates whether found the minimum within tolerance.\n            - num_func_calls (int): number of objective function called.\n            - position (Tensor): the position of the last iteration. If the search converged, this value is the argmin of the objective function regrading to the initial position.\n            - objective_value (Tensor): objective function value at the `position`.\n            - objective_gradient (Tensor): objective function gradient at the `position`.\n            - inverse_hessian_estimate (Tensor): the estimate of inverse hessian at the `position`.\n\n    Examples:\n        .. code-block:: python\n            :name: code-example1\n\n            >>> # Example1: 1D Grid Parameters\n            >>> import paddle\n            >>> # Randomly simulate a batch of input data\n            >>> inputs = paddle. normal(shape=(100, 1))\n            >>> labels = inputs * 2.0\n            >>> # define the loss function\n            >>> def loss(w):\n            ...     y = w * inputs\n            ...     return paddle.nn.functional.square_error_cost(y, labels).mean()\n            >>> # Initialize weight parameters\n            >>> w = paddle.normal(shape=(1,))\n            >>> # Call the bfgs method to solve the weight that makes the loss the smallest, and update the parameters\n            >>> for epoch in range(0, 10):\n            ...     # Call the bfgs method to optimize the loss, note that the third parameter returned represents the weight\n            ...     w_update = paddle.incubate.optimizer.functional.minimize_bfgs(loss, w)[2]\n            ...     # Use paddle.assign to update parameters in place\n            ...     paddle. assign(w_update, w)\n\n        .. code-block:: python\n            :name: code-example2\n\n            >>> # Example2: Multidimensional Grid Parameters\n            >>> import paddle\n            >>> def flatten(x):\n            ...     return x. flatten()\n            >>> def unflatten(x):\n            ...     return x.reshape((2,2))\n            >>> # Assume the network parameters are more than one dimension\n            >>> def net(x):\n            ...     assert len(x.shape) > 1\n            ...     return x.square().mean()\n            >>> # function to be optimized\n            >>> def bfgs_f(flatten_x):\n            ...     return net(unflatten(flatten_x))\n            >>> x = paddle.rand([2,2])\n            >>> for i in range(0, 10):\n            ...     # Flatten x before using minimize_bfgs\n            ...     x_update = paddle.incubate.optimizer.functional.minimize_bfgs(bfgs_f, flatten(x))[2]\n            ...     # unflatten x_update, then update parameters\n            ...     paddle.assign(unflatten(x_update), x)\n    "
    if dtype not in ['float32', 'float64']:
        raise ValueError(f"The dtype must be 'float32' or 'float64', but the specified is {dtype}.")
    op_name = 'minimize_bfgs'
    check_input_type(initial_position, 'initial_position', op_name)
    I = paddle.eye(initial_position.shape[0], dtype=dtype)
    if initial_inverse_hessian_estimate is None:
        initial_inverse_hessian_estimate = I
    else:
        check_input_type(initial_inverse_hessian_estimate, 'initial_inverse_hessian_estimate', op_name)
        check_initial_inverse_hessian_estimate(initial_inverse_hessian_estimate)
    Hk = paddle.assign(initial_inverse_hessian_estimate)
    xk = paddle.assign(initial_position.detach())
    (value, g1) = _value_and_gradient(objective_func, xk)
    num_func_calls = paddle.full(shape=[1], fill_value=1, dtype='int64')
    k = paddle.full(shape=[1], fill_value=0, dtype='int64')
    done = paddle.full(shape=[1], fill_value=False, dtype='bool')
    is_converge = paddle.full(shape=[1], fill_value=False, dtype='bool')

    def cond(k, done, is_converge, num_func_calls, xk, value, g1, Hk):
        if False:
            for i in range(10):
                print('nop')
        return (k < max_iters) & ~done

    def body(k, done, is_converge, num_func_calls, xk, value, g1, Hk):
        if False:
            while True:
                i = 10
        pk = -paddle.matmul(Hk, g1)
        if line_search_fn == 'strong_wolfe':
            (alpha, value, g2, ls_func_calls) = strong_wolfe(f=objective_func, xk=xk, pk=pk, max_iters=max_line_search_iters, initial_step_length=initial_step_length, dtype=dtype)
        else:
            raise NotImplementedError("Currently only support line_search_fn = 'strong_wolfe', but the specified is '{}'".format(line_search_fn))
        num_func_calls += ls_func_calls
        sk = alpha * pk
        yk = g2 - g1
        xk = xk + sk
        g1 = g2
        sk = paddle.unsqueeze(sk, 0)
        yk = paddle.unsqueeze(yk, 0)
        rhok_inv = paddle.dot(yk, sk)
        rhok = paddle.static.nn.cond(rhok_inv == 0.0, lambda : paddle.full(shape=[1], fill_value=1000.0, dtype=dtype), lambda : 1.0 / rhok_inv)
        Vk_transpose = I - rhok * sk * yk.t()
        Vk = I - rhok * yk * sk.t()
        Hk = paddle.matmul(paddle.matmul(Vk_transpose, Hk), Vk) + rhok * sk * sk.t()
        k += 1
        gnorm = paddle.linalg.norm(g1, p=np.inf)
        pk_norm = paddle.linalg.norm(pk, p=np.inf)
        paddle.assign(done | (gnorm < tolerance_grad) | (pk_norm < tolerance_change), done)
        paddle.assign(done, is_converge)
        paddle.assign(done | (alpha == 0.0), done)
        return [k, done, is_converge, num_func_calls, xk, value, g1, Hk]
    paddle.static.nn.while_loop(cond=cond, body=body, loop_vars=[k, done, is_converge, num_func_calls, xk, value, g1, Hk])
    return (is_converge, num_func_calls, xk, value, g1, Hk)