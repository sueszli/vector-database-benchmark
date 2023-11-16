import paddle

def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    if False:
        print('Hello World!')
    "Cubic interpolation between (x1, f1, g1) and (x2, f2, g2).\n        Use two points and their gradient to determine a cubic function and get the minimun point\n        between them in the cubic curve.\n\n    Reference:\n        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.\n        pp59: formula 3.59\n\n    Args:\n        x1, f1, g1: point1's position, value and gradient.\n        x2, f2, g2: point2's position, value and gradient.\n        bounds: bounds of interpolation area\n\n    Returns:\n        min_pos: the minimun point between the specified points in the cubic curve.\n    "
    if bounds is not None:
        (xmin_bound, xmax_bound) = bounds
    else:
        (xmin_bound, xmax_bound) = (x1, x2) if x1 <= x2 else (x2, x1)
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 ** 2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0

def _strong_wolfe(obj_func, xk, alpha, d, loss, grad, gtd, c1=0.0001, c2=0.9, tolerance_change=1e-09, max_ls=25):
    if False:
        i = 10
        return i + 15
    "Implements of line search algorithm that satisfies the strong Wolfe conditions using double zoom.\n\n    Reference:\n        Jorge Nocedal, Stephen J. Wright, Numerical Optimization, Second Edition, 2006.\n        pp60: Algorithm 3.5 (Line Search Algorithm).\n\n    Args:\n        obj_func: the objective function to minimize. ```` accepts a multivariate input and returns a scalar.\n        xk (Tensor): the starting point of the iterates.\n        alpha (Scalar): the initial step size.\n        d (Tensor): search direction.\n        loss (scalar): the initial loss\n        grad (Tensor): the initial grad\n        c1 (Scalar): parameter for sufficient decrease condition.\n        c2 (Scalar): parameter for curvature condition.\n        tolerance_change (Scalar): terminates if the change of function value/position/parameter between\n            two iterations is smaller than this value.\n        max_ls(int): max iteration of line search.\n        alpha_max (float): max step length.\n\n    Returns:\n        loss_new (Scaler): loss of obj_func at final alpha.\n        grad_new, (Tensor): derivative of obj_func at final alpha.\n        alpha(Tensor): optimal step length, or 0. if the line search algorithm did not converge.\n        ls_func_evals (Scaler): number of objective function called in line search process.\n\n    Following summarizes the essentials of the strong Wolfe line search algorithm.\n    Some notations used in the description:\n\n        - `func` denotes the objective function.\n        - `obi_func` is a function of step size alpha, restricting `obj_func` on a line.\n\n            obi_func = func(xk + alpha * d),\n            where xk is the position of k'th iterate, d is the line search direction(decent direction),\n            and a is the step size.\n        - alpha : substitute of alpha\n        - a1 is alpha of last iteration, which is alpha_(i-1).\n        - a2 is alpha of current iteration, which is alpha_i.\n        - a_lo is alpha in left position when calls zoom, which is alpha_low.\n        - a_hi is alpha in right position when calls zoom, which is alpha_high.\n\n    Line Search Algorithm:\n        repeat\n            Compute obi_func(a2) and derphi(a2).\n            1. If obi_func(a2) > obi_func(0) + c_1 * a2 * obi_func'(0) or [obi_func(a2) >= obi_func(a1) and i > 1],\n                alpha= zoom(a1, a2) and stop;\n\n            2. If |obi_func'(a2)| <= -c_2 * obi_func'(0),\n                alpha= a2 and stop;\n\n            3. If obi_func'(a2) >= 0,\n                alpha= zoom(a2, a1) and stop;\n\n            a1 = a2\n            a2 = min(2 * a2, a2)\n            i = i + 1\n        end(repeat)\n\n    zoom(a_lo, a_hi) Algorithm:\n        repeat\n            aj = cubic_interpolation(a_lo, a_hi)\n            Compute obi_func(aj) and derphi(aj).\n            1. If obi_func(aj) > obi_func(0) + c_1 * aj * obi_func'(0) or obi_func(aj) >= obi_func(a_lo),\n                then a_hi <- aj;\n            2.\n                2.1. If |obi_func'(aj)| <= -c_2 * obi_func'(0), then alpha= a2 and stop;\n\n                2.2. If obi_func'(aj) * (a2 - a1) >= 0, then a_hi = a_lo\n\n                a_lo = aj;\n        end(repeat)\n    "
    d_norm = d.abs().max()
    grad = grad.clone()
    (loss_new, grad_new) = obj_func(xk, alpha, d)
    ls_func_evals = 1
    gtd_new = paddle.dot(grad_new, d)
    (t_prev, f_prev, g_prev, gtd_prev) = (paddle.to_tensor(0, dtype=grad.dtype), loss, grad, gtd)
    done = False
    ls_iter = 0
    while ls_iter < max_ls:
        if loss_new > loss + c1 * alpha * gtd or (ls_iter > 1 and loss_new >= f_prev):
            bracket = [t_prev, alpha]
            bracket_f = [f_prev, loss_new]
            bracket_g = [g_prev, grad_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break
        if paddle.abs(gtd_new) <= -c2 * gtd:
            bracket = [alpha]
            bracket_f = [loss_new]
            bracket_g = [grad_new]
            done = True
            break
        if gtd_new >= 0:
            bracket = [t_prev, alpha]
            bracket_f = [f_prev, loss_new]
            bracket_g = [g_prev, grad_new.clone()]
            bracket_gtd = [gtd_prev, gtd_new]
            break
        min_step = alpha + 0.01 * (alpha - t_prev)
        max_step = alpha * 10
        tmp = alpha
        alpha = _cubic_interpolate(t_prev, f_prev, gtd_prev, alpha, loss_new, gtd_new, bounds=(min_step, max_step))
        t_prev = tmp
        f_prev = loss_new
        g_prev = grad_new.clone()
        gtd_prev = gtd_new
        (loss_new, grad_new) = obj_func(xk, alpha, d)
        ls_func_evals += 1
        gtd_new = grad_new.dot(d)
        ls_iter += 1
    if ls_iter == max_ls:
        bracket = [0, alpha]
        bracket_f = [loss, loss_new]
        bracket_g = [grad, grad_new]
    insuf_progress = False
    (low_pos, high_pos) = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)
    while not done and ls_iter < max_ls:
        if paddle.abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:
            break
        alpha = _cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0], bracket[1], bracket_f[1], bracket_gtd[1])
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - alpha, alpha - min(bracket)) < eps:
            if insuf_progress or alpha >= max(bracket) or alpha <= min(bracket):
                if paddle.abs(alpha - max(bracket)) < paddle.abs(alpha - min(bracket)):
                    alpha = max(bracket) - eps
                else:
                    alpha = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False
        (loss_new, grad_new) = obj_func(xk, alpha, d)
        ls_func_evals += 1
        gtd_new = grad_new.dot(d)
        ls_iter += 1
        if loss_new > loss + c1 * alpha * gtd or loss_new >= bracket_f[low_pos]:
            bracket[high_pos] = alpha
            bracket_f[high_pos] = loss_new
            bracket_g[high_pos] = grad_new.clone()
            bracket_gtd[high_pos] = gtd_new
            (low_pos, high_pos) = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            if paddle.abs(gtd_new) <= -c2 * gtd:
                done = True
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
            bracket[low_pos] = alpha
            bracket_f[low_pos] = loss_new
            bracket_g[low_pos] = grad_new.clone()
            bracket_gtd[low_pos] = gtd_new
    alpha = bracket[low_pos]
    loss_new = bracket_f[low_pos]
    grad_new = bracket_g[low_pos]
    return (loss_new, grad_new, alpha, ls_func_evals)