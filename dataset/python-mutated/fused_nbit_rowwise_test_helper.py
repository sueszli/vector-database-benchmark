import numpy as np

def param_search_greedy(x, bit_rate, n_bins=200, ratio=0.16):
    if False:
        print('Hello World!')
    (xmin, xmax) = (np.min(x), np.max(x))
    stepsize = (xmax - xmin) / np.float32(n_bins)
    min_bins = np.float32(n_bins) * (np.float32(1) - np.float32(ratio))
    (xq, loss) = _compress_uniform_simplified(x, bit_rate, xmin, xmax)
    solutions = []
    (cur_min, cur_max, cur_loss) = (xmin, xmax, loss)
    thr = min_bins * stepsize
    while cur_min + thr < cur_max:
        (xq, loss1) = _compress_uniform_simplified(x, bit_rate, cur_min + stepsize, cur_max)
        (xq, loss2) = _compress_uniform_simplified(x, bit_rate, cur_min, cur_max - stepsize)
        if cur_loss < loss1 and cur_loss < loss2:
            solutions.append((cur_min, cur_max, cur_loss))
        if loss1 < loss2:
            (cur_min, cur_max, cur_loss) = (cur_min + stepsize, cur_max, loss1)
        else:
            (cur_min, cur_max, cur_loss) = (cur_min, cur_max - stepsize, loss2)
    if len(solutions):
        best = solutions[0]
        for solution in solutions:
            if solution[-1] < best[-1]:
                best = solution
        return (best[0], best[1])
    return (xmin, xmax)

def _compress_uniform_simplified(X, bit_rate, xmin, xmax, fp16_scale_bias=True):
    if False:
        return 10
    if fp16_scale_bias:
        xmin = xmin.astype(np.float16).astype(np.float32)
    data_range = xmax - xmin
    scale = np.where(data_range == 0, np.float32(1), data_range / np.float32(2 ** bit_rate - 1))
    if fp16_scale_bias:
        scale = scale.astype(np.float16).astype(np.float32)
    inverse_scale = np.float32(1) / scale
    Xq = np.clip(np.round((X - xmin) * inverse_scale), 0, np.float32(2 ** bit_rate - 1))
    Xq = Xq * scale + xmin
    vlen = 8
    loss_v = np.zeros(vlen).astype(np.float32)
    for i in range(len(Xq) // vlen * vlen):
        loss_v[i % vlen] += (X[i] - Xq[i]) * (X[i] - Xq[i])
    loss = np.float32(0)
    for i in range(vlen):
        loss += loss_v[i]
    for i in range(len(Xq) // vlen * vlen, len(Xq)):
        loss += (X[i] - Xq[i]) * (X[i] - Xq[i])
    loss = np.sqrt(loss)
    return (Xq, loss)