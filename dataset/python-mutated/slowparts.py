from numpy import power, tanh, zeros

def slowparts(d, re, preDz, preWz, SRW, RSW, yxV, xyU, resid):
    if False:
        while True:
            i = 10
    'computes the linear algebra intensive part of the gradients of the grae'

    def fprime(x):
        if False:
            while True:
                i = 10
        return 1 - power(tanh(x), 2)
    partialDU = zeros((d + 1, re, 2 * d, d))
    for k in range(2 * d):
        for i in range(d):
            partialDU[:, :, k, i] = fprime(preDz[k]) * fprime(preWz[i]) * (SRW[i, k] + RSW[i, k]) * yxV[:, :, i]
    return partialDU