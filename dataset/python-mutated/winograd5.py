from __future__ import division
import numpy as np
from struct import pack, unpack
from neon import logger as neon_logger

def ceil_div(x, y):
    if False:
        while True:
            i = 10
    return -(-x // y)

def out_dim(S, X, padding, strides):
    if False:
        while True:
            i = 10
    return ceil_div(X - S + 1 + 2 * padding, strides)

def strip_mantissa(val):
    if False:
        return 10
    i = unpack('I', pack('f', val))[0] & 2139095040
    f = unpack('f', pack('I', i))[0]
    return f

def immediate(val):
    if False:
        i = 10
        return i + 15
    i = unpack('I', pack('f', val))[0] & 2147479552
    f = unpack('f', pack('I', i))[0]
    return f

def quantize(ary, bits, sign=1):
    if False:
        i = 10
        return i + 15
    maxval = float(np.max(np.absolute(ary)))
    scale = strip_mantissa(maxval) / float(1 << bits - sign - 1)
    ary = np.around(ary * (1.0 / scale)).astype(np.int64)
    return (ary, np.float64(scale))

def fconv_slice(q, S, X, padding, strides):
    if False:
        for i in range(10):
            print('nop')
    f1 = 0
    f2 = S - 1
    x1 = q * strides - padding
    x2 = x1 + f2
    if x1 < 0:
        f1 = -x1
        x1 = 0
    if x2 >= X:
        dif = x2 - X + 1
        f2 -= dif
        x2 -= dif
    return (slice(f1, f2 + 1), slice(x1, x2 + 1), f2 - f1 + 1)

def bconv_slice(x, S, Q, padding, strides):
    if False:
        print('Hello World!')
    qs = x - (S - padding - 1)
    firstF = None
    for s in range(S):
        q = qs + s
        if q % strides == 0:
            q //= strides
            if q >= 0 and q < Q:
                if firstF is None:
                    firstF = s
                    firstE = q
                lastF = s
                lastE = q
    if firstF is None:
        return (slice(0, 0, 1), slice(0, 0, 1), 0)
    return (slice(firstF, lastF + 1, strides), slice(firstE, lastE + 1, 1), 0)

def xprop_direct(I, F, O, padding, strides, backward=False):
    if False:
        while True:
            i = 10
    if all((x == 1 for x in F.shape[1:3])):
        C = F.shape[0]
        K = F.shape[4]
        if backward:
            O[:] = np.dot(F.reshape((C, -1)), I.reshape((K, -1))).reshape(O.shape)
        else:
            O[:] = np.dot(F.reshape((C, -1)).T, I.reshape((C, -1))).reshape(O.shape)
        return
    if backward:
        F = np.transpose(F[:, ::-1, ::-1, :], (3, 1, 2, 0)).copy()
        xconv_slice = bconv_slice
    else:
        xconv_slice = fconv_slice
    (C, Y, X, N) = I.shape
    (C, R, S, K) = F.shape
    (K, P, Q, N) = O.shape
    qSlice = [xconv_slice(q, S, X, padding[0], strides[0]) for q in range(Q)]
    for p in range(P):
        (sliceR, sliceY, _) = xconv_slice(p, R, Y, padding[1], strides[1])
        for q in range(Q):
            (sliceS, sliceX, _) = qSlice[q]
            slicedF = F[:, sliceR, sliceS, :].reshape((-1, K))
            slicedI = I[:, sliceY, sliceX, :].reshape((-1, N))
            O[:, p, q, :] = np.dot(slicedF.T, slicedI)

def updat_direct(I, E, U, padding, strides):
    if False:
        print('Hello World!')
    (C, Y, X, N) = I.shape
    (K, P, Q, N) = E.shape
    (C, R, S, K) = U.shape
    if all((x == 1 for x in (R, S))):
        U[:] = np.dot(I.reshape((C, -1)), E.reshape((K, -1)).T).reshape(U.shape)
        return
    U.fill(0.0)
    qSlice = [fconv_slice(q, S, X, padding[0], strides[0]) for q in range(Q)]
    for p in range(P):
        (sliceR, sliceY, rlen) = fconv_slice(p, R, Y, padding[1], strides[1])
        for q in range(Q):
            (sliceS, sliceX, slen) = qSlice[q]
            slicedI = I[:, sliceY, sliceX, :].reshape((-1, N))
            slicedE = E[:, p, q, :]
            U[:, sliceR, sliceS, :] += np.dot(slicedI, slicedE.T).reshape((C, rlen, slen, K))
I_2x2_5x5 = (np.array([[1.0, 0.0], [1.0, 0.75], [1.0, -0.75], [1.0, 1.5], [1.0, -1.5], [0.0, 1.0]]),)
F_2x2_5x5 = (np.array([[64.0 / 81.0, 0.0, 0.0, 0.0, 0.0], [-128.0 / 243.0, -32.0 / 81.0, -8.0 / 27.0, -2.0 / 9.0, -1.0 / 6.0], [-128.0 / 243.0, 32.0 / 81.0, -8.0 / 27.0, 2.0 / 9.0, -1.0 / 6.0], [32.0 / 243.0, 16.0 / 81.0, 8.0 / 27.0, 4.0 / 9.0, 2.0 / 3.0], [32.0 / 243.0, -16.0 / 81.0, 8.0 / 27.0, -4.0 / 9.0, 2.0 / 3.0], [0.0, 0.0, 0.0, 0.0, 1.0]]),)
O_2x2_5x5 = (np.array([[1.265625, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.6875, 1.6875, -0.84375, 0.84375, 1.265625], [-2.8125, -2.25, -2.25, -0.5625, -0.5625, 0.0], [0.0, 0.75, -0.75, 1.5, -1.5, -2.8125], [1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),)

def trans_I_2x2_5x5(Iw, I, minimal=False, trans=False):
    if False:
        return 10
    if minimal:
        T0 = np.empty((6, 2))
        T1 = np.empty((6, 6))
        for (O, I) in ((T0, I), (T1, T0.T)):
            O[0, :] = I[0, :]
            O[1, :] = I[0, :] + I[1, :] * 0.75
            O[2, :] = I[0, :] - I[1, :] * 0.75
            O[3, :] = I[0, :] + I[1, :] * 1.5
            O[4, :] = I[0, :] - I[1, :] * 1.5
            O[5, :] = I[1, :]
        Iw[:] = T1.T
    else:
        Iw[:] = np.dot(np.dot(I_2x2_5x5[trans[0]], I), I_2x2_5x5[trans[1]].T)

def trans_F_2x2_5x5(Fw, F, minimal=False, trans=False):
    if False:
        return 10
    if minimal:
        T0 = np.empty((6, 5))
        T1 = np.empty((6, 6))
        for (O, I) in ((T0, F[::-1, ::-1]), (T1, T0.T)):
            t0 = I[2, :] * 8.0 / 27.0
            t1 = I[1, :] * 32.0 / 81.0 + I[3, :] * 2.0 / 9.0
            t2 = I[1, :] * 16.0 / 81.0 + I[3, :] * 4.0 / 9.0
            t3 = I[0, :] * -128.0 / 243.0 - I[4, :] * 1.0 / 6.0 - t0
            t4 = I[0, :] * 32.0 / 243.0 + I[4, :] * 2.0 / 3.0 + t0
            O[0, :] = I[0, :] * 64.0 / 81.0
            O[1, :] = t3 - t1
            O[2, :] = t3 + t1
            O[3, :] = t4 + t2
            O[4, :] = t4 - t2
            O[5, :] = I[4, :]
        Fw[:] = T1.T
    else:
        Fw[:] = np.dot(np.dot(F_2x2_5x5[trans[0]], F[::-1, ::-1]), F_2x2_5x5[trans[1]].T)

def trans_O_2x2_5x5(Mw, minimal=False, trans=False):
    if False:
        print('Hello World!')
    if minimal:
        T0 = np.empty((6, 6))
        T1 = np.empty((6, 6))
        for (O, I) in ((T0, Mw), (T1, T0.T)):
            t0 = I[1, :] + I[2, :]
            t1 = I[3, :] + I[4, :]
            t2 = I[1, :] - I[2, :]
            t3 = I[3, :] - I[4, :]
            O[0, :] = I[0, :] * 1.265625
            O[4, :] = I[0, :] + t0 + t1
            O[2, :] = t0 * -2.25 + t1 * -0.5625 + I[0, :] * -2.8125
            O[1, :] = t2 * -1.6875 + t3 * -0.84375 + I[5, :] * 1.265625
            O[3, :] = t2 * 0.75 + t3 * 1.5 + I[5, :] * -2.8125
            O[5, :] = I[5, :]
        return T1.T
    else:
        return np.dot(np.dot(O_2x2_5x5[trans[0]], Mw), O_2x2_5x5[trans[1]].T)

def image_slice(x, X, B):
    if False:
        for i in range(10):
            print('nop')
    x0 = x * B
    x1 = x0 + B
    if x1 > X:
        return (slice(x0, X, 1), (0, 1))
    return (slice(x0, x1, 1), (0, 0))

def output_slice(x, P, B, D, pad):
    if False:
        print('Hello World!')
    p0 = x * B + pad - 4
    p1 = p0 + D
    if p0 < 0:
        m0 = -p0
        p0 = 0
    else:
        m0 = 0
    if p1 > P:
        m1 = D - (p1 - P)
        p1 = P
    else:
        m1 = D
    return (slice(p0, p1, 1), slice(m0, m1, 1))

def xprop_winograd(I, F, O, padding, minimal=False, trans=False, backward=False):
    if False:
        print('Hello World!')
    if backward:
        F = np.transpose(F[:, ::-1, ::-1, :], (3, 1, 2, 0)).copy()
        padding = [4 - p for p in padding]
    (C, Y, X, N) = I.shape
    (K, P, Q, N) = O.shape
    B = 2
    D = 6
    Yw = ceil_div(Y, B)
    Xw = ceil_div(X, B)
    Fw = np.empty((D, D, C, K))
    Iw = np.empty((D, D, C, Yw, Xw, N))
    Mw = np.empty((D, D, K, Yw, Xw, N))
    O.fill(0.0)
    for c in range(C):
        for k in range(K):
            trans_F_2x2_5x5(Fw[:, :, c, k], F[c, :, :, k], minimal, trans)
    for y in range(Yw):
        (slice_y, pad_y) = image_slice(y, Y, B)
        for x in range(Xw):
            (slice_x, pad_x) = image_slice(x, X, B)
            sliceI = I[:, slice_y, slice_x, :]
            if pad_y[1] or pad_x[1]:
                sliceI = np.pad(sliceI, ((0, 0), pad_y, pad_x, (0, 0)), 'constant')
            for c in range(C):
                for n in range(N):
                    trans_I_2x2_5x5(Iw[:, :, c, y, x, n], sliceI[c, :, :, n], minimal, trans)
    for s in range(D):
        for t in range(D):
            Mw[s, t] = np.dot(Fw[s, t].T, Iw[s, t].reshape(C, -1)).reshape((K, Yw, Xw, N))
    for y in range(Yw):
        (slice_p, slice_y) = output_slice(y, P, B, D, padding[0])
        for x in range(Xw):
            (slice_q, slice_x) = output_slice(x, Q, B, D, padding[1])
            for k in range(K):
                for n in range(N):
                    Out = trans_O_2x2_5x5(Mw[:, :, k, y, x, n], minimal, trans)
                    O[k, slice_p, slice_q, n] += Out[slice_y, slice_x]
np.set_printoptions(threshold=8192 * 4, linewidth=600, formatter={'float': lambda x: '%4.0f' % x})
minimal = 1
trans = (0, 0)
ones = 0
N = 4
(C, K) = (4, 4)
(Y, X) = (6, 6)
(R, S) = (5, 5)
strides = (1, 1)
padding = (2, 2)
P = out_dim(R, Y, padding[0], strides[0])
Q = out_dim(S, X, padding[1], strides[1])
neon_logger.display('{}'.format(P, Q))
dimI = (C, Y, X, N)
dimF = (C, R, S, K)
dimO = (K, P, Q, N)
if ones:
    I = np.ones(dimI)
    F = np.ones(dimF)
    E = np.ones(dimO)
    F[0, :, :, 0] = np.arange(0, 25).reshape(5, 5)
else:
    I = np.random.uniform(-1.0, 1.0, dimI)
    F = np.random.uniform(-1.0, 1.0, dimF)
    E = np.random.uniform(-1.0, 1.0, dimO)
Od = np.empty(dimO)
Ow = np.empty(dimO)
Bd = np.empty(dimI)
Bw = np.empty(dimI)
xprop_direct(I, F, Od, padding, strides)
xprop_winograd(I, F, Ow, padding, minimal=minimal, trans=trans)
xprop_direct(E, F, Bd, padding, strides, backward=True)
xprop_winograd(E, F, Bw, padding, minimal=minimal, trans=trans, backward=True)
difO = Od - Ow
difB = Bd - Bw
neon_logger.display(abs(difO).max() / Od.max())
neon_logger.display(abs(difB).max() / Bd.max())