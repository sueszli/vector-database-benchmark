from __future__ import division
import numpy as np
from struct import pack, unpack
from neon import logger as neon_logger

def ceil_div(x, y):
    if False:
        print('Hello World!')
    return -(-x // y)

def out_dim(S, X, padding, strides):
    if False:
        print('Hello World!')
    return ceil_div(X - S + 1 + 2 * padding, strides)

def strip_mantissa(val):
    if False:
        while True:
            i = 10
    i = unpack('I', pack('f', val))[0] & 2139095040
    f = unpack('f', pack('I', i))[0]
    return f

def quantize(ary, bits):
    if False:
        i = 10
        return i + 15
    maxval = float(np.max(np.absolute(ary)))
    scale = strip_mantissa(maxval) / float(1 << bits - 2)
    ary = np.around(ary * (1.0 / scale)).astype(np.int64)
    return (ary, np.float32(scale))

def fconv_slice(q, S, X, padding, strides):
    if False:
        return 10
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
        return 10
    if all((x == 1 for x in F.shape[1:3])):
        C = F.shape[0]
        if backward:
            O[:] = np.dot(F.reshape((C, -1)), I.reshape((C, -1))).reshape(O.shape)
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
        while True:
            i = 10
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
I_2x2_3x3 = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0]])
F_2x2_3x3 = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]])
O_2x2_3x3 = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0]])
half = np.float(0.5)
quarter = np.float(0.25)

def trans_I_2x2_3x3(Iw, I, minimal=False):
    if False:
        print('Hello World!')
    if minimal:
        T0 = np.empty((4, 4))
        T1 = np.empty((4, 4))
        for (O, I) in ((T0, I), (T1, T0.T)):
            O[0, :] = I[0, :] - I[2, :]
            O[1, :] = I[1, :] + I[2, :]
            O[2, :] = I[2, :] - I[1, :]
            O[3, :] = I[1, :] - I[3, :]
        Iw[:] = T1.T
    else:
        Iw[:] = np.dot(np.dot(I_2x2_3x3, I), I_2x2_3x3.T)

def trans_F_2x2_3x3(Fw, F, minimal=False):
    if False:
        return 10
    if minimal:
        T0 = np.empty((4, 3))
        T1 = np.empty((4, 4))
        for (O, I) in ((T0, F), (T1, T0.T)):
            t0 = (I[0, :] + I[2, :]) * 0.5
            O[0, :] = I[0, :]
            O[1, :] = t0 + I[1, :] * 0.5
            O[2, :] = t0 - I[1, :] * 0.5
            O[3, :] = I[2, :]
        Fw[:] = T1.T
    else:
        Fw[:] = np.dot(np.dot(F_2x2_3x3, F), F_2x2_3x3.T)

def trans_O_2x2_3x3(Mw, minimal=False):
    if False:
        while True:
            i = 10
    if minimal:
        T0 = np.empty((2, 4))
        T1 = np.empty((2, 2))
        for (O, I) in ((T0, Mw), (T1, T0.T)):
            t0 = I[0, :] + I[1, :]
            t1 = I[1, :] - I[2, :]
            O[0, :] = t0 + I[2, :]
            O[1, :] = t1 - I[3, :]
        return T1.T
    else:
        return np.dot(np.dot(O_2x2_3x3, Mw), O_2x2_3x3.T)
I_3x3_2x2 = np.array([[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 0.0], [0.0, -1.0, 0.0, 1.0]])
F_3x3_2x2 = np.array([[1.0, 0.0], [0.5, 0.5], [0.5, -0.5], [0.0, 1.0]])
O_3x3_2x2 = np.array([[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, -1.0, 0.0], [0.0, 1.0, 1.0, 1.0]])

def trans_I_3x3_2x2(Iw, I, minimal=False):
    if False:
        i = 10
        return i + 15
    if minimal:
        T0 = np.empty((4, 4))
        T1 = np.empty((4, 4))
        for (O, I) in ((T0, I), (T1, T0.T)):
            O[0, :] = I[0, :] - I[2, :]
            O[1, :] = I[1, :] + I[2, :]
            O[2, :] = I[2, :] - I[1, :]
            O[3, :] = I[3, :] - I[1, :]
        Iw[:] = T1.T
    else:
        Iw[:] = np.dot(np.dot(I_3x3_2x2, I), I_3x3_2x2.T)

def trans_F_3x3_2x2(Fw, F, minimal=False):
    if False:
        print('Hello World!')
    if minimal:
        T0 = np.empty((4, 2))
        T1 = np.empty((4, 4))
        for (O, I) in ((T0, F), (T1, T0.T)):
            O[0, :] = I[0, :]
            O[1, :] = (I[0, :] + I[1, :]) * 0.5
            O[2, :] = (I[0, :] - I[1, :]) * 0.5
            O[3, :] = I[1, :]
        Fw[:] = T1.T
    else:
        Fw[:] = np.dot(np.dot(F_3x3_2x2, F), F_3x3_2x2.T)

def trans_O_3x3_2x2(Mw, minimal=False):
    if False:
        print('Hello World!')
    if minimal:
        T0 = np.empty((3, 4))
        T1 = np.empty((3, 3))
        for (O, I) in ((T0, Mw), (T1, T0.T)):
            t0 = I[1, :] + I[2, :]
            O[0, :] = t0 + I[0, :]
            O[1, :] = I[1, :] - I[2, :]
            O[2, :] = t0 + I[3, :]
        return T1.T
    else:
        return np.dot(np.dot(O_3x3_2x2, Mw), O_3x3_2x2.T)

def image_slice(x, X, B, D, pad=0):
    if False:
        for i in range(10):
            print('nop')
    start = x * B - pad
    stop = start + D
    pad = [0, 0]
    if start < 0:
        pad[0] = -start
        start = 0
    if stop - 1 >= X:
        pad[1] = stop - X
    return (start, stop, pad)

def xprop_winograd(I, F, O, padding, minimal=False, backward=False):
    if False:
        while True:
            i = 10
    if backward:
        F = np.transpose(F[:, ::-1, ::-1, :], (3, 1, 2, 0)).copy()
        padding = [2 - p for p in padding]
    (C, Y, X, N) = I.shape
    (K, P, Q, N) = O.shape
    B = 2
    D = B + 2
    Yw = ceil_div(P, B)
    Xw = ceil_div(Q, B)
    Fw = np.empty((D, D, C, K))
    Iw = np.empty((D, D, C, Yw, Xw, N))
    Mw = np.empty((D, D, K, Yw, Xw, N))
    for c in range(C):
        for k in range(K):
            trans_F_2x2_3x3(Fw[:, :, c, k], F[c, :, :, k], minimal)
    for y in range(Yw):
        (start_y, stop_y, pad_y) = image_slice(y, Y, B, D, padding[0])
        for x in range(Xw):
            (start_x, stop_x, pad_x) = image_slice(x, X, B, D, padding[1])
            sliceI = I[:, start_y:stop_y, start_x:stop_x, :]
            if any(pad_y) or any(pad_x):
                sliceI = np.pad(sliceI, ((0, 0), pad_y, pad_x, (0, 0)), 'constant')
            for c in range(C):
                for n in range(N):
                    trans_I_2x2_3x3(Iw[:, :, c, y, x, n], sliceI[c, :, :, n], minimal)
    for s in range(D):
        for t in range(D):
            Mw[s, t] = np.dot(Fw[s, t].T, Iw[s, t].reshape(C, -1)).reshape((K, Yw, Xw, N))
    for y in range(Yw):
        p = y * B
        plen = 2 if p + 1 < P else 1
        for x in range(Xw):
            q = x * B
            qlen = 2 if q + 1 < Q else 1
            for k in range(K):
                for n in range(N):
                    O[k, p:p + plen, q:q + qlen, n] = trans_O_2x2_3x3(Mw[:, :, k, y, x, n], minimal)[0:plen, 0:qlen]

def updat_winograd(I, E, U, padding, minimal=False, inner=False):
    if False:
        print('Hello World!')
    (C, Y, X, N) = I.shape
    (K, P, Q, N) = E.shape
    B = 2
    D = B + 2
    Yw = ceil_div(P, B)
    Xw = ceil_div(Q, B)
    Iw = np.empty((D, D, N, C))
    Ew = np.empty((D, D, N, K))
    if inner:
        Mw = np.empty((D, D, C, K))
        U.fill(0.0)
    else:
        Mw = np.zeros((D, D, C, K))
    for y in range(Yw):
        (start_y, stop_y, pad_y) = image_slice(y, Y, B, D, padding[0])
        (start_p, stop_p, pad_p) = image_slice(y, P, B, B)
        for x in range(Xw):
            (start_x, stop_x, pad_x) = image_slice(x, X, B, D, padding[1])
            (start_q, stop_q, pad_q) = image_slice(x, Q, B, B)
            sliceI = I[:, start_y:stop_y, start_x:stop_x, :]
            sliceE = E[:, start_p:stop_p, start_q:stop_q, :]
            if any(pad_y) or any(pad_x):
                sliceI = np.pad(sliceI, ((0, 0), pad_y, pad_x, (0, 0)), 'constant')
            if any(pad_p) or any(pad_q):
                sliceE = np.pad(sliceE, ((0, 0), pad_p, pad_q, (0, 0)), 'constant')
            for c in range(C):
                for n in range(N):
                    trans_I_3x3_2x2(Iw[:, :, n, c], sliceI[c, :, :, n], minimal)
            for k in range(K):
                for n in range(N):
                    trans_F_3x3_2x2(Ew[:, :, n, k], sliceE[k, :, :, n], minimal)
            for s in range(D):
                for t in range(D):
                    if inner:
                        Mw[s, t] = np.dot(Iw[s, t].T, Ew[s, t])
                    else:
                        Mw[s, t] += np.dot(Iw[s, t].T, Ew[s, t])
            if inner:
                for c in range(C):
                    for k in range(K):
                        U[c, :, :, k] += trans_O_3x3_2x2(Mw[:, :, c, k], minimal)
    if not inner:
        for c in range(C):
            for k in range(K):
                U[c, :, :, k] = trans_O_3x3_2x2(Mw[:, :, c, k], minimal)
np.set_printoptions(threshold=8192 * 4, linewidth=600, formatter={'float': lambda x: '%6.3f' % x})
minimal = 1
ones = 0
N = 32
(C, K) = (32, 32)
(Y, X) = (4, 4)
(R, S) = (3, 3)
strides = (1, 1)
padding = (1, 1)
P = out_dim(R, Y, padding[0], strides[0])
Q = out_dim(S, X, padding[1], strides[1])
dimI = (C, Y, X, N)
dimF = (C, R, S, K)
dimO = (K, P, Q, N)
if ones:
    I = np.ones(dimI)
    F = np.ones(dimF)
    E = np.ones(dimO)
else:
    I = np.maximum(np.random.uniform(-1.0, 1.0, dimI), 0)
    F = np.random.normal(0.0, 0.1, dimF)
    E = np.random.uniform(-1.0, 1.0, dimO)
Od = np.empty(dimO)
Ow = np.empty(dimO)
Bd = np.empty(dimI)
Bw = np.empty(dimI)
Ud = np.empty(dimF)
Uw = np.empty(dimF)
xprop_direct(I, F, Od, padding, strides)
xprop_winograd(I, F, Ow, padding, minimal=minimal)
xprop_direct(E, F, Bd, padding, strides, backward=True)
xprop_winograd(E, F, Bw, padding, minimal=minimal, backward=True)
updat_direct(I, E, Ud, padding, strides)
updat_winograd(I, E, Uw, padding, minimal=minimal, inner=True)
difO = Od - Ow
difB = Bd - Bw
difU = Ud - Uw
neon_logger.display(abs(difO).max() / Od.max())
neon_logger.display(abs(difB).max() / Bd.max())
neon_logger.display(abs(difU).max() / Ud.max())