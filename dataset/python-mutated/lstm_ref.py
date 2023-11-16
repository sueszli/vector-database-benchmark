"""
This is a reference LSTM numpy implementation adapted from Karpathy's code:

The adaptation includes
  - interface to use the same initialization values
  - being able to read out intermediate values to compare with another LSTM
    implementation
"""
from builtins import input
import numpy as np
from neon import logger as neon_logger
from utils import allclose_with_out

class LSTM(object):

    @staticmethod
    def init(input_size, hidden_size):
        if False:
            return 10
        '\n        Initialize parameters of the LSTM (both weights and biases in one matrix)\n        to be ones\n        '
        a = input_size + hidden_size + 1
        b = 4 * hidden_size
        WLSTM = np.ones((a, b))
        return WLSTM

    @staticmethod
    def forward(X, WLSTM, c0=None, h0=None):
        if False:
            return 10
        '\n        X should be of shape (n,b,input_size), where n = length of sequence, b = batch size\n        '
        (n, b, input_size) = X.shape
        d = WLSTM.shape[1] // 4
        if c0 is None:
            c0 = np.zeros((b, d))
        if h0 is None:
            h0 = np.zeros((b, d))
        xphpb = WLSTM.shape[0]
        Hin = np.zeros((n, b, xphpb))
        Hout = np.zeros((n, b, d))
        IFOG = np.zeros((n, b, d * 4))
        IFOGf = np.zeros((n, b, d * 4))
        C = np.zeros((n, b, d))
        Ct = np.zeros((n, b, d))
        for t in range(n):
            prevh = Hout[t - 1] if t > 0 else h0
            Hin[t, :, 0] = 1
            Hin[t, :, 1:input_size + 1] = X[t]
            Hin[t, :, input_size + 1:] = prevh
            IFOG[t] = Hin[t].dot(WLSTM)
            IFOGf[t, :, :3 * d] = 1.0 / (1.0 + np.exp(-IFOG[t, :, :3 * d]))
            IFOGf[t, :, 3 * d:] = np.tanh(IFOG[t, :, 3 * d:])
            prevc = C[t - 1] if t > 0 else c0
            C[t] = IFOGf[t, :, :d] * IFOGf[t, :, 3 * d:] + IFOGf[t, :, d:2 * d] * prevc
            Ct[t] = np.tanh(C[t])
            Hout[t] = IFOGf[t, :, 2 * d:3 * d] * Ct[t]
        cache = {}
        cache['WLSTM'] = WLSTM
        cache['Hout'] = Hout
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['C'] = C
        cache['Ct'] = Ct
        cache['Hin'] = Hin
        cache['c0'] = c0
        cache['h0'] = h0
        return (Hout, C[t], Hout[t], cache)

    @staticmethod
    def backward(dHout_in, cache, dcn=None, dhn=None):
        if False:
            for i in range(10):
                print('nop')
        WLSTM = cache['WLSTM']
        Hout = cache['Hout']
        IFOGf = cache['IFOGf']
        IFOG = cache['IFOG']
        C = cache['C']
        Ct = cache['Ct']
        Hin = cache['Hin']
        c0 = cache['c0']
        (n, b, d) = Hout.shape
        input_size = WLSTM.shape[0] - d - 1
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((n, b, input_size))
        dh0 = np.zeros((b, d))
        dc0 = np.zeros((b, d))
        dHout = dHout_in.copy()
        if dcn is not None:
            dC[n - 1] += dcn.copy()
        if dhn is not None:
            dHout[n - 1] += dhn.copy()
        for t in reversed(range(n)):
            tanhCt = Ct[t]
            dIFOGf[t, :, 2 * d:3 * d] = tanhCt * dHout[t]
            dC[t] += (1 - tanhCt ** 2) * (IFOGf[t, :, 2 * d:3 * d] * dHout[t])
            if t > 0:
                dIFOGf[t, :, d:2 * d] = C[t - 1] * dC[t]
                dC[t - 1] += IFOGf[t, :, d:2 * d] * dC[t]
            else:
                dIFOGf[t, :, d:2 * d] = c0 * dC[t]
                dc0 = IFOGf[t, :, d:2 * d] * dC[t]
            dIFOGf[t, :, :d] = IFOGf[t, :, 3 * d:] * dC[t]
            dIFOGf[t, :, 3 * d:] = IFOGf[t, :, :d] * dC[t]
            dIFOG[t, :, 3 * d:] = (1 - IFOGf[t, :, 3 * d:] ** 2) * dIFOGf[t, :, 3 * d:]
            y = IFOGf[t, :, :3 * d]
            dIFOG[t, :, :3 * d] = y * (1.0 - y) * dIFOGf[t, :, :3 * d]
            dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())
            dX[t] = dHin[t, :, 1:input_size + 1]
            if t > 0:
                dHout[t - 1, :] += dHin[t, :, input_size + 1:]
            else:
                dh0 += dHin[t, :, input_size + 1:]
        return (dX, dWLSTM, dc0, dh0)

    @staticmethod
    def runBatchFpropWithGivenInput(hidden_size, X):
        if False:
            i = 10
            return i + 15
        '\n        run the LSTM model through the given input data. The data has dimension\n        (seq_len, batch_size, hidden_size)\n\n        '
        input_size = X.shape[2]
        WLSTM = LSTM.init(input_size, hidden_size)
        (Hout, cprev, hprev, batch_cache) = LSTM.forward(X, WLSTM)
        IFOGf = batch_cache['IFOGf']
        Ct = batch_cache['Ct']
        return (Hout, IFOGf, Ct, batch_cache)

    @staticmethod
    def runBatchBpropWithGivenDelta(hidden_size, batch_cache, delta):
        if False:
            i = 10
            return i + 15
        '\n        run the LSTM model through the given input errors. The data has dimension\n        (seq_len, batch_size, hidden_size)\n\n        '
        dH = delta
        (dX, dWLSTM, dc0, dh0) = LSTM.backward(dH, batch_cache)
        input_size = dWLSTM.shape[0] - hidden_size - 1
        dWrecur = dWLSTM[-hidden_size:, :]
        dWinput = dWLSTM[1:input_size + 1, :]
        db = dWLSTM[0, :]
        return (dX, dWrecur, dWinput, db, dWLSTM)

def checkSequentialMatchesBatch():
    if False:
        return 10
    ' check LSTM I/O forward/backward interactions '
    (n, b, d) = (5, 3, 4)
    input_size = 10
    WLSTM = LSTM.init(input_size, d)
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)
    c0 = np.random.randn(b, d)
    cprev = c0
    hprev = h0
    caches = [{} for t in range(n)]
    Hcat = np.zeros((n, b, d))
    for t in range(n):
        xt = X[t:t + 1]
        (_, cprev, hprev, cache) = LSTM.forward(xt, WLSTM, cprev, hprev)
        caches[t] = cache
        Hcat[t] = hprev
    (H, _, _, batch_cache) = LSTM.forward(X, WLSTM, c0, h0)
    assert allclose_with_out(H, Hcat), 'Sequential and Batch forward dont match!'
    wrand = np.random.randn(*Hcat.shape)
    dH = wrand
    (BdX, BdWLSTM, Bdc0, Bdh0) = LSTM.backward(dH, batch_cache)
    dX = np.zeros_like(X)
    dWLSTM = np.zeros_like(WLSTM)
    dc0 = np.zeros_like(c0)
    dh0 = np.zeros_like(h0)
    dcnext = None
    dhnext = None
    for t in reversed(range(n)):
        dht = dH[t].reshape(1, b, d)
        (dx, dWLSTMt, dcprev, dhprev) = LSTM.backward(dht, caches[t], dcnext, dhnext)
        dhnext = dhprev
        dcnext = dcprev
        dWLSTM += dWLSTMt
        dX[t] = dx[0]
        if t == 0:
            dc0 = dcprev
            dh0 = dhprev
    neon_logger.display('Making sure batched version agrees with sequential version: (should all be True)')
    neon_logger.display(np.allclose(BdX, dX))
    neon_logger.display(np.allclose(BdWLSTM, dWLSTM))
    neon_logger.display(np.allclose(Bdc0, dc0))
    neon_logger.display(np.allclose(Bdh0, dh0))

def checkBatchGradient():
    if False:
        return 10
    ' check that the batch gradient is correct '
    (n, b, d) = (5, 3, 4)
    input_size = 10
    WLSTM = LSTM.init(input_size, d)
    X = np.random.randn(n, b, input_size)
    h0 = np.random.randn(b, d)
    c0 = np.random.randn(b, d)
    (H, Ct, Ht, cache) = LSTM.forward(X, WLSTM, c0, h0)
    wrand = np.random.randn(*H.shape)
    dH = wrand
    (dX, dWLSTM, dc0, dh0) = LSTM.backward(dH, cache)

    def fwd():
        if False:
            for i in range(10):
                print('nop')
        (h, _, _, _) = LSTM.forward(X, WLSTM, c0, h0)
        return np.sum(h * wrand)
    delta = 1e-05
    rel_error_thr_warning = 0.01
    rel_error_thr_error = 1
    tocheck = [X, WLSTM, c0, h0]
    grads_analytic = [dX, dWLSTM, dc0, dh0]
    names = ['X', 'WLSTM', 'c0', 'h0']
    for j in range(len(tocheck)):
        mat = tocheck[j]
        dmat = grads_analytic[j]
        name = names[j]
        for i in range(mat.size):
            old_val = mat.flat[i]
            mat.flat[i] = old_val + delta
            loss0 = fwd()
            mat.flat[i] = old_val - delta
            loss1 = fwd()
            mat.flat[i] = old_val
            grad_analytic = dmat.flat[i]
            grad_numerical = (loss0 - loss1) / float(2 * delta)
            if grad_numerical == 0 and grad_analytic == 0:
                rel_error = 0
                status = 'OK'
            elif abs(grad_numerical) < 1e-07 and abs(grad_analytic) < 1e-07:
                rel_error = 0
                status = 'VAL SMALL WARNING'
            else:
                rel_error = abs(grad_analytic - grad_numerical) / abs(float(grad_numerical + grad_analytic))
                status = 'OK'
                if rel_error > rel_error_thr_warning:
                    status = 'WARNING'
                if rel_error > rel_error_thr_error:
                    status = '!!!!! NOTOK'
            neon_logger.display('%s checking param %s index %s (val = %+8f), analytic = %+8f,' + 'numerical = %+8f, relative error = %+8f' % (status, name, repr(np.unravel_index(i, mat.shape)), old_val, grad_analytic, grad_numerical, rel_error))
if __name__ == '__main__':
    checkSequentialMatchesBatch()
    input('check OK, press key to continue to gradient check')
    checkBatchGradient()
    neon_logger.display('every line should start with OK. Have a nice day!')