from gnuradio import gr, gr_unittest, filter
import sys
import math

def stopband_atten_to_dev(atten_db):
    if False:
        return 10
    'Convert a stopband attenuation in dB to an absolute value'
    return 10 ** (-atten_db / 20)

def passband_ripple_to_dev(ripple_db):
    if False:
        for i in range(10):
            print('nop')
    'Convert passband ripple spec expressed in dB to an absolute value'
    return (10 ** (ripple_db / 20) - 1) / (10 ** (ripple_db / 20) + 1)

def remezord(fcuts, mags, devs, fsamp=2):
    if False:
        while True:
            i = 10
    '\n    FIR order estimator (lowpass, highpass, bandpass, mulitiband).\n    '
    fcuts = fcuts[:]
    mags = mags[:]
    devs = devs[:]
    for i in range(len(fcuts)):
        fcuts[i] = float(fcuts[i]) / fsamp
    nf = len(fcuts)
    nm = len(mags)
    nd = len(devs)
    nbands = nm
    if nm != nd:
        raise ValueError('Length of mags and devs must be equal')
    if nf != 2 * (nbands - 1):
        raise ValueError('Length of f must be 2 * len (mags) - 2')
    for i in range(len(mags)):
        if mags[i] != 0:
            devs[i] = devs[i] / mags[i]
    f1 = fcuts[0::2]
    f2 = fcuts[1::2]
    n = 0
    min_delta = 2
    for i in range(len(f1)):
        if f2[i] - f1[i] < min_delta:
            n = i
            min_delta = f2[i] - f1[i]
    if nbands == 2:
        l = lporder(f1[n], f2[n], devs[0], devs[1])
    else:
        l = 0
        for i in range(1, nbands - 1):
            l1 = lporder(f1[i - 1], f2[i - 1], devs[i], devs[i - 1])
            l2 = lporder(f1[i], f2[i], devs[i], devs[i + 1])
            l = max(l, l1, l2)
    n = int(math.ceil(l)) - 1
    ff = [0] + fcuts + [1]
    for i in range(1, len(ff) - 1):
        ff[i] *= 2
    aa = []
    for a in mags:
        aa = aa + [a, a]
    max_dev = max(devs)
    wts = [1] * len(devs)
    for i in range(len(wts)):
        wts[i] = max_dev / devs[i]
    return (n, ff, aa, wts)

def lporder(freq1, freq2, delta_p, delta_s):
    if False:
        print('Hello World!')
    '\n    FIR lowpass filter length estimator.\n    '
    df = abs(freq2 - freq1)
    ddp = math.log10(delta_p)
    dds = math.log10(delta_s)
    a1 = 0.005309
    a2 = 0.07114
    a3 = -0.4761
    a4 = -0.00266
    a5 = -0.5941
    a6 = -0.4278
    b1 = 11.01217
    b2 = 0.5124401
    t1 = a1 * ddp * ddp
    t2 = a2 * ddp
    t3 = a4 * ddp * ddp
    t4 = a5 * ddp
    dinf = (t1 + t2 + a3) * dds + (t3 + t4 + a6)
    ff = b1 + b2 * (ddp - dds)
    n = dinf / df - ff * df + 1
    return n

class test_pm_remez(gr_unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_001_low_pass(self):
        if False:
            print('Hello World!')
        gain = 1
        Fs = 1
        freq1 = 0.1
        freq2 = 0.2
        passband_ripple_db = 0.01
        stopband_atten_db = 60
        passband_dev = passband_ripple_to_dev(passband_ripple_db)
        stopband_dev = stopband_atten_to_dev(stopband_atten_db)
        desired_ampls = (gain, 0)
        (n, fo, ao, w) = remezord([freq1, freq2], desired_ampls, [passband_dev, stopband_dev], Fs)
        new_taps = filter.pm_remez(n + 2, fo, ao, w, 'bandpass')
        known_taps = (-0.0008370135734511828, -0.0006622211673134374, 0.0008501079576365787, 0.003059609130249229, 0.003202235537205373, -0.001000899296974219, -0.007589728680590891, -0.009790921118281865, -0.001524210202628562, 0.014373535837200111, 0.02392881326993834, 0.011798133085019008, -0.021954446348997188, -0.05293436740264934, -0.04375787096766848, 0.028038890498420392, 0.14612655590172896, 0.25738578419108626, 0.302967004188747, 0.25738578419108626, 0.14612655590172896, 0.028038890498420392, -0.04375787096766848, -0.05293436740264934, -0.021954446348997188, 0.011798133085019008, 0.02392881326993834, 0.014373535837200111, -0.001524210202628562, -0.009790921118281865, -0.007589728680590891, -0.001000899296974219, 0.003202235537205373, 0.003059609130249229, 0.0008501079576365787, -0.0006622211673134374, -0.0008370135734511828)
        self.assertFloatTuplesAlmostEqual(known_taps, new_taps, 5)

    def test_002_test_invalid_args(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            filter.pm_remez(2, [0, 0.1], [1, 0.0002], [1, 0.25], 'bandpass')
        with self.assertRaises(ValueError):
            filter.pm_remez(200, [0.1, 0], [1, 0.0002], [1, 0.25], 'bandpass')
        with self.assertRaises(ValueError):
            filter.pm_remez(200, [0, 0.1], [1, 0.0002, 2e-06], [1, 0.25], 'bandpass')
if __name__ == '__main__':
    gr_unittest.run(test_pm_remez)