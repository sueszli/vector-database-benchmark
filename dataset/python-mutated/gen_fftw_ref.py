from subprocess import Popen, PIPE, STDOUT
import numpy as np
SZ = [2, 3, 4, 8, 12, 15, 16, 17, 32, 64, 128, 256, 512, 1024]

def gen_data(dt):
    if False:
        print('Hello World!')
    arrays = {}
    if dt == np.float128:
        pg = './fftw_longdouble'
    elif dt == np.float64:
        pg = './fftw_double'
    elif dt == np.float32:
        pg = './fftw_single'
    else:
        raise ValueError('unknown: %s' % dt)
    for type in [1, 2, 3, 4, 5, 6, 7, 8]:
        arrays[type] = {}
        for sz in SZ:
            a = Popen([pg, str(type), str(sz)], stdout=PIPE, stderr=STDOUT)
            st = [i.decode('ascii').strip() for i in a.stdout.readlines()]
            arrays[type][sz] = np.fromstring(','.join(st), sep=',', dtype=dt)
    return arrays
data = gen_data(np.float32)
filename = 'fftw_single_ref'
d = {'sizes': SZ}
for type in [1, 2, 3, 4]:
    for sz in SZ:
        d['dct_%d_%d' % (type, sz)] = data[type][sz]
d['sizes'] = SZ
for type in [5, 6, 7, 8]:
    for sz in SZ:
        d['dst_%d_%d' % (type - 4, sz)] = data[type][sz]
np.savez(filename, **d)
data = gen_data(np.float64)
filename = 'fftw_double_ref'
d = {'sizes': SZ}
for type in [1, 2, 3, 4]:
    for sz in SZ:
        d['dct_%d_%d' % (type, sz)] = data[type][sz]
d['sizes'] = SZ
for type in [5, 6, 7, 8]:
    for sz in SZ:
        d['dst_%d_%d' % (type - 4, sz)] = data[type][sz]
np.savez(filename, **d)
data = gen_data(np.float128)
filename = 'fftw_longdouble_ref'
d = {'sizes': SZ}
for type in [1, 2, 3, 4]:
    for sz in SZ:
        d['dct_%d_%d' % (type, sz)] = data[type][sz]
d['sizes'] = SZ
for type in [5, 6, 7, 8]:
    for sz in SZ:
        d['dst_%d_%d' % (type - 4, sz)] = data[type][sz]
np.savez(filename, **d)