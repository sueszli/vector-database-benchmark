"""
Script to show the results of the two marching cubes algorithms on different
data.
"""
import time
from contextlib import contextmanager
import numpy as np
from skimage.measure import marching_cubes_classic, marching_cubes_lewiner
from skimage.draw import ellipsoid

def main(select=3, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Script main function.\n\n    select: int\n        1: Medical data\n        2: Blocky data, different every time\n        3: Two donuts\n        4: Ellipsoid\n\n    '
    import visvis as vv
    if select == 1:
        vol = vv.volread('stent')
        isovalue = kwargs.pop('level', 800)
    elif select == 2:
        vol = vv.aVolume(20, 128)
        isovalue = kwargs.pop('level', 0.2)
    elif select == 3:
        with timer('computing donuts'):
            vol = donuts()
        isovalue = kwargs.pop('level', 0.0)
    elif select == 4:
        vol = ellipsoid(4, 3, 2, levelset=True)
        isovalue = kwargs.pop('level', 0.0)
    else:
        raise ValueError('invalid selection')
    with timer('finding surface lewiner'):
        (vertices1, faces1) = marching_cubes_lewiner(vol, isovalue, **kwargs)[:2]
    with timer('finding surface classic'):
        (vertices2, faces2) = marching_cubes_classic(vol, isovalue, **kwargs)
    vv.figure(1)
    vv.clf()
    a1 = vv.subplot(121)
    vv.title('Lewiner')
    m1 = vv.mesh(np.fliplr(vertices1), faces1)
    a2 = vv.subplot(122)
    vv.title('Classic')
    m2 = vv.mesh(np.fliplr(vertices2), faces2)
    a1.camera = a2.camera
    m1.cullFaces = m2.cullFaces = 'front'
    vv.use().Run()

def donuts():
    if False:
        for i in range(10):
            print('nop')
    'Return volume of two donuts based on a formula by Thomas Lewiner.'
    n = 48
    a = 2.5 / n * 8.0
    b = -1.25 * 8.0
    c = 16.0 - 1.85 * 1.85
    d = 64.0
    i = np.arange(n, dtype=int)
    ia_plus_b = i * a + b
    ia_plus_b_square = ia_plus_b ** 2
    z = ia_plus_b_square[:, np.newaxis, np.newaxis]
    zc = z + c
    y1 = ((ia_plus_b - 2) ** 2)[np.newaxis, :, np.newaxis]
    y2 = ((ia_plus_b + 2) ** 2)[np.newaxis, :, np.newaxis]
    x = ia_plus_b_square[np.newaxis, np.newaxis, :]
    x1 = (x + y1 + zc) ** 2
    x2 = (x + y2 + zc) ** 2
    return (x1 - d * (x + y1)) * (x2 - d * (z + y2)) + 1025

@contextmanager
def timer(message):
    if False:
        while True:
            i = 10
    'Context manager for timing execution speed of body.'
    print(message, end=' ')
    start = time.time()
    yield
    print(f'took {1000 * (time.time() - start):1.0f} ms')
if __name__ == '__main__':
    main()