import time
from PIL import PyAccess
from .helper import hopper

def iterate_get(size, access):
    if False:
        for i in range(10):
            print('nop')
    (w, h) = size
    for x in range(w):
        for y in range(h):
            access[x, y]

def iterate_set(size, access):
    if False:
        for i in range(10):
            print('nop')
    (w, h) = size
    for x in range(w):
        for y in range(h):
            access[x, y] = (x % 256, y % 256, 0)

def timer(func, label, *args):
    if False:
        print('Hello World!')
    iterations = 5000
    starttime = time.time()
    for x in range(iterations):
        func(*args)
        if time.time() - starttime > 10:
            break
    endtime = time.time()
    print('{}: completed {} iterations in {:.4f}s, {:.6f}s per iteration'.format(label, x + 1, endtime - starttime, (endtime - starttime) / (x + 1.0)))

def test_direct():
    if False:
        while True:
            i = 10
    im = hopper()
    im.load()
    caccess = im.im.pixel_access(False)
    access = PyAccess.new(im, False)
    assert caccess[0, 0] == access[0, 0]
    print(f'Size: {im.width}x{im.height}')
    timer(iterate_get, 'PyAccess - get', im.size, access)
    timer(iterate_set, 'PyAccess - set', im.size, access)
    timer(iterate_get, 'C-api - get', im.size, caccess)
    timer(iterate_set, 'C-api - set', im.size, caccess)