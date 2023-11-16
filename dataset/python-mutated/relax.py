import ctypes
import os
so = os.path.join(os.path.dirname(__file__), 'maths.so')
try:
    lib = ctypes.CDLL(so)
    COMPILED = True
except OSError:
    COMPILED = False
if COMPILED:
    lib.relax.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]

    def relax(pos, links, mrel1, mrel2, lengths, push, pull, iters):
        if False:
            for i in range(10):
                print('nop')
        nlinks = links.shape[0]
        lib.relax(pos.ctypes, links.ctypes, mrel1.ctypes, mrel2.ctypes, lengths.ctypes, push.ctypes, pull.ctypes, nlinks, iters)
else:

    def relax(pos, links, mrel1, mrel2, lengths, push, pull, iters):
        if False:
            return 10
        lengths2 = lengths ** 2
        for i in range(iters):
            for l in range(links.shape[0]):
                (p1, p2) = links[l]
                x1 = pos[p1]
                x2 = pos[p2]
                dx = x2 - x1
                dist2 = (dx ** 2).sum()
                if push[l] and dist2 < lengths2[l] or (pull[l] and dist2 > lengths2[l]):
                    dist = dist2 ** 0.5
                    change = (lengths[l] - dist) / dist
                    dx *= change
                    pos[p1] -= mrel2[l] * dx
                    pos[p2] += mrel1[l] * dx