import numpy as np
import pytest
from neon import logger as neon_logger

def slicable(dim, pad=0):
    if False:
        print('Hello World!')
    dim0 = np.prod(dim[:-1]) + pad
    return (dim0, dim[-1])

def test_pooling_mkl(backend_pair_bench_mkl):
    if False:
        for i in range(10):
            print('nop')
    (nm, nc) = backend_pair_bench_mkl
    layer_args = dict(dtype=np.float32, N=122, C=16, D=1, H=32, W=32, J=5)
    pool_test_args = dict(ones=0, cpu=1, nm=nm, nc=nc, alpha=1.0, ascale=1.2, beta=0.0, bpower=0.5, layer_m=nm.lrn_layer(**layer_args), layer_c=nc.lrn_layer(**layer_args), **layer_args)
    lrn_helper_mkl(**pool_test_args)

def lrn_helper_mkl(dtype, ones, cpu, alpha, beta, ascale, bpower, nm, nc, layer_m, layer_c, N, C, D, H, W, J):
    if False:
        print('Hello World!')
    dimI = layer_m.dimI
    dimO = layer_m.dimO
    if ones:
        cpuI = np.ones(slicable(dimI), dtype=np.float32)
        cpuB = np.ones(slicable(dimI), dtype=np.float32)
        cpuE = np.ones(dimO, dtype=np.float32)
        cpuO = np.ones(dimO, dtype=np.float32)
    else:
        cpuI = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuB = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuE = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)
        cpuO = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)
    devI = nm.array(cpuI.reshape(dimI), dtype=dtype)
    devB = nm.array(cpuB.reshape(dimI), dtype=dtype)
    devE = nm.array(cpuE, dtype=dtype)
    devO = nm.array(cpuO, dtype=dtype)
    devD = nm.empty(dimO, dtype=dtype)
    cccI = nc.array(cpuI.reshape(dimI), dtype=dtype)
    cccB = nc.array(cpuB.reshape(dimI), dtype=dtype)
    cccE = nc.array(cpuE, dtype=dtype)
    cccO = nc.array(cpuO, dtype=dtype)
    cccD = nc.empty(dimO, dtype=dtype)
    nm.fprop_lrn(layer_m, devI, devO, devD, alpha, beta, ascale, bpower)
    nc.fprop_lrn(layer_c, cccI, cccO, cccD, None, None, ascale, bpower)
    neon_logger.display('== denom ==')
    neon_logger.display('CPU fprop')
    neon_logger.display(cccD.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('MKL fprop')
    neon_logger.display(devD.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('== output ==')
    neon_logger.display('CPU fprop')
    neon_logger.display(cccO.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('MKL fprop')
    neon_logger.display(devO.get().reshape(C * D * H * W, N)[0:4, 0:4])
    nm.bprop_lrn(layer_m, devI, devO, devE, devB, devD, alpha, beta, ascale, bpower)
    nc.bprop_lrn(layer_c, cccI, cccO, cccE, cccB, cccD, None, None, ascale, bpower)
    neon_logger.display('== bprop ==')
    neon_logger.display('CPU bprop')
    neon_logger.display(cccB.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('MKL bprop')
    neon_logger.display(devB.get().reshape(C * D * H * W, N)[0:4, 0:4])

@pytest.mark.hasgpu
def test_pooling(backend_pair_bench):
    if False:
        i = 10
        return i + 15
    (ng, nc) = backend_pair_bench
    layer_args = dict(dtype=np.float32, N=122, C=16, D=1, H=32, W=32, J=5)
    pool_test_args = dict(ones=0, cpu=1, ng=ng, nc=nc, alpha=1.0, ascale=1.2, beta=0.0, bpower=0.5, layer_g=ng.lrn_layer(**layer_args), layer_c=nc.lrn_layer(**layer_args), **layer_args)
    lrn_helper(**pool_test_args)

def lrn_helper(dtype, ones, cpu, alpha, beta, ascale, bpower, ng, nc, layer_g, layer_c, N, C, D, H, W, J):
    if False:
        for i in range(10):
            print('nop')
    dimI = layer_g.dimI
    dimO = layer_g.dimO
    if ones:
        cpuI = np.ones(slicable(dimI), dtype=np.float32)
        cpuB = np.ones(slicable(dimI), dtype=np.float32)
        cpuE = np.ones(dimO, dtype=np.float32)
        cpuO = np.ones(dimO, dtype=np.float32)
    else:
        cpuI = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuB = np.random.uniform(-1.0, 1.0, slicable(dimI)).astype(np.float16).astype(np.float32)
        cpuE = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)
        cpuO = np.random.uniform(-1.0, 1.0, dimO).astype(np.float16).astype(np.float32)
    devI = ng.array(cpuI.reshape(dimI), dtype=dtype)
    devB = ng.array(cpuB.reshape(dimI), dtype=dtype)
    devE = ng.array(cpuE, dtype=dtype)
    devO = ng.array(cpuO, dtype=dtype)
    devD = ng.empty(dimO, dtype=dtype)
    cccI = nc.array(cpuI.reshape(dimI), dtype=dtype)
    cccB = nc.array(cpuB.reshape(dimI), dtype=dtype)
    cccE = nc.array(cpuE, dtype=dtype)
    cccO = nc.array(cpuO, dtype=dtype)
    cccD = nc.empty(dimO, dtype=dtype)
    ng.fprop_lrn(layer_g, devI, devO, devD, alpha, beta, ascale, bpower)
    nc.fprop_lrn(layer_c, cccI, cccO, cccD, None, None, ascale, bpower)
    neon_logger.display('== denom ==')
    neon_logger.display('CPU fprop')
    neon_logger.display(cccD.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('GPU fprop')
    neon_logger.display(devD.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('== output ==')
    neon_logger.display('CPU fprop')
    neon_logger.display(cccO.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('GPU fprop')
    neon_logger.display(devO.get().reshape(C * D * H * W, N)[0:4, 0:4])
    ng.bprop_lrn(layer_g, devI, devO, devE, devB, devD, alpha, beta, ascale, bpower)
    nc.bprop_lrn(layer_c, cccI, cccO, cccE, cccB, cccD, None, None, ascale, bpower)
    neon_logger.display('== bprop ==')
    neon_logger.display('CPU bprop')
    neon_logger.display(cccB.get().reshape(C * D * H * W, N)[0:4, 0:4])
    neon_logger.display('GPU bprop')
    neon_logger.display(devB.get().reshape(C * D * H * W, N)[0:4, 0:4])
if __name__ == '__main__':
    test_pooling(0)
    test_pooling_mkl(0)