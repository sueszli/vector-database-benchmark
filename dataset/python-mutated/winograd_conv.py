"""
Python code to wrap convolution kernels
"""
from __future__ import division
from future.utils import native_str
import sys
import logging
import numpy as np
import pycuda.driver as drv
from pycuda.tools import context_dependent_memoize
from . import kernel_specs
from neon.backends.cuda_templates import _common_round, _common_fp16_to_fp32, _ew_types
from neon.backends.util.source_module import SourceModule
import os.path
from .convolution import KernelGroup, NoopTransform, CompoundOps, UpdateConvReduce, BatchNormSum, ConvertDataType, FilterDimShuffle, _get_sm_count, _ceil_div, _magic64, _div64, _magic32, _flatten, _closest_divisor
from neon.util.shelver import atomic_shelve
logger = logging.getLogger(__name__)

class XpropWinograd_2x2_3x3(KernelGroup):

    def __init__(self, op, lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, filter_extern=None, bprop=False):
        if False:
            i = 10
            return i + 15
        super(XpropWinograd_2x2_3x3, self).__init__(lib, dtype, N, C, K, 1, H, W, 1, 3, 3, 1, P, Q, 0, pad_h, pad_w, 1, 1, 1, 1, 1, 1, bprop)
        SMs = _get_sm_count()
        self.autotune_key = ' '.join((native_str(x) for x in (op + '_2x2_3x3', SMs, dtype.itemsize, N, C, K, H, W, P, Q)))
        self.autotune_db_file = os.path.join(lib.cache_dir, 'autotune%d.db' % sys.version_info[0])
        self.warmup = min(max(int(5000000000000.0 / (P * Q * K * N * C * 9 * 2.0) * (SMs / 24.0)), 1), 1000)
        if filter_extern is None:
            self.init()
        else:
            self.initialized = True
            self.init(autotune=1, filter_extern=filter_extern)
        lib.set_scratch_size(self.filter_trans.size, self.bsum.size)

    def init(self, autotune=0, filter_extern=0):
        if False:
            while True:
                i = 10
        (N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params
        itemsize = self.dtype.itemsize
        if not autotune:
            with atomic_shelve(self.autotune_db_file) as autotune_db:
                if self.autotune_key in autotune_db:
                    filter_extern = autotune_db[self.autotune_key]
                    self.initialized = True
                else:
                    filter_extern = True
                    self.initialized = False
        self.filter_extern = filter_extern
        if N == 1:
            shiftN = 0
        elif N < 32:
            shiftN = len(bin(N - 1)) - 2
        else:
            shiftN = 5
        blockN = 1 << shiftN
        (superP, shiftP, superQ, shiftQ, superN) = {1: (515, 3, 768, 4, 0), 2: (515, 3, 513, 3, 1), 4: (260, 2, 514, 3, 3), 8: (260, 2, 259, 2, 7), 16: (0, 1, 260, 2, 15), 32: (0, 1, 0, 1, 31)}.get(blockN)
        blockP = 1 << shiftP
        blockQ = 1 << shiftQ
        gridP = _ceil_div(P, blockP)
        gridQ = _ceil_div(Q, blockQ)
        gridN = _ceil_div(N, blockN)
        gridK = _ceil_div(K, 32)
        gridP2 = max(gridP // 2, 1)
        gridQ2 = gridQ * 2
        n = _closest_divisor(gridN, 2)
        k = _closest_divisor(gridK, 4)
        nk = n * k
        Qnk = gridQ2 * nk
        magic_Qnk = _magic64(Qnk)
        magic_nk = _magic32(Qnk, nk)
        magic_k = _magic32(nk, k)
        gridPQ = gridP * gridQ
        grid = (gridPQ * nk, gridK // k, gridN // n)
        block = (256, 1, 1)
        options = list()
        if filter_extern:
            options.append('FX')
        options.append(('K', K))
        options.append(('W', W))
        options.append(('Q', Q))
        options.append(('N', N))
        self.kernel_opts = tuple(options)
        self.kernel_name = '%s_winograd_2x2_3x3_32x32' % self.clss
        self.kernel_args = [grid, block, None, None, None, None, None, None, None, None, None]
        self.kernel_args.extend(_flatten([C, H, P, pad_h, pad_w, H * W * N, W * N, P * Q * N, Q * N, Qnk, nk, n, k, magic_Qnk, magic_nk, magic_k, R * S * K, 4 * R * S * K * itemsize, 4 * H * W * N * itemsize, gridK, gridP2, gridQ, gridN, gridQ * gridN, gridPQ * gridN, superP, superQ, superN, shiftP, shiftQ, shiftN]))
        self.bsum = BatchNormSum(self.lib, K, gridPQ * gridN)

    def autotune(self, I, F, O):
        if False:
            while True:
                i = 10
        (start, stop) = self.lib.get_events()
        if not self.lib.warmup:
            self.lib.warmup = True
            self.init(autotune=1, filter_extern=1)
            self.bind_params(I, F, O, no_op=1)
            self.execute(repeat=self.warmup, unbind=False)
        results = []
        for external in (0, 1):
            self.init(autotune=1, filter_extern=external)
            self.bind_params(I, F, O, no_op=1)
            start.record(stream=self.lib.stream)
            self.execute(repeat=self.warmup, unbind=False)
            stop.record(stream=self.lib.stream)
            stop.synchronize()
            msecs = stop.time_since(start) / self.warmup
            results.append((msecs, external))
        results.sort()
        external = results[0][1]
        with atomic_shelve(self.autotune_db_file) as autotune_db:
            autotune_db[self.autotune_key] = external
        self.init(autotune=0, filter_extern=external)

    def bind_params(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0, relu=False, brelu=False, slope=0.0, no_op=0):
        if False:
            print('Hello World!')
        assert I.dtype == O.dtype == self.dtype
        if not self.initialized:
            self.initialized = True
            self.autotune(I, F, O)
        self.lib.scratch_buffer_init()
        filter_data = self.filter_trans.bind_params(F)
        (bsum_data, x_data) = self.xprop_params(O, X, bias, bsum, beta, relu, brelu, slope)
        self.kernel_args[2:11] = (self.lib.stream, bsum_data, x_data, O.gpudata, I.gpudata, filter_data, alpha, beta or slope, no_op)

    def execute(self, repeat=1, unbind=True):
        if False:
            print('Hello World!')
        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_options)
        for r in range(repeat):
            self.filter_trans.execute()
            kernel.prepared_async_call(*self.kernel_args)
            self.bsum.execute()
        if unbind:
            self.filter_trans.unbind()
            self.bsum.unbind()
            self.kernel_args[2:11] = (None,) * 9

class FpropWinograd_2x2_3x3(XpropWinograd_2x2_3x3):

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w, filter_extern=None):
        if False:
            print('Hello World!')
        super(FpropWinograd_2x2_3x3, self).__init__('fprop', lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, filter_extern)

    def init(self, autotune=0, filter_extern=0):
        if False:
            return 10
        super(FpropWinograd_2x2_3x3, self).init(autotune, filter_extern)
        if self.filter_extern:
            (C, K) = self.params[1:3]
            self.filter_trans = FpropFilter_2x2_3x3(self.lib, self.dtype, C, K)
        else:
            self.filter_trans = NoopTransform()

class BpropWinograd_2x2_3x3(XpropWinograd_2x2_3x3):

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w, filter_extern=None):
        if False:
            i = 10
            return i + 15
        super(BpropWinograd_2x2_3x3, self).__init__('bprop', lib, dtype, N, K, C, P, Q, H, W, 2 - pad_h, 2 - pad_w, filter_extern, bprop=True)

    def init(self, autotune=0, filter_extern=0):
        if False:
            print('Hello World!')
        super(BpropWinograd_2x2_3x3, self).init(autotune, filter_extern)
        (K, C) = self.params[1:3]
        if self.filter_extern:
            self.filter_trans = BpropFilter_2x2_3x3(self.lib, self.dtype, C, K)
        else:
            self.filter_trans = FilterDimShuffle(self.lib, self.dtype, C, 1, 3, 3, K)

class UpdateWinograd_3x3_2x2(KernelGroup):
    internal_superblock = {1: (2, 2, 257, 256, 0), 2: (1, 2, 0, 257, 256), 4: (1, 1, 0, 0, 512)}
    external_superblock = {1: (1, 1, 262, 261, 0), 2: (0, 1, 0, 262, 261), 4: (0, 0, 0, 0, 517)}

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w):
        if False:
            i = 10
            return i + 15
        assert N in (1, 2) or N % 4 == 0
        super(UpdateWinograd_3x3_2x2, self).__init__(lib, dtype, N, C, K, 1, H, W, 1, 3, 3, 1, P, Q, 0, pad_h, pad_w, 1, 1, 1, 1, 1, 1)
        SMs = _get_sm_count()
        self.autotune_key = [native_str(x) for x in ('update_3x3_2x2', SMs, 0, dtype.itemsize, N, C, K, H, W, P, Q)]
        self.autotune_db_file = os.path.join(lib.cache_dir, 'autotune%d.db' % sys.version_info[0])
        self.init()
        lib.set_scratch_size(self.image_trans.size, self.output_trans.size)
        self.warmup = min(max(int(5000000000000.0 / (P * Q * K * N * C * 9 * 2.0) * (SMs / 24.0)), 1), 1000)

    def init(self, autotune=False):
        if False:
            print('Hello World!')
        (N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params
        loopN = 4 if N >= 4 else N
        blkN = 4 if N >= 3 else N
        superI = UpdateWinograd_3x3_2x2.external_superblock.get(blkN)
        superE = UpdateWinograd_3x3_2x2.internal_superblock.get(blkN)
        blkXI = 1 << superI[1]
        blkY = 1 << superE[0]
        blkX = 1 << superE[1]
        GY = _ceil_div(P, 2)
        GX = _ceil_div(Q, 2)
        GYS = self.GYS = _ceil_div(P, blkY)
        GXS = self.GXS = _ceil_div(Q, blkX)
        if autotune:
            (strideY, strideX, external) = autotune
        else:
            with atomic_shelve(self.autotune_db_file) as autotune_db:
                autotune_key = ' '.join(self.autotune_key)
                if autotune_key in autotune_db:
                    (strideY, strideX, external) = autotune_db[autotune_key]
                    self.initialized = True
                else:
                    if GYS * GXS > 768:
                        strideY = 768
                        strideX = 1
                    else:
                        strideY = GYS
                        strideX = GXS
                    external = True
                    self.initialized = False
        loopXI = N * (strideX * blkXI - 1)
        loopXE = N * (strideX * blkX - 1)
        Np = N * self.dtype.itemsize
        XNp = W * N * self.dtype.itemsize
        XN2p = W * N * 2 * self.dtype.itemsize
        QNp = Q * N * self.dtype.itemsize
        gridK = _ceil_div(K, 32)
        gridC = _ceil_div(C, 32)
        k_size = _closest_divisor(gridK, 4)
        c_size = _closest_divisor(gridC, 4)
        C_size = gridC // c_size
        kc = k_size * c_size
        Qkc = strideX * kc
        PQkc = strideY * Qkc
        CPQkc = C_size * PQkc
        gridPQ = strideY * strideX
        gridPQKC = gridPQ * gridK * gridC
        magic_CPQkc = _magic64(CPQkc)
        magic_PQkc = _magic64(PQkc)
        magic_Qkc = _magic64(Qkc)
        magic_kc = _magic32(Qkc, kc)
        magic_c = _magic32(kc, c_size)
        CRSK = C * R * S * K
        self.blocksCK = gridK * gridC
        options = list()
        if external:
            options.append('IX')
            WN = GX * N
            HWN = GY * WN
            self.image_trans = UpdateImage_3x3_2x2(self.lib, self.dtype, N, C, K, H, W, P, Q, pad_h, pad_w)
        else:
            WN = W * N
            HWN = H * WN
            superI = superE
            self.image_trans = NoopTransform()
        if gridPQ == 1 or self.lib.deterministic:
            self.output_trans = UpdateConvReduce(self.lib, gridPQ, CRSK)
            self.zero = False
            options.append('D')
        else:
            self.output_trans = UpdateConvReduce(self.lib, 1, CRSK)
            self.zero = True
        self.kernel_opts = tuple(options)
        self.kernel_name = '%s_winograd_3x3_2x2_32x32' % self.clss
        self.kernel_args = [(gridPQKC, 1, 1), (256, 1, 1), None, None, None, None, 1.0]
        self.kernel_args.extend(_flatten([H, W, P, Q, C, K, N, pad_h, pad_w, GY, GX, GYS, GXS, superI, superE, loopXI, loopXE, loopN, strideY, strideX, WN, HWN, Q * N, P * Q * N, S * K, R * S * K, Np, XNp, XN2p, QNp, CPQkc, PQkc, Qkc, kc, c_size, k_size, magic_CPQkc, magic_PQkc, magic_Qkc, magic_kc, magic_c, CRSK]))

    def autotune(self, I, E, O):
        if False:
            i = 10
            return i + 15
        autotune_key = ' '.join(self.autotune_key)
        (start, stop) = self.lib.get_events()
        if not self.lib.warmup:
            self.lib.warmup = True
            self.init(autotune=(self.GYS, 1, 1))
            self.bind_params(I, E, O, no_op=True)
            self.execute(repeat=self.warmup, unbind=False)
        block_slots = _get_sm_count()
        loopsN = max(self.params[0] // 4, 1)
        modes = (0, 1) if self.params[0] <= 4 else (1,)
        gys = float(self.GYS)
        gxs = float(self.GYS)
        small_set = gys * gxs <= 512
        results = []
        sys.stdout.write('Autotune ' + native_str(self))
        progress = 0
        for threshold in (True, False):
            for external in modes:
                for strideY in range(1, self.GYS + 1):
                    for strideX in range(1, self.GXS + 1):
                        if progress % 32 == 0:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                        progress += 1
                        outputs = strideY * strideX
                        blocks = self.blocksCK * strideY * strideX
                        depth = gys / strideY * (gxs / strideX) * loopsN
                        filters = outputs <= 768 and blocks >= block_slots and (depth >= 32.0)
                        if small_set or (threshold and filters) or (not threshold and (not filters)):
                            settings = (strideY, strideX, external)
                            self.init(autotune=settings)
                            self.bind_params(I, E, O, no_op=True)
                            start.record(stream=self.lib.stream)
                            self.execute(repeat=2, unbind=False)
                            stop.record(stream=self.lib.stream)
                            stop.synchronize()
                            msecs = stop.time_since(start) / 2.0
                            results.append((msecs, settings))
            if len(results) > 0:
                break
        sys.stdout.write('\n')
        results.sort()
        settings = results[0][1]
        with atomic_shelve(self.autotune_db_file) as autotune_db:
            autotune_db[autotune_key] = settings
            if settings[0] * settings[1] <= 8:
                self.autotune_key[2] = native_str(1 - int(self.autotune_key[2]))
                autotune_key = ' '.join(self.autotune_key)
                autotune_db[autotune_key] = settings
        self.init(autotune=settings)

    def bind_params(self, I, E, O, alpha=1.0, beta=0.0, no_op=False):
        if False:
            while True:
                i = 10
        assert I.dtype == E.dtype == self.dtype
        if not self.initialized:
            self.initialized = True
            self.autotune(I, E, O)
        self.lib.scratch_buffer_init()
        image_data = self.image_trans.bind_params(I)
        output_data = self.output_trans.bind_params(O, alpha, beta, no_op)
        if self.zero:
            self.zero_args = (output_data, 0, O.size, self.lib.stream)
        self.kernel_args[2:6] = (self.lib.stream, output_data, image_data, E.gpudata)

    def execute(self, repeat=1, unbind=True):
        if False:
            while True:
                i = 10
        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_opts)
        for r in range(repeat):
            if self.zero:
                drv.memset_d32_async(*self.zero_args)
            self.image_trans.execute()
            kernel.prepared_async_call(*self.kernel_args)
            self.output_trans.execute()
        if unbind:
            self.image_trans.unbind()
            self.output_trans.unbind()
            self.zero_args = None
            self.kernel_args[2:6] = (None,) * 4

class XpropWinograd_4x4_3x3(KernelGroup):

    def __init__(self, op, lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, external=None, bprop=False):
        if False:
            while True:
                i = 10
        super(XpropWinograd_4x4_3x3, self).__init__(lib, dtype, N, C, K, 1, H, W, 1, 3, 3, 1, P, Q, 0, pad_h, pad_w, 1, 1, 1, 1, 1, 1, bprop)
        SMs = _get_sm_count()
        self.autotune_key = ' '.join((native_str(x) for x in (op + '_4x4_3x3', SMs, dtype.itemsize, N, C, K, H, W, P, Q)))
        self.autotune_db_file = os.path.join(lib.cache_dir, 'autotune%d.db' % sys.version_info[0])
        self.warmup = min(max(int(5000000000000.0 / (P * Q * K * N * C * 9 * 2.0) * (SMs / 24.0)), 1), 1000)
        if external is None:
            self.init()
        else:
            self.initialized = True
            self.init(autotune=1, external=external)
        lib.set_scratch_size(self.image_size, self.filter_trans.size, self.bsum.size)

    def init(self, autotune=0, external=1):
        if False:
            i = 10
            return i + 15
        (N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params
        if not autotune:
            with atomic_shelve(self.autotune_db_file) as autotune_db:
                if self.autotune_key in autotune_db:
                    external = autotune_db[self.autotune_key]
                    self.initialized = True
                else:
                    external = True
                    self.initialized = False
        external = True
        self.initialized = True
        self.external = external
        if N == 1:
            shlN = 0
        elif N < 32:
            shlN = len(bin(N - 1)) - 2
        else:
            shlN = 5
        (shlY, shlX, maskY, shrY, maskX, shrX, maskN, supY, supX) = {0: (4, 5, 24, 3, 7, 0, 0, 515, 768), 1: (4, 4, 24, 3, 6, 1, 1, 515, 513), 2: (3, 4, 16, 4, 12, 2, 3, 260, 514), 3: (3, 3, 16, 4, 8, 3, 7, 260, 259), 4: (2, 3, 0, 0, 16, 4, 15, 0, 260), 5: (2, 2, 0, 0, 0, 0, 31, 0, 0)}.get(shlN)
        itemsize = self.dtype.itemsize
        GYS = _ceil_div(P, 1 << shlY)
        GXS = _ceil_div(Q, 1 << shlX)
        GN = _ceil_div(N, 1 << shlN)
        GK = _ceil_div(K, 32)
        GYS2 = max(GYS // 2, 1)
        GXS2 = GXS * 2
        k = _closest_divisor(GK, 4)
        self.kernel_args = [(GYS * GXS * k, GK // k, GN), (640, 1, 1), None, None, None, None, None, None, None, None, None]
        options = list()
        if external:
            self.kernel_name = '%s_winograd_4x4_3x3_32x32_X' % self.clss
            options.append(('Q', Q))
            options.append(('N', N))
            Xk = GXS * k
            magic_GXS2 = _magic64(GXS2)
            magic_Xk = _magic64(Xk)
            magic_k = _magic32(Xk, k)
            self.image_size = itemsize * 1152 * C * GXS * GYS * GN
            self.image_args = [(GN, GYS * GXS, C), (32, 1, 1), None, None, None, H, W, N, pad_h, pad_w, GXS, GYS2, GXS2, magic_GXS2[0], magic_GXS2[1], shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN, H * W * N, W * N, GYS * GXS * C * 1152, GXS * C * 1152, C * 1152]
            self.kernel_args.extend(_flatten([C, K, N, Xk, k, magic_Xk, magic_k, C * 1152, GXS * C * 1152, GYS * GXS * C * 1152, P, Q, Q * N, P * Q * N, P * Q * N * 15, maskN, shlX, shlY, supX, supY, GN, GXS * GN, GYS * GXS * GN]))
        else:
            self.kernel_name = '%s_winograd_4x4_3x3_32x32' % self.clss
            options.append(('K', K))
            options.append(('W', W))
            options.append(('Q', Q))
            options.append(('N', N))
            self.image_size = 0
            Xk = GXS2 * k
            magic_Xk = _magic64(Xk)
            magic_k = _magic32(Xk, k)
            self.kernel_args.extend(_flatten([C, K, N, H, W, H * W * N, W * N, GYS2, GXS, Xk, k, magic_Xk, magic_k, P, Q, Q * N, P * Q * N, P * Q * N * 15, maskN, shlX, shlY, supX, supY, pad_w, pad_h, R * S * K, R * S * K * 2 * itemsize, H * W * N * 2 * itemsize, GN, GXS * GN, GYS * GXS * GN]))
        self.kernel_opts = tuple(options)
        self.bsum = BatchNormSum(self.lib, K, GYS * GXS * GN)

    def autotune(self, I, F, O):
        if False:
            i = 10
            return i + 15
        (start, stop) = self.lib.get_events()
        if not self.lib.warmup:
            self.lib.warmup = True
            self.init(autotune=1, external=1)
            self.bind_params(I, F, O, no_op=1)
            self.execute(repeat=self.warmup, unbind=False)
        results = []
        for external in (0, 1):
            self.init(autotune=1, external=external)
            self.bind_params(I, F, O, no_op=1)
            start.record(stream=self.lib.stream)
            self.execute(repeat=self.warmup, unbind=False)
            stop.record(stream=self.lib.stream)
            stop.synchronize()
            msecs = stop.time_since(start) / self.warmup
            results.append((msecs, external))
        results.sort()
        external = results[0][1]
        with atomic_shelve(self.autotune_db_file) as autotune_db:
            autotune_db[self.autotune_key] = external
        self.init(autotune=0, external=external)

    def bind_params(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0, relu=False, brelu=False, slope=0.0, no_op=0):
        if False:
            for i in range(10):
                print('nop')
        assert I.dtype == O.dtype == self.dtype
        if not self.initialized:
            self.initialized = True
            self.autotune(I, F, O)
        self.lib.scratch_buffer_init()
        if self.image_size:
            image_data = self.lib.scratch_buffer_offset(self.image_size)
            self.image_args[2:5] = (self.lib.stream, image_data, I.gpudata)
        else:
            image_data = I.gpudata
        filter_data = self.filter_trans.bind_params(F)
        (bsum_data, x_data) = self.xprop_params(O, X, bias, bsum, beta, relu, brelu, slope)
        self.kernel_args[2:11] = (self.lib.stream, bsum_data, x_data, O.gpudata, image_data, filter_data, alpha, beta or slope, no_op)

    def execute(self, repeat=1, unbind=True):
        if False:
            for i in range(10):
                print('nop')
        if self.image_size:
            image_kernel = _get_xprop_image_4x4_3x3_kernel(self.dtype_str)
        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_options)
        for r in range(repeat):
            if self.image_size:
                image_kernel.prepared_async_call(*self.image_args)
            self.filter_trans.execute()
            kernel.prepared_async_call(*self.kernel_args)
            self.bsum.execute()
        if unbind:
            self.filter_trans.unbind()
            self.bsum.unbind()
            self.kernel_args[2:11] = (None,) * 9
            if self.image_size:
                self.image_args[2:5] = (None,) * 3

class FpropWinograd_4x4_3x3(XpropWinograd_4x4_3x3):

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w, external=None):
        if False:
            while True:
                i = 10
        super(FpropWinograd_4x4_3x3, self).__init__('fprop', lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, external)

    def init(self, autotune=0, external=1):
        if False:
            print('Hello World!')
        super(FpropWinograd_4x4_3x3, self).init(autotune, external)
        (C, K) = self.params[1:3]
        if self.external:
            self.filter_trans = FpropFilter_4x4_3x3(self.lib, self.dtype, C, K)
        elif self.dtype.itemsize != 4:
            self.filter_trans = ConvertDataType(self.lib, self.dtype, C * 9 * K, out_mode=False)
        else:
            self.filter_trans = NoopTransform()

class BpropWinograd_4x4_3x3(XpropWinograd_4x4_3x3):

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w, external=None):
        if False:
            while True:
                i = 10
        super(BpropWinograd_4x4_3x3, self).__init__('bprop', lib, dtype, N, K, C, P, Q, H, W, 2 - pad_h, 2 - pad_w, external, bprop=True)

    def init(self, autotune=0, external=1):
        if False:
            i = 10
            return i + 15
        super(BpropWinograd_4x4_3x3, self).init(autotune, external)
        (K, C) = self.params[1:3]
        if self.external:
            self.filter_trans = BpropFilter_4x4_3x3(self.lib, self.dtype, C, K)
        else:
            self.filter_trans = FilterDimShuffle(self.lib, self.dtype, C, 1, 3, 3, K)

class UpdateWinograd_3x3_4x4(KernelGroup):

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w):
        if False:
            return 10
        super(UpdateWinograd_3x3_4x4, self).__init__(lib, dtype, N, C, K, 1, H, W, 1, 3, 3, 1, P, Q, 0, pad_h, pad_w, 1, 1, 1, 1, 1, 1)
        SMs = _get_sm_count()
        self.autotune_key = [native_str(x) for x in ('update_3x3_4x4', SMs, 0, dtype.itemsize, N, C, K, H, W, P, Q)]
        self.autotune_db_file = os.path.join(lib.cache_dir, 'autotune%d.db' % sys.version_info[0])
        self.init()
        lib.set_scratch_size(self.image_size, self.delta_size, self.output_trans.size)
        self.warmup = min(max(int(5000000000000.0 / (P * Q * K * N * C * 9 * 2.0) * (SMs / 24.0)), 1), 1000)

    def init(self, autotune=False):
        if False:
            i = 10
            return i + 15
        (N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params
        itemsize = self.dtype.itemsize
        if N == 1:
            shlN = 0
        elif N < 16:
            shlN = len(bin(N - 1)) - 2
        else:
            shlN = 4
        GC32 = _ceil_div(C, 32)
        GK32 = _ceil_div(K, 32)
        GC16 = _ceil_div(GC32 * 32, 16)
        GK16 = _ceil_div(GK32 * 32, 16)
        GY = _ceil_div(P, 4)
        GX = _ceil_div(Q, 4)
        if shlN > 1 or GX >= 32:
            (shlY, shlX, maskY, shrY, maskX, shrX, maskN) = {0: (2, 6, 0, 0, 15, 0, 0), 1: (2, 5, 0, 0, 14, 1, 1), 2: (2, 4, 0, 0, 12, 2, 3), 3: (2, 3, 0, 0, 8, 3, 7), 4: (2, 2, 0, 0, 0, 0, 15)}.get(shlN)
        elif shlN > 0 or GX >= 16:
            (shlY, shlX, maskY, shrY, maskX, shrX, maskN) = {0: (3, 5, 8, 3, 7, 0, 0), 1: (3, 4, 8, 3, 6, 1, 1)}.get(shlN)
        else:
            (shlY, shlX, maskY, shrY, maskX, shrX, maskN) = {0: (4, 4, 12, 2, 3, 0, 0)}.get(shlN)
        GYS = _ceil_div(P, 1 << shlY)
        GXS = _ceil_div(Q, 1 << shlX)
        GN16 = _ceil_div(N, 1 << shlN)
        GYS2 = max(GYS // 2, 1)
        GXS2 = GXS * 2
        groupC = 2
        groupK = 2
        groupN = 1 if GN16 & 1 else 2
        shift_c = groupC - 1
        shift_k = groupK - 1
        shift_n = groupN - 1
        X2cn = GXS2 * groupC * groupN
        Xkn = GXS * groupK * groupN
        magic_X2cn = _magic64(X2cn)
        magic_Xkn = _magic64(Xkn)
        self.image_size = GC32 * GY * GX * N * 1152 * itemsize
        self.image_args = [(GYS * GXS * groupC * groupN, GN16 // groupN, GC16 // groupC), (256, 1, 1), None, None, None, C, H, W, N, pad_h, pad_w, GY, GX, GXS, GYS2, X2cn, magic_X2cn[0], magic_X2cn[1], shift_c, shift_n, shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN, H * W * N, W * N, GY * GX * N * 1152, GX * N * 1152, N * 1152]
        self.delta_size = GK32 * GY * GX * N * 1152 * itemsize
        self.delta_args = [(GYS * GXS * groupK * groupN, GN16 // groupN, GK16 // groupK), (256, 1, 1), None, None, None, K, P, Q, N, GY, GX, Xkn, magic_Xkn[0], magic_Xkn[1], shift_k, shift_n, shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN, P * Q * N, Q * N, GY * GX * N * 1152, GX * N * 1152, N * 1152]
        Gc = _closest_divisor(GC32, 4)
        Gk = _closest_divisor(GK32, 4)
        GC = GC32 // Gc
        GK = GK32 // Gk
        kc = Gk * Gc
        YXN = GY * GX * N
        YXN2 = self.YXN2 = _ceil_div(YXN, 2)
        self.maxYXN2 = max(1, YXN2 // N)
        if autotune:
            strideYXN = autotune
        else:
            with atomic_shelve(self.autotune_db_file) as autotune_db:
                autotune_key = ' '.join(self.autotune_key)
                if autotune_key in autotune_db:
                    strideYXN = autotune_db[autotune_key]
                    self.initialized = True
                else:
                    strideYXN = self.maxYXN2
                    self.initialized = False
        self.blocksCK = GC32 * GK32
        magic_sYXN = _magic64(strideYXN)
        magic_kc = _magic64(kc)
        magic_c = _magic32(kc, Gc)
        CRSK = C * R * S * K
        options = list()
        if strideYXN == 1 or self.lib.deterministic:
            self.output_trans = UpdateConvReduce(self.lib, strideYXN, CRSK)
            self.zero = False
            options.append('D')
        else:
            self.output_trans = UpdateConvReduce(self.lib, 1, CRSK)
            self.zero = True
        self.kernel_opts = tuple(options)
        self.kernel_name = '%s_winograd_3x3_4x4_32x32' % self.clss
        self.kernel_args = [(strideYXN * Gk * Gc, GC, GK), (640, 1, 1), None, None, None, None, 1.0]
        self.kernel_args.extend(_flatten([K, C, Gk, Gc, kc, magic_kc, magic_c, YXN2, strideYXN, magic_sYXN, strideYXN * 2 * 1152 * itemsize, YXN, YXN * 1152, R * S * K, CRSK, K * 4, S * K * 4, (R * S * K * 15 - S * K * 2) * 4]))

    def autotune(self, I, E, O):
        if False:
            return 10
        autotune_key = ' '.join(self.autotune_key)
        if not self.lib.warmup:
            self.lib.warmup = True
            self.init(autotune=self.maxYXN2)
            self.bind_params(I, E, O, no_op=True)
            self.execute(repeat=self.warmup, unbind=False)
        (start, stop) = self.lib.get_events()
        block_slots = _get_sm_count()
        small_set = self.YXN2 < 512
        YXN2 = float(self.YXN2)
        results = []
        sys.stdout.write('Autotune ' + native_str(self))
        progress = 0
        for threshold in (True, False):
            for strideYXN in range(1, self.maxYXN2 + 1):
                if progress % 32 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                progress += 1
                blocks = self.blocksCK * strideYXN
                depth = YXN2 / strideYXN
                filters = blocks >= block_slots and blocks <= 24 * block_slots and (depth >= 32.0)
                if small_set or (threshold and filters) or (not threshold and (not filters)):
                    self.init(autotune=strideYXN)
                    self.bind_params(I, E, O, no_op=True)
                    start.record(stream=self.lib.stream)
                    self.execute(repeat=2, unbind=False)
                    stop.record(stream=self.lib.stream)
                    stop.synchronize()
                    msecs = stop.time_since(start) / 2.0
                    results.append((msecs, strideYXN))
            if len(results) > 0:
                break
        sys.stdout.write('\n')
        results.sort()
        strideYXN = results[0][1]
        with atomic_shelve(self.autotune_db_file) as autotune_db:
            autotune_db[autotune_key] = strideYXN
            if strideYXN <= 8:
                self.autotune_key[2] = native_str(1 - int(self.autotune_key[2]))
                autotune_key = ' '.join(self.autotune_key)
                autotune_db[autotune_key] = strideYXN
        self.init(autotune=strideYXN)

    def bind_params(self, I, E, O, alpha=1.0, beta=0.0, no_op=False):
        if False:
            while True:
                i = 10
        assert I.dtype == E.dtype == self.dtype
        if not self.initialized:
            self.initialized = True
            self.autotune(I, E, O)
        self.lib.scratch_buffer_init()
        image_data = self.lib.scratch_buffer_offset(self.image_size)
        delta_data = self.lib.scratch_buffer_offset(self.delta_size)
        output_data = self.output_trans.bind_params(O, alpha, beta, no_op)
        self.image_args[2:5] = (self.lib.stream, image_data, I.gpudata)
        self.delta_args[2:5] = (self.lib.stream, delta_data, E.gpudata)
        if self.zero:
            self.zero_args = [output_data, 0, O.size, self.lib.stream]
        self.kernel_args[2:6] = (self.lib.stream, output_data, image_data, delta_data)

    def execute(self, repeat=1, unbind=True):
        if False:
            print('Hello World!')
        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_opts)
        image_kernel = _get_update_image_3x3_4x4_kernel(self.dtype_str)
        delta_kernel = _get_update_delta_3x3_4x4_kernel(self.dtype_str)
        for r in range(repeat):
            if self.zero:
                drv.memset_d32_async(*self.zero_args)
            image_kernel.prepared_async_call(*self.image_args)
            delta_kernel.prepared_async_call(*self.delta_args)
            kernel.prepared_async_call(*self.kernel_args)
            self.output_trans.execute()
        if unbind:
            self.output_trans.unbind()
            self.zero_args = None
            self.kernel_args[2:6] = (None,) * 4
            self.image_args[2:5] = (None,) * 3
            self.delta_args[2:5] = (None,) * 3

class XpropWinograd_2x2_5x5(KernelGroup):

    def __init__(self, op, lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w, bprop=False):
        if False:
            print('Hello World!')
        super(XpropWinograd_2x2_5x5, self).__init__(lib, dtype, N, C, K, 1, H, W, 1, 5, 5, 1, P, Q, 0, pad_h, pad_w, 1, 1, 1, 1, 1, 1, bprop)
        self.init()
        lib.set_scratch_size(self.filter_trans.size, self.output_trans.size)

    def init(self):
        if False:
            return 10
        (N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w) = self.params
        if N == 1:
            shlN = 0
        elif N < 32:
            shlN = len(bin(N - 1)) - 2
        else:
            shlN = 5
        itemsize = self.dtype.itemsize
        if itemsize == 4:
            (shlY, shlX, supY, supX, supN, SupY, SupX, SupN) = {0: (3, 4, 515, 768, 0, 515, 768, 0), 1: (3, 3, 515, 513, 1, 515, 513, 1), 2: (2, 3, 260, 514, 3, 260, 514, 3), 3: (2, 2, 260, 259, 7, 260, 259, 7), 4: (1, 2, 0, 260, 15, 0, 260, 15), 5: (1, 1, 0, 0, 31, 0, 0, 31)}.get(shlN)
        else:
            (shlY, shlX, supY, supX, supN, SupY, SupX, SupN) = {0: (3, 4, 515, 768, 0, 514, 512, 0), 1: (3, 3, 515, 513, 1, 514, 512, 0), 2: (2, 3, 260, 514, 3, 259, 513, 1), 3: (2, 2, 260, 259, 7, 259, 258, 3), 4: (1, 2, 0, 260, 15, 0, 259, 7), 5: (1, 1, 0, 0, 31, 0, 0, 15)}.get(shlN)
        GYS = _ceil_div(H, 1 << shlY)
        GXS = _ceil_div(W, 1 << shlX)
        GN = _ceil_div(N, 1 << shlN)
        GK = _ceil_div(K, 32)
        GYS2 = max(GYS // 2, 1)
        GXS2 = GXS * 2
        k = _closest_divisor(GK, 2)
        Xk = GXS2 * k
        magic_Xk = _magic64(Xk)
        magic_k = _magic32(Xk, k)
        options = list()
        options.append(('W', W))
        options.append(('Q', Q))
        options.append(('N', N))
        self.kernel_args = [(GYS * GXS * k, GK // k, GN), (640, 1, 1), None, None, None, None, None, None]
        self.kernel_name = '%s_winograd_2x2_5x5_32x32' % self.clss
        self.kernel_opts = tuple(options)
        self.zero_args = None
        self.kernel_args.extend(_flatten([C, K, N, H, W, H * W * N, W * N, GYS2, GXS, Xk, k, magic_Xk, magic_k, P, Q, Q * N, P * Q * N, P * Q * N * itemsize, P * Q * N * 15 * itemsize, shlY, shlX, shlN, supY, supX, supN, SupY, SupX, SupN, pad_w, pad_h, H * W * N * 2 * itemsize, C * 1152]))
        self.output_trans = CompoundOps(self.lib, self.dtype, K, P * Q * N)

    def bind_params(self, I, F, O, X=None, bias=None, bsum=None, alpha=1.0, beta=0.0, relu=False, brelu=False, slope=0.0, no_op=0):
        if False:
            i = 10
            return i + 15
        assert I.dtype == O.dtype == self.dtype
        self.lib.scratch_buffer_init()
        filter_data = self.filter_trans.bind_params(F)
        if beta == 1.0 and X is None:
            self.zero_args = None
            self.kernel_args[2:8] = (self.lib.stream, O.gpudata, I.gpudata, filter_data, alpha, no_op)
            self.output_trans.kernel = None
        else:
            output_data = self.output_trans.bind_params(O, X, bias, bsum, alpha, beta or slope, relu, brelu)
            self.zero_args = (output_data, 0, O.nbytes, self.lib.stream)
            self.kernel_args[2:8] = (self.lib.stream, output_data, I.gpudata, filter_data, 1.0, no_op)

    def execute(self, repeat=1, unbind=True):
        if False:
            i = 10
            return i + 15
        kernel = kernel_specs.get_kernel(self.kernel_name, self.kernel_opts)
        for r in range(repeat):
            if self.zero_args is not None:
                drv.memset_d8_async(*self.zero_args)
            self.filter_trans.execute()
            kernel.prepared_async_call(*self.kernel_args)
            self.output_trans.execute()
        if unbind:
            self.filter_trans.unbind()
            self.output_trans.unbind()
            self.kernel_args[2:8] = (None,) * 6

class FpropWinograd_2x2_5x5(XpropWinograd_2x2_5x5):

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w):
        if False:
            for i in range(10):
                print('nop')
        super(FpropWinograd_2x2_5x5, self).__init__('fprop', lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w)

    def init(self):
        if False:
            return 10
        super(FpropWinograd_2x2_5x5, self).init()
        (C, K) = self.params[1:3]
        self.filter_trans = FpropFilter_2x2_5x5(self.lib, self.dtype, C, K)

class BpropWinograd_2x2_5x5(XpropWinograd_2x2_5x5):

    def __init__(self, lib, dtype, N, C, K, D, H, W, T, R, S, M, P, Q, pad_d, pad_h, pad_w, str_d, str_h, str_w, dil_d, dil_h, dil_w):
        if False:
            for i in range(10):
                print('nop')
        super(BpropWinograd_2x2_5x5, self).__init__('bprop', lib, dtype, N, K, C, P, Q, H, W, 4 - pad_h, 4 - pad_w, bprop=True)

    def init(self):
        if False:
            print('Hello World!')
        super(BpropWinograd_2x2_5x5, self).init()
        (K, C) = self.params[1:3]
        self.filter_trans = BpropFilter_2x2_5x5(self.lib, self.dtype, C, K)

class UpdateImage_3x3_2x2(object):

    def __init__(self, lib, dtype, N, C, K, H, W, P, Q, pad_h, pad_w):
        if False:
            i = 10
            return i + 15
        if N == 1:
            shlN = 0
        elif N < 16:
            shlN = len(bin(N - 1)) - 2
        else:
            shlN = 4
        GY = _ceil_div(P, 2)
        GX = _ceil_div(Q, 2)
        if shlN > 1 or GX >= 32:
            (shlY, shlX, maskY, shrY, maskX, shrX, maskN) = {0: (1, 5, 0, 0, 15, 0, 0), 1: (1, 4, 0, 0, 14, 1, 1), 2: (1, 3, 0, 0, 12, 2, 3), 3: (1, 2, 0, 0, 8, 3, 7), 4: (1, 1, 0, 0, 0, 0, 15)}.get(shlN)
        elif shlN > 0 or GX >= 16:
            (shlY, shlX, maskY, shrY, maskX, shrX, maskN) = {0: (2, 4, 8, 3, 7, 0, 0), 1: (2, 3, 8, 3, 6, 1, 1)}.get(shlN)
        else:
            (shlY, shlX, maskY, shrY, maskX, shrX, maskN) = {0: (3, 3, 12, 2, 3, 0, 0)}.get(shlN)
        blkN = 1 << shlN
        blkY = 1 << shlY
        blkX = 1 << shlX
        gridC = _ceil_div(C, 32)
        GC = _ceil_div(gridC * 32, 16)
        GYS = _ceil_div(P, blkY)
        GXS = _ceil_div(Q, blkX)
        GN = _ceil_div(N, blkN)
        GYS2 = GYS // 2
        GXS2 = GXS * 2
        groupC = 2
        groupN = 1 if GN & 1 else 2
        shift_c = groupC - 1
        shift_n = groupN - 1
        X2cn = GXS2 * groupC * groupN
        magic_X2cn = _magic64(X2cn)
        self.lib = lib
        self.dtype = dtype.str[1:]
        self.dim = (gridC, GY, GX, N, 4, 4, 32)
        self.size = int(np.prod(self.dim)) * dtype.itemsize
        self.args = [(GYS * GXS * groupC * groupN, GN // groupN, GC // groupC), (256, 1, 1), None, None, None, C, H, W, N, pad_h, pad_w, GY, GX, GXS, GYS2, X2cn, magic_X2cn[0], magic_X2cn[1], shift_c, shift_n, shlY, shlX, maskY, shrY, maskX, shrX, shlN, maskN, H * W * N, W * N, GY * GX * N * 512, GX * N * 512, N * 512]

    def bind_params(self, I):
        if False:
            i = 10
            return i + 15
        self.data = self.lib.scratch_buffer_offset(self.size)
        self.args[2:5] = (self.lib.stream, self.data, I.gpudata)
        self.kernel = _get_update_image_3x3_2x2_kernel(self.dtype)
        return self.data

    def execute(self):
        if False:
            return 10
        self.kernel.prepared_async_call(*self.args)

    def unbind(self):
        if False:
            i = 10
            return i + 15
        self.kernel = None
        self.args[2:5] = (None,) * 3

@context_dependent_memoize
def _get_update_image_3x3_2x2_kernel(dtype):
    if False:
        return 10
    code = "\n%(common)s\n\n__global__ void update_image_3x3_2x2(\n    %(type)s* Out, const %(type)s* In,\n    int C, int Y, int X, int N, int pad_y, int pad_x,\n    int GY, int GX, int GXS, int GYS2,\n    int X2cn, int magic_X2cn, int shift_X2cn, int shift_c, int shift_n,\n    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,\n    int YXN, int XN, int GYGXN512, int GXN512, int N512)\n{\n    // Add padding to avoid all but 1 shared bank conflict on loads\n    %(type)s __shared__ share[16][16*16 + 2];\n\n    int tid        = threadIdx.x;\n    int blk_YXcn   = blockIdx.x;\n    int blk_N      = blockIdx.y;\n    int blk_C      = blockIdx.z;\n\n    // unpack y,x,c,n from blockIdx.x\n    int gy2     = div64(blk_YXcn, magic_X2cn, shift_X2cn);\n    int blk_Xcn = blk_YXcn - gy2*X2cn;\n\n    int shift_cn = shift_c + shift_n;\n\n    int gx2    = blk_Xcn >> shift_cn;\n    int blk_cn = blk_Xcn - (gx2 << shift_cn);\n\n    int blk_c = blk_cn >> shift_n;\n    int blk_n = blk_cn - (blk_c << shift_n);\n\n    int blkN = (blk_N << shift_n) + blk_n;\n    int blkC = (blk_C << shift_c) + blk_c;\n\n    // Implement a square wave block id remapping\n    // (for all but last row (if odd number of rows))\n    int gy = gy2 << 1;\n    int gx = gx2;\n    if (gy2 != GYS2)\n    {\n        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);\n        gx  = gx2 >> 1;\n    }\n    // Scan backwards on odd rows\n    if (gy2 & 1)\n        gx = GXS - gx - 1;\n\n    int ns = tid & 15;\n    int cs = tid >> 4;\n\n    // Super block YXN coordinates\n    int y0 = (gy << shlY) + (((ns & maskY) >> shrY) << 1) - pad_y;\n    int x0 = (gx << shlX) + (((ns & maskX) >> shrX) << 1) - pad_x;\n    int n  = (blkN << shlN) + (ns & maskN);\n    int c  = (blkC << 4) + cs;\n\n    bool valid = c < C && n < N;\n\n    bool xin[4], yin[4];\n    float I[4][4], T[4][4];\n\n    #pragma unroll\n    for (int i = 0; i < 4; i++)\n    {\n        xin[i] = x0 + i >= 0 && x0 + i < X && valid;\n        yin[i] = y0 + i >= 0 && y0 + i < Y;\n    }\n\n    int offset = c*YXN + y0*XN + x0*N + n;\n\n    #pragma unroll\n    for (int y = 0; y < 4; y++)\n    {\n        if (y) offset += XN;\n\n        #pragma unroll\n        for (int x = 0; x < 4; x++)\n        {\n            %(type)s val = 0;\n            if (yin[y] && xin[x])\n                val = __ldg(In + x*N + offset);\n            I[y][x] = %(cvt_in)s(val);\n        }\n    }\n    #pragma unroll\n    for (int i = 0; i < 4; i++)\n    {\n        T[0][i] = I[0][i] - I[2][i];\n        T[1][i] = I[1][i] + I[2][i];\n        T[2][i] = I[2][i] - I[1][i];\n        T[3][i] = I[3][i] - I[1][i];\n    }\n    #pragma unroll\n    for (int i = 0; i < 4; i++)\n    {\n        share[ns][cs + 16*(i*4 + 0)] = %(cvt_out)s(T[i][0] - T[i][2]);\n        share[ns][cs + 16*(i*4 + 1)] = %(cvt_out)s(T[i][1] + T[i][2]);\n        share[ns][cs + 16*(i*4 + 2)] = %(cvt_out)s(T[i][2] - T[i][1]);\n        share[ns][cs + 16*(i*4 + 3)] = %(cvt_out)s(T[i][3] - T[i][1]);\n    }\n    __syncthreads();\n\n    // now make c contiguous\n    cs = tid & 15;\n    ns = tid >> 4;\n\n    // apply the super block to just grid coordinates this time\n    gy = (gy << (shlY-1)) + ((ns & maskY) >> shrY);\n    gx = (gx << (shlX-1)) + ((ns & maskX) >> shrX);\n    n  = (blkN << shlN)   + (ns  & maskN);\n\n    if (n < N && gy < GY && gx < GX)\n    {\n        // output dim: (C32,GY,GX,N,c32)\n        // where c32 is the 512 element transform data for 32 values of c\n        // We group two blkC's to form each transform row\n\n        Out += (blkC >> 1)*GYGXN512 + gy*GXN512 + gx*N512 + n*512 + (blkC & 1)*16 + cs;\n\n        #pragma unroll\n        for (int i = 0; i < 16; i++)\n            Out[32*i] = share[ns][cs + 16*i];\n    }\n}\n"
    common = _common_round['nearest'].get(dtype, '') + _div64
    if dtype == 'f2':
        common += _common_fp16_to_fp32
    code = code % {'common': common, 'type': _ew_types[dtype]['type'], 'cvt_in': _ew_types[dtype]['cvt'], 'cvt_out': _ew_types[dtype]['cvt_out']}
    module = SourceModule(code)
    kernel = module.get_function('update_image_3x3_2x2')
    kernel.prepare('PPIIIIIIIIIIIIIIIIIIIIIIIIIIII')
    return kernel

@context_dependent_memoize
def _get_xprop_image_4x4_3x3_kernel(dtype):
    if False:
        for i in range(10):
            print('nop')
    code = '\n%(common)s\n\n__global__ void xprop_image_4x4_3x3(\n    %(type)s* Out, const %(type)s* In,\n    int Y, int X, int N, int pad_y, int pad_x,\n    int GXS, int GYS2, int GXS2, int magic_GXS2, int shift_GXS2,\n    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,\n    int YXN, int XN, int GYS_GXS_C_1152, int GXS_C_1152, int C_1152)\n{\n    int tid   = threadIdx.x;\n    int blkN  = gridDim.x - blockIdx.x - 1;\n    int blkYX = gridDim.y - blockIdx.y - 1;\n    int c     = gridDim.z - blockIdx.z - 1;\n\n    // unpack y,x from blockIdx.x\n    int gy2 = div64(blkYX, magic_GXS2, shift_GXS2);\n    int gx2 = blkYX - gy2*GXS2;\n\n    // Implement a square wave block id remapping\n    // (for all but last row (if odd number of rows))\n    int gy = gy2 << 1;\n    int gx = gx2;\n    if (gy2 != GYS2)\n    {\n        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);\n        gx  = gx2 >> 1;\n    }\n    // Scan backwards on odd rows\n    if (gy2 & 1)\n        gx = GXS - gx - 1;\n\n    // Super block YXN coordinates\n    int y0 = (gy << shlY) + (((tid & maskY) >> shrY) << 2) - pad_y;\n    int x0 = (gx << shlX) + (((tid & maskX) >> shrX) << 2) - pad_x;\n    int n  = (blkN << shlN) + (tid & maskN);\n\n    Out += blkN*GYS_GXS_C_1152 + gy*GXS_C_1152 + gx*C_1152 + c*1152 + tid;\n\n    bool valid = n < N;\n\n    bool xin[6], yin[6];\n    float I[6][6];\n\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        xin[i] = x0 + i >= 0 && x0 + i < X && valid;\n        yin[i] = y0 + i >= 0 && y0 + i < Y;\n    }\n    int offset = c*YXN + y0*XN + x0*N + n;\n\n    #pragma unroll\n    for (int y = 0; y < 6; y++)\n    {\n        if (y) offset += XN;\n\n        #pragma unroll\n        for (int x = 0; x < 6; x++)\n        {\n            %(type)s val = 0;\n            if (yin[y] && xin[x])\n                val = __ldg(In + x*N + offset);\n            I[y][x] = %(cvt_in)s(val);\n        }\n    }\n    float f1_1025 = 1.1025f;\n    float f2_74   = 2.7400f;\n    float f0_70   = 0.7000f;\n    float f0_49   = 0.4900f;\n    float T[6][6];\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        float t0 = __fmaf_rn(I[2][i], -2.25f, I[4][i]);\n        float t1 = __fmaf_rn(I[1][i], -2.25f, I[3][i]);\n        float t2 = __fmaf_rn(I[2][i], -f0_49, I[4][i]);\n        float t3 = __fmaf_rn(I[1][i], -f0_49, I[3][i]);\n        float t4 = __fmaf_rn(I[2][i], -f2_74, I[4][i]);\n        float t5 = __fmaf_rn(I[3][i], -f2_74, I[5][i]);\n\n        T[0][i] = __fmaf_rn(I[0][i], f1_1025, t4);\n        T[1][i] = __fmaf_rn(t1,  f0_70, t0);\n        T[2][i] = __fmaf_rn(t1, -f0_70, t0);\n        T[3][i] = __fmaf_rn(t3,  1.5f,  t2);\n        T[4][i] = __fmaf_rn(t3, -1.5f,  t2);\n        T[5][i] = __fmaf_rn(I[1][i], f1_1025, t5);\n    }\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        float t0 = __fmaf_rn(T[i][2], -2.25f, T[i][4]);\n        float t1 = __fmaf_rn(T[i][1], -2.25f, T[i][3]);\n        float t2 = __fmaf_rn(T[i][2], -f0_49, T[i][4]);\n        float t3 = __fmaf_rn(T[i][1], -f0_49, T[i][3]);\n        float t4 = __fmaf_rn(T[i][2], -f2_74, T[i][4]);\n        float t5 = __fmaf_rn(T[i][3], -f2_74, T[i][5]);\n\n        Out[32*(i*6 + 0)] = %(cvt_out)s(__fmaf_rn(T[i][0], f1_1025, t4));\n        Out[32*(i*6 + 1)] = %(cvt_out)s(__fmaf_rn(t1,  f0_70, t0));\n        Out[32*(i*6 + 2)] = %(cvt_out)s(__fmaf_rn(t1, -f0_70, t0));\n        Out[32*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(t3,  1.5f, t2));\n        Out[32*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(t3, -1.5f, t2));\n        Out[32*(i*6 + 5)] = %(cvt_out)s(__fmaf_rn(T[i][1], f1_1025, t5));\n    }\n}\n'
    common = _common_round['nearest'].get(dtype, '') + _div64
    if dtype == 'f2':
        common += _common_fp16_to_fp32
    code = code % {'common': common, 'type': _ew_types[dtype]['type'], 'type4': _ew_types[dtype]['type4'], 'cvt_in': _ew_types[dtype]['cvt'], 'cvt_out': _ew_types[dtype]['cvt_out']}
    module = SourceModule(code)
    kernel = module.get_function('xprop_image_4x4_3x3')
    kernel.prepare('PPIIIIIIIIIIIIIIIIIIIIIII')
    return kernel

@context_dependent_memoize
def _get_update_image_3x3_4x4_kernel(dtype):
    if False:
        for i in range(10):
            print('nop')
    code = "\n%(common)s\n\n__global__ void update_image_3x3_4x4(\n    %(type)s* O, const %(type)s* In,\n    int C, int Y, int X, int N, int pad_y, int pad_x,\n    int GY, int GX, int GXS, int GYS2,\n    int X2cn, int magic_X2cn, int shift_X2cn, int shift_c, int shift_n,\n    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,\n    int YXN, int XN, int GYGXN1152, int GXN1152, int N1152)\n{\n    // Add padding to avoid all but 1 shared bank conflict on loads\n    %(type)s __shared__ share[16][16*36 + 2];\n\n    int tid        = threadIdx.x;\n    int blk_YXcn   = blockIdx.x;\n    int blk_N      = blockIdx.y;\n    int blk_C      = blockIdx.z;\n\n    // unpack y,x,c,n from blockIdx.x\n    int gy2     = div64(blk_YXcn, magic_X2cn, shift_X2cn);\n    int blk_Xcn = blk_YXcn - gy2*X2cn;\n\n    int shift_cn = shift_c + shift_n;\n\n    int gx2    = blk_Xcn >> shift_cn;\n    int blk_cn = blk_Xcn - (gx2 << shift_cn);\n\n    int blk_c = blk_cn >> shift_n;\n    int blk_n = blk_cn - (blk_c << shift_n);\n\n    int blkN = (blk_N << shift_n) + blk_n;\n    int blkC = (blk_C << shift_c) + blk_c;\n\n    // Implement a square wave block id remapping\n    // (for all but last row (if odd number of rows))\n    int gy = gy2 << 1;\n    int gx = gx2;\n    if (gy2 != GYS2)\n    {\n        gy += (gx2 & 1) ^ ((gx2 & 2) >> 1);\n        gx  = gx2 >> 1;\n    }\n    // Scan backwards on odd rows\n    if (gy2 & 1)\n        gx = GXS - gx - 1;\n\n    int ns = tid & 15;\n    int cs = tid >> 4;\n\n    // Super block YXN coordinates\n    int y0 = (gy << shlY) + (((ns & maskY) >> shrY) << 2) - pad_y;\n    int x0 = (gx << shlX) + (((ns & maskX) >> shrX) << 2) - pad_x;\n    int n  = (blkN << shlN) + (ns & maskN);\n    int c  = (blkC << 4) + cs;\n\n    bool valid = c < C && n < N;\n\n    bool xin[6], yin[6];\n    float I[6][6], T[6][6];\n\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        xin[i] = x0 + i >= 0 && x0 + i < X && valid;\n        yin[i] = y0 + i >= 0 && y0 + i < Y;\n    }\n\n    int offset = c*YXN + y0*XN + x0*N + n;\n\n    #pragma unroll\n    for (int y = 0; y < 6; y++)\n    {\n        if (y) offset += XN;\n\n        #pragma unroll\n        for (int x = 0; x < 6; x++)\n        {\n            %(type)s val = 0;\n            if (yin[y] && xin[x])\n                val = __ldg(In + offset + x*N);\n            I[y][x] = %(cvt_in)s(val);\n        }\n    }\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        float t0 = __fmaf_rn(I[2][i], -2.250000f, I[4][i]);\n        float t1 = __fmaf_rn(I[1][i], -2.250000f, I[3][i]);\n        float t2 = __fmaf_rn(I[2][i], -0.390625f, I[4][i]);\n        float t3 = __fmaf_rn(I[1][i], -0.390625f, I[3][i]);\n        float t4 = __fmaf_rn(I[2][i], -2.640625f, I[4][i]);\n        float t5 = __fmaf_rn(I[3][i], -2.640625f, I[5][i]);\n\n        T[0][i] = __fmaf_rn(I[0][i], 0.87890625f, t4);\n        T[1][i] = __fmaf_rn(t1,  0.625f, t0);\n        T[2][i] = __fmaf_rn(t1, -0.625f, t0);\n        T[3][i] = __fmaf_rn(t3,  1.500f, t2);\n        T[4][i] = __fmaf_rn(t3, -1.500f, t2);\n        T[5][i] = __fmaf_rn(I[1][i], 0.87890625f, t5);\n    }\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        float t0 = __fmaf_rn(T[i][2], -2.250000f, T[i][4]);\n        float t1 = __fmaf_rn(T[i][1], -2.250000f, T[i][3]);\n        float t2 = __fmaf_rn(T[i][2], -0.390625f, T[i][4]);\n        float t3 = __fmaf_rn(T[i][1], -0.390625f, T[i][3]);\n        float t4 = __fmaf_rn(T[i][2], -2.640625f, T[i][4]);\n        float t5 = __fmaf_rn(T[i][3], -2.640625f, T[i][5]);\n\n        share[ns][cs + 16*(i*6 + 0)] = %(cvt_out)s(__fmaf_rn(T[i][0], 0.87890625f, t4));\n        share[ns][cs + 16*(i*6 + 1)] = %(cvt_out)s(__fmaf_rn(t1,  0.625f, t0));\n        share[ns][cs + 16*(i*6 + 2)] = %(cvt_out)s(__fmaf_rn(t1, -0.625f, t0));\n        share[ns][cs + 16*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(t3,  1.500f, t2));\n        share[ns][cs + 16*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(t3, -1.500f, t2));\n        share[ns][cs + 16*(i*6 + 5)] = %(cvt_out)s(__fmaf_rn(T[i][1], 0.87890625f, t5));\n    }\n    __syncthreads();\n\n    // now make c contiguous\n    cs = tid & 15;\n    ns = tid >> 4;\n\n    // apply the super block to just grid coordinates this time\n    gy = (gy << (shlY-2)) + ((ns & maskY) >> shrY);\n    gx = (gx << (shlX-2)) + ((ns & maskX) >> shrX);\n    n  = (blkN << shlN)   + (ns  & maskN);\n\n    if (n < N && gy < GY && gx < GX)\n    {\n        // output dim: (C32,GY,GX,N,c32)\n        // where c32 is the 1152 element transform data for 32 values of c\n        // We group two blkC's to form each transform row\n\n        O += (blkC >> 1)*GYGXN1152 + gy*GXN1152 + gx*N1152 + n*1152 + (blkC & 1)*16 + cs;\n\n        #pragma unroll\n        for (int i = 0; i < 36; i++)\n            O[32*i] = share[ns][cs + 16*i];\n    }\n}\n"
    common = _common_round['nearest'].get(dtype, '') + _div64
    if dtype == 'f2':
        common += _common_fp16_to_fp32
    code = code % {'common': common, 'type': _ew_types[dtype]['type'], 'cvt_in': _ew_types[dtype]['cvt'], 'cvt_out': _ew_types[dtype]['cvt_out']}
    module = SourceModule(code)
    kernel = module.get_function('update_image_3x3_4x4')
    kernel.prepare('PPIIIIIIIIIIIIIIIIIIIIIIIIIIII')
    return kernel

@context_dependent_memoize
def _get_update_delta_3x3_4x4_kernel(dtype):
    if False:
        for i in range(10):
            print('nop')
    code = "\n%(common)s\n\n__global__ void update_delta_3x3_4x4(\n    %(type)s* O, const %(type)s* In,\n    int K, int Y, int X, int N, int GY, int GX,\n    int Xkn, int magic_Xkn, int shift_Xkn, int shift_k, int shift_n,\n    int shlY, int shlX, int maskY, int shrY, int maskX, int shrX, int shlN, int maskN,\n    int YXN, int XN, int GYGXN1152, int GXN1152, int N1152)\n{\n    // Add padding to avoid all but 1 shared bank conflict on loads\n    %(type)s __shared__ share[16][16*36 + 2];\n\n    int tid        = threadIdx.x;\n    int blk_YXkn   = blockIdx.x;\n    int blk_N      = blockIdx.y;\n    int blk_K      = blockIdx.z;\n\n    // unpack y,x,k,n from blockIdx.x\n    int gy      = div64(blk_YXkn, magic_Xkn, shift_Xkn);\n    int blk_Xkn = blk_YXkn - gy*Xkn;\n\n    int shift_kn = shift_k + shift_n;\n\n    int gx     = blk_Xkn >> shift_kn;\n    int blk_kn = blk_Xkn - (gx << shift_kn);\n\n    int blk_k = blk_kn >> shift_n;\n    int blk_n = blk_kn - (blk_k << shift_n);\n\n    int blkN = (blk_N << shift_n) + blk_n;\n    int blkK = (blk_K << shift_k) + blk_k;\n\n    int ns = tid & 15;\n    int ks = tid >> 4;\n\n    // Super block YXN coordinates\n    int y0 = (gy << shlY) + (((ns & maskY) >> shrY) << 2);\n    int x0 = (gx << shlX) + (((ns & maskX) >> shrX) << 2);\n    int n  = (blkN << shlN) + (ns & maskN);\n    int k  = (blkK << 4) + ks;\n\n    bool valid = k < K && n < N;\n\n    bool xin[4], yin[4];\n    float I[4][4];\n\n    #pragma unroll\n    for (int i = 0; i < 4; i++)\n    {\n        xin[i] = x0 + i >= 0 && x0 + i < X && valid;\n        yin[i] = y0 + i >= 0 && y0 + i < Y;\n    }\n\n    const %(type)s* In0 = In + k*YXN + y0*XN + x0*N + n;\n\n    #pragma unroll\n    for (int y = 0; y < 4; y++)\n    {\n        if (y)\n            In0 += XN;\n        const %(type)s* In1 = In0 + N;\n        const %(type)s* In2 = In1 + N;\n        const %(type)s* In3 = In2 + N;\n\n        I[y][0] = yin[y] && xin[0] ? %(cvt_in)s(__ldg(In0)) : 0.0f;\n        I[y][1] = yin[y] && xin[1] ? %(cvt_in)s(__ldg(In1)) : 0.0f;\n        I[y][2] = yin[y] && xin[2] ? %(cvt_in)s(__ldg(In2)) : 0.0f;\n        I[y][3] = yin[y] && xin[3] ? %(cvt_in)s(__ldg(In3)) : 0.0f;\n    }\n\n    float T[6][4];\n    float coeff0  =  0.26890756302521f;\n    float coeff1  = -0.688403361344538f;\n    float coeff2  =  0.119514472455649f;\n    float coeff3  =  0.430252100840336f;\n    float coeff4  =  0.179271708683473f;\n    float coeff5  =  0.168067226890756f;\n    float coeff6  =  0.403361344537815f;\n    float coeff7  =  1.13777777777778f;\n\n    #pragma unroll\n    for (int i = 0; i < 4; i++)\n    {\n        float t0 = I[2][i] * coeff0;\n        float t1 = __fmaf_rn(I[0][i], coeff1, -t0);\n        float t2 = __fmaf_rn(I[0][i], coeff2,  t0);\n        float t3 = __fmaf_rn(I[1][i], coeff3,  I[3][i] * coeff5);\n        float t4 = __fmaf_rn(I[1][i], coeff4,  I[3][i] * coeff6);\n\n        T[0][i] = I[0][i]*coeff7;\n        T[1][i] = t1 - t3;\n        T[2][i] = t1 + t3;\n        T[3][i] = t2 + t4;\n        T[4][i] = t2 - t4;\n        T[5][i] = I[3][i];\n    }\n\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        float t0 = T[i][2] * coeff0;\n        float t1 = __fmaf_rn(T[i][0], coeff1, -t0);\n        float t2 = __fmaf_rn(T[i][0], coeff2,  t0);\n        float t3 = __fmaf_rn(T[i][1], coeff3,  T[i][3] * coeff5);\n        float t4 = __fmaf_rn(T[i][1], coeff4,  T[i][3] * coeff6);\n\n        share[ns][ks + 16*(i*6 + 0)] = %(cvt_out)s(T[i][0]*coeff7);\n        share[ns][ks + 16*(i*6 + 1)] = %(cvt_out)s(t1 - t3);\n        share[ns][ks + 16*(i*6 + 2)] = %(cvt_out)s(t1 + t3);\n        share[ns][ks + 16*(i*6 + 3)] = %(cvt_out)s(t2 + t4);\n        share[ns][ks + 16*(i*6 + 4)] = %(cvt_out)s(t2 - t4);\n        share[ns][ks + 16*(i*6 + 5)] = %(cvt_out)s(T[i][3]);\n    }\n    __syncthreads();\n\n    // now make k contiguous\n    ks = tid & 15;\n    ns = tid >> 4;\n\n    // apply the super block to just grid coordinates this time\n    gy = (gy << (shlY-2)) + ((ns & maskY) >> shrY);\n    gx = (gx << (shlX-2)) + ((ns & maskX) >> shrX);\n    n  = (blkN << shlN)   + (ns  & maskN);\n\n    if (n < N && gy < GY && gx < GX)\n    {\n        // output dim: (K32,GY,GX,N,k32)\n        // where k32 is the 1152 element transform data for 32 values of k\n        // We group two blkK's to form each transform row\n\n        O += (blkK >> 1)*GYGXN1152 + gy*GXN1152 + gx*N1152 + n*1152 + (blkK & 1)*16 + ks;\n\n        #pragma unroll\n        for (int i = 0; i < 36; i++)\n            O[32*i] = share[ns][ks + 16*i];\n    }\n}\n"
    common = _common_round['nearest'].get(dtype, '') + _div64
    if dtype == 'f2':
        common += _common_fp16_to_fp32
    code = code % {'common': common, 'type': _ew_types[dtype]['type'], 'cvt_in': _ew_types[dtype]['cvt'], 'cvt_out': _ew_types[dtype]['cvt_out']}
    module = SourceModule(code)
    kernel = module.get_function('update_delta_3x3_4x4')
    kernel.prepare('PPIIIIIIIIIIIIIIIIIIIIIIII')
    return kernel

class FilterTransform(object):

    def __init__(self, lib, dtype, prop, Rt, St, R, S, mode):
        if False:
            return 10
        self.lib = lib
        self.size = int(np.prod(self.dim)) * dtype.itemsize
        self.kargs = [dtype.str[1:], None, prop, Rt, St, R, S, mode]

    def bind_params(self, F):
        if False:
            i = 10
            return i + 15
        self.data = self.lib.scratch_buffer_offset(self.size)
        self.args[2:5] = (self.lib.stream, self.data, F.gpudata)
        self.kargs[1] = F.dtype.str[1:]
        self.kernel = _get_filter_kernel(*self.kargs)
        return self.data

    def execute(self):
        if False:
            print('Hello World!')
        self.kernel.prepared_async_call(*self.args)

    def unbind(self):
        if False:
            while True:
                i = 10
        self.kernel = None
        self.args[2:5] = (None,) * 3

class FpropFilterTransform(FilterTransform):

    def __init__(self, lib, dtype, C, K, Rt, St, R, S, mode):
        if False:
            return 10
        GK32 = _ceil_div(K, 32)
        self.dim = (GK32, C, R + Rt - 1, S + St - 1, 32)
        self.args = [(GK32, C, 1), (32, 1, 1), None, None, None, R * S * K, S * K, K, C]
        super(FpropFilterTransform, self).__init__(lib, dtype, 'fprop', Rt, St, R, S, mode)

class BpropFilterTransform(FilterTransform):

    def __init__(self, lib, dtype, C, K, Rt, St, R, S, mode):
        if False:
            i = 10
            return i + 15
        GC32 = _ceil_div(C, 32)
        GK32 = _ceil_div(K, 32)
        GC16 = _ceil_div(GC32 * 32, 16)
        GK16 = _ceil_div(K, 16)
        self.dim = (GC32, K, R + Rt - 1, S + St - 1, 32)
        self.args = [(GK16, GC16, 1), (256, 1, 1), None, None, None, R * S * K, S * K, K, C]
        super(BpropFilterTransform, self).__init__(lib, dtype, 'bprop', Rt, St, R, S, mode)

class FpropFilter_2x2_3x3(FpropFilterTransform):

    def __init__(self, lib, dtype, C, K):
        if False:
            return 10
        super(FpropFilter_2x2_3x3, self).__init__(lib, dtype, C, K, 2, 2, 3, 3, 'cyclic')

class BpropFilter_2x2_3x3(BpropFilterTransform):

    def __init__(self, lib, dtype, C, K):
        if False:
            while True:
                i = 10
        super(BpropFilter_2x2_3x3, self).__init__(lib, dtype, C, K, 2, 2, 3, 3, 'cyclic')

class FpropFilter_4x4_3x3(FpropFilterTransform):

    def __init__(self, lib, dtype, C, K):
        if False:
            while True:
                i = 10
        super(FpropFilter_4x4_3x3, self).__init__(lib, dtype, C, K, 4, 4, 3, 3, 'cyclic')

class BpropFilter_4x4_3x3(BpropFilterTransform):

    def __init__(self, lib, dtype, C, K):
        if False:
            i = 10
            return i + 15
        super(BpropFilter_4x4_3x3, self).__init__(lib, dtype, C, K, 4, 4, 3, 3, 'cyclic')

class FpropFilter_2x2_5x5(FpropFilterTransform):

    def __init__(self, lib, dtype, C, K):
        if False:
            print('Hello World!')
        super(FpropFilter_2x2_5x5, self).__init__(lib, dtype, C, K, 2, 2, 5, 5, 'linear')

class BpropFilter_2x2_5x5(BpropFilterTransform):

    def __init__(self, lib, dtype, C, K):
        if False:
            return 10
        super(BpropFilter_2x2_5x5, self).__init__(lib, dtype, C, K, 2, 2, 5, 5, 'linear')

@context_dependent_memoize
def _get_filter_kernel(type_out, type_in, prop, Rt, St, R, S, conv_mode):
    if False:
        i = 10
        return i + 15
    transforms = {'2x2_3x3': '\n\n    float T[4][3];\n    #pragma unroll\n    for (int i = 0; i < 3; i++)\n    {\n        float t0 = (I[0][i] + I[2][i]) * 0.5f;\n        T[0][i] = I[0][i];\n        T[1][i] = __fmaf_rn(I[1][i],  0.5f, t0);\n        T[2][i] = __fmaf_rn(I[1][i], -0.5f, t0);\n        T[3][i] = I[2][i];\n    }\n    #pragma unroll\n    for (int i = 0; i < 4; i++)\n    {\n        float t0 = (T[i][0] + T[i][2]) * 0.5f;\n        %(output)s*(i*4 + 0)] = %(cvt_out)s(T[i][0]);\n        %(output)s*(i*4 + 1)] = %(cvt_out)s(__fmaf_rn(T[i][1],  0.5f, t0));\n        %(output)s*(i*4 + 2)] = %(cvt_out)s(__fmaf_rn(T[i][1], -0.5f, t0));\n        %(output)s*(i*4 + 3)] = %(cvt_out)s(T[i][2]);\n    }\n', '4x4_3x3': '\n\n    float f25_88    =  25.0f /   88.0f;\n    float f25_132   =  25.0f /  132.0f;\n    float f25_198   =  25.0f /  198.0f;\n    float f125_308  = 125.0f /  308.0f;\n    float f400_441  = 400.0f /  441.0f;\n    float f625_1078 = 625.0f / 1078.0f;\n    float T[6][3];\n    #pragma unroll\n    for (int i = 0; i < 3; i++)\n    {\n        float t0 = f25_88 * I[2][i];\n        float t1 = __fmaf_rn(I[0][i], -f625_1078, -t0);\n        float t2 = __fmaf_rn(I[0][i],  f25_198,    t0);\n        T[0][i] = f400_441 * I[0][i];\n        T[1][i] = __fmaf_rn(I[1][i], -f125_308, t1);\n        T[2][i] = __fmaf_rn(I[1][i],  f125_308, t1);\n        T[3][i] = __fmaf_rn(I[1][i],  f25_132,  t2);\n        T[4][i] = __fmaf_rn(I[1][i], -f25_132,  t2);\n        T[5][i] = I[2][i];\n    }\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        float t0 = f25_88 *  T[i][2];\n        float t1 = __fmaf_rn(T[i][0], -f625_1078, -t0);\n        float t2 = __fmaf_rn(T[i][0],  f25_198,    t0);\n        %(output)s*(i*6 + 0)] = %(cvt_out)s(f400_441 * T[i][0]);\n        %(output)s*(i*6 + 1)] = %(cvt_out)s(__fmaf_rn(T[i][1], -f125_308, t1));\n        %(output)s*(i*6 + 2)] = %(cvt_out)s(__fmaf_rn(T[i][1],  f125_308, t1));\n        %(output)s*(i*6 + 3)] = %(cvt_out)s(__fmaf_rn(T[i][1],  f25_132,  t2));\n        %(output)s*(i*6 + 4)] = %(cvt_out)s(__fmaf_rn(T[i][1], -f25_132,  t2));\n        %(output)s*(i*6 + 5)] = %(cvt_out)s(T[i][2]);\n    }\n', '2x2_5x5': '\n\n    float f64_81   =   64.0f /  81.0f;\n    float f128_243 = -128.0f / 243.0f;\n    float f32_243  =   32.0f / 243.0f;\n    float f32_81   =   32.0f /  81.0f;\n    float f16_81   =   16.0f /  81.0f;\n    float f8_27    =    8.0f /  27.0f;\n    float f2_9     =    2.0f /   9.0f;\n    float f4_9     =    4.0f /   9.0f;\n    float f1_6     =   -1.0f /   6.0f;\n    float f2_3     =    2.0f /   3.0f;\n\n    float T[6][5];\n    #pragma unroll\n    for (int i = 0; i < 5; i++)\n    {\n        float t0 = I[2][i] * f8_27;\n        float t1 = __fmaf_rn(I[1][i], f32_81, I[3][i] * f2_9);\n        float t2 = __fmaf_rn(I[1][i], f16_81, I[3][i] * f4_9);\n        float t3 = __fmaf_rn(I[0][i], f128_243, __fmaf_rn(I[4][i], f1_6, -t0));\n        float t4 = __fmaf_rn(I[0][i], f32_243,  __fmaf_rn(I[4][i], f2_3,  t0));\n        T[0][i]  = I[0][i] * f64_81;\n        T[1][i]  = t3 - t1;\n        T[2][i]  = t3 + t1;\n        T[3][i]  = t4 + t2;\n        T[4][i]  = t4 - t2;\n        T[5][i]  = I[4][i];\n    }\n    #pragma unroll\n    for (int i = 0; i < 6; i++)\n    {\n        float t0 = T[i][2] * f8_27;\n        float t1 = __fmaf_rn(T[i][1], f32_81,   T[i][3] * f2_9);\n        float t2 = __fmaf_rn(T[i][1], f16_81,   T[i][3] * f4_9);\n        float t3 = __fmaf_rn(T[i][0], f128_243, __fmaf_rn(T[i][4], f1_6, -t0));\n        float t4 = __fmaf_rn(T[i][0], f32_243,  __fmaf_rn(T[i][4], f2_3,  t0));\n        %(output)s*(i*6 + 0)] = %(cvt_out)s(T[i][0] * f64_81);\n        %(output)s*(i*6 + 1)] = %(cvt_out)s(t3 - t1);\n        %(output)s*(i*6 + 2)] = %(cvt_out)s(t3 + t1);\n        %(output)s*(i*6 + 3)] = %(cvt_out)s(t4 + t2);\n        %(output)s*(i*6 + 4)] = %(cvt_out)s(t4 - t2);\n        %(output)s*(i*6 + 5)] = %(cvt_out)s(T[i][4]);\n    }\n'}
    outputs = {'fprop': 'Out[32', 'bprop': 'share[ks][cs + 16'}
    kernels = {'fprop': '\n%(common)s\n\n__global__ void %(prop)s_filter_%(trans_name)s(\n    %(type_out)s* Out, const %(type_in)s* In, int RSK, int SK, int K, int C)\n{\n    int tid  = threadIdx.x;\n    int blkK = gridDim.x - blockIdx.x - 1;\n    int c    = gridDim.y - blockIdx.y - 1;\n    int k    = (blkK<<5) + tid;\n\n    bool valid_k = k < K;\n\n    int offset = c*RSK + k;\n\n    Out += blkK*C*32*%(RSt)s + c*32*%(RSt)s + tid;\n\n    float I[%(R)s][%(S)s];\n    #pragma unroll\n    for (int r = 0; r < %(R)s; r++)\n    {\n        if (r > 0) offset += SK;\n\n        #pragma unroll\n        for (int s = 0; s < %(S)s; s++)\n        {\n            %(type_in)s val = 0;\n            if (valid_k)\n                val = __ldg(In + K*s + offset);\n            %(conv_mode)s = %(cvt_in)s(val);\n        }\n    }\n    %(trans_code)s\n}\n', 'bprop': '\n%(common)s\n\n__global__ void %(prop)s_filter_%(trans_name)s(\n    %(type_out)s* Out, const %(type_in)s* In, int RSK, int SK, int K, int C)\n{\n    // Add padding to avoid all but 1 shared bank conflict on loads\n    %(type_out)s __shared__ share[16][16*%(RSt)s + 2];\n\n    int tid  = threadIdx.x;\n    int blkK = gridDim.x - blockIdx.x - 1;\n    int blkC = gridDim.y - blockIdx.y - 1;\n\n    int cs = tid >> 4;\n    int ks = tid & 15;\n\n    int c = blkC * 16 + cs;\n    int k = blkK * 16 + ks;\n\n    bool valid_ck = c < C && k < K;\n\n    int offset = c*RSK + k;\n\n    float I[%(R)s][%(S)s];\n    #pragma unroll\n    for (int r = 0; r < %(R)s; r++)\n    {\n        if (r > 0) offset += SK;\n\n        #pragma unroll\n        for (int s = 0; s < %(S)s; s++)\n        {\n            %(type_in)s val = 0;\n            if (valid_ck)\n                val = __ldg(In + K*s + offset);\n            %(conv_mode)s = %(cvt_in)s(val);\n        }\n    }\n    %(trans_code)s\n\n    __syncthreads();\n\n    // now make c contiguous\n    cs = tid & 15;\n    ks = tid >> 4;\n\n    k = blkK*16 + ks;\n\n    if (k < K)\n    {\n        Out += (blkC>>1)*K*32*%(RSt)s + k*32*%(RSt)s + (blkC&1)*16 + cs;\n\n        #pragma unroll\n        for (int i = 0; i < %(RSt)s; i++)\n            Out[32*i] = share[ks][cs + 16*i];\n    }\n}\n'}
    filter_map = {'cyclic': {'fprop': 'I[r][s]', 'bprop': 'I[%d-r-1][%d-s-1]' % (R, S)}, 'linear': {'bprop': 'I[r][s]', 'fprop': 'I[%d-r-1][%d-s-1]' % (R, S)}}
    common = _common_round['nearest'].get(type_out, '')
    if type_in == 'f2':
        common += _common_fp16_to_fp32
    trans_name = '%sx%s_%sx%s' % (Rt, St, R, S)
    trans_code = transforms[trans_name] % {'output': outputs[prop], 'cvt_out': _ew_types[type_out]['cvt_out']}
    code = kernels[prop] % {'common': common, 'type_in': _ew_types[type_in]['type'], 'type_out': _ew_types[type_out]['type'], 'cvt_in': _ew_types[type_in]['cvt'], 'conv_mode': filter_map[conv_mode][prop], 'trans_name': trans_name, 'trans_code': trans_code, 'prop': prop, 'RSt': (R + Rt - 1) * (S + St - 1), 'R': R, 'S': S}
    module = SourceModule(code)
    kernel = module.get_function(prop + '_filter_' + trans_name)
    kernel.prepare('PPIIII')
    return kernel