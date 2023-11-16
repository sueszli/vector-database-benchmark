from .common import Benchmark, get_squares_, get_indexes_rand, TYPES1
import numpy as np

class Eindot(Benchmark):

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.a = np.arange(60000.0).reshape(150, 400)
        self.ac = self.a.copy()
        self.at = self.a.T
        self.atc = self.a.T.copy()
        self.b = np.arange(240000.0).reshape(400, 600)
        self.c = np.arange(600)
        self.d = np.arange(400)
        self.a3 = np.arange(480000.0).reshape(60, 80, 100)
        self.b3 = np.arange(192000.0).reshape(80, 60, 40)

    def time_dot_a_b(self):
        if False:
            while True:
                i = 10
        np.dot(self.a, self.b)

    def time_dot_d_dot_b_c(self):
        if False:
            return 10
        np.dot(self.d, np.dot(self.b, self.c))

    def time_dot_trans_a_at(self):
        if False:
            while True:
                i = 10
        np.dot(self.a, self.at)

    def time_dot_trans_a_atc(self):
        if False:
            for i in range(10):
                print('nop')
        np.dot(self.a, self.atc)

    def time_dot_trans_at_a(self):
        if False:
            i = 10
            return i + 15
        np.dot(self.at, self.a)

    def time_dot_trans_atc_a(self):
        if False:
            while True:
                i = 10
        np.dot(self.atc, self.a)

    def time_einsum_i_ij_j(self):
        if False:
            i = 10
            return i + 15
        np.einsum('i,ij,j', self.d, self.b, self.c)

    def time_einsum_ij_jk_a_b(self):
        if False:
            for i in range(10):
                print('nop')
        np.einsum('ij,jk', self.a, self.b)

    def time_einsum_ijk_jil_kl(self):
        if False:
            return 10
        np.einsum('ijk,jil->kl', self.a3, self.b3)

    def time_inner_trans_a_a(self):
        if False:
            print('Hello World!')
        np.inner(self.a, self.a)

    def time_inner_trans_a_ac(self):
        if False:
            i = 10
            return i + 15
        np.inner(self.a, self.ac)

    def time_matmul_a_b(self):
        if False:
            i = 10
            return i + 15
        np.matmul(self.a, self.b)

    def time_matmul_d_matmul_b_c(self):
        if False:
            print('Hello World!')
        np.matmul(self.d, np.matmul(self.b, self.c))

    def time_matmul_trans_a_at(self):
        if False:
            print('Hello World!')
        np.matmul(self.a, self.at)

    def time_matmul_trans_a_atc(self):
        if False:
            while True:
                i = 10
        np.matmul(self.a, self.atc)

    def time_matmul_trans_at_a(self):
        if False:
            i = 10
            return i + 15
        np.matmul(self.at, self.a)

    def time_matmul_trans_atc_a(self):
        if False:
            print('Hello World!')
        np.matmul(self.atc, self.a)

    def time_tensordot_a_b_axes_1_0_0_1(self):
        if False:
            print('Hello World!')
        np.tensordot(self.a3, self.b3, axes=([1, 0], [0, 1]))

class Linalg(Benchmark):
    params = set(TYPES1) - set(['float16'])
    param_names = ['dtype']

    def setup(self, typename):
        if False:
            for i in range(10):
                print('nop')
        np.seterr(all='ignore')
        self.a = get_squares_()[typename]

    def time_svd(self, typename):
        if False:
            i = 10
            return i + 15
        np.linalg.svd(self.a)

    def time_pinv(self, typename):
        if False:
            print('Hello World!')
        np.linalg.pinv(self.a)

    def time_det(self, typename):
        if False:
            print('Hello World!')
        np.linalg.det(self.a)

class LinalgNorm(Benchmark):
    params = TYPES1
    param_names = ['dtype']

    def setup(self, typename):
        if False:
            print('Hello World!')
        self.a = get_squares_()[typename]

    def time_norm(self, typename):
        if False:
            for i in range(10):
                print('nop')
        np.linalg.norm(self.a)

class LinalgSmallArrays(Benchmark):
    """ Test overhead of linalg methods for small arrays """

    def setup(self):
        if False:
            print('Hello World!')
        self.array_5 = np.arange(5.0)
        self.array_5_5 = np.reshape(np.arange(25.0), (5, 5))

    def time_norm_small_array(self):
        if False:
            i = 10
            return i + 15
        np.linalg.norm(self.array_5)

    def time_det_small_array(self):
        if False:
            for i in range(10):
                print('nop')
        np.linalg.det(self.array_5_5)

class Lstsq(Benchmark):

    def setup(self):
        if False:
            i = 10
            return i + 15
        self.a = get_squares_()['float64']
        self.b = get_indexes_rand()[:100].astype(np.float64)

    def time_numpy_linalg_lstsq_a__b_float64(self):
        if False:
            print('Hello World!')
        np.linalg.lstsq(self.a, self.b, rcond=-1)

class Einsum(Benchmark):
    param_names = ['dtype']
    params = [[np.float32, np.float64]]

    def setup(self, dtype):
        if False:
            print('Hello World!')
        self.one_dim_small = np.arange(600, dtype=dtype)
        self.one_dim = np.arange(3000, dtype=dtype)
        self.one_dim_big = np.arange(480000, dtype=dtype)
        self.two_dim_small = np.arange(1200, dtype=dtype).reshape(30, 40)
        self.two_dim = np.arange(240000, dtype=dtype).reshape(400, 600)
        self.three_dim_small = np.arange(10000, dtype=dtype).reshape(10, 100, 10)
        self.three_dim = np.arange(24000, dtype=dtype).reshape(20, 30, 40)
        self.non_contiguous_dim1_small = np.arange(1, 80, 2, dtype=dtype)
        self.non_contiguous_dim1 = np.arange(1, 4000, 2, dtype=dtype)
        self.non_contiguous_dim2 = np.arange(1, 2400, 2, dtype=dtype).reshape(30, 40)
        self.non_contiguous_dim3 = np.arange(1, 48000, 2, dtype=dtype).reshape(20, 30, 40)

    def time_einsum_outer(self, dtype):
        if False:
            i = 10
            return i + 15
        np.einsum('i,j', self.one_dim, self.one_dim, optimize=True)

    def time_einsum_multiply(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.einsum('..., ...', self.two_dim_small, self.three_dim, optimize=True)

    def time_einsum_sum_mul(self, dtype):
        if False:
            return 10
        np.einsum(',i...->', 300, self.three_dim_small, optimize=True)

    def time_einsum_sum_mul2(self, dtype):
        if False:
            i = 10
            return i + 15
        np.einsum('i...,->', self.three_dim_small, 300, optimize=True)

    def time_einsum_mul(self, dtype):
        if False:
            i = 10
            return i + 15
        np.einsum('i,->i', self.one_dim_big, 300, optimize=True)

    def time_einsum_contig_contig(self, dtype):
        if False:
            print('Hello World!')
        np.einsum('ji,i->', self.two_dim, self.one_dim_small, optimize=True)

    def time_einsum_contig_outstride0(self, dtype):
        if False:
            return 10
        np.einsum('i->', self.one_dim_big, optimize=True)

    def time_einsum_noncon_outer(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.einsum('i,j', self.non_contiguous_dim1, self.non_contiguous_dim1, optimize=True)

    def time_einsum_noncon_multiply(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.einsum('..., ...', self.non_contiguous_dim2, self.non_contiguous_dim3, optimize=True)

    def time_einsum_noncon_sum_mul(self, dtype):
        if False:
            return 10
        np.einsum(',i...->', 300, self.non_contiguous_dim3, optimize=True)

    def time_einsum_noncon_sum_mul2(self, dtype):
        if False:
            print('Hello World!')
        np.einsum('i...,->', self.non_contiguous_dim3, 300, optimize=True)

    def time_einsum_noncon_mul(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.einsum('i,->i', self.non_contiguous_dim1, 300, optimize=True)

    def time_einsum_noncon_contig_contig(self, dtype):
        if False:
            i = 10
            return i + 15
        np.einsum('ji,i->', self.non_contiguous_dim2, self.non_contiguous_dim1_small, optimize=True)

    def time_einsum_noncon_contig_outstride0(self, dtype):
        if False:
            for i in range(10):
                print('nop')
        np.einsum('i->', self.non_contiguous_dim1, optimize=True)

class LinAlgTransposeVdot(Benchmark):
    params = [[(16, 16), (32, 32), (64, 64)], TYPES1]
    param_names = ['shape', 'npdtypes']

    def setup(self, shape, npdtypes):
        if False:
            while True:
                i = 10
        self.xarg = np.random.uniform(-1, 1, np.dot(*shape)).reshape(shape)
        self.xarg = self.xarg.astype(npdtypes)
        self.x2arg = np.random.uniform(-1, 1, np.dot(*shape)).reshape(shape)
        self.x2arg = self.x2arg.astype(npdtypes)
        if npdtypes.startswith('complex'):
            self.xarg += self.xarg.T * 1j
            self.x2arg += self.x2arg.T * 1j

    def time_transpose(self, shape, npdtypes):
        if False:
            print('Hello World!')
        np.transpose(self.xarg)

    def time_vdot(self, shape, npdtypes):
        if False:
            return 10
        np.vdot(self.xarg, self.x2arg)