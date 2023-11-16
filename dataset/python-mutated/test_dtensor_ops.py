import unittest
import warnings
import torch
import torch.distributed as dist
import torch.testing._internal.common_methods_invocations as common_ops
from torch.distributed._tensor import DeviceMesh, DTensor
from torch.overrides import resolve_name
from torch.testing._internal.common_device_type import instantiate_device_type_tests, ops
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_utils import run_tests, suppress_warnings, TEST_WITH_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorConverter, DTensorOpTestBase
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
common_ops.L = 24
common_ops.M = 12
common_ops.S = 4
common_ops.XS = 2

def xfail(op_name, variant_name='', *, device_type=None, dtypes=None):
    if False:
        for i in range(10):
            print('nop')
    return (op_name, variant_name, device_type, dtypes, True)

def skip(op_name, variant_name='', *, device_type=None, dtypes=None):
    if False:
        while True:
            i = 10
    return (op_name, variant_name, device_type, dtypes, False)

def skipOps(test_case_name, base_test_name, to_skip):
    if False:
        for i in range(10):
            print('nop')
    all_opinfos = op_db
    for xfail in to_skip:
        (op_name, variant_name, device_type, dtypes, expected_failure) = xfail
        matching_opinfos = [o for o in all_opinfos if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for opinfo in matching_opinfos:
            decorators = list(opinfo.decorators)
            if expected_failure:
                decorator = DecorateInfo(unittest.expectedFailure, test_case_name, base_test_name, device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(unittest.skip('Skipped!'), test_case_name, base_test_name, device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            opinfo.decorators = tuple(decorators)

    def wrapped(fn):
        if False:
            while True:
                i = 10
        return fn
    return wrapped
dtensor_fails = {xfail('__getitem__'), xfail('__rsub__'), xfail('_native_batch_norm_legit'), xfail('_softmax_backward_data'), xfail('_upsample_bilinear2d_aa'), xfail('addbmm'), xfail('addmv'), xfail('addr'), xfail('all'), xfail('allclose'), xfail('amax'), xfail('amin'), xfail('aminmax'), xfail('any'), xfail('arange'), xfail('argmax'), xfail('argmin'), xfail('argsort'), xfail('as_strided'), xfail('as_strided', 'partial_views'), xfail('as_strided_scatter'), xfail('bernoulli'), xfail('block_diag'), xfail('broadcast_shapes'), xfail('cauchy'), xfail('cartesian_prod'), xfail('cdist'), xfail('cholesky'), xfail('cholesky_inverse'), xfail('cholesky_solve'), xfail('chunk'), xfail('clamp'), xfail('clamp_max'), xfail('clamp_min'), xfail('combinations'), xfail('complex'), xfail('constant_pad_nd'), xfail('corrcoef'), xfail('count_nonzero'), xfail('cov'), xfail('cross'), xfail('cummax'), xfail('cummin'), xfail('cumsum'), xfail('cumulative_trapezoid'), xfail('diag'), xfail('diag_embed'), xfail('diagflat'), xfail('diagonal'), xfail('diagonal_copy'), xfail('diagonal_scatter'), xfail('dist'), xfail('dot'), xfail('einsum'), xfail('empty'), xfail('empty_like'), xfail('empty_permuted'), xfail('exponential'), xfail('eye'), xfail('fft.fft2'), xfail('fft.fft'), xfail('fft.fftn'), xfail('fft.fftshift'), xfail('fft.ifft2'), xfail('fft.ifft'), xfail('fft.ifftshift'), xfail('fft.ihfft2'), xfail('fft.ihfft'), xfail('fft.ihfftn'), xfail('fft.irfft2'), xfail('fft.irfftn'), xfail('fft.rfft2'), xfail('fft.rfft'), xfail('fft.rfftn'), xfail('fill'), xfail('flip'), xfail('fliplr'), xfail('flipud'), xfail('floor_divide'), xfail('fmax'), xfail('fmin'), xfail('frexp'), xfail('full'), xfail('full_like'), xfail('gather'), xfail('geometric'), xfail('geqrf'), xfail('grid_sampler_2d'), xfail('gradient'), xfail('heaviside'), xfail('histc'), xfail('histogram'), xfail('histogramdd'), xfail('index_add'), xfail('index_copy'), xfail('index_fill'), xfail('index_put'), xfail('index_reduce'), xfail('index_select'), xfail('isin'), xfail('isinf'), xfail('isneginf'), xfail('isposinf'), xfail('kthvalue'), xfail('linalg.cholesky'), xfail('linalg.cholesky_ex'), xfail('linalg.cond'), xfail('linalg.cross'), xfail('linalg.det'), xfail('linalg.det', 'singular'), xfail('linalg.diagonal'), xfail('linalg.eig'), xfail('linalg.eigh'), xfail('linalg.eigvals'), xfail('linalg.eigvalsh'), xfail('linalg.householder_product'), xfail('linalg.inv'), xfail('linalg.inv_ex'), xfail('linalg.ldl_factor'), xfail('linalg.ldl_factor_ex'), xfail('linalg.ldl_solve'), xfail('linalg.lstsq'), xfail('linalg.lstsq', 'grad_oriented'), xfail('linalg.lu'), xfail('linalg.lu_factor'), xfail('linalg.lu_factor_ex'), xfail('linalg.lu_solve'), xfail('linalg.matrix_norm'), xfail('linalg.matrix_power'), xfail('linalg.matrix_rank'), xfail('linalg.matrix_rank', 'hermitian'), xfail('linalg.multi_dot'), xfail('linalg.norm'), xfail('linalg.norm', 'subgradients_at_zero'), xfail('linalg.pinv'), xfail('linalg.pinv', 'hermitian'), xfail('linalg.qr'), xfail('linalg.slogdet'), xfail('linalg.solve'), xfail('linalg.solve_ex'), xfail('linalg.solve_triangular'), xfail('linalg.svd'), xfail('linalg.svdvals'), xfail('linalg.tensorinv'), xfail('linalg.tensorsolve'), xfail('linalg.vander'), xfail('linalg.vecdot'), xfail('linalg.vector_norm'), xfail('linspace'), xfail('log_normal'), xfail('logcumsumexp'), xfail('logdet'), xfail('logspace'), xfail('logsumexp'), xfail('lu'), xfail('lu_solve'), xfail('lu_unpack'), xfail('masked_fill'), xfail('masked_scatter'), xfail('masked_select'), xfail('masked.amax'), xfail('masked.amin'), xfail('masked.argmax'), xfail('masked.argmin'), xfail('masked.cumprod'), xfail('masked.cumsum'), xfail('masked.logsumexp'), xfail('masked.median'), xfail('masked.norm'), xfail('matrix_exp'), xfail('max', 'binary'), xfail('max', 'reduction_with_dim'), xfail('maximum'), xfail('median'), xfail('min', 'binary'), xfail('min', 'reduction_with_dim'), xfail('minimum'), xfail('mode'), xfail('msort'), xfail('multinomial'), xfail('mv'), xfail('max_pool2d_with_indices_backward', ''), xfail('nanmean'), xfail('nanmedian'), xfail('nanquantile'), xfail('nansum'), xfail('native_batch_norm'), xfail('narrow_copy'), xfail('ne'), xfail('new_empty'), xfail('new_empty_strided'), xfail('transpose'), xfail('nn.functional.adaptive_avg_pool1d'), xfail('nn.functional.adaptive_avg_pool2d'), xfail('nn.functional.adaptive_avg_pool3d'), xfail('nn.functional.adaptive_max_pool1d'), xfail('nn.functional.adaptive_max_pool2d'), xfail('nn.functional.adaptive_max_pool3d'), xfail('nn.functional.alpha_dropout'), xfail('nn.functional.avg_pool1d'), xfail('nn.functional.avg_pool2d'), xfail('nn.functional.avg_pool3d'), xfail('nn.functional.batch_norm'), xfail('nn.functional.batch_norm', 'without_cudnn'), xfail('nn.functional.bilinear'), xfail('nn.functional.binary_cross_entropy'), xfail('nn.functional.binary_cross_entropy_with_logits'), xfail('nn.functional.celu'), xfail('nn.functional.conv1d'), xfail('nn.functional.conv2d'), xfail('nn.functional.conv_transpose1d'), xfail('nn.functional.conv_transpose2d'), xfail('nn.functional.conv_transpose3d'), xfail('nn.functional.cosine_similarity'), xfail('nn.functional.cross_entropy'), xfail('nn.functional.ctc_loss'), xfail('nn.functional.dropout'), xfail('nn.functional.dropout2d'), xfail('nn.functional.dropout3d'), xfail('nn.functional.elu'), xfail('nn.functional.fractional_max_pool2d'), xfail('nn.functional.fractional_max_pool3d'), xfail('nn.functional.gaussian_nll_loss'), xfail('nn.functional.glu'), xfail('nn.functional.grid_sample'), xfail('nn.functional.group_norm'), xfail('nn.functional.hardshrink'), xfail('nn.functional.hardsigmoid'), xfail('nn.functional.hardswish'), xfail('nn.functional.hardtanh'), xfail('nn.functional.huber_loss'), xfail('nn.functional.instance_norm'), xfail('nn.functional.interpolate', 'area'), xfail('nn.functional.interpolate', 'bicubic'), xfail('nn.functional.interpolate', 'bilinear'), xfail('nn.functional.interpolate', 'linear'), xfail('nn.functional.interpolate', 'nearest'), xfail('nn.functional.interpolate', 'trilinear'), xfail('nn.functional.leaky_relu'), xfail('nn.functional.linear'), xfail('nn.functional.local_response_norm'), xfail('nn.functional.logsigmoid'), xfail('nn.functional.margin_ranking_loss'), xfail('nn.functional.max_pool1d'), xfail('nn.functional.max_pool2d'), xfail('nn.functional.max_pool3d'), xfail('nn.functional.max_unpool1d'), xfail('nn.functional.max_unpool1d', 'grad'), xfail('nn.functional.max_unpool2d'), xfail('nn.functional.max_unpool2d', 'grad'), xfail('nn.functional.max_unpool3d'), xfail('nn.functional.max_unpool3d', 'grad'), xfail('nn.functional.mish'), xfail('nn.functional.mse_loss'), xfail('nn.functional.multi_margin_loss'), xfail('nn.functional.multilabel_margin_loss'), xfail('nn.functional.multilabel_soft_margin_loss'), xfail('nn.functional.nll_loss'), xfail('nn.functional.normalize'), xfail('nn.functional.pad', 'constant'), xfail('nn.functional.pad', 'reflect'), xfail('nn.functional.pad', 'replicate'), xfail('nn.functional.pairwise_distance'), xfail('nn.functional.pdist'), xfail('nn.functional.pixel_shuffle'), xfail('nn.functional.pixel_unshuffle'), xfail('nn.functional.poisson_nll_loss'), xfail('nn.functional.prelu'), xfail('nn.functional.relu6'), xfail('nn.functional.rrelu'), xfail('nn.functional.selu'), xfail('nn.functional.silu'), xfail('nn.functional.smooth_l1_loss'), xfail('nn.functional.soft_margin_loss'), xfail('nn.functional.softplus'), xfail('nn.functional.softshrink'), xfail('nn.functional.threshold'), xfail('nn.functional.triplet_margin_loss'), xfail('nn.functional.triplet_margin_with_distance_loss'), xfail('nn.functional.unfold'), xfail('nn.functional.upsample_bilinear'), xfail('nn.functional.upsample_nearest'), xfail('nonzero'), xfail('norm'), xfail('norm', 'fro'), xfail('norm', 'inf'), xfail('norm', 'nuc'), xfail('normal'), xfail('normal', 'number_mean'), xfail('normal', 'in_place'), xfail('ormqr'), xfail('ones'), xfail('pca_lowrank'), xfail('pinverse'), xfail('polar'), xfail('put'), xfail('qr'), xfail('quantile'), xfail('rand_like'), xfail('randint_like'), xfail('randint'), xfail('randn'), xfail('randn_like'), xfail('renorm'), xfail('repeat_interleave'), xfail('resize_'), xfail('resize_as_'), xfail('roll'), xfail('rot90'), xfail('rsub'), xfail('scalar_tensor'), xfail('scatter_add'), xfail('scatter'), xfail('scatter_reduce', 'amax'), xfail('scatter_reduce', 'amin'), xfail('scatter_reduce', 'mean'), xfail('scatter_reduce', 'prod'), xfail('scatter_reduce', 'sum'), xfail('searchsorted'), xfail('select'), xfail('select_scatter'), xfail('sort'), xfail('sparse.sampled_addmm'), xfail('sparse.mm', 'reduce'), xfail('special.airy_ai'), xfail('special.bessel_j0'), xfail('special.bessel_j1'), xfail('special.bessel_y0'), xfail('special.bessel_y1'), xfail('special.chebyshev_polynomial_t'), xfail('special.chebyshev_polynomial_u'), xfail('special.entr'), xfail('special.erfcx'), xfail('special.hermite_polynomial_h'), xfail('special.hermite_polynomial_he'), xfail('special.i0e'), xfail('special.i1'), xfail('special.i1e'), xfail('special.laguerre_polynomial_l'), xfail('special.log_ndtr'), xfail('special.modified_bessel_i0'), xfail('special.modified_bessel_i1'), xfail('special.modified_bessel_k0'), xfail('special.modified_bessel_k1'), xfail('special.ndtri'), xfail('special.scaled_modified_bessel_k0'), xfail('special.scaled_modified_bessel_k1'), xfail('special.spherical_bessel_j0'), xfail('special.xlog1py'), xfail('special.zeta'), xfail('squeeze', 'multiple'), xfail('signal.windows.bartlett'), xfail('signal.windows.blackman'), xfail('signal.windows.cosine'), xfail('signal.windows.exponential'), xfail('signal.windows.gaussian'), xfail('signal.windows.general_cosine'), xfail('signal.windows.general_hamming'), xfail('signal.windows.hamming'), xfail('signal.windows.hann'), xfail('signal.windows.nuttall'), xfail('signal.windows.kaiser'), xfail('stack'), xfail('std'), xfail('std', 'unbiased'), xfail('std_mean'), xfail('std_mean', 'unbiased'), xfail('stft'), xfail('svd'), xfail('svd_lowrank'), xfail('t'), xfail('take_along_dim'), xfail('take'), xfail('tensor_split'), xfail('to_sparse'), xfail('topk'), xfail('trace'), xfail('trapezoid'), xfail('trapz'), xfail('triangular_solve'), xfail('tril'), xfail('triu'), xfail('unbind'), xfail('unfold'), xfail('unfold_copy'), xfail('uniform'), xfail('unflatten'), xfail('unique_consecutive'), xfail('unique'), xfail('unsafe_split'), xfail('var_mean'), xfail('var_mean', 'unbiased'), xfail('vdot'), xfail('view_copy'), xfail('view_as_complex'), xfail('zeros'), skip('argwhere'), skip('cumprod'), skip('__rmatmul__'), skip('meshgrid', 'list_of_tensors'), skip('meshgrid', 'variadic_tensors'), skip('nn.functional.scaled_dot_product_attention'), skip('nn.functional.softmin'), skip('nn.functional.embedding'), skip('nn.functional.embedding_bag'), skip('nn.functional.feature_alpha_dropout', 'with_train'), skip('nn.functional.feature_alpha_dropout', 'without_train'), skip('nn.functional.hinge_embedding_loss'), skip('nn.functional.cosine_embedding_loss'), skip('fft.hfft'), skip('fft.hfft2'), skip('fft.hfft2'), skip('fft.hfftn'), skip('fft.ifftn'), skip('fft.irfft'), skip('istft'), skip('isclose'), skip('isreal'), skip('matmul'), skip('masked.mean'), skip('masked.var'), skip('masked.std'), skip('masked.normalize'), skip('prod'), skip('_segment_reduce', 'lengths'), skip('_segment_reduce', 'offsets'), skip('squeeze')}
skip_bw = [None, 'torch.bucketize', 'torch.conj_physical', 'torch.eq', 'torch.isfinite', 'torch.isnan', 'torch.native_layer_norm', 'torch.nn.functional.layer_norm']
OP_DB_WORLD_SIZE = 4
DEVICE_TYPE = 'cpu'

class TestDTensorOps(DTensorOpTestBase):

    @property
    def world_size(self) -> int:
        if False:
            while True:
                i = 10
        return OP_DB_WORLD_SIZE

    @unittest.skipIf(TEST_WITH_ASAN, 'Skipped under ASAN')
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float,))
    @skipOps('TestDTensorOps', 'test_dtensor_op_db', dtensor_fails)
    def test_dtensor_op_db(self, dtype, op):
        if False:
            for i in range(10):
                print('nop')
        self.mesh = DeviceMesh(DEVICE_TYPE, torch.arange(self.world_size))

        def test():
            if False:
                print('Hello World!')
            samples = op.sample_inputs(DEVICE_TYPE, dtype, requires_grad=True)
            for sample_input in samples:
                args = [sample_input.input] + list(sample_input.args)
                kwargs = sample_input.kwargs
                self.run_dtensor_crossref(op.op, args, kwargs)
        self.check_dtensor_func(test, op)

    def assert_ref_dtensor_equal(self, dtensor_rs, rs):
        if False:
            while True:
                i = 10
        flat_dtensor_rs = pytree.tree_leaves(dtensor_rs)
        flat_rs = pytree.tree_leaves(rs)
        self.assertEqual(len(flat_dtensor_rs), len(flat_rs))
        for (dtensor_r, r) in zip(flat_dtensor_rs, flat_rs):
            if not isinstance(r, torch.Tensor):
                continue
            self.assertIsInstance(dtensor_r, torch.Tensor)
            self.assertEqualOnRank(dtensor_r.shape, r.shape, f'Shape mismatch! original shape:{r.shape}, dtensor shape: {dtensor_r.shape}')
            self.assertEqualOnRank(dtensor_r.requires_grad, r.requires_grad, f'op result requires_grad mismatch!original requires_grad: {r.requires_grad}, dtensor requires_grad: {dtensor_r.requires_grad}')
            self.assertEqualOnRank(dtensor_r, r)

    def run_dtensor_crossref(self, func, args, kwargs):
        if False:
            return 10
        to_dtensor = DTensorConverter(self.mesh, args, kwargs)

        def concat_res_if_necessary(func, res: object) -> object:
            if False:
                return 10
            if resolve_name(func) is not None and 'split' in resolve_name(func):
                dim = args[2] if len(args) == 3 else 0
                return torch.cat(res, dim=dim)
            else:
                return res
        rs = func(*args, **kwargs)
        rs = concat_res_if_necessary(func, rs)

        def to_replicate(e: object) -> object:
            if False:
                for i in range(10):
                    print('nop')
            return e.full_tensor() if isinstance(e, DTensor) else e
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for (dtensor_args, dtensor_kwargs) in to_dtensor:
                    if to_dtensor.successful():
                        dtensor_rs = func(*dtensor_args, **dtensor_kwargs)
                        flat_args = pytree.tree_leaves(dtensor_rs)
                        if any((isinstance(e, torch.Tensor) and e.numel() == 0 for e in flat_args)):
                            continue
                        dtensor_rs = tree_map(to_replicate, dtensor_rs)
                        dtensor_rs = concat_res_if_necessary(func, dtensor_rs)
                        try:
                            if resolve_name(func) not in skip_bw:
                                if isinstance(dtensor_rs, DTensor):
                                    dtensor_rs.to_local().sum().backward()
                                elif isinstance(dtensor_rs, tuple):
                                    dtensor_rs[0].to_local().sum().backward()
                        except Exception as e:
                            if torch.distributed.get_rank() == 0:
                                print(f'failed to run BW: {resolve_name(func)}, {func}, {str(e)})')
                        self.assert_ref_dtensor_equal(dtensor_rs, rs)
                    else:
                        raise RuntimeError(f'failed to convert args to DTensor; originally (*{args}, **{kwargs})')
        except Exception as e:
            raise RuntimeError(f'failed to run: {resolve_name(func)}, with (*{args}, **{kwargs})') from e
        return rs

    def check_dtensor_func(self, test_func, opinfo, dry_run=False):
        if False:
            for i in range(10):
                print('nop')
        try:
            test_func()
        except Exception:
            if not dry_run:
                raise
            if dist.get_rank() == 0:
                if opinfo.variant_test_name:
                    print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
                else:
                    print(f"xfail('{opinfo.name}'),")
instantiate_device_type_tests(TestDTensorOps, globals(), only_for=(DEVICE_TYPE,))
if __name__ == '__main__':
    if torch.cuda.is_available():
        run_tests()