import caffe2.python.hypothesis_test_util as hu
from caffe2.python import core, workspace
from hypothesis import given
from caffe2.quantization.server import dnnlowp_pybind11

class TestInt8QuantSchemeBlobFillOperator(hu.HypothesisTestCase):

    @given(**hu.gcs_cpu_only)
    def test_int8_quant_scheme_blob_fill_op(self, gc, dc):
        if False:
            for i in range(10):
                print('nop')
        gen_quant_scheme_net = core.Net('gen_quant_scheme')
        gen_quant_scheme_op = core.CreateOperator('Int8QuantSchemeBlobFill', [], ['quant_scheme'], quantization_kind='MIN_MAX_QUANTIZATION', preserve_sparsity=False, device_option=gc)
        gen_quant_scheme_net.Proto().op.extend([gen_quant_scheme_op])
        assert workspace.RunNetOnce(gen_quant_scheme_net), 'Failed to run the gen_quant_scheme net'
        (quantization_kind, preserve_sparsity) = dnnlowp_pybind11.ObserveInt8QuantSchemeBlob('quant_scheme')
        assert quantization_kind == 'MIN_MAX_QUANTIZATION'
        assert not preserve_sparsity