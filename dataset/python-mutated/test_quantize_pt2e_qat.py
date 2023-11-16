import copy
import operator
import unittest
from typing import Any, Optional, Tuple, Type
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization import default_fake_quant, FusedMovingAvgObsFakeQuantize, MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver, QConfigMapping
from torch.ao.quantization.backend_config import get_qnnpack_backend_config
from torch.ao.quantization.qconfig import default_per_channel_symmetric_qnnpack_qat_qconfig, default_symmetric_qnnpack_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.ao.quantization.quantize_pt2e import _convert_to_reference_decomposed_fx, convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer import DerivedQuantizationSpec, QuantizationAnnotation, QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.xnnpack_quantizer import get_symmetric_quantization_config, XNNPACKQuantizer
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_quantization import NodeSpec as ns, QuantizationTestCase, skip_if_no_torchvision, skipIfNoQNNPACK
from torch.testing._internal.common_quantized import override_quantized_engine

class PT2EQATTestCase(QuantizationTestCase):
    """
    Base QuantizationTestCase for PT2E QAT with some helper methods.
    """

    class _BaseConvBnModel(torch.nn.Module):

        def __init__(self, conv_class: Type[torch.nn.Module], bn_class: Type[torch.nn.Module], has_conv_bias: bool, has_bn: bool, has_relu: bool, **conv_kwargs):
            if False:
                print('Hello World!')
            super().__init__()
            self.conv = conv_class(3, 3, 3, bias=has_conv_bias, **conv_kwargs)
            self.bn = bn_class(3) if has_bn else None
            self.relu = torch.nn.ReLU() if has_relu else None

        def forward(self, x):
            if False:
                while True:
                    i = 10
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x

    def _get_conv_bn_model(self, has_conv_bias: bool=True, has_bn: bool=True, has_relu: bool=False, **conv_kwargs):
        if False:
            print('Hello World!')
        '\n        Return an instance of a simple test model containing the\n        conv[-bn][-relu] pattern. By default, this returns a\n        conv-bn model with conv bias.\n        '
        return self._BaseConvBnModel(self.conv_class, self.bn_class, has_conv_bias, has_bn, has_relu, **conv_kwargs)

    def _verify_symmetric_xnnpack_qat_numerics(self, model: torch.nn.Module, example_inputs: Tuple[Any, ...]):
        if False:
            for i in range(10):
                print('nop')
        self._verify_symmetric_xnnpack_qat_numerics_helper(model, example_inputs, is_per_channel=True)
        self._verify_symmetric_xnnpack_qat_numerics_helper(model, example_inputs, is_per_channel=False)

    def _verify_symmetric_xnnpack_qat_numerics_helper(self, model: torch.nn.Module, example_inputs: Tuple[Any, ...], is_per_channel: bool, verify_convert: bool=True):
        if False:
            print('Hello World!')
        '\n        Helper method to verify that the QAT numerics for PT2E quantization match those of\n        FX graph mode quantization for symmetric qnnpack.\n        '
        torch._dynamo.reset()
        MANUAL_SEED = 100
        model_pt2e = copy.deepcopy(model)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=is_per_channel, is_qat=True))
        model_pt2e = capture_pre_autograd_graph(model_pt2e, example_inputs)
        model_pt2e = prepare_qat_pt2e(model_pt2e, quantizer)
        torch.manual_seed(MANUAL_SEED)
        after_prepare_result_pt2e = model_pt2e(*example_inputs)
        model_fx = copy.deepcopy(model)
        if is_per_channel:
            default_qconfig = default_per_channel_symmetric_qnnpack_qat_qconfig
        else:
            default_qconfig = default_symmetric_qnnpack_qat_qconfig
        qconfig_mapping = QConfigMapping().set_global(default_qconfig)
        backend_config = get_qnnpack_backend_config()
        model_fx = prepare_qat_fx(model_fx, qconfig_mapping, example_inputs, backend_config=backend_config)
        torch.manual_seed(MANUAL_SEED)
        after_prepare_result_fx = model_fx(*example_inputs)
        self.assertEqual(after_prepare_result_pt2e, after_prepare_result_fx)
        if verify_convert:
            torch.ao.quantization.move_exported_model_to_eval(model_pt2e)
            model_pt2e = convert_pt2e(model_pt2e)
            quant_result_pt2e = model_pt2e(*example_inputs)
            model_fx.eval()
            model_fx = _convert_to_reference_decomposed_fx(model_fx, backend_config=backend_config)
            quant_result_fx = model_fx(*example_inputs)
            self.assertEqual(quant_result_pt2e, quant_result_fx)

    def _verify_symmetric_xnnpack_qat_graph(self, m: torch.fx.GraphModule, example_inputs: Tuple[Any, ...], has_relu: bool, has_bias: bool=True, is_cuda: bool=False, expected_conv_literal_args: Optional[Tuple[Any, ...]]=None):
        if False:
            for i in range(10):
                print('nop')
        self._verify_symmetric_xnnpack_qat_graph_helper(m, example_inputs, is_per_channel=True, has_relu=has_relu, has_bias=has_bias, is_cuda=is_cuda, expected_conv_literal_args=expected_conv_literal_args)
        self._verify_symmetric_xnnpack_qat_graph_helper(m, example_inputs, is_per_channel=False, has_relu=has_relu, has_bias=has_bias, is_cuda=is_cuda, expected_conv_literal_args=expected_conv_literal_args)

    def _verify_symmetric_xnnpack_qat_graph_helper(self, m: torch.fx.GraphModule, example_inputs: Tuple[Any, ...], is_per_channel: bool, has_relu: bool, has_bias: bool=True, is_cuda: bool=False, expected_conv_literal_args: Optional[Tuple[Any, ...]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that the graph module matches the fused QAT [conv - bn (- relu)] pattern\n        with fake quantizes inserted into the correct places.\n        # TODO: also verify that metadata is copied over to the new nodes.\n        '
        m = copy.deepcopy(m)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel, is_qat=True))
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_qat_pt2e(m, quantizer)
        m(*example_inputs)
        output_node = list(m.graph.nodes)[-1]
        output_fq_node = output_node.args[0][0]
        self.assertTrue(output_fq_node.target.startswith('activation_post_process_'))
        output_fq_mod = getattr(m, output_fq_node.target)
        self.assertEqual(type(output_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(type(output_fq_mod.activation_post_process), MovingAverageMinMaxObserver)
        self.assertEqual(output_fq_mod.dtype, torch.int8)
        self.assertEqual(output_fq_mod.quant_min, -128)
        self.assertEqual(output_fq_mod.quant_max, 127)
        if has_relu:
            relu_node = output_fq_node.args[0]
            getitem_node = relu_node.args[0]
            self.assertEqual(relu_node.target, torch.ops.aten.relu.default)
        else:
            relu_node = None
            getitem_node = output_fq_node.args[0]
        bn_node = getitem_node.args[0]
        if is_cuda:
            if torch.version.cuda is not None:
                expected_bn_op = torch.ops.aten.cudnn_batch_norm.default
            elif torch.version.hip is not None:
                expected_bn_op = torch.ops.aten.miopen_batch_norm.default
        else:
            expected_bn_op = torch.ops.aten._native_batch_norm_legit.default
        self.assertEqual(getitem_node.target, operator.getitem)
        self.assertEqual(bn_node.target, expected_bn_op)
        if has_bias:
            add_bias_node = bn_node.args[0]
            (div_scale_factor_node, bias_reshape_node) = add_bias_node.args
            self.assertEqual(add_bias_node.target, torch.ops.aten.add.Tensor)
            self.assertEqual(bias_reshape_node.target, torch.ops.aten.reshape.default)
        else:
            div_scale_factor_node = bn_node.args[0]
        (conv_node, scale_factor_reshape_node) = div_scale_factor_node.args
        self.assertEqual(div_scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertEqual(conv_node.target, torch.ops.aten.conv2d.default)
        self.assertEqual(scale_factor_reshape_node.target, torch.ops.aten.reshape.default)
        if expected_conv_literal_args is not None:
            assert len(expected_conv_literal_args) == 6, 'wrong num conv args, bad test setup'
            for i in range(6):
                if i + 3 < len(conv_node.args):
                    self.assertEqual(conv_node.args[i + 3], expected_conv_literal_args[i])
        conv_input_fq_node = conv_node.args[0]
        conv_input_node = conv_input_fq_node.args[0]
        self.assertTrue(conv_input_fq_node.target.startswith('activation_post_process_'))
        conv_input_fq_mod = getattr(m, conv_input_fq_node.target)
        self.assertEqual(type(conv_input_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(type(conv_input_fq_mod.activation_post_process), MovingAverageMinMaxObserver)
        self.assertEqual(conv_input_fq_mod.dtype, torch.int8)
        self.assertEqual(conv_input_fq_mod.quant_min, -128)
        self.assertEqual(conv_input_fq_mod.quant_max, 127)
        self.assertTrue(conv_input_node.op, 'placeholder')
        conv_weight_fq_node = conv_node.args[1]
        self.assertTrue(conv_weight_fq_node.target.startswith('activation_post_process_'))
        conv_weight_fq_mod = getattr(m, conv_weight_fq_node.target)
        if is_per_channel:
            expected_weight_observer_type = MovingAveragePerChannelMinMaxObserver
        else:
            expected_weight_observer_type = MovingAverageMinMaxObserver
        self.assertEqual(type(conv_weight_fq_mod), FusedMovingAvgObsFakeQuantize)
        self.assertEqual(type(conv_weight_fq_mod.activation_post_process), expected_weight_observer_type)
        self.assertEqual(conv_weight_fq_mod.dtype, torch.int8)
        self.assertEqual(conv_weight_fq_mod.quant_min, -127)
        self.assertEqual(conv_weight_fq_mod.quant_max, 127)
        zero_bias_node = conv_node.args[2] if len(conv_node.args) > 2 else None
        mul_weight_scale_factor_node = conv_weight_fq_node.args[0]
        (conv_weight_fq_node, scale_factor_reshape_node) = mul_weight_scale_factor_node.args
        if has_bias:
            self.assertEqual(zero_bias_node.target, torch.ops.aten.zeros_like.default)
        else:
            self.assertTrue(zero_bias_node is None)
        self.assertEqual(mul_weight_scale_factor_node.target, torch.ops.aten.mul.Tensor)
        self.assertEqual(scale_factor_reshape_node.target, torch.ops.aten.reshape.default)
        scale_factor_node = scale_factor_reshape_node.args[0]
        (bn_weight_node, sqrt_node) = scale_factor_node.args
        bn_running_var_add_node = sqrt_node.args[0]
        (bn_running_var_node, eps) = bn_running_var_add_node.args
        self.assertEqual(scale_factor_node.target, torch.ops.aten.div.Tensor)
        self.assertTrue('param_constant' in bn_weight_node.target)
        self.assertEqual(sqrt_node.target, torch.ops.aten.sqrt.default)
        self.assertEqual(bn_running_var_add_node.target, torch.ops.aten.add.Tensor)
        self.assertTrue('tensor_constant' in bn_running_var_node.target)
        self.assertEqual(eps, 1e-05)

class BaseTestQuantizePT2EQAT_ConvBn(PT2EQATTestCase):
    """
    Base TestCase to be used for all conv-bn[-relu] fusion patterns.
    """

    def test_qat_conv_no_bias(self):
        if False:
            for i in range(10):
                print('nop')
        m1 = self._get_conv_bn_model(has_conv_bias=False, has_bn=False, has_relu=True)
        m2 = self._get_conv_bn_model(has_conv_bias=False, has_bn=False, has_relu=False)
        self._verify_symmetric_xnnpack_qat_numerics(m1, self.example_inputs)
        self._verify_symmetric_xnnpack_qat_numerics(m2, self.example_inputs)

    def test_qat_conv_bn_fusion(self):
        if False:
            print('Hello World!')
        m = self._get_conv_bn_model()
        self._verify_symmetric_xnnpack_qat_graph(m, self.example_inputs, has_relu=False)
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_qat_conv_bn_fusion_cuda(self):
        if False:
            while True:
                i = 10
        m = self._get_conv_bn_model().cuda()
        example_inputs = (self.example_inputs[0].cuda(),)
        self._verify_symmetric_xnnpack_qat_graph(m, example_inputs, has_relu=False, is_cuda=True)
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    def test_qat_conv_bn_fusion_literal_args(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3, stride=(2, 2), padding=(4, 4))
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                x = self.bn(x)
                return x
        example_inputs = (torch.randn(1, 3, 5, 5),)
        conv_args = ((2, 2), (4, 4), (1, 1), False, (0, 0), 1)
        self._verify_symmetric_xnnpack_qat_graph(M(), example_inputs, has_relu=False, expected_conv_literal_args=conv_args)
        self._verify_symmetric_xnnpack_qat_numerics(M(), example_inputs)

    def test_qat_conv_bn_fusion_no_conv_bias(self):
        if False:
            for i in range(10):
                print('nop')

        class M2(torch.nn.Module):
            """
            Mixed conv + BN with and without conv bias.
            """

            def __init__(self, conv_class, bn_class):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = conv_class(3, 3, 3, bias=False)
                self.bn1 = bn_class(3)
                self.conv2 = conv_class(3, 3, 3, bias=True)
                self.bn2 = bn_class(3)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.conv2(x)
                x = self.bn2(x)
                return x
        m1 = self._get_conv_bn_model(has_conv_bias=False)
        m2 = M2(self.conv_class, self.bn_class)
        example_inputs = (torch.randn(3, 3, 5, 5),)
        self._verify_symmetric_xnnpack_qat_graph(m1, example_inputs, has_relu=False, has_bias=False)
        self._verify_symmetric_xnnpack_qat_numerics(m1, example_inputs)
        self._verify_symmetric_xnnpack_qat_numerics(m2, example_inputs)

    def test_qat_conv_bn_relu_fusion(self):
        if False:
            while True:
                i = 10
        m = self._get_conv_bn_model(has_relu=True)
        self._verify_symmetric_xnnpack_qat_graph(m, self.example_inputs, has_relu=True)
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_qat_conv_bn_relu_fusion_cuda(self):
        if False:
            return 10
        m = self._get_conv_bn_model(has_relu=True).cuda()
        example_inputs = (self.example_inputs[0].cuda(),)
        self._verify_symmetric_xnnpack_qat_graph(m, example_inputs, has_relu=True, is_cuda=True)
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    def test_qat_conv_bn_relu_fusion_no_conv_bias(self):
        if False:
            return 10
        m = self._get_conv_bn_model(has_conv_bias=False, has_relu=True)
        self._verify_symmetric_xnnpack_qat_graph(m, self.example_inputs, has_relu=True, has_bias=False)
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)

    def test_qat_inplace_add_relu(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self, conv_class):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = conv_class(1, 1, 1)
                self.relu = torch.nn.ReLU(inplace=True)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x0 = x
                x = self.conv(x)
                x += x0
                x = self.relu(x)
                return x
        m = M(self.conv_class)
        example_inputs = (torch.randn(1, 1, 3, 3),)
        self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    def test_prepare_qat_conv_bn_fusion_getitem_placeholder(self):
        if False:
            while True:
                i = 10
        '\n        Test the case where the placeholder node for the [conv - bn - getitem] pattern\n        is also a getitem node:\n\n          some_op -> unrelated_getitem -> conv -> bn -> conv_bn_getitem\n\n        We want the metadata to be copied from the `conv_bn_getitem` node, not from\n        the `unrelated_getitem` node, which is not part of the conv-bn pattern but\n        is returned as part of the match anyway (as a placeholder).\n        '

        class M(torch.nn.Module):

            def __init__(self, conv_class, bn_class):
                if False:
                    return 10
                super().__init__()
                self.bn1 = bn_class(3)
                self.conv = conv_class(3, 3, 3)
                self.bn2 = bn_class(3)

            def forward(self, x):
                if False:
                    return 10
                x = self.bn1(x)
                x = self.conv(x)
                x = self.bn2(x)
                return x

        def _get_getitem_nodes(m: torch.fx.GraphModule):
            if False:
                print('Hello World!')
            '\n            Return a 2-tuple of (unrelated_getitem_node, conv_bn_getitem_node) from the graph.\n            '
            (unrelated_getitem_node, conv_bn_getitem_node) = (None, None)
            for node in m.graph.nodes:
                if node.target != operator.getitem or node.args[0].target != torch.ops.aten._native_batch_norm_legit.default:
                    continue
                if node.args[0].args[0].op == 'placeholder':
                    unrelated_getitem_node = node
                else:
                    conv_bn_getitem_node = node
            assert unrelated_getitem_node is not None, 'did not find unrelated getitem node, bad test setup'
            assert conv_bn_getitem_node is not None, 'did not find conv bn getitem node, bad test setup'
            return (unrelated_getitem_node, conv_bn_getitem_node)
        m = M(self.conv_class, self.bn_class)
        m = capture_pre_autograd_graph(m, self.example_inputs)
        m.graph.eliminate_dead_code()
        m.recompile()
        (_, original_conv_bn_getitem_node) = _get_getitem_nodes(m)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=False, is_qat=True))
        m = prepare_qat_pt2e(m, quantizer)
        (unrelated_getitem_node, conv_bn_getitem_node) = _get_getitem_nodes(m)
        original_conv_bn_getitem_meta = original_conv_bn_getitem_node.meta['quantization_annotation']
        conv_bn_getitem_meta = conv_bn_getitem_node.meta['quantization_annotation']
        self.assertEqual(conv_bn_getitem_meta, original_conv_bn_getitem_meta)
        self.assertTrue('quantization_annotation' not in unrelated_getitem_node.meta)

    def test_qat_update_shared_qspec(self):
        if False:
            while True:
                i = 10
        '\n        Test the case where nodes used in SharedQuantizationSpec were replaced\n        during QAT subgraph rewriting.\n        '

        class M(torch.nn.Module):

            def __init__(self, conv_class, bn_class):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = conv_class(3, 3, 3)
                self.bn = bn_class(3)
                self.hardtanh = torch.nn.Hardtanh()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(x)
                x = self.bn(x)
                x = self.hardtanh(x)
                return x
        m = M(self.conv_class, self.bn_class)
        self._verify_symmetric_xnnpack_qat_numerics(m, self.example_inputs)

    def test_qat_preserve_source_fn_stack(self):
        if False:
            i = 10
            return i + 15
        '\n        Test whether `source_fn_stack` is preserved after QAT fusion.\n        '

        class M(torch.nn.Module):

            def __init__(self, conv_class, bn_class, backbone):
                if False:
                    return 10
                super().__init__()
                self.conv = conv_class(5, 3, 3)
                self.bn = bn_class(3)
                self.relu = torch.nn.ReLU()
                self.backbone = backbone

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.backbone(x)
                return x
        backbone = self._get_conv_bn_model(has_relu=True)
        m = M(self.conv_class, self.bn_class, backbone)
        example_inputs = (torch.randn(1, 5, 10, 10),)
        quantizer = XNNPACKQuantizer()
        quantizer.set_global(get_symmetric_quantization_config(is_qat=True))
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_qat_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        (first_conv, first_relu, second_conv, second_relu) = (None, None, None, None)
        for n in m.graph.nodes:
            if n.target == torch.ops.aten.relu.default:
                if first_relu is None:
                    assert first_conv is None, 'bad test setup'
                    first_relu = n
                    first_conv = n.args[0]
                else:
                    assert second_conv is None, 'bad test setup'
                    second_relu = n
                    second_conv = n.args[0]

        def get_conv_weight_and_bias(conv_node: torch.fx.Node):
            if False:
                for i in range(10):
                    print('nop')
            weight_dq_node = conv_node.args[1]
            weight_q_node = weight_dq_node.args[0]
            weight_node = weight_q_node.args[0]
            bias_node = conv_node.args[2]
            assert isinstance(weight_node, torch.fx.Node)
            assert isinstance(bias_node, torch.fx.Node)
            return (weight_node, bias_node)
        (first_conv_weight, first_conv_bias) = get_conv_weight_and_bias(first_conv)
        (second_conv_weight, second_conv_bias) = get_conv_weight_and_bias(second_conv)

        def get_source_fn(node: torch.fx.Node):
            if False:
                print('Hello World!')
            return node.meta['source_fn_stack'][0][0]
        self.assertEqual(get_source_fn(first_conv), get_source_fn(first_conv_weight))
        self.assertEqual(get_source_fn(first_conv), get_source_fn(first_conv_bias))
        self.assertEqual(get_source_fn(second_conv), get_source_fn(second_conv_weight))
        self.assertEqual(get_source_fn(second_conv), get_source_fn(second_conv_bias))
        self.assertNotEqual(get_source_fn(first_conv), get_source_fn(first_relu))
        self.assertNotEqual(get_source_fn(first_conv), get_source_fn(second_conv))
        self.assertNotEqual(get_source_fn(second_conv), get_source_fn(second_relu))
        self.assertNotEqual(get_source_fn(first_relu), get_source_fn(second_relu))
        self.assertTrue('backbone' not in get_source_fn(first_conv))
        self.assertTrue('backbone' not in get_source_fn(first_relu))
        self.assertTrue('backbone' in get_source_fn(second_conv))
        self.assertTrue('backbone' in get_source_fn(second_relu))

    def test_qat_conv_bn_bias_derived_qspec(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._get_conv_bn_model()
        example_inputs = self.example_inputs
        m = capture_pre_autograd_graph(m, example_inputs)
        quantizer = ConvBnDerivedBiasQuantizer()
        m = prepare_qat_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        m(*example_inputs)
        (conv_node, _, _) = _get_conv_bn_getitem_nodes(m)
        weight_dq = conv_node.args[1]
        bias_dq = conv_node.args[2]
        self.assertEqual(weight_dq.target, torch.ops.quantized_decomposed.dequantize_per_tensor.default)
        self.assertEqual(bias_dq.target, torch.ops.quantized_decomposed.dequantize_per_tensor.default)
        weight_q = weight_dq.args[0]
        bias_q = bias_dq.args[0]
        self.assertEqual(weight_q.target, torch.ops.quantized_decomposed.quantize_per_tensor.default)
        self.assertEqual(bias_q.target, torch.ops.quantized_decomposed.quantize_per_tensor.default)
        input_dq = conv_node.args[0]
        input_scale = input_dq.args[1]
        bias_scale = bias_dq.args[1]
        weight_scale = weight_dq.args[1]
        self.assertEqual(bias_scale, input_scale * weight_scale)
        (bias_qmin, bias_qmax, bias_dtype) = bias_dq.args[3:]
        self.assertEqual(bias_qmin, -2 ** 31)
        self.assertEqual(bias_qmax, 2 ** 31 - 1)
        self.assertEqual(bias_dtype, torch.int32)

    def test_qat_per_channel_weight_custom_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        m = self._get_conv_bn_model()
        example_inputs = self.example_inputs
        m = capture_pre_autograd_graph(m, example_inputs)
        quantizer = ConvBnInt32WeightQuantizer()
        m = prepare_qat_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m)
        m(*example_inputs)
        (conv_node, _, _) = _get_conv_bn_getitem_nodes(m)
        weight_dq = conv_node.args[1]
        self.assertEqual(weight_dq.target, torch.ops.quantized_decomposed.dequantize_per_channel.default)
        weight_q = weight_dq.args[0]
        self.assertEqual(weight_q.target, torch.ops.quantized_decomposed.quantize_per_channel.default)
        (q_axis, q_qmin, q_qmax, q_dtype) = weight_q.args[3:]
        (dq_axis, dq_qmin, dq_qmax, dq_dtype) = weight_dq.args[3:]
        self.assertEqual(q_axis, 0)
        self.assertEqual(dq_axis, 0)
        self.assertEqual(q_qmin, 0)
        self.assertEqual(dq_qmin, 0)
        self.assertEqual(q_qmax, 2 ** 31 - 1)
        self.assertEqual(dq_qmax, 2 ** 31 - 1)
        self.assertEqual(q_dtype, torch.int32)
        self.assertEqual(dq_dtype, torch.int32)

@skipIfNoQNNPACK
class _TestQuantizePT2EQAT_ConvBn1d(BaseTestQuantizePT2EQAT_ConvBn):
    example_inputs = (torch.randn(1, 3, 5),)
    conv_class = torch.nn.Conv1d
    bn_class = torch.nn.BatchNorm1d

@skipIfNoQNNPACK
class TestQuantizePT2EQAT_ConvBn2d(BaseTestQuantizePT2EQAT_ConvBn):
    example_inputs = (torch.randn(1, 3, 5, 5),)
    conv_class = torch.nn.Conv2d
    bn_class = torch.nn.BatchNorm2d

def _get_conv_bn_getitem_nodes(model: torch.fx.GraphModule):
    if False:
        while True:
            i = 10
    '\n    Return a 3-tuple of (conv, bn, getitem) nodes from the graph.\n    '
    model.graph.eliminate_dead_code()
    model.recompile()
    conv_node = None
    bn_node = None
    getitem_node = None
    for n in model.graph.nodes:
        if n.target == torch.ops.aten.conv2d.default:
            conv_node = n
        if n.target == torch.ops.aten._native_batch_norm_legit.default:
            bn_node = n
        if n.target == operator.getitem:
            getitem_node = n
    assert conv_node is not None, 'bad test setup'
    return (conv_node, bn_node, getitem_node)

class ConvBnInt32WeightQuantizer(Quantizer):
    """
    Dummy quantizer that annotates conv bn in such a way that the weights
    are quantized per channel to int32.
    """

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        if False:
            print('Hello World!')
        (conv_node, _, getitem_node) = _get_conv_bn_getitem_nodes(model)
        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, observer_or_fake_quant_ctr=default_fake_quant)
        weight_qspec = QuantizationSpec(dtype=torch.int32, quant_min=0, quant_max=2 ** 31 - 1, qscheme=torch.per_channel_affine, observer_or_fake_quant_ctr=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAveragePerChannelMinMaxObserver))
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={conv_node.args[0]: act_qspec, conv_node.args[1]: weight_qspec}, _annotated=True)
        getitem_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=act_qspec, _annotated=True)
        return model

    def validate(self, model: torch.fx.GraphModule):
        if False:
            return 10
        pass

class ConvBnDerivedBiasQuantizer(Quantizer):
    """
    Dummy quantizer that annotates conv bn in such a way that the bias qparams are
    derived from the conv input activation and weight qparams.
    """

    def _derive_bias_qparams_from_act_and_weight_qparams(self, obs_or_fqs):
        if False:
            print('Hello World!')
        (act_scale, _) = obs_or_fqs[0].calculate_qparams()
        (weight_scale, _) = obs_or_fqs[1].calculate_qparams()
        bias_scale = torch.tensor([act_scale * weight_scale], dtype=torch.float32)
        bias_zero_point = torch.tensor([0], dtype=torch.int32)
        return (bias_scale, bias_zero_point)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        if False:
            for i in range(10):
                print('nop')
        (conv_node, _, getitem_node) = _get_conv_bn_getitem_nodes(model)
        act_and_weight_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, observer_or_fake_quant_ctr=default_fake_quant)
        bias_qspec = DerivedQuantizationSpec(derived_from=[(conv_node.args[0], conv_node), (conv_node.args[1], conv_node)], derive_qparams_fn=self._derive_bias_qparams_from_act_and_weight_qparams, dtype=torch.int32, quant_min=-2 ** 31, quant_max=2 ** 31 - 1, qscheme=torch.per_tensor_affine)
        conv_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={conv_node.args[0]: act_and_weight_qspec, conv_node.args[1]: act_and_weight_qspec, conv_node.args[2]: bias_qspec}, _annotated=True)
        getitem_node.meta['quantization_annotation'] = QuantizationAnnotation(output_qspec=act_and_weight_qspec, _annotated=True)
        return model

    def validate(self, model: torch.fx.GraphModule):
        if False:
            while True:
                i = 10
        pass

@skipIfNoQNNPACK
class TestQuantizePT2EQATModels(PT2EQATTestCase):

    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_qat_resnet18(self):
        if False:
            return 10
        import torchvision
        with override_quantized_engine('qnnpack'):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.resnet18()
            self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

    @skip_if_no_torchvision
    @skipIfNoQNNPACK
    def test_qat_mobilenet_v2(self):
        if False:
            while True:
                i = 10
        import torchvision
        with override_quantized_engine('qnnpack'):
            example_inputs = (torch.randn(1, 3, 224, 224),)
            m = torchvision.models.mobilenet_v2()
            self._verify_symmetric_xnnpack_qat_numerics(m, example_inputs)

class TestQuantizeMixQATAndPTQ(QuantizationTestCase):

    class TwoLinear(torch.nn.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super().__init__()
            self.linear1 = torch.nn.Linear(16, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            if False:
                while True:
                    i = 10
            return self.linear2(self.linear1(x))

    class QATPTQTestModule(torch.nn.Module):

        def __init__(self):
            if False:
                print('Hello World!')
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linears = TestQuantizeMixQATAndPTQ.TwoLinear()
            self.my_linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            if False:
                for i in range(10):
                    print('nop')
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            linear_out = self.linears(permute_out)
            my_linear_out = self.my_linear(linear_out)
            return torch.nn.functional.hardtanh(my_linear_out)

    def _prepare_qat_linears(self, model):
        if False:
            i = 10
            return i + 15
        for (name, child) in model.named_children():
            if isinstance(child, (torch.nn.Linear, TestQuantizeMixQATAndPTQ.TwoLinear)):
                if isinstance(child, torch.nn.Linear):
                    in_channels = child.weight.size(1)
                else:
                    in_channels = child.linear1.weight.size(1)
                example_input = (torch.rand((1, in_channels)),)
                traced_child = capture_pre_autograd_graph(child, example_input)
                quantizer = XNNPACKQuantizer()
                quantization_config = get_symmetric_quantization_config(is_per_channel=True, is_qat=True)
                quantizer.set_global(quantization_config)
                traced_child_prepared = prepare_qat_pt2e(traced_child, quantizer)
                setattr(model, name, traced_child_prepared)
            else:
                self._prepare_qat_linears(child)

    def _convert_qat_linears(self, model):
        if False:
            i = 10
            return i + 15
        for (name, child) in model.named_children():
            if isinstance(child, torch.fx.GraphModule):
                torch.ao.quantization.move_exported_model_to_eval(child)
                converted_child = convert_pt2e(child, fold_quantize=True)
                setattr(model, name, converted_child)
            else:
                self._convert_qat_linears(child)

    def test_mixing_qat_ptq(self):
        if False:
            return 10
        example_inputs = (torch.randn(2, 3, 4, 4),)
        model = TestQuantizeMixQATAndPTQ.QATPTQTestModule()
        self._prepare_qat_linears(model)
        after_prepare_result_pt2e = model(*example_inputs)
        self._convert_qat_linears(model)
        quant_result_pt2e = model(*example_inputs)
        model_pt2e = capture_pre_autograd_graph(model, example_inputs)
        quantizer = XNNPACKQuantizer()
        quantizer.set_module_type(torch.nn.Linear, None)
        quantization_config = get_symmetric_quantization_config()
        quantizer.set_global(quantization_config)
        model_pt2e = prepare_pt2e(model_pt2e, quantizer)
        after_prepare_result_pt2e = model_pt2e(*example_inputs)
        model_pt2e = convert_pt2e(model_pt2e)
        quant_result_pt2e = model_pt2e(*example_inputs)
        exported_model = torch.export.export(model_pt2e, example_inputs)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 9, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 9, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 3}
        self.checkGraphModuleNodes(exported_model.graph_module, expected_node_occurrence=node_occurrence)