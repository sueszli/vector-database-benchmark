import copy
from typing import List, Tuple
import torch
from torch._export import capture_pre_autograd_graph
from torch import Tensor
from torch.ao.quantization import observer, ObserverOrFakeQuantize, QConfigMapping
from torch.ao.quantization.quantizer import DerivedQuantizationSpec, FixedQParamsQuantizationSpec, QuantizationAnnotation, QuantizationSpec, Quantizer, SharedQuantizationSpec
from torch.ao.quantization.quantizer.xnnpack_quantizer import XNNPACKQuantizer, get_symmetric_quantization_config
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OP_TO_ANNOTATOR, QuantizationConfig
from torch.ao.quantization.quantizer.composable_quantizer import ComposableQuantizer
from torch.ao.quantization.quantizer.embedding_quantizer import EmbeddingQuantizer
from torch.ao.quantization.quantize_pt2e import _convert_to_reference_decomposed_fx, convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.backend_config import get_executorch_backend_config
from torch.ao.quantization.qconfig import default_per_channel_symmetric_qnnpack_qconfig, float_qparams_weight_only_qconfig, per_channel_weight_observer_range_neg_127_to_127, QConfig, weight_observer_range_neg_127_to_127
from torch.ao.quantization.quantize_fx import prepare_fx
from torch.fx import Node
from torch.testing._internal.common_quantization import NodeSpec as ns, QuantizationTestCase, skipIfNoQNNPACK, TestHelperModules
from torch.testing._internal.common_utils import TemporaryFileName
from torch._export import dynamic_dim

class PT2EQuantizationTestCase(QuantizationTestCase):
    """
    Base QuantizationTestCase for PT2 with some helper methods.
    """
    _MAP_TO_FX_TRACED_OPS = {torch.ops.quantized_decomposed.quantize_per_tensor: torch.ops.quantized_decomposed.quantize_per_tensor.default, torch.ops.quantized_decomposed.dequantize_per_tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.default, torch.ops.quantized_decomposed.quantize_per_channel: torch.ops.quantized_decomposed.quantize_per_channel.default, torch.ops.quantized_decomposed.dequantize_per_channel: torch.ops.quantized_decomposed.dequantize_per_channel.default, torch.ops.quantized_decomposed.quantize_per_tensor.tensor: torch.ops.quantized_decomposed.quantize_per_tensor.tensor, torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: torch.ops.quantized_decomposed.dequantize_per_tensor.tensor}

    def _test_quantizer(self, model, example_inputs, quantizer, expected_node_occurrence, expected_node_list=None, check_against_fx_quant=False, fx_qconfig_mapping=None, export_with_dynamic_shape=False):
        if False:
            while True:
                i = 10
        torch._dynamo.reset()
        m_eager = model.eval()
        m = copy.deepcopy(m_eager)
        m = capture_pre_autograd_graph(m, example_inputs, constraints=[dynamic_dim(example_inputs[0], 0)] if export_with_dynamic_shape else [])
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)
        pt2_quant_output = m(*example_inputs)
        node_occurrence = {ns.call_function(k): v for (k, v) in expected_node_occurrence.items()}
        if expected_node_list is None:
            expected_node_list = []
        node_list = [ns.call_function(n) for n in expected_node_list]
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence, expected_node_list=node_list)
        if check_against_fx_quant:
            qconfig_mapping = fx_qconfig_mapping
            backend_config = get_executorch_backend_config()
            m_copy = copy.deepcopy(m_eager)
            m_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs, backend_config=backend_config)
            m_fx(*example_inputs)
            m_fx = _convert_to_reference_decomposed_fx(m_fx, backend_config=backend_config)
            m_fx = capture_pre_autograd_graph(m_fx, example_inputs, constraints=[dynamic_dim(example_inputs[0], 0)] if export_with_dynamic_shape else [])
            node_occurrence = {}
            for (k, v) in PT2EQuantizationTestCase._MAP_TO_FX_TRACED_OPS.items():
                if k in expected_node_occurrence:
                    node_occurrence[ns.call_function(v)] = expected_node_occurrence[k]
            self.checkGraphModuleNodes(m_fx, expected_node_occurrence=node_occurrence)
            fx_quant_output = m_fx(*example_inputs)
            self.assertEqual(fx_quant_output, pt2_quant_output)

    def _quantize(self, m, quantizer, example_inputs):
        if False:
            i = 10
            return i + 15
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)
        return m

    def _get_pt2e_quantized_linear(self, is_per_channel=False) -> torch.fx.GraphModule:
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.linear(x)
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=is_per_channel)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        return self._quantize(m, quantizer, example_inputs)

@skipIfNoQNNPACK
class TestQuantizePT2E(PT2EQuantizationTestCase):

    def test_simple_quantizer(self):
        if False:
            while True:
                i = 10

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    i = 10
                    return i + 15
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)
                        bias_qspec = QuantizationSpec(dtype=torch.float32, is_dynamic=False, observer_or_fake_quant_ctr=observer.PlaceholderObserver)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, output_qspec=act_qspec, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                pass
        example_inputs = (torch.randn(1, 3, 5, 5),)
        node_occurrence = {torch.ops.quantized_decomposed.quantize_per_tensor.default: 2, torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3}
        node_list = [torch.ops.quantized_decomposed.dequantize_per_tensor.default, torch.ops.quantized_decomposed.dequantize_per_tensor.default, torch.ops.aten.conv2d.default, torch.ops.quantized_decomposed.quantize_per_tensor.default]
        self._test_quantizer(TestHelperModules.ConvWithBNRelu(relu=False, bn=False), example_inputs, BackendAQuantizer(), node_occurrence, node_list)

    def test_wo_annotate_conv_output_quantizer(self):
        if False:
            i = 10
            return i + 15

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    return 10
                act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)
                bias_qspec = QuantizationSpec(dtype=torch.float32, is_dynamic=False, observer_or_fake_quant_ctr=observer.PlaceholderObserver)
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    print('Hello World!')
                pass
        m = torch.nn.Conv2d(2, 2, 1)
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 1, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.conv2d.default)]
        self.checkGraphModuleNodes(m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    def test_max_pool2d_quantizer(self):
        if False:
            print('Hello World!')

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    return 10
                act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)
                bias_qspec = QuantizationSpec(dtype=torch.float32, is_dynamic=False, observer_or_fake_quant_ctr=observer.PlaceholderObserver)
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, _annotated=True)
                    if node.op == 'call_function' and node.target == torch.ops.aten.max_pool2d.default:
                        maxpool_node = node
                        input_act = maxpool_node.args[0]
                        assert isinstance(input_act, Node)
                        maxpool_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec}, output_qspec=SharedQuantizationSpec((input_act, maxpool_node)), _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    while True:
                        i = 10
                pass
        m = TestHelperModules.ConvMaxPool2d()
        x = torch.rand(1, 2, 14, 14)
        example_inputs = (x,)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 3, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 4}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.conv2d.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.max_pool2d.default)]
        self.checkGraphModuleNodes(m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    def test_derived_qspec(self):
        if False:
            return 10

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    i = 10
                    return i + 15
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)

                        def derive_qparams_fn(obs_or_fqs: List[ObserverOrFakeQuantize]) -> Tuple[Tensor, Tensor]:
                            if False:
                                i = 10
                                return i + 15
                            assert len(obs_or_fqs) == 2, f'Expecting two obs/fqs, one for activation and one for weight, got: {len(obs_or_fq)}'
                            act_obs_or_fq = obs_or_fqs[0]
                            weight_obs_or_fq = obs_or_fqs[1]
                            (act_scale, act_zp) = act_obs_or_fq.calculate_qparams()
                            (weight_scale, weight_zp) = weight_obs_or_fq.calculate_qparams()
                            return (torch.tensor([act_scale * weight_scale]).to(torch.float32), torch.tensor([0]).to(torch.int32))
                        bias_qspec = DerivedQuantizationSpec(derived_from=[(input_act, node), (weight, node)], derive_qparams_fn=derive_qparams_fn, dtype=torch.int32, quant_min=-2 ** 31, quant_max=2 ** 31 - 1, qscheme=torch.per_tensor_symmetric)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, output_qspec=act_qspec, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    print('Hello World!')
                pass
        m = TestHelperModules.ConvWithBNRelu(relu=False, bn=False).eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 4}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.conv2d.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default)]
        self.checkGraphModuleNodes(m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    def test_derived_qspec_per_channel(self):
        if False:
            return 10

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    for i in range(10):
                        print('nop')
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_affine, is_dynamic=False, ch_axis=0, observer_or_fake_quant_ctr=observer.default_per_channel_weight_observer)

                        def derive_qparams_fn(obs_or_fqs: List[ObserverOrFakeQuantize]) -> Tuple[Tensor, Tensor]:
                            if False:
                                i = 10
                                return i + 15
                            assert len(obs_or_fqs) == 1, f'Expecting one weight obs/fq, got: {len(obs_or_fq)}'
                            weight_obs_or_fq = obs_or_fqs[0]
                            (weight_scale, weight_zp) = weight_obs_or_fq.calculate_qparams()
                            return (weight_scale, torch.zeros_like(weight_scale))
                        bias_qspec = DerivedQuantizationSpec(derived_from=[(weight, node)], derive_qparams_fn=derive_qparams_fn, dtype=torch.int32, quant_min=-2 ** 31, quant_max=2 ** 31 - 1, qscheme=torch.per_channel_symmetric, ch_axis=0)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, output_qspec=act_qspec, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    i = 10
                    return i + 15
                pass
        m = TestHelperModules.ConvWithBNRelu(relu=False, bn=False).eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.quantize_per_channel.default): 0, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 2}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default), ns.call_function(torch.ops.aten.conv2d.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default)]
        self.checkGraphModuleNodes(m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    def test_fixed_qparams_qspec(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.sigmoid(x)

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    i = 10
                    return i + 15
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.sigmoid.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        act_qspec = FixedQParamsQuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, scale=1.0 / 256.0, zero_point=0)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec}, output_qspec=act_qspec, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    return 10
                pass
        m = M().eval()
        example_inputs = (torch.randn(1, 3, 5, 5),)
        m = self._quantize(m, BackendAQuantizer(), example_inputs)
        fixed_scale = 1.0 / 256.0
        fixed_zero_point = 0
        for n in m.graph.nodes:
            if n.op == 'call_function':
                if n.target == torch.ops.quantized_decomposed.quantize_per_tensor.default:
                    scale_0 = n.args[1]
                    zero_point_0 = n.args[2]
                if n.target == torch.ops.quantized_decomposed.dequantize_per_tensor.default:
                    scale_1 = n.args[1]
                    zero_point_1 = n.args[2]
        self.assertEqual(scale_0, fixed_scale)
        self.assertEqual(zero_point_0, fixed_zero_point)
        self.assertEqual(scale_1, fixed_scale)
        self.assertEqual(zero_point_1, fixed_zero_point)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.sigmoid.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default)]
        self.checkGraphModuleNodes(m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    def test_shared_qspec(self):
        if False:
            i = 10
            return i + 15

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    print('Hello World!')
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)
                        bias_qspec = QuantizationSpec(dtype=torch.float32, is_dynamic=False, observer_or_fake_quant_ctr=observer.PlaceholderObserver)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, output_qspec=act_qspec, _annotated=True)
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        input_qspec_map[first_input_node] = act_qspec
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec((first_input_node, cat_node))
                        for input_node in input_nodes[1:]:
                            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec
                        cat_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=share_qparams_with_input_act0_qspec, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    print('Hello World!')
                pass
        m = TestHelperModules.Conv2dWithCat().eval()
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, BackendAQuantizer())
        conv_output_obs = []
        for n in m.graph.nodes:
            if n.op == 'call_function' and n.target == torch.ops.aten.conv2d.default:
                conv_output_obs.append(getattr(m, list(n.users)[0].target))
            if n.op == 'call_function' and n.target == torch.ops.aten.cat.default:
                inputs = n.args[0]
                input0 = inputs[0]
                input1 = inputs[1]
                assert input0.op == 'call_module'
                assert input1.op == 'call_module'
                obs_ins0 = getattr(m, input0.target)
                obs_ins1 = getattr(m, input1.target)
                assert obs_ins0 == obs_ins1
        assert len(conv_output_obs) == 2, 'expecting two observer that follows conv2d ops'
        assert conv_output_obs[0] == conv_output_obs[1]
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 5, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 7}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.cat.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default)]
        self.checkGraphModuleNodes(m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    def _test_transitive_sharing_with_cat_helper(self, quantizer):
        if False:
            for i in range(10):
                print('nop')
        m = TestHelperModules.Conv2dWithTwoCat().eval()
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5), torch.randn(1, 6, 3, 3), torch.randn(1, 6, 3, 3))
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        conv_output_obs = []
        for n in m.graph.nodes:
            if n.op == 'call_function' and n.target == torch.ops.aten.conv2d.default:
                conv_output_obs.append(getattr(m, list(n.users)[0].target))
            if n.op == 'call_function' and n.target == torch.ops.aten.cat.default:
                inputs = n.args[0]
                input0 = inputs[0]
                input1 = inputs[1]
                assert input0.op == 'call_module'
                assert input1.op == 'call_module'
                obs_ins0 = getattr(m, input0.target)
                obs_ins1 = getattr(m, input1.target)
                assert obs_ins0 == obs_ins1
                output_obs = list(n.users)[0]
                assert output_obs.op == 'call_module'
                obs_ins2 = getattr(m, output_obs.target)
                assert obs_ins0 == obs_ins2, 'input observer does not match output'
        assert len(conv_output_obs) == 2, 'expecting two observer that follows conv2d ops'
        assert conv_output_obs[0] == conv_output_obs[1]
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 7, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 9}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.cat.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.cat.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default)]
        self.checkGraphModuleNodes(m, expected_node_list=node_list, expected_node_occurrence=node_occurrence)

    def test_shared_qspec_transitivity(self):
        if False:
            print('Hello World!')
        'This tests the transitivity of SharedQuantizationSpec, that is\n        if A is shared with B, B is shared with C, then C should be shared with A as well\n\n        x1 -> conv1 -> cat1 -----> cat2\n        x2 -> conv2 -/            /\n                       x3 -> add /\n                       x4  /\n\n        both cat has shared input and output, and because of cat and (cat1 -> cat2) is the same Tensor\n        so there is an implicit sharing here, all tensors connect to cat1 and cat2 are in the same\n        sharing group after transitive sharing\n        '

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    i = 10
                    return i + 15
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)
                        bias_qspec = QuantizationSpec(dtype=torch.float32, is_dynamic=False, observer_or_fake_quant_ctr=observer.PlaceholderObserver)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, output_qspec=act_qspec, _annotated=True)
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        input_qspec_map[first_input_node] = act_qspec
                        share_qparams_with_input_act0_qspec = SharedQuantizationSpec((first_input_node, cat_node))
                        for input_node in input_nodes[1:]:
                            input_qspec_map[input_node] = share_qparams_with_input_act0_qspec
                        cat_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=share_qparams_with_input_act0_qspec, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    while True:
                        i = 10
                pass
        self._test_transitive_sharing_with_cat_helper(BackendAQuantizer())

    def test_shared_qspec_transitivity_case_2(self):
        if False:
            print('Hello World!')
        'This tests the transitivity of SharedQuantizationSpec, that is\n        if A is shared with B, B is shared with C, then C should be shared with A as well\n\n        x1 -> conv1 -> cat1 -----> cat2\n        x2 -> conv2 -/            /\n                       x3 -> add /\n                       x4  /\n\n        both cat has shared input and output, and because of cat and (cat1 -> cat2) is the same Tensor\n        so there is an implicit sharing here, all tensors connect to cat1 and cat2 are in the same\n        sharing group after transitive sharing\n\n        the difference is that for this one, all edges and nodes are shared with the second input edge of cat\n        instead of the first input edge of cat as in previous example\n        '

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    while True:
                        i = 10
                for node in model.graph.nodes:
                    if node.op == 'call_function' and node.target == torch.ops.aten.conv2d.default:
                        input_act = node.args[0]
                        assert isinstance(input_act, Node)
                        weight = node.args[1]
                        assert isinstance(weight, Node)
                        bias = node.args[2]
                        assert isinstance(bias, Node)
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        weight_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)
                        bias_qspec = QuantizationSpec(dtype=torch.float32, is_dynamic=False, observer_or_fake_quant_ctr=observer.PlaceholderObserver)
                        node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map={input_act: act_qspec, weight: weight_qspec, bias: bias_qspec}, output_qspec=act_qspec, _annotated=True)
                    elif node.target is torch.ops.aten.cat.default:
                        cat_node = node
                        input_nodes = cat_node.args[0]
                        first_input_node = input_nodes[0]
                        second_input_node = input_nodes[1]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        input_qspec_map[second_input_node] = act_qspec
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec((second_input_node, cat_node))
                        input_qspec_map[first_input_node] = share_qparams_with_input_act1_qspec
                        cat_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=share_qparams_with_input_act1_qspec, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    i = 10
                    return i + 15
                pass
        self._test_transitive_sharing_with_cat_helper(BackendAQuantizer())

    def test_allow_implicit_sharing(self):
        if False:
            for i in range(10):
                print('nop')
        "This tests the allow_transitive_sharing flag of QuantizationAnnotation, that is\n        if a node is configured with allow_implicit_sharing=False, we will not have implicit sharing\n        for node and (node, consumer) even they refer to the same Tensor\n\n        x1 -> add1 -----> add3\n        x2 -/              /\n               x3 -> add2 /\n               x4 -/\n\n        all add has shared input and output, and second input is using shared quantization spec pointing\n        to first input, but we set allow_implicit_sharing to False for all add nodes so input and output of add1,\n        add2 and add3 will each belong to one sharing group, so we'll have:\n\n        x1 -> obs1 -> add1 -> obs1 -> obs3--> add3 -> obs3\n        x2 -> obs1 -/                         /\n               x3 -> obs2 -> add2 -> obs2 -> obs3\n               x4 -> obs2 -/\n        "

        class BackendAQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    while True:
                        i = 10
                for node in model.graph.nodes:
                    if node.target is torch.ops.aten.add.Tensor:
                        add_node = node
                        first_input_node = add_node.args[0]
                        second_input_node = add_node.args[1]
                        input_qspec_map = {}
                        act_qspec = QuantizationSpec(dtype=torch.uint8, quant_min=0, quant_max=255, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                        input_qspec_map[second_input_node] = act_qspec
                        share_qparams_with_input_act1_qspec = SharedQuantizationSpec((second_input_node, add_node))
                        input_qspec_map[first_input_node] = share_qparams_with_input_act1_qspec
                        add_node.meta['quantization_annotation'] = QuantizationAnnotation(input_qspec_map=input_qspec_map, output_qspec=share_qparams_with_input_act1_qspec, allow_implicit_sharing=False, _annotated=True)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                pass
        m = TestHelperModules.ThreeAdd().eval()
        example_inputs = (torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5), torch.randn(1, 3, 5, 5))
        m = capture_pre_autograd_graph(m, example_inputs)
        quantizer = BackendAQuantizer()
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        observers = []
        for n in m.graph.nodes:
            if n.target == torch.ops.aten.add.Tensor:
                input_obs1 = getattr(m, n.args[0].target)
                input_obs2 = getattr(m, n.args[1].target)
                output_obs = getattr(m, list(n.users)[0].target)
                self.assertIs(input_obs1, input_obs2)
                self.assertIs(input_obs1, output_obs)
                observers.append(input_obs1)
        assert len(observers) == 3
        self.assertIsNot(observers[0], observers[1])
        self.assertIsNot(observers[0], observers[2])
        self.assertIsNot(observers[1], observers[2])

    def test_int16(self):
        if False:
            for i in range(10):
                print('nop')

        class Int16ActQuantizer(Quantizer):

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    return 10
                int16_qspec = QuantizationSpec(dtype=torch.int16, quant_min=-2 ** 15, quant_max=2 ** 15 - 1, qscheme=torch.per_tensor_affine, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_observer)
                int8_qspec = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_tensor_symmetric, is_dynamic=False, observer_or_fake_quant_ctr=observer.default_weight_observer)
                quantization_config = QuantizationConfig(input_activation=int16_qspec, weight=int8_qspec, bias=None, output_activation=int16_qspec)
                OP_TO_ANNOTATOR['conv'](model, quantization_config)

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    i = 10
                    return i + 15
                pass

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.conv(x)
        quantizer = Int16ActQuantizer()
        node_occurrence = {torch.ops.quantized_decomposed.quantize_per_tensor.default: 2, torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3}
        node_list = [torch.ops.quantized_decomposed.dequantize_per_tensor.default, torch.ops.quantized_decomposed.dequantize_per_tensor.default, torch.ops.aten.conv2d.default, torch.ops.quantized_decomposed.quantize_per_tensor.default]
        example_inputs = (torch.randn(1, 3, 3, 3),)
        self._test_quantizer(M().eval(), example_inputs, Int16ActQuantizer(), node_occurrence, node_list)

    def test_fold_quantize(self):
        if False:
            i = 10
            return i + 15
        'Test to make sure the quantized model gets quantized weight (quantize_per_tensor op is folded)\n        '
        m = self._get_pt2e_quantized_linear()
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 3}
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_fold_quantize_per_channel(self):
        if False:
            return 10
        'Test to make sure the quantized model gets quantized weight (quantize_per_channel op is folded)\n        '
        m = self._get_pt2e_quantized_linear(is_per_channel=True)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 1, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 2}
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_dont_fold_other_constant(self):
        if False:
            return 10
        'Make sure the constant propagation does not apply to things unrelated to\n        quantization\n        '

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.dont_fold_me = torch.nn.Parameter(torch.randn(2, 2))

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                t = self.dont_fold_me.t()
                return self.linear(x) + t
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_module_type(torch.nn.Linear, operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 3, ns.call_function(torch.ops.aten.t.default): 1}
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_fold_all_ops_before_quantize(self):
        if False:
            i = 10
            return i + 15
        "Test folding all ops that's before quantized operator:\n        Before:\n            get_attr(weight) -> transpose -> quantize -> dequantize\n        After:\n            get_attr(folded_weight) -> dequantize\n        "

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.weight = torch.randn(2, 2)

            def forward(self, x):
                if False:
                    return 10
                t = self.weight.t()
                return torch.nn.functional.linear(x, t)
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config(is_per_channel=False)
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = self._quantize(m, quantizer, example_inputs)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 2, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 3}
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_constant_prop_preserve_metadata(self):
        if False:
            print('Hello World!')
        'Test to make sure the get_attr node for const propagated weight Tensor gets the correct\n        metadata (from original get_attr node from weight)\n        '

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.linear(x)
        quantizer = XNNPACKQuantizer()
        operator_config = get_symmetric_quantization_config()
        quantizer.set_global(operator_config)
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()
        m = capture_pre_autograd_graph(m, example_inputs)
        weight_meta = None
        for n in m.graph.nodes:
            if n.op == 'get_attr' and list(n.users)[0].target == torch.ops.aten.linear.default:
                weight_meta = n.meta
                break
        assert weight_meta is not None, 'Expect to find metadata for weight node'
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        m = convert_pt2e(m, fold_quantize=True)
        for n in m.graph.nodes:
            if n.op == 'get_attr' and 'frozen_param' in n.target:
                self.assertIn('stack_trace', n.meta)
                for key in n.meta:
                    self.assertEqual(n.meta[key], weight_meta[key])

    def test_save_load(self):
        if False:
            for i in range(10):
                print('nop')
        'Test save/load a quantized model\n        '
        m = self._get_pt2e_quantized_linear()
        example_inputs = (torch.randn(2, 2),)
        ref_res = m(*example_inputs)
        with TemporaryFileName() as fname:
            quantized_ep = torch.export.export(m, example_inputs)
            torch.export.save(quantized_ep, fname)
            loaded_ep = torch.export.load(fname)
            loaded_quantized_model = loaded_ep.module()
            res = loaded_quantized_model(*example_inputs)
            self.assertEqual(ref_res, res)

    def test_composable_quantizer_throw(self):
        if False:
            return 10

        class BadQuantizer(Quantizer):

            def annotate(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    i = 10
                    return i + 15
                for n in gm.graph.nodes:
                    n.meta['quantization_annotation'] = None

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                pass
        quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        quantizer.set_global(quantization_config)
        bad_quantizer = BadQuantizer()
        composable_quantizer = ComposableQuantizer([quantizer, bad_quantizer])
        m_eager = TestHelperModules.ConvLinearWPermute().eval()
        example_inputs = (torch.randn(2, 3, 4, 4),)
        self.assertRaises(RuntimeError, lambda : self._test_quantizer(m_eager, example_inputs, composable_quantizer, {}))

    def test_transform_for_annotation(self):
        if False:
            i = 10
            return i + 15

        class TestQuantizer(Quantizer):

            def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    print('Hello World!')
                for n in model.graph.nodes:
                    if n.target == torch.ops.aten.add.Tensor:
                        n.target = torch.ops.aten.mul.Tensor
                return model

            def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
                if False:
                    print('Hello World!')
                return model

            def validate(self, model: torch.fx.GraphModule) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + 3
        m = M().eval()
        quantizer = TestQuantizer()
        example_inputs = (torch.randn(1, 2, 3, 3),)
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        m(*example_inputs)
        node_occurrence = {ns.call_function(torch.ops.aten.add.Tensor): 0, ns.call_function(torch.ops.aten.mul.Tensor): 1}
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence)

    def test_embedding_quantizer(self):
        if False:
            print('Hello World!')
        m_eager = TestHelperModules.EmbeddingModule().eval()
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        example_inputs = (indices,)
        quantizer = EmbeddingQuantizer()
        node_occurrence = {torch.ops.quantized_decomposed.quantize_per_channel.default: 0, torch.ops.quantized_decomposed.dequantize_per_channel.default: 1}
        node_list = [torch.ops.quantized_decomposed.dequantize_per_channel.default, torch.ops.aten.embedding.default]
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping = qconfig_mapping.set_object_type(torch.nn.Embedding, float_qparams_weight_only_qconfig)
        self._test_quantizer(m_eager, example_inputs, quantizer, node_occurrence, node_list, True, qconfig_mapping)

    def test_composable_quantizer_linear_conv(self):
        if False:
            i = 10
            return i + 15
        dynamic_quantizer = XNNPACKQuantizer()
        quantization_config_dynamic = get_symmetric_quantization_config(is_per_channel=False, is_dynamic=True)
        dynamic_quantizer.set_global(quantization_config_dynamic)
        static_quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_global(quantization_config)
        composable_quantizer = ComposableQuantizer([dynamic_quantizer, static_quantizer])
        m_eager = TestHelperModules.ConvLinearWPermute().eval()
        node_occurrence = {torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1, torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1, torch.ops.quantized_decomposed.quantize_per_tensor.default: 3, torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4, torch.ops.quantized_decomposed.quantize_per_channel.default: 0, torch.ops.quantized_decomposed.dequantize_per_channel.default: 1}
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine, quant_min=-128, quant_max=127, eps=2 ** (-12), is_dynamic=True)
        dynamic_qconfig = QConfig(activation=act_affine_quant_obs, weight=weight_observer_range_neg_127_to_127)
        example_inputs = (torch.randn(2, 3, 4, 4),)
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping.set_object_type(torch.nn.Linear, dynamic_qconfig)
        self._test_quantizer(m_eager, example_inputs, composable_quantizer, node_occurrence, [], False, qconfig_mapping)

    def test_embedding_conv_linear_quantization(self):
        if False:
            print('Hello World!')
        m_eager = TestHelperModules.EmbeddingConvLinearModule().eval()
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        indices = torch.unsqueeze(indices, 0)
        example_inputs = (indices,)
        embedding_quantizer = EmbeddingQuantizer()
        dynamic_quantizer = XNNPACKQuantizer()
        quantization_config_dynamic = get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
        dynamic_quantizer.set_global(quantization_config_dynamic)
        static_quantizer = XNNPACKQuantizer()
        quantization_config = get_symmetric_quantization_config(is_per_channel=True)
        static_quantizer.set_global(quantization_config)
        composed_quantizer = ComposableQuantizer([embedding_quantizer, dynamic_quantizer, static_quantizer])
        act_affine_quant_obs = observer.PlaceholderObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine, quant_min=-128, quant_max=127, eps=2 ** (-12), is_dynamic=True)
        dynamic_qconfig = QConfig(activation=act_affine_quant_obs, weight=per_channel_weight_observer_range_neg_127_to_127)
        qconfig = default_per_channel_symmetric_qnnpack_qconfig
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        qconfig_mapping.set_object_type(torch.nn.Linear, dynamic_qconfig)
        qconfig_mapping = qconfig_mapping.set_object_type(torch.nn.Embedding, float_qparams_weight_only_qconfig)
        node_occurrence = {torch.ops.quantized_decomposed.quantize_per_tensor.default: 4, torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4, torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1, torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1, torch.ops.quantized_decomposed.quantize_per_channel.default: 0, torch.ops.quantized_decomposed.dequantize_per_channel.default: 3}
        self._test_quantizer(m_eager, example_inputs, composed_quantizer, node_occurrence, [], True, qconfig_mapping)

    def test_move_exported_model_to_eval(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.dropout(x)
        example_inputs = (torch.randn(1),)
        m = M().train()
        m = capture_pre_autograd_graph(m, example_inputs)
        dropout_node = None
        for n in m.graph.nodes:
            if n.target == torch.ops.aten.native_dropout.default:
                dropout_node = n
                break
        self.assertTrue(dropout_node is not None)
        self.assertTrue(dropout_node.args[2])
        torch.ao.quantization.move_exported_model_to_eval(m)
        targets = [n.target for n in m.graph.nodes]
        self.assertTrue(torch.ops.aten.clone.default in targets)
        self.assertTrue(torch.ops.aten.native_dropout.default not in targets)

    def test_disallow_eval_train(self):
        if False:
            while True:
                i = 10
        m = TestHelperModules.ConvWithBNRelu(relu=True)
        example_inputs = (torch.rand(3, 3, 5, 5),)
        m.eval()
        m.train()
        m = capture_pre_autograd_graph(m, example_inputs)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()
        quantizer = XNNPACKQuantizer()
        m = prepare_qat_pt2e(m, quantizer)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()
        m = convert_pt2e(m, fold_quantize=True)
        with self.assertRaises(NotImplementedError):
            m.eval()
        with self.assertRaises(NotImplementedError):
            m.train()

    def test_reentrant(self):
        if False:
            while True:
                i = 10
        'Test we can safely call quantization apis multiple times'
        m = TestHelperModules.ConvBnReLU2dAndLinearReLU()
        example_inputs = (torch.randn(3, 3, 10, 10),)
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_qat=True))
        m.conv_bn_relu = capture_pre_autograd_graph(m.conv_bn_relu, example_inputs)
        m.conv_bn_relu = prepare_qat_pt2e(m.conv_bn_relu, quantizer)
        m(*example_inputs)
        m.conv_bn_relu = convert_pt2e(m.conv_bn_relu, fold_quantize=True)
        quantizer = XNNPACKQuantizer().set_module_type(torch.nn.Linear, get_symmetric_quantization_config(is_per_channel=False))
        m = capture_pre_autograd_graph(m, example_inputs)
        m = prepare_pt2e(m, quantizer)
        m = convert_pt2e(m, fold_quantize=True)
        node_occurrence = {ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default): 4, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default): 5, ns.call_function(torch.ops.quantized_decomposed.dequantize_per_channel.default): 1}
        node_list = [ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.conv2d.default), ns.call_function(torch.ops.aten.relu.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default), ns.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default), ns.call_function(torch.ops.aten.linear.default), ns.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default)]
        self.checkGraphModuleNodes(m, expected_node_occurrence=node_occurrence, expected_node_list=node_list)