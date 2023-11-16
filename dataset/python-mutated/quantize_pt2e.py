import torch
from torch.fx import GraphModule
from torch.fx import Node
from .pt2e.prepare import prepare
from .pt2e.qat_utils import _fuse_conv_bn_qat, _fold_conv_bn_qat
from .pt2e.utils import _get_node_name_to_scope, _fuse_conv_bn_, _disallow_eval_train
from .pt2e.representation import reference_representation_rewrite
from .quantize_fx import _convert_to_reference_decomposed_fx
from torch.ao.quantization.quantizer import Quantizer, QuantizationSpecBase, QuantizationSpec, FixedQParamsQuantizationSpec, SharedQuantizationSpec, DerivedQuantizationSpec, QuantizationAnnotation
from torch.fx.passes.infra.pass_manager import PassManager
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch._inductor.constant_folding import constant_fold
__all__ = ['prepare_pt2e', 'prepare_qat_pt2e', 'convert_pt2e']

def prepare_pt2e(model: GraphModule, quantizer: Quantizer) -> GraphModule:
    if False:
        for i in range(10):
            print('nop')
    "Prepare a model for post training quantization\n\n    Args:\n      * `model` (torch.fx.GraphModule): a model captured by `torch.export` API\n        in the short term we are using `torch._export.capture_pre_autograd_graph`,\n        in the long term we'll migrate to some `torch.export` API\n      * `quantizer`: A backend specific quantizer that conveys how user want the\n        model to be quantized. Tutorial for how to write a quantizer can be found here:\n        https://pytorch.org/tutorials/prototype/pt2e_quantizer.html\n\n    Return:\n      A GraphModule with observer (based on quantizer annotation), ready for calibration\n\n    Example::\n\n        import torch\n        from torch.ao.quantization.quantize_pt2e import prepare_pt2e\n        from torch._export import capture_pre_autograd_graph\n        from torch.ao.quantization.quantizer import (\n            XNNPACKQuantizer,\n            get_symmetric_quantization_config,\n        )\n\n        class M(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.linear = torch.nn.Linear(5, 10)\n\n           def forward(self, x):\n               return self.linear(x)\n\n        # initialize a floating point model\n        float_model = M().eval()\n\n        # define calibration function\n        def calibrate(model, data_loader):\n            model.eval()\n            with torch.no_grad():\n                for image, target in data_loader:\n                    model(image)\n\n        # Step 1. program capture\n        # NOTE: this API will be updated to torch.export API in the future, but the captured\n        # result shoud mostly stay the same\n        m = capture_pre_autograd_graph(m, *example_inputs)\n        # we get a model with aten ops\n\n        # Step 2. quantization\n        # backend developer will write their own Quantizer and expose methods to allow\n        # users to express how they\n        # want the model to be quantized\n        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())\n        m = prepare_pt2e(m, quantizer)\n\n        # run calibration\n        # calibrate(m, sample_inference_data)\n    "
    torch._C._log_api_usage_once('quantization_api.quantize_pt2e.prepare_pt2e')
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    _fuse_conv_bn_(model)
    quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    model = prepare(model, node_name_to_scope, is_qat=False)
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model

def prepare_qat_pt2e(model: GraphModule, quantizer: Quantizer) -> GraphModule:
    if False:
        while True:
            i = 10
    'Prepare a model for quantization aware training\n\n    Args:\n      * `model` (torch.fx.GraphModule): see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`\n      * `quantizer`: see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`\n\n    Return:\n      A GraphModule with fake quant modules (based on quantizer annotation), ready for\n      quantization aware training\n\n    Example::\n        import torch\n        from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e\n        from torch._export import capture_pre_autograd_graph\n        from torch.ao.quantization.quantizer import (\n            XNNPACKQuantizer,\n            get_symmetric_quantization_config,\n        )\n\n        class M(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.linear = torch.nn.Linear(5, 10)\n\n           def forward(self, x):\n               return self.linear(x)\n\n        # initialize a floating point model\n        float_model = M().eval()\n\n        # define the training loop for quantization aware training\n        def train_loop(model, train_data):\n            model.train()\n            for image, target in data_loader:\n                ...\n\n        # Step 1. program capture\n        # NOTE: this API will be updated to torch.export API in the future, but the captured\n        # result shoud mostly stay the same\n        m = capture_pre_autograd_graph(m, *example_inputs)\n        # we get a model with aten ops\n\n        # Step 2. quantization\n        # backend developer will write their own Quantizer and expose methods to allow\n        # users to express how they\n        # want the model to be quantized\n        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())\n        m = prepare_qat_pt2e(m, quantizer)\n\n        # run quantization aware training\n        train_loop(prepared_model, train_loop)\n\n    '
    torch._C._log_api_usage_once('quantization_api.quantize_pt2e.prepare_qat_pt2e')
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    _fuse_conv_bn_qat(model)
    model = prepare(model, node_name_to_scope, is_qat=True)
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model
_QUANT_OPS = [torch.ops.quantized_decomposed.quantize_per_tensor.default, torch.ops.quantized_decomposed.quantize_per_tensor.tensor, torch.ops.quantized_decomposed.quantize_per_channel.default]

def _quant_node_constraint(n: Node) -> bool:
    if False:
        return 10
    "If there is any pure ops between get_attr and quantize op they will be const propagated\n    e.g. get_attr(weight) -> transpose -> quantize -> dequantize*\n    (Note: dequantize op is not going to be constant propagated)\n\n    This filter is added because we don't want to constant fold the things that are not\n    related to quantization\n    "
    return n.op == 'call_function' and n.target in _QUANT_OPS

def convert_pt2e(model: GraphModule, use_reference_representation: bool=False, fold_quantize: bool=False) -> GraphModule:
    if False:
        i = 10
        return i + 15
    "Convert a calibrated/trained model to a quantized model\n\n    Args:\n      * `model` (torch.fx.GraphModule): calibrated/trained model\n      * `use_reference_representation` (bool): boolean flag to indicate whether to produce referece representation or not\n      * `fold_quantize` (bool): boolean flag to indicate whether fold the quantize op or not\n\n    Note: please set `fold_quantize` to True whenever you can, we'll deprecate this flag and\n    make True the default option in the future, to make sure the change doesn't break BC for you, it's\n    better to set the flag to True now.\n\n    Returns:\n        quantized model, either in q/dq representation or reference representation\n\n    Example::\n\n        # prepared_model: the model produced by `prepare_pt2e`/`prepare_qat_pt2e` and calibration/training\n        # `convert_pt2e` produces a quantized model that represents quantized computation with\n        # quantize dequantize ops and fp32 ops by default.\n        # Please refer to\n        # https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html#convert-the-calibrated-model-to-a-quantized-model\n        # for detailed explanation of output quantized model\n        quantized_model = convert_pt2e(prepared_model)\n\n    "
    torch._C._log_api_usage_once('quantization_api.quantize_pt2e.convert_pt2e')
    original_graph_meta = model.meta
    model = _convert_to_reference_decomposed_fx(model)
    model = _fold_conv_bn_qat(model)
    pm = PassManager([DuplicateDQPass()])
    model = pm(model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module
    if fold_quantize:
        constant_fold(model, _quant_node_constraint)
    if use_reference_representation:
        model = reference_representation_rewrite(model)
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model