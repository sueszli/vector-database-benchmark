from __future__ import annotations
from typing import Any, Callable, Mapping, Optional, Sequence, Union
import torch._dynamo
import torch.fx
import torch.onnx
from torch.onnx._internal import _beartype, exporter, io_adapter
from torch.onnx._internal.diagnostics import infra

class TorchExport(exporter.FXGraphExtractor):
    """Generates a FX GraphModule using torch.export API
    Args:
        aten_graph: If True, exports a graph with ATen operators.
                    If False, exports a graph with Python operators.
    """

    def __init__(self, aten_graph: Optional[bool]=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.aten_graph = aten_graph or True

    def generate_fx(self, options: exporter.ResolvedExportOptions, model: 'ExportedProgram', model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
        if False:
            while True:
                i = 10
        model = model.run_decompositions(options.decomposition_table)
        self.input_adapter.append_step(io_adapter.FlattenInputWithTreeSpecValidationInputStep())
        self.input_adapter.append_step(io_adapter.PrependParamsAndBuffersAotAutogradInputStep(model))
        options.fx_tracer.input_adapter.append_step(io_adapter.RemoveNoneInputStep())
        updated_model_args = self.input_adapter.apply(*model_args, **model_kwargs)
        options.fx_tracer.output_adapter.append_step(io_adapter.FlattenOutputStep())
        return self.pre_export_passes(options, model, model.graph_module, updated_model_args)

    @_beartype.beartype
    def pre_export_passes(self, options: exporter.ResolvedExportOptions, original_model: Union[torch.nn.Module, Callable], fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
        if False:
            while True:
                i = 10
        from torch.onnx._internal.fx import analysis, passes
        diagnostic_context = options.diagnostic_context
        fx_module = passes.InsertTypePromotion(diagnostic_context, fx_module).run()
        analysis.UnsupportedFxNodesAnalysis(diagnostic_context, fx_module, options.onnxfunction_dispatcher).analyze(infra.levels.ERROR)
        return fx_module