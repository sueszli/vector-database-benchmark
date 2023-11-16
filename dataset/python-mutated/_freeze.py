"""Freezing.

This is not intended to be imported directly; please use the exposed
functionalities in `torch.jit`.
"""
from typing import List, Optional
import torch
from torch.jit._script import RecursiveScriptModule, ScriptModule

def freeze(mod, preserved_attrs: Optional[List[str]]=None, optimize_numerics: bool=True):
    if False:
        while True:
            i = 10
    'Freeze ScriptModule, inline submodules, and attributes as constants.\n\n    Freezing a :class:`ScriptModule` will clone it and attempt to inline the cloned\n    module\'s submodules, parameters, and attributes as constants in the TorchScript IR Graph.\n    By default, `forward` will be preserved, as well as attributes & methods specified in\n    `preserved_attrs`. Additionally, any attribute that is modified within a preserved\n    method will be preserved.\n\n    Freezing currently only accepts ScriptModules that are in eval mode.\n\n    Freezing applies generic optimization that will speed up your model regardless of machine.\n    To further optimize using server-specific settings, run `optimize_for_inference` after\n    freezing.\n\n    Args:\n        mod (:class:`ScriptModule`): a module to be frozen\n        preserved_attrs (Optional[List[str]]): a list of attributes to preserve in addition to the forward method.\n            Attributes modified in preserved methods will also be preserved.\n        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly\n            preserve numerics. Full details of optimization can be found at `torch.jit.run_frozen_optimizations`.\n\n    Returns:\n        Frozen :class:`ScriptModule`.\n\n    Example (Freezing a simple module with a Parameter):\n\n    .. testcode::\n        import torch\n        class MyModule(torch.nn.Module):\n            def __init__(self, N, M):\n                super().__init__()\n                self.weight = torch.nn.Parameter(torch.rand(N, M))\n                self.linear = torch.nn.Linear(N, M)\n\n            def forward(self, input):\n                output = self.weight.mm(input)\n                output = self.linear(output)\n                return output\n\n        scripted_module = torch.jit.script(MyModule(2, 3).eval())\n        frozen_module = torch.jit.freeze(scripted_module)\n        # parameters have been removed and inlined into the Graph as constants\n        assert len(list(frozen_module.named_parameters())) == 0\n        # See the compiled graph as Python code\n        print(frozen_module.code)\n\n    Example (Freezing a module with preserved attributes)\n\n    .. testcode::\n        import torch\n        class MyModule2(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n                self.modified_tensor = torch.tensor(10.)\n                self.version = 1\n\n            def forward(self, input):\n                self.modified_tensor += 1\n                return input + self.modified_tensor\n\n        scripted_module = torch.jit.script(MyModule2().eval())\n        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["version"])\n        # we\'ve manually preserved `version`, so it still exists on the frozen module and can be modified\n        assert frozen_module.version == 1\n        frozen_module.version = 2\n        # `modified_tensor` is detected as being mutated in the forward, so freezing preserves\n        # it to retain model semantics\n        assert frozen_module(torch.tensor(1)) == torch.tensor(12)\n        # now that we\'ve run it once, the next result will be incremented by one\n        assert frozen_module(torch.tensor(1)) == torch.tensor(13)\n\n    Note:\n        Freezing submodule attributes is also supported:\n        frozen_module = torch.jit.freeze(scripted_module, preserved_attrs=["submodule.version"])\n\n    Note:\n        If you\'re not sure why an attribute is not being inlined as a constant, you can run\n        `dump_alias_db` on frozen_module.forward.graph to see if freezing has detected the\n        attribute is being modified.\n\n    Note:\n        Because freezing makes weights constants and removes module hierarchy, `to` and other\n        nn.Module methods to manipulate device or dtype no longer work. As a workaround,\n        You can remap devices by specifying `map_location` in `torch.jit.load`, however\n        device-specific logic may have been baked into the model.\n    '
    if not isinstance(mod, ScriptModule):
        raise RuntimeError("Freezing expects a ScriptModule as input. Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'.")
    if mod.training:
        raise RuntimeError('Freezing is currently only implemented for modules in eval mode. Please call .eval() on your module before freezing.')
    preserved_attrs = preserved_attrs if preserved_attrs is not None else []
    out = RecursiveScriptModule(torch._C._freeze_module(mod._c, preserved_attrs))
    RecursiveScriptModule._finalize_scriptmodule(out)
    preserved_methods = [x for x in preserved_attrs if mod._c._has_method(x)]
    run_frozen_optimizations(out, optimize_numerics, preserved_methods)
    return out

def run_frozen_optimizations(mod, optimize_numerics: bool=True, preserved_methods: Optional[List[str]]=None):
    if False:
        i = 10
        return i + 15
    '\n    Run a series of optimizations looking for patterns that occur in frozen graphs.\n\n    The current set of optimizations includes:\n        - Dropout Removal\n        - Pretranspose Linear Layers\n        - Concat Linear Layers with same input Tensor\n        - Conv -> Batchnorm folding\n        - Conv -> Add/Sub folding\n        - Conv -> Mul/Div folding\n\n    Args:\n        mod (:class:`ScriptModule`): a frozen module to be optimized\n\n        optimize_numerics (bool): If ``True``, a set of optimization passes will be run that does not strictly\n        preserve numerics. These optimizations preserve default rtol and atol of `torch.testing.assert_close`\n        when applied on a single transformation, however in a module where many transformations are applied\n        the rtol or atol may no longer fall within the default `assert_close` tolerance. Conv -> Batchnorm folding,\n        Conv-Add/Sub, and Conv -> Mul/Div folding all may alter numerics.\n\n    Returns:\n        None\n\n    Note:\n        In rare occassions, this can result in slower execution.\n\n    Example (Freezing a module with Conv->Batchnorm)\n    .. code-block:: python\n        import torch\n        in_channels, out_channels = 3, 32\n        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)\n        bn = torch.nn.BatchNorm2d(out_channels, eps=.001)\n        mod = torch.nn.Sequential(conv, bn)\n        # set optimize to False here, by default freezing runs run_frozen_optimizations\n        frozen_mod = torch.jit.freeze(torch.jit.script(mod.eval()), optimize=False)\n        # inspect frozen mod\n        assert "batch_norm" in str(frozen_mod.graph)\n        torch.jit.run_frozen_optimizations(frozen_mod)\n        assert "batch_norm" not in str(frozen_mod.graph)\n\n    '
    if mod._c._has_method('forward'):
        torch._C._jit_pass_optimize_frozen_graph(mod.graph, optimize_numerics)
    if preserved_methods is None:
        preserved_methods = []
    for method in preserved_methods:
        torch._C._jit_pass_optimize_frozen_graph(mod.__getattr__(method).graph, optimize_numerics)

def optimize_for_inference(mod: ScriptModule, other_methods: Optional[List[str]]=None) -> ScriptModule:
    if False:
        i = 10
        return i + 15
    '\n    Perform a set of optimization passes to optimize a model for the purposes of inference.\n\n    If the model is not already frozen, optimize_for_inference\n    will invoke `torch.jit.freeze` automatically.\n\n    In addition to generic optimizations that should speed up your model regardless\n    of environment, prepare for inference will also bake in build specific settings\n    such as the presence of CUDNN or MKLDNN, and may in the future make transformations\n    which speed things up on one machine but slow things down on another. Accordingly,\n    serialization is not implemented following invoking `optimize_for_inference` and\n    is not guaranteed.\n\n    This is still in prototype, and may have the potential to slow down your model.\n    Primary use cases that have been targeted so far have been vision models on cpu\n    and gpu to a lesser extent.\n\n    Example (optimizing a module with Conv->Batchnorm)::\n\n        import torch\n        in_channels, out_channels = 3, 32\n        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=True)\n        bn = torch.nn.BatchNorm2d(out_channels, eps=.001)\n        mod = torch.nn.Sequential(conv, bn)\n        frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(mod.eval()))\n        assert "batch_norm" not in str(frozen_mod.graph)\n        # if built with MKLDNN, convolution will be run with MKLDNN weights\n        assert "MKLDNN" in frozen_mod.graph\n    '
    if not isinstance(mod, ScriptModule):
        raise RuntimeError("optimize_for_inference expects a ScriptModule as input. Please use torch.jit.script or torch.jit.trace to script your 'nn.Module'.")
    if other_methods is None:
        other_methods = []
    if hasattr(mod, 'training'):
        mod = freeze(mod.eval(), preserved_attrs=other_methods)
    torch._C._jit_pass_optimize_for_inference(mod._c, other_methods)
    return mod