"""Logic for converting human-readable benchmarks into executable form.

This is mostly string manipulation, with just a bit of importlib magic.
"""
import importlib.abc
import importlib.util
import itertools as it
import os
import re
import textwrap
import uuid
from typing import List, Optional, Tuple, TYPE_CHECKING
import torch
if TYPE_CHECKING:
    from torch.utils.benchmark.utils.timer import Language
else:
    from torch.utils.benchmark import Language
from core.api import AutogradMode, AutoLabels, GroupedBenchmark, RuntimeMode, TimerArgs
from core.types import FlatDefinition, FlatIntermediateDefinition, Label
from core.utils import get_temp_dir
_ALL_MODES = tuple(it.product(RuntimeMode, AutogradMode, Language))

def _generate_torchscript_file(model_src: str, name: str) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    "Returns the path a saved model if one can be constructed from `spec`.\n\n    Because TorchScript requires actual source code in order to script a\n    model, we can't simply `eval` an appropriate model string. Instead, we\n    must write the correct source to a temporary Python file and then import\n    the TorchScript model from that temporary file.\n\n    `model_src` must contain `jit_model = ...`, which `materialize` will supply.\n    "
    assert 'jit_model = ' in model_src, f'Missing jit_model definition:\n{model_src}'
    model_src = f'import torch\n{model_src}'
    model_root = os.path.join(get_temp_dir(), 'TorchScript_models')
    os.makedirs(model_root, exist_ok=True)
    module_path = os.path.join(model_root, f'torchscript_{name}.py')
    artifact_path = os.path.join(model_root, f'torchscript_{name}.pt')
    if os.path.exists(module_path):
        raise ValueError(f'File {module_path} already exists.')
    with open(module_path, 'w') as f:
        f.write(model_src)
    module_spec = importlib.util.spec_from_file_location(f'torchscript__{name}', module_path)
    assert module_spec is not None
    module = importlib.util.module_from_spec(module_spec)
    loader = module_spec.loader
    assert loader is not None
    loader.exec_module(module)
    jit_model = module.jit_model
    assert isinstance(jit_model, (torch.jit.ScriptFunction, torch.jit.ScriptModule)), f'Expected ScriptFunction or ScriptModule, got: {type(jit_model)}'
    jit_model.save(artifact_path)
    os.remove(module_path)
    return artifact_path

def _get_stmt(benchmark: GroupedBenchmark, runtime: RuntimeMode, autograd: AutogradMode, language: Language) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    'Specialize a GroupedBenchmark for a particular configuration.'
    is_python = language == Language.PYTHON
    if runtime == RuntimeMode.EAGER:
        stmts = (benchmark.py_fwd_stmt, benchmark.cpp_fwd_stmt)
    else:
        assert runtime == RuntimeMode.JIT
        assert benchmark.signature_args is not None
        stmts = GroupedBenchmark._make_model_invocation(benchmark.signature_args, benchmark.signature_output, RuntimeMode.JIT)
    stmt = stmts[0 if is_python else 1]
    if autograd == AutogradMode.FORWARD_BACKWARD and stmt is not None:
        assert benchmark.signature_output is not None
        backward = f"{benchmark.signature_output}{('.toTensor()' if runtime == RuntimeMode.JIT and language == Language.CPP else '')}.backward(){(';' if language == Language.CPP else '')}"
        stmt = f'{stmt}\n{backward}'
    return stmt

def _get_setup(benchmark: GroupedBenchmark, runtime: RuntimeMode, language: Language, stmt: str, model_path: Optional[str]) -> str:
    if False:
        i = 10
        return i + 15
    'Specialize a GroupedBenchmark for a particular configuration.\n\n    Setup requires two extra pieces of information:\n      1) The benchmark stmt. This is needed to warm up the model and avoid\n         measuring lazy initialization.\n      2) The model path so we can load it during the benchmark.\n\n    These are only used when `runtime == RuntimeMode.JIT`.\n    '
    if language == Language.PYTHON:
        setup = benchmark.setup.py_setup
        model_setup = benchmark.py_model_setup
    else:
        assert language == Language.CPP
        setup = benchmark.setup.cpp_setup
        model_setup = benchmark.cpp_model_setup
    if runtime == RuntimeMode.EAGER:
        return '\n'.join([setup, model_setup or ''])
    assert runtime == RuntimeMode.JIT
    assert model_path is not None
    assert '"' not in model_path
    if language == Language.PYTHON:
        setup_template: str = textwrap.dedent(f'\n            jit_model = torch.jit.load("{model_path}")\n\n            # Warmup `jit_model`\n            for _ in range(3):\n            {{stmt}}\n        ')
    else:
        assert language == Language.CPP
        setup_template = textwrap.dedent(f'\n            const std::string fpath = "{model_path}";\n            auto jit_model = torch::jit::load(fpath);\n\n            // Warmup `jit_model`\n            for (int i = 0; i < 3; i++) {{{{\n            {{stmt}}\n            }}}}\n        ')
    model_load = setup_template.format(stmt=textwrap.indent(stmt, ' ' * 4))
    return '\n'.join([setup, model_load])

def materialize(benchmarks: FlatIntermediateDefinition) -> FlatDefinition:
    if False:
        for i in range(10):
            print('nop')
    'Convert a heterogeneous benchmark into an executable state.\n\n    This entails generation of TorchScript model artifacts, splitting\n    GroupedBenchmarks into multiple TimerArgs, and tagging the results with\n    AutoLabels.\n    '
    results: List[Tuple[Label, AutoLabels, TimerArgs]] = []
    for (label, args) in benchmarks.items():
        if isinstance(args, TimerArgs):
            auto_labels = AutoLabels(RuntimeMode.EXPLICIT, AutogradMode.EXPLICIT, args.language)
            results.append((label, auto_labels, args))
        else:
            assert isinstance(args, GroupedBenchmark)
            model_path: Optional[str] = None
            if args.py_model_setup and args.torchscript:
                model_setup = f'{args.py_model_setup}\njit_model = torch.jit.script(model)'
                name: str = re.sub('[^a-z0-9_]', '_', '_'.join(label).lower())
                name = f'{name}_{uuid.uuid4()}'
                model_path = _generate_torchscript_file(model_setup, name=name)
            for ((runtime, autograd, language), num_threads) in it.product(_ALL_MODES, args.num_threads):
                if runtime == RuntimeMode.EXPLICIT or autograd == AutogradMode.EXPLICIT:
                    continue
                if runtime == RuntimeMode.JIT and (not args.torchscript):
                    continue
                if autograd == AutogradMode.FORWARD_BACKWARD and (not args.autograd):
                    continue
                stmt = _get_stmt(args, runtime, autograd, language)
                if stmt is None:
                    continue
                setup = _get_setup(args, runtime, language, stmt, model_path)
                global_setup: str = ''
                if language == Language.CPP and runtime == RuntimeMode.JIT:
                    global_setup = textwrap.dedent('\n                        #include <string>\n                        #include <vector>\n                        #include <torch/script.h>\n                    ')
                autolabels = AutoLabels(runtime, autograd, language)
                timer_args = TimerArgs(stmt=stmt, setup=setup, global_setup=global_setup, num_threads=num_threads, language=language)
                results.append((label, autolabels, timer_args))
    return tuple(results)