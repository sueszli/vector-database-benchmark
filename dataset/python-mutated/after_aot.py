import argparse
import copy
import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import uuid
from importlib import import_module
from tempfile import TemporaryFile
from typing import Any, Callable, Dict, Union
import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo.debug_utils import _cuda_system_info_comment, AccuracyError, backend_accuracy_fails, BuckTargetWriter, cast_to_fp64, extra_imports, generate_config_string, helper_for_dump_minify, InputReader, InputWriter, MAX_CONSTANT_NUMEL_INLINE, minifier_dir, NNModuleToString, NopInputReader, same_two_models
from torch._dynamo.utils import clone_inputs, counters, same
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets, has_free_symbols
from torch.hub import tqdm
from .. import config
log = logging.getLogger(__name__)
inductor_config = import_module('torch._inductor.config')
use_buck = inductor_config.is_fbcode()

def wrap_compiler_debug(unconfigured_compiler_fn, compiler_name: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both\n    forward and backward call separately with the backend compiler_fn - like\n    inductor or nvfuser. Intercepting after Aot Autograd presents neat\n    abstraction, where all the params are lifted as graph inputs, making it easy\n    to save the graph as a string.\n    '

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(gm, example_inputs, **kwargs):
        if False:
            while True:
                i = 10
        from torch._subclasses import FakeTensorMode
        compiler_fn = functools.partial(unconfigured_compiler_fn, **kwargs)
        from torch._functorch.aot_autograd import get_aot_graph_name
        graph_name = get_aot_graph_name()
        orig_graph = copy.deepcopy(gm.graph)
        assert config.repro_after in ('dynamo', 'aot', None)
        try:
            inner_compiled_fn = compiler_fn(gm, example_inputs)
        except Exception as e:
            if config.repro_after == 'aot':
                if config.repro_level == 1:
                    dump_compiler_graph_state(fx.GraphModule(gm, orig_graph), example_inputs, compiler_name)
                elif config.repro_level == 2:
                    dump_to_minify(fx.GraphModule(gm, orig_graph), example_inputs, compiler_name)
                log.error('CompilerError')
            raise

        def deferred_for_real_inputs(real_inputs):
            if False:
                i = 10
                return i + 15
            if config.repro_after != 'aot':
                return inner_compiled_fn(real_inputs)
            with config.patch(repro_after=None):
                return inner_debug_fn(real_inputs)

        def inner_debug_fn(real_inputs):
            if False:
                i = 10
                return i + 15
            '\n            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,\n            example_inputs can be fake tensors. We can call compiler_fn (which is\n            inductor or nvfuser) with fake tensors but the actually compiled_fn\n            should be called with real tensors. Therefore, the actual invocation\n            is deferred.\n            '
            fake_mode = FakeTensorMode()
            copy_tensor_attrs = [fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x for x in real_inputs]
            if config.repro_level == 3:
                dump_to_minify(fx.GraphModule(gm, orig_graph), real_inputs, compiler_name)
            if config.repro_level == 4:
                if compiler_name != 'inductor':
                    raise NotImplementedError('Accuracy minification is supported for inductor only')
                if backend_aot_accuracy_fails(gm, real_inputs, compiler_fn):
                    log.warning('Accuracy failed for the AOT Autograd graph %s', graph_name)
                    dump_compiler_graph_state(fx.GraphModule(gm, orig_graph), real_inputs, f'{compiler_name}_accuracy')
                    dump_to_minify(fx.GraphModule(gm, orig_graph), real_inputs, f'{compiler_name}_accuracy')
                    raise AccuracyError('Bad accuracy detected')
                else:
                    return inner_compiled_fn(real_inputs)
            else:
                try:
                    out = inner_compiled_fn(real_inputs)
                    for arg in example_inputs:
                        if isinstance(arg, torch.Tensor) and arg.is_cuda:
                            torch.cuda.synchronize()
                            break
                    return out
                except Exception as e:
                    if config.repro_level == 1:
                        dump_compiler_graph_state(fx.GraphModule(gm, orig_graph), copy_tensor_attrs, compiler_name)
                    elif config.repro_level == 2:
                        dump_to_minify(fx.GraphModule(gm, orig_graph), copy_tensor_attrs, compiler_name)
                    raise
        if config.repro_after == 'aot':
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True
            return compiled_fn
        else:
            return inner_compiled_fn
    return debug_wrapper

def generate_compiler_repro_string(gm, args, *, stable_output=False, save_dir=None):
    if False:
        while True:
            i = 10
    model_str = textwrap.dedent(f'\nimport torch\nfrom torch import tensor, device\nimport torch.fx as fx\nfrom torch._dynamo.testing import rand_strided\nfrom math import inf\nimport torch._inductor.inductor_prims\n\n{generate_config_string(stable_output=stable_output)}\n\nisolate_fails_code_str = None\n\n{extra_imports}\n\n        ')
    if not stable_output:
        model_str += f'# torch version: {torch.version.__version__}\n'
        if hasattr(torch.version, 'cuda'):
            model_str += f'# torch cuda version: {torch.version.cuda}\n'
        if hasattr(torch.version, 'git_version'):
            model_str += f'# torch git version: {torch.version.git_version}\n\n\n'
        model_str += _cuda_system_info_comment()
    model_str += NNModuleToString.convert(gm)

    def hint_if_symint(x):
        if False:
            for i in range(10):
                print('nop')
        return tuple((i.node.hint if isinstance(i, torch.SymInt) else i for i in x))
    writer = InputWriter(save_dir)
    for (placeholder, arg) in zip(fx_placeholder_targets(gm), args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            writer.tensor(placeholder, arg)
        else:
            raise TypeError(f'arg is neither SymInt/int nor torch.Tensor, {arg}')
    model_str += '\n'.join(writer.lines()) + '\n'
    model_str += 'mod = Repro()\n'
    return model_str

def save_graph_repro(fd, gm, args, compiler_name, *, stable_output=False, save_dir=None, command='run', accuracy=None, tracing_mode=None, check_str=None):
    if False:
        while True:
            i = 10
    fd.write(generate_compiler_repro_string(gm, args, stable_output=stable_output, save_dir=save_dir))
    if accuracy is None:
        accuracy = '_accuracy' in compiler_name
    if tracing_mode is None:
        tracing_mode = 'real'
        if any((has_free_symbols(a) for a in args)):
            tracing_mode = 'symbolic'
    fd.write("if __name__ == '__main__':\n")
    fd.write('    from torch._dynamo.repro.after_aot import run_repro\n')
    fd.write(f'    with torch.no_grad():        run_repro(mod, load_args, accuracy={accuracy!r}, command={command!r}, save_dir={save_dir!r}, tracing_mode={tracing_mode!r}, check_str={check_str!r})\n')

def dump_compiler_graph_state(gm, args, compiler_name, *, accuracy=None):
    if False:
        print('Hello World!')
    subdir = os.path.join(minifier_dir(), 'checkpoints')
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f'{len(gm.graph.nodes)}.py')
    log.warning('Writing checkpoint with %s nodes to %s', len(gm.graph.nodes), file_name)
    with open(file_name, 'w') as fd:
        save_graph_repro(fd, gm, args, compiler_name, save_dir=subdir, accuracy=accuracy)
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, 'repro.py')
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning('Copying repro file for convenience to %s', repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning('No write permissions for %s', repro_path)
        pass

def dump_to_minify(gm, args, compiler_name: str):
    if False:
        while True:
            i = 10
    out = io.StringIO()
    subdir = os.path.join(minifier_dir(), 'checkpoints')
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    save_graph_repro(out, gm, args, compiler_name, save_dir=subdir, command='minify')
    return helper_for_dump_minify(out.getvalue())

def isolate_fails(fx_g, args, compiler_name: str, env=None, save_dir=None, accuracy=None, tracing_mode=None, check_str=None):
    if False:
        for i in range(10):
            print('nop')
    if env is None:
        env = {}
    subdir = os.path.join(os.getcwd(), 'isolate')
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f'{str(uuid.uuid4())[:5]}.py')
    with open(file_name, 'w') as fd:
        save_graph_repro(fd, fx_g, args, compiler_name, save_dir=save_dir, command='minifier-query', accuracy=accuracy, tracing_mode=tracing_mode, check_str=check_str)
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    (stdout, stderr) = (TemporaryFile(), TemporaryFile())
    if use_buck:
        cmd = BuckTargetWriter(file_name).write(print_msg=False)
    else:
        cmd = ['python', file_name]
    p = subprocess.Popen(cmd, cwd=subdir, stdout=stdout, stderr=stderr, env=new_env)
    p.wait()
    stdout.seek(0)
    stderr.seek(0)
    print(textwrap.indent(stdout.read().decode('utf-8'), prefix='>>  '), file=sys.stdout)
    print(textwrap.indent(stderr.read().decode('utf-8'), prefix='>>  '), file=sys.stderr)
    return p.returncode != 0

def inductor_fails(fx_g, args, check_str=None):
    if False:
        return 10
    has_cuda = False
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_cuda:
            has_cuda = True
            break

    def sync():
        if False:
            return 10
        if has_cuda:
            torch.cuda.synchronize()
    from torch._inductor.compile_fx import compile_fx_inner
    try:
        result = fx_g(*args)
        assert isinstance(result, (tuple, list))
        assert not any((isinstance(x, (tuple, list)) for x in result))
    except Exception:
        return False
    sync()
    try:
        compile_mod = compile_fx_inner(fx_g, args)
        compile_mod(args)
        sync()
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False

def inductor_accuracy_fails(fx_g, args, check_str=None, *, require_fp64=False, ignore_non_fp=False):
    if False:
        i = 10
        return i + 15
    from torch._inductor.compile_fx import compile_fx_inner
    return backend_aot_accuracy_fails(fx_g, args, compile_fx_inner, require_fp64=require_fp64, ignore_non_fp=ignore_non_fp)
backend_aot_accuracy_fails = functools.partial(backend_accuracy_fails, only_fwd=True)

def repro_common(options, mod, load_args):
    if False:
        i = 10
        return i + 15
    assert not any(mod.named_parameters())
    for (n, b) in mod.named_buffers():
        if b.numel() > MAX_CONSTANT_NUMEL_INLINE:
            log.warning('Constant %s was not serialized, generated random data instead. If you think this is affecting you, please comment on https://github.com/pytorch/pytorch/issues/100468', n)
    if not hasattr(load_args, '_version'):
        log.warning('load_args does not have a _version attribute, please file a bug to PyTorch and describe how you generate this repro script')
    elif load_args._version > 0:
        log.warning('load_args is version %s, but this version of PyTorch only supports version 0.  We will try to run it anyway but there may be an incompatibility; if so, try upgrading your version of PyTorch.', load_args._version)
    nop_reader = NopInputReader()
    load_args(nop_reader)
    with tqdm(desc='Loading inputs', total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args
    mod = make_fx(mod, tracing_mode=options.tracing_mode)(*args)
    torch._inductor.config.generate_intermediate_hooks = True
    return (mod, args)
ACCURACY_FAILS: Dict[str, Callable[[nn.Module, Any], bool]] = {'': inductor_fails, 'accuracy': functools.partial(inductor_accuracy_fails, require_fp64=True, ignore_non_fp=True), 'strict_accuracy': inductor_accuracy_fails}

def repro_minifier_query(options, mod, load_args):
    if False:
        print('Hello World!')
    (mod, args) = repro_common(options, mod, load_args)
    fail_fn = functools.partial(ACCURACY_FAILS[options.accuracy], check_str=options.check_str)
    if fail_fn(mod, args):
        sys.exit(1)
    else:
        sys.exit(0)

def repro_minify(options, mod, load_args):
    if False:
        return 10
    from functorch.compile import minifier
    (mod, args) = repro_common(options, mod, load_args)
    compiler_name = 'inductor_accuracy' if options.accuracy != '' else 'inductor'
    favored_device = 1 if torch.cuda.device_count() >= 2 else 0
    env_variables = {'CUDA_VISIBLE_DEVICES': str(favored_device)}
    module_fails: Any
    if options.isolate:
        module_fails = functools.partial(isolate_fails, env=env_variables, compiler_name=compiler_name, save_dir=options.save_dir, accuracy=options.accuracy, tracing_mode=options.tracing_mode)
    else:
        module_fails = ACCURACY_FAILS[options.accuracy]
    minifier(mod, args, module_fails=functools.partial(module_fails, check_str=options.check_str), dump_state=functools.partial(dump_compiler_graph_state, compiler_name=compiler_name), save_dir=options.save_dir, offload_to_disk=options.offload_to_disk, skip_offload=options.skip_saving_eager_intermediates, skip_sanity=options.skip_sanity, max_granularity=options.max_granularity)

def repro_analyze(options, mod, load_args):
    if False:
        return 10
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.hooks import intermediate_hook
    (mod, args) = repro_common(options, mod, load_args)
    with tqdm(desc='Compiling'):
        compiled = compile_fx_inner(mod, args)
    total = counters['inductor']['intermediate_hooks']
    known_names = set()

    def save_hook(name, val):
        if False:
            while True:
                i = 10
        known_names.add(name)
        if not options.skip_saving_inductor_intermediates:
            writer.write_tensor(os.path.join('inductor', name), val)
        pbar.update(1)
    writer = torch.utils._content_store.ContentStoreWriter(options.save_dir, stable_hash=options.stable_hash)
    reader = torch.utils._content_store.ContentStoreReader(options.save_dir)
    new_args = clone_inputs(args)
    with intermediate_hook(save_hook), tqdm(desc='Saving inductor intermediates', total=total) as pbar:
        compiled(new_args)
        assert not new_args

    def compare_tuples(tuple1, tuple2):
        if False:
            while True:
                i = 10
        diff_indices = [i for i in range(len(tuple1)) if tuple1[i] != tuple2[i]]
        diff_values = [(tuple1[i], tuple2[i]) for i in diff_indices]
        if not diff_values:
            return None
        else:
            return ' and '.join((f'{a} != {b}' for (a, b) in diff_values))

    def check_hook(name, val):
        if False:
            while True:
                i = 10
        meta = writer.compute_tensor_metadata(val)
        meta2 = reader.read_tensor_metadata(os.path.join('inductor', name))
        reason = compare_tuples(meta, meta2)
        if reason is not None:
            pbar.write(f'NONDETERMINISTIC INDUCTOR at {name} ({reason})')
        pbar.update(1)
    if not options.skip_check_deterministic:
        new_args = clone_inputs(args)
        with intermediate_hook(check_hook), tqdm(desc='Checking inductor determinism', total=total) as pbar:
            compiled(new_args)
            assert not new_args

    class WriterInterp(fx.Interpreter):

        def __init__(self, mod, subdir):
            if False:
                i = 10
                return i + 15
            super().__init__(mod)
            self.subdir = subdir

        def run_node(self, n):
            if False:
                i = 10
                return i + 15
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                pbar.update(1)
                writer.write_tensor(os.path.join(self.subdir, name), r)
            return r
    if not options.skip_saving_float64_intermediates:
        (new_mod, new_args) = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))
        with tqdm(desc='Saving float64 intermediates', total=total) as pbar:
            WriterInterp(new_mod, 'float64').boxed_run(new_args)
        assert not new_args

    class ExactReaderInterp(fx.Interpreter):

        def run_node(self, n):
            if False:
                return 10
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                meta = writer.compute_tensor_metadata(r)
                meta2 = reader.read_tensor_metadata(os.path.join('float64', name))
                reason = compare_tuples(meta, meta2)
                if reason is not None:
                    pbar.write(f'NONDETERMINISTIC FLOAT64 at {name} ({reason})')
                pbar.update(1)
            return r
    if not options.skip_check_deterministic:
        (new_mod, new_args) = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))
        with tqdm(desc='Checking float64 determinism', total=total) as pbar:
            ExactReaderInterp(new_mod).boxed_run(new_args)
            assert not new_args

    class ReaderInterp(fx.Interpreter):

        def run_node(self, n):
            if False:
                for i in range(10):
                    print('nop')
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                inductor = reader.read_tensor(os.path.join('inductor', name))
                float64 = reader.read_tensor(os.path.join('float64', name))
                logged = False

                def log_error(msg, *args):
                    if False:
                        return 10
                    nonlocal logged
                    logged = True
                    pbar.write(f'DIVERGED at {name}: {msg % args}')
                if not same(r, inductor, float64, tol=torch._dynamo.config.repro_tolerance, equal_nan=True, log_error=log_error):
                    assert logged
                pbar.update(1)
            return r
    with tqdm(desc='Checking divergence', total=total) as pbar:
        ReaderInterp(mod).boxed_run(args)
    assert not args

def repro_run(options, mod, load_args):
    if False:
        i = 10
        return i + 15
    from torch._inductor.compile_fx import compile_fx_inner
    (mod, args) = repro_common(options, mod, load_args)
    from torch.cuda import synchronize
    compiled = compile_fx_inner(mod, args)
    if options.accuracy != '':
        if not same_two_models(mod, compiled, args, only_fwd=True):
            raise AccuracyError('Bad accuracy detected')
    else:
        need_sync = False
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.is_cuda:
                need_sync = True
                break
        ref = compiled(args)
        if need_sync:
            synchronize()

def run_repro(mod, load_args, *, command='run', accuracy: Union[bool, str]='', save_dir=None, tracing_mode=None, patch_code=None, check_str=None, **kwargs):
    if False:
        return 10
    for k in kwargs:
        log.warning('Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch', k)
    if accuracy is True:
        accuracy = 'accuracy'
    elif accuracy is False:
        accuracy = ''
    if patch_code is not None:
        log.warning('patch_code no longer works on this version of PyTorch, silently ignoring')
    parser = argparse.ArgumentParser(description=f"An after_aot repro script, typically triggering a bug in PyTorch Inductor.\nWhen run with no arguments, this script defaults to running '{command}'.\nExtra flags may be available; to find out more, try '{command} --help'.\nThere are also alternate subcommands available, see below.\n\ndefault settings on this script:\n  accuracy={accuracy!r}\n  tracing_mode={tracing_mode!r}\n  save_dir={save_dir!r}\n  check_str={check_str!r}\n", formatter_class=argparse.RawTextHelpFormatter)

    def common_flags(parser):
        if False:
            for i in range(10):
                print('nop')
        accuracy_group = parser.add_mutually_exclusive_group()
        accuracy_group.add_argument('--no-accuracy', dest='accuracy', action='store_const', const='', default=accuracy, help='do not test accuracy, just run the module and see if it errors')
        accuracy_group.add_argument('--accuracy', action='store_const', const='accuracy', default=accuracy, help="test if the RMSE between the compiled module and the fp64 reference is greater\nthan eager and the fp64 reference. This is usually more reliable than the\nstandard allclose test, as we expect numeric differences from compiling, often\nimproving accuracy over eager.  RMSE test allows for compiled module to\ndiverge greatly from eager, as long as this divergence moves it closer to the\n'true' mathematical value of the network.  Caveats: (1) double precision can\nstill suffer from rounding error, so it is not a perfect reference (see for\nexample 'Herbie: Automatically Improving Floating Point Accuracy') for\napproaches that detect the necessary working precision and compute it in\narbitrary precision floating point; unfortunately, this is not practical for\ntensor computation; (2) if there are not enough samples in the output being\ncompared, we may get unlucky and have an unlucky greater RMSE than eager; this\ncould be overcome by applying a more rigorous statistical test at some\np-value, which we leave for future work.\n")
        accuracy_group.add_argument('--strict-accuracy', dest='accuracy', action='store_const', const='strict_accuracy', default=accuracy, help='by default, when doing accuracy minification we will reject reductions which\nchange the divergence from a floating point divergence to a integral/boolean\ndivergence.  This is because some operations like ReLU involve temporarily\nsharp boundaries that smooth out again afterwards; without requiring\ndivergence on floating point, the minifier will often fixate on divergent\nboolean tensor even though this is not the true source of the divergence.\nHowever, rejecting these reductions makes it more difficult for the minifier\nto make process.  Using this option will let the minifier progress for ALL\ndivergences--you just might not end up with a useful repro in the end.')
        parser.add_argument('--save-dir', type=str, default=save_dir, metavar='DIR', help='directory where saved inputs live')
        parser.add_argument('--no-save-dir', dest='save_dir', action='store_const', const=None, help="don't use any directory for saved inputs")
        parser.add_argument('--tracing-mode', type=str, metavar='{real,fake,symbolic}', default=tracing_mode, help='how to trace the repro module into a GraphModule with metadata')
    subparsers = parser.add_subparsers(dest='command', metavar='{run,minify,analyze}', required=True)
    parser_run = subparsers.add_parser('run', help='just run the repro')
    common_flags(parser_run)
    parser_minify = subparsers.add_parser('minify', help='run the minifier on the repro')
    common_flags(parser_minify)
    parser_minify_isolate = parser_minify.add_mutually_exclusive_group()
    parser_minify_isolate.add_argument('--isolate', action='store_true', default=True, help='run in separate processes to avoid interference (default)')
    parser_minify_isolate.add_argument('--no-isolate', dest='isolate', action='store_false', help='speed up by running all compilation in same process')
    parser_minify.add_argument('--skip-saving-eager-intermediates', action='store_true', help='skip saving eager intermediates on --minify')
    parser_minify.add_argument('--offload-to-disk', action='store_true', help="during minification, offload delta debugging intermediates to disk.  Use if you're OOMing")
    parser_minify.add_argument('--skip-sanity', action='store_true', help='skip sanity check at beginning of minification on original graph')
    parser_minify.add_argument('--max-granularity', type=int, default=None, help='start at this granularity and work down; must be power of 2')
    parser_minify.add_argument('--check-str', type=str, default=check_str, help='require minified program to fail with error containing this string')
    parser_analyze = subparsers.add_parser('analyze', help='run the accuracy analyzer on the repro')
    common_flags(parser_analyze)
    parser_analyze.add_argument('--skip-saving-inductor-intermediates', action='store_true', help='skip saving inductor intermediates on --analyze')
    parser_analyze.add_argument('--skip-saving-float64-intermediates', action='store_true', help='skip saving float64 intermediates')
    parser_analyze.add_argument('--skip-check-deterministic', action='store_true', help='skip checking that the network is deterministic')
    parser_analyze.add_argument('--stable-hash', action='store_true', help='use SHA-1 checksum instead of fast (but possibly unsound) hash')
    parser_minifier_query = subparsers.add_parser('minifier-query')
    common_flags(parser_minifier_query)
    parser_minifier_query.add_argument('--check-str', type=str, default=check_str, help='require minified program to fail with error containing this string')
    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]
    options = parser.parse_args(args)
    COMMAND_FNS = {'minify': repro_minify, 'analyze': repro_analyze, 'minifier-query': repro_minifier_query, 'run': repro_run}
    COMMAND_FNS[options.command](options, mod, load_args)