"""
To run this file by hand from the root of the PyTorch
repository, run:

python -m tools.autograd.gen_autograd        aten/src/ATen/native/native_functions.yaml        aten/src/ATen/native/tags.yaml        $OUTPUT_DIR        tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/csrc/autograd/generated/
"""
import argparse
import os
from typing import List
from torchgen.api import cpp
from torchgen.api.autograd import match_differentiability_info, NativeFunctionWithDifferentiabilityInfo
from torchgen.gen import parse_native_yaml
from torchgen.selective_build.selector import SelectiveBuilder
from . import gen_python_functions
from .gen_autograd_functions import gen_autograd_functions_lib, gen_autograd_functions_python
from .gen_inplace_or_view_type import gen_inplace_or_view_type
from .gen_trace_type import gen_trace_type
from .gen_variable_factories import gen_variable_factories
from .gen_variable_type import gen_variable_type
from .load_derivatives import load_derivatives

def gen_autograd(native_functions_path: str, tags_path: str, out: str, autograd_dir: str, operator_selector: SelectiveBuilder, disable_autograd: bool=False) -> None:
    if False:
        while True:
            i = 10
    (differentiability_infos, used_dispatch_keys) = load_derivatives(os.path.join(autograd_dir, 'derivatives.yaml'), native_functions_path, tags_path)
    template_path = os.path.join(autograd_dir, 'templates')
    native_funcs = parse_native_yaml(native_functions_path, tags_path).native_functions
    fns = sorted(filter(operator_selector.is_native_function_selected_for_training, native_funcs), key=lambda f: cpp.name(f.func))
    fns_with_diff_infos: List[NativeFunctionWithDifferentiabilityInfo] = match_differentiability_info(fns, differentiability_infos)
    if not disable_autograd:
        gen_variable_type(out, native_functions_path, tags_path, fns_with_diff_infos, template_path, used_dispatch_keys)
        gen_inplace_or_view_type(out, native_functions_path, tags_path, fns_with_diff_infos, template_path)
        gen_trace_type(out, native_funcs, template_path)
    gen_autograd_functions_lib(out, differentiability_infos, template_path)
    gen_variable_factories(out, native_functions_path, tags_path, template_path)

def gen_autograd_python(native_functions_path: str, tags_path: str, out: str, autograd_dir: str) -> None:
    if False:
        print('Hello World!')
    (differentiability_infos, _) = load_derivatives(os.path.join(autograd_dir, 'derivatives.yaml'), native_functions_path, tags_path)
    template_path = os.path.join(autograd_dir, 'templates')
    gen_autograd_functions_python(out, differentiability_infos, template_path)
    deprecated_path = os.path.join(autograd_dir, 'deprecated.yaml')
    gen_python_functions.gen(out, native_functions_path, tags_path, deprecated_path, template_path)

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Generate autograd C++ files script')
    parser.add_argument('native_functions', metavar='NATIVE', help='path to native_functions.yaml')
    parser.add_argument('tags', metavar='NATIVE', help='path to tags.yaml')
    parser.add_argument('out', metavar='OUT', help='path to output directory')
    parser.add_argument('autograd', metavar='AUTOGRAD', help='path to autograd directory')
    args = parser.parse_args()
    gen_autograd(args.native_functions, args.tags, args.out, args.autograd, SelectiveBuilder.get_nop_selector())
if __name__ == '__main__':
    main()