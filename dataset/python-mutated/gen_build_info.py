"""Generates a Python module containing information about the build."""
import argparse
try:
    from cuda.cuda import cuda_config
except ImportError:
    cuda_config = None
try:
    from tensorrt.tensorrt import tensorrt_config
except ImportError:
    tensorrt_config = None

def write_build_info(filename, key_value_list):
    if False:
        for i in range(10):
            print('nop')
    'Writes a Python that describes the build.\n\n  Args:\n    filename: filename to write to.\n    key_value_list: A list of "key=value" strings that will be added to the\n      module\'s "build_info" dictionary as additional entries.\n  '
    build_info = {}
    if cuda_config:
        build_info.update(cuda_config.config)
    if tensorrt_config:
        build_info.update(tensorrt_config.config)
    for arg in key_value_list:
        (key, value) = arg.split('=')
        if value.lower() == 'true':
            build_info[key] = True
        elif value.lower() == 'false':
            build_info[key] = False
        else:
            build_info[key] = value.format(**build_info)
    sorted_build_info_pairs = sorted(build_info.items())
    contents = '\n# Copyright 2020 The TensorFlow Authors. All Rights Reserved.\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# you may not use this file except in compliance with the License.\n# You may obtain a copy of the License at\n#\n#     http://www.apache.org/licenses/LICENSE-2.0\n#\n# Unless required by applicable law or agreed to in writing, software\n# distributed under the License is distributed on an "AS IS" BASIS,\n# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n# See the License for the specific language governing permissions and\n# limitations under the License.\n# ==============================================================================\n"""Auto-generated module providing information about the build."""\nimport collections\n\nbuild_info = collections.OrderedDict(%s)\n' % sorted_build_info_pairs
    open(filename, 'w').write(contents)
parser = argparse.ArgumentParser(description='Build info injection into the PIP package.')
parser.add_argument('--raw_generate', type=str, help='Generate build_info.py')
parser.add_argument('--key_value', type=str, nargs='*', help='List of key=value pairs.')
args = parser.parse_args()
if args.raw_generate:
    write_build_info(args.raw_generate, args.key_value)
else:
    raise RuntimeError('--raw_generate must be used.')