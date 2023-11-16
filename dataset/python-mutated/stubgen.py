import re
import sys
COPYRIGHT_NOTICE = '\n// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.\n//\n// Licensed under the Apache License, Version 2.0 (the "License");\n// you may not use this file except in compliance with the License.\n// You may obtain a copy of the License at\n//\n//     http://www.apache.org/licenses/LICENSE-2.0\n//\n// Unless required by applicable law or agreed to in writing, software\n// distributed under the License is distributed on an "AS IS" BASIS,\n// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\n// See the License for the specific language governing permissions and\n// limitations under the License.\n\n// THIS IS A STUB IMPLEMENTATION IMPLEMENTATION FILE ONLY MEANT FOR BUILD\n// PURPOSES.\n\n'

def stubgen(header_filepath, out_file=sys.stdout):
    if False:
        for i in range(10):
            print('nop')
    header_text = ''
    with open(header_filepath, 'r') as file:
        header_text = file.read()
    print(COPYRIGHT_NOTICE, file=out_file)
    print('#include "{}"\n\n'.format(header_filepath), file=out_file)
    FUNCTION_DECL_PATTERN = 'DLL_PUBLIC[\\s]+(.*)[\\s]+(.*)\\(([^\\)]*?)\\);'
    for entry in re.finditer(FUNCTION_DECL_PATTERN, header_text):
        ret_type = entry.group(1)
        func_name = entry.group(2)
        args = entry.group(3)
        print('{} {}({}) {{\n}}\n\n'.format(ret_type, func_name, args), file=out_file)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Produces an empty stub implementation of a C header')
    parser.add_argument('header_filepath', metavar='header', type=str, help='Path to the header file')
    parser.add_argument('--output', metavar='output', type=str, help='Path to the output file')
    args = parser.parse_args()
    f = open(args.output, 'w+') if args.output is not None else sys.stdout
    stubgen(args.header_filepath, out_file=f)