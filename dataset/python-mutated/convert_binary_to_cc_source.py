"""Simple script to convert binary file to C++ source code for embedding."""
import argparse
import datetime
import sys

def _convert_bytes_to_cc_source(data, array_name, max_line_width=80, include_guard=None, include_path=None, use_tensorflow_license=False):
    if False:
        return 10
    'Returns strings representing a C++ constant array containing `data`.\n\n  Args:\n    data: Byte array that will be converted into a C++ constant.\n    array_name: String to use as the variable name for the constant array.\n    max_line_width: The longest line length, for formatting purposes.\n    include_guard: Name to use for the include guard macro definition.\n    include_path: Optional path to include in the source file.\n    use_tensorflow_license: Whether to include the standard TensorFlow Apache2\n      license in the generated files.\n\n  Returns:\n    Text that can be compiled as a C++ source file to link in the data as a\n    literal array of values.\n    Text that can be used as a C++ header file to reference the literal array.\n  '
    starting_pad = '   '
    array_lines = []
    array_line = starting_pad
    for value in bytearray(data):
        if len(array_line) + 4 > max_line_width:
            array_lines.append(array_line + '\n')
            array_line = starting_pad
        array_line += ' 0x%02x,' % value
    if len(array_line) > len(starting_pad):
        array_lines.append(array_line + '\n')
    array_values = ''.join(array_lines)
    if include_guard is None:
        include_guard = 'TENSORFLOW_LITE_UTIL_' + array_name.upper() + '_DATA_H_'
    if include_path is not None:
        include_line = '#include "{include_path}"\n'.format(include_path=include_path)
    else:
        include_line = ''
    if use_tensorflow_license:
        license_text = '\n/* Copyright {year} The TensorFlow Authors. All Rights Reserved.\n\nLicensed under the Apache License, Version 2.0 (the "License");\nyou may not use this file except in compliance with the License.\nYou may obtain a copy of the License at\n\n    http://www.apache.org/licenses/LICENSE-2.0\n\nUnless required by applicable law or agreed to in writing, software\ndistributed under the License is distributed on an "AS IS" BASIS,\nWITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.\nSee the License for the specific language governing permissions and\nlimitations under the License.\n==============================================================================*/\n'.format(year=datetime.date.today().year)
    else:
        license_text = ''
    source_template = '{license_text}\n// This is a binary file that has been converted into a C++ data array using the\n// //tensorflow/lite/experimental/acceleration/compatibility/convert_binary_to_cc_source.py\n// script. This form is useful for compiling into a binary to simplify\n// deployment on mobile devices\n\n{include_line}\n// We need to keep the data array aligned on some architectures.\n#ifdef __has_attribute\n#define HAVE_ATTRIBUTE(x) __has_attribute(x)\n#else\n#define HAVE_ATTRIBUTE(x) 0\n#endif\n#if HAVE_ATTRIBUTE(aligned) || (defined(__GNUC__) && !defined(__clang__))\n#define DATA_ALIGN_ATTRIBUTE __attribute__((aligned(16)))\n#else\n#define DATA_ALIGN_ATTRIBUTE\n#endif\n\nextern const unsigned char {array_name}[] DATA_ALIGN_ATTRIBUTE = {{\n{array_values}}};\nextern const int {array_name}_len = {array_length};\n'
    source_text = source_template.format(array_name=array_name, array_length=len(data), array_values=array_values, license_text=license_text, include_line=include_line)
    header_template = '\n{license_text}\n\n// This is a binary file that has been converted into a C++ data array using the\n// //tensorflow/lite/experimental/acceleration/compatibility/convert_binary_to_cc_source.py\n// script. This form is useful for compiling into a binary to simplify\n// deployment on mobile devices\n\n#ifndef {include_guard}\n#define {include_guard}\n\nextern const unsigned char {array_name}[];\nextern const int {array_name}_len;\n\n#endif  // {include_guard}\n'
    header_text = header_template.format(array_name=array_name, include_guard=include_guard, license_text=license_text)
    return (source_text, header_text)

def main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Binary to C++ source converter')
    parser.add_argument('--input_binary_file', type=str, help='Full filepath of input binary.', required=True)
    parser.add_argument('--output_header_file', type=str, help='Full filepath of output header.', required=True)
    parser.add_argument('--array_variable_name', type=str, help='Full filepath of output source.', required=True)
    parser.add_argument('--output_source_file', type=str, help='Name of global variable that will contain the binary data.', required=True)
    (flags, _) = parser.parse_known_args(args=sys.argv[1:])
    with open(flags.input_binary_file, 'rb') as input_handle:
        input_data = input_handle.read()
    (source, header) = _convert_bytes_to_cc_source(data=input_data, array_name=flags.array_variable_name, use_tensorflow_license=True)
    with open(flags.output_source_file, 'w') as source_handle:
        source_handle.write(source)
    with open(flags.output_header_file, 'w') as header_handle:
        header_handle.write(header)
if __name__ == '__main__':
    main()