"""Generates dynamic loading stubs for functions in CUDA and HIP APIs."""
from __future__ import absolute_import
from __future__ import print_function
import argparse
import re
import json
import clang.cindex

def function_header(return_type, name, args):
    if False:
        while True:
            i = 10
    args_expr = []
    for (arg_type, arg_name) in args:
        match = re.search('\\[|\\)', arg_type)
        if match:
            pos = match.span()[0]
            print(arg_type[:pos])
            args_expr.append(f'{arg_type[:pos]} {arg_name}{arg_type[pos:]}')
        else:
            args_expr.append(f'{arg_type} {arg_name}')
    arg_str = ', '.join(args_expr)
    ret = f'{return_type} {name}({arg_str})'
    return ret

def main():
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description='Generate dynamic loading stubs for CUDA and HIP APIs.')
    parser.add_argument('--unique_prefix', default='', type=str, help='Unique prefix for used in the stub')
    parser.add_argument('input', nargs='?', type=argparse.FileType('r'))
    parser.add_argument('output', nargs='?', type=argparse.FileType('w'))
    parser.add_argument('header', nargs='?', type=str, default=None)
    parser.add_argument('extra_args', nargs='*', type=str, default=None)
    args = parser.parse_args()
    config = json.load(args.input)
    function_impl = '\n{return_type} %s {1}NotFound({2}) {{\n  return {not_found_error};\n}}\n\n{0} {{\n  using FuncPtr = {return_type} (%s *)({2});\n  static auto func_ptr = reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("{1}")) ?\n                           reinterpret_cast<FuncPtr>(LOAD_SYMBOL_FUNC("{1}")) :\n                           {1}NotFound;\n  return func_ptr({3});\n}}\n' % (config['calling_conv'], config['calling_conv'])
    prolog = '\nvoid *{0}LoadSymbol(const char *name);\n\n#define LOAD_SYMBOL_FUNC {0}##LoadSymbol\n\n'
    index = clang.cindex.Index.create()
    header = args.header
    extra_args = args.extra_args
    translation_unit = index.parse(header, args=extra_args)
    for diag in translation_unit.diagnostics:
        if diag.severity in [diag.Warning, diag.Fatal]:
            raise Exception(str(diag))
    for extra_i in config['extra_include']:
        args.output.write('#include {}\n'.format(extra_i))
    args.output.write(prolog.format(args.unique_prefix))
    all_definition = set()
    all_declaration = set()
    for cursor in translation_unit.cursor.get_children():
        if cursor.is_definition():
            all_definition.add(cursor.spelling)
        if cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            all_declaration.add(cursor.spelling)
    for cursor in translation_unit.cursor.get_children():
        if cursor.kind != clang.cindex.CursorKind.FUNCTION_DECL:
            continue
        function_name = cursor.spelling
        if function_name not in config['functions'] or function_name in all_definition or function_name not in all_declaration:
            continue
        all_declaration.remove(function_name)
        arg_types = [arg.type.spelling for arg in cursor.get_arguments()]
        arg_names = [arg.spelling for arg in cursor.get_arguments()]
        return_type = config['functions'][function_name].get('return_type', config['return_type'])
        not_found_error = config['functions'][function_name].get('not_found_error', config['not_found_error'])
        header = function_header(return_type, function_name, zip(arg_types, arg_names))
        implementation = function_impl.format(header, function_name, ', '.join(arg_types), ', '.join(arg_names), return_type=return_type, not_found_error=not_found_error)
        args.output.write(implementation)
if __name__ == '__main__':
    main()