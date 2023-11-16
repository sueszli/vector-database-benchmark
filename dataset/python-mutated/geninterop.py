"""
TypeOffset is a C# class that mirrors the in-memory layout of heap
allocated Python objects.

This script parses the Python C headers and outputs the TypeOffset
C# class.

Requirements:
    - pycparser
    - clang
"""
import os
import shutil
import sys
import sysconfig
import subprocess
from io import StringIO
from pathlib import Path
from pycparser import c_ast, c_parser
_typeoffset_member_renames = {'ht_name': 'name', 'ht_qualname': 'qualname', 'getitem': 'spec_cache_getitem'}

def _check_output(*args, **kwargs):
    if False:
        return 10
    return subprocess.check_output(*args, **kwargs, encoding='utf8')

class AstParser(object):
    """Walk an AST and determine the members of all structs"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._typedefs = {}
        self._typedecls = {}
        self._structs = {}
        self._struct_stack = []
        self._struct_members_stack = []
        self._ptr_decl_depth = 0
        self._struct_members = {}
        self._decl_names = {}

    def get_struct_members(self, name):
        if False:
            for i in range(10):
                print('nop')
        'return a list of (name, type) of struct members'
        defs = self._typedefs.get(name)
        if defs is None:
            return None
        node = self._get_leaf_node(defs)
        name = node.name
        if name is None:
            name = defs.declname
        return self._struct_members.get(name)

    def visit(self, node):
        if False:
            print('Hello World!')
        if isinstance(node, c_ast.FileAST):
            self.visit_ast(node)
        elif isinstance(node, c_ast.Typedef):
            self.visit_typedef(node)
        elif isinstance(node, c_ast.TypeDecl):
            self.visit_typedecl(node)
        elif isinstance(node, c_ast.Struct):
            self.visit_struct(node)
        elif isinstance(node, c_ast.Decl):
            self.visit_decl(node)
        elif isinstance(node, c_ast.FuncDecl):
            self.visit_funcdecl(node)
        elif isinstance(node, c_ast.PtrDecl):
            self.visit_ptrdecl(node)
        elif isinstance(node, c_ast.IdentifierType):
            self.visit_identifier(node)
        elif isinstance(node, c_ast.Union):
            self.visit_union(node)

    def visit_ast(self, ast):
        if False:
            while True:
                i = 10
        for (_name, node) in ast.children():
            self.visit(node)

    def visit_typedef(self, typedef):
        if False:
            i = 10
            return i + 15
        self._typedefs[typedef.name] = typedef.type
        self.visit(typedef.type)

    def visit_typedecl(self, typedecl):
        if False:
            return 10
        self._decl_names[typedecl.type] = typedecl.declname
        self.visit(typedecl.type)

    def visit_struct(self, struct):
        if False:
            while True:
                i = 10
        if struct.decls:
            self._structs[self._get_struct_name(struct)] = struct
            self._struct_stack.insert(0, struct)
            for decl in struct.decls:
                self._struct_members_stack.insert(0, decl.name)
                self.visit(decl)
                self._struct_members_stack.pop(0)
            self._struct_stack.pop(0)
        elif self._ptr_decl_depth or self._struct_members_stack:
            self._add_struct_member(struct.name)

    def visit_decl(self, decl):
        if False:
            for i in range(10):
                print('nop')
        self.visit(decl.type)

    def visit_funcdecl(self, funcdecl):
        if False:
            for i in range(10):
                print('nop')
        self.visit(funcdecl.type)

    def visit_ptrdecl(self, ptrdecl):
        if False:
            while True:
                i = 10
        self._ptr_decl_depth += 1
        self.visit(ptrdecl.type)
        self._ptr_decl_depth -= 1

    def visit_identifier(self, identifier):
        if False:
            i = 10
            return i + 15
        type_name = ' '.join(identifier.names)
        self._add_struct_member(type_name)

    def visit_union(self, union):
        if False:
            print('Hello World!')
        if self._struct_members_stack and union.decls:
            decl = union.decls[0]
            self._struct_members_stack.pop(0)
            self._struct_members_stack.insert(0, decl.name)
            self.visit(decl)

    def _add_struct_member(self, type_name):
        if False:
            i = 10
            return i + 15
        if not (self._struct_stack and self._struct_members_stack):
            return
        current_struct = self._struct_stack[0]
        member_name = self._struct_members_stack[0]
        struct_members = self._struct_members.setdefault(self._get_struct_name(current_struct), [])
        node = None
        if type_name in self._typedefs:
            node = self._get_leaf_node(self._typedefs[type_name])
            if isinstance(node, c_ast.Struct) and node.decls is None:
                if node.name in self._structs:
                    node = self._structs[node.name]
        elif type_name in self._structs:
            node = self._structs[type_name]
        if not self._ptr_decl_depth and isinstance(node, c_ast.Struct):
            for decl in node.decls or []:
                self._struct_members_stack.insert(0, decl.name)
                self.visit(decl)
                self._struct_members_stack.pop(0)
        else:
            struct_members.append((member_name, type_name))

    def _get_leaf_node(self, node):
        if False:
            return 10
        if isinstance(node, c_ast.Typedef):
            return self._get_leaf_node(node.type)
        if isinstance(node, c_ast.TypeDecl):
            return self._get_leaf_node(node.type)
        return node

    def _get_struct_name(self, node):
        if False:
            while True:
                i = 10
        return node.name or self._decl_names.get(node) or '_struct_%d' % id(node)

class Writer(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self._stream = StringIO()

    def append(self, indent=0, code=''):
        if False:
            i = 10
            return i + 15
        self._stream.write('%s%s\n' % (indent * '    ', code))

    def extend(self, s):
        if False:
            while True:
                i = 10
        self._stream.write(s)

    def to_string(self):
        if False:
            return 10
        return self._stream.getvalue()

def preprocess_python_headers(*, cc=None, include_py=None):
    if False:
        print('Hello World!')
    'Return Python.h pre-processed, ready for parsing.\n    Requires clang.\n    '
    this_path = Path(__file__).parent
    fake_libc_include = this_path / 'fake_libc_include'
    include_dirs = [fake_libc_include]
    if cc is None:
        cc = shutil.which('clang')
    if cc is None:
        cc = shutil.which('gcc')
    if cc is None:
        raise RuntimeError('No suitable C compiler found, need clang or gcc')
    if include_py is None:
        include_py = sysconfig.get_config_var('INCLUDEPY')
    include_py = Path(include_py)
    include_dirs.append(include_py)
    include_args = [c for p in include_dirs for c in ['-I', str(p)]]
    defines = ['-D', '__attribute__(x)=', '-D', '__inline__=inline', '-D', '__asm__=;#pragma asm', '-D', '__int64=long long', '-D', '_POSIX_THREADS']
    if sys.platform == 'win32':
        defines.extend(['-D', '__inline=inline', '-D', '__ptr32=', '-D', '__ptr64=', '-D', '__declspec(x)='])
    if hasattr(sys, 'abiflags'):
        if 'd' in sys.abiflags:
            defines.extend(('-D', 'PYTHON_WITH_PYDEBUG'))
        if 'u' in sys.abiflags:
            defines.extend(('-D', 'PYTHON_WITH_WIDE_UNICODE'))
    python_h = include_py / 'Python.h'
    cmd = [cc, '-pthread'] + include_args + defines + ['-E', str(python_h)]
    lines = []
    for line in _check_output(cmd).splitlines():
        if line.startswith('#'):
            line = line.replace('\\', '/')
        lines.append(line)
    return '\n'.join(lines)

def gen_interop_head(writer, version, abi_flags):
    if False:
        for i in range(10):
            print('nop')
    filename = os.path.basename(__file__)
    class_definition = f"\n// Auto-generated by {filename}.\n// DO NOT MODIFY BY HAND.\n\n// Python {'.'.join(map(str, version[:2]))}: ABI flags: '{abi_flags}'\n\n// ReSharper disable InconsistentNaming\n// ReSharper disable IdentifierTypo\n\nusing System;\nusing System.Diagnostics.CodeAnalysis;\nusing System.Runtime.InteropServices;\n\nusing Python.Runtime.Native;\n\nnamespace Python.Runtime\n{{"
    writer.extend(class_definition)

def gen_interop_tail(writer):
    if False:
        for i in range(10):
            print('nop')
    tail = '}\n'
    writer.extend(tail)

def gen_heap_type_members(parser, writer, type_name):
    if False:
        i = 10
        return i + 15
    'Generate the TypeOffset C# class'
    members = parser.get_struct_members('PyHeapTypeObject')
    class_definition = f'\n    [SuppressMessage("Style", "IDE1006:Naming Styles",\n                     Justification = "Following CPython",\n                     Scope = "type")]\n\n    [StructLayout(LayoutKind.Sequential)]\n    internal class {type_name} : GeneratedTypeOffsets, ITypeOffsets\n    {{\n        public {type_name}() {{ }}\n        // Auto-generated from PyHeapTypeObject in Python.h\n'
    for (name, _type) in members:
        name = _typeoffset_member_renames.get(name, name)
        class_definition += '        public int %s  { get; private set; }\n' % name
    class_definition += '    }\n'
    writer.extend(class_definition)

def gen_structure_code(parser, writer, type_name, indent):
    if False:
        return 10
    members = parser.get_struct_members(type_name)
    if members is None:
        return False
    out = writer.append
    out(indent, '[StructLayout(LayoutKind.Sequential)]')
    out(indent, f'internal struct {type_name}')
    out(indent, '{')
    for (name, _type) in members:
        out(indent + 1, f'public IntPtr {name};')
    out(indent, '}')
    out()
    return True

def main(*, cc=None, include_py=None, version=None, out=None):
    if False:
        return 10
    python_h = preprocess_python_headers(cc=cc, include_py=include_py)
    parser = c_parser.CParser()
    ast = parser.parse(python_h)
    ast_parser = AstParser()
    ast_parser.visit(ast)
    writer = Writer()
    if include_py and (not version):
        raise RuntimeError('If the include path is overridden, version must be defined')
    if version:
        version = version.split('.')
    else:
        version = sys.version_info
    abi_flags = getattr(sys, 'abiflags', '').replace('m', '')
    gen_interop_head(writer, version, abi_flags)
    type_name = f'TypeOffset{version[0]}{version[1]}{abi_flags}'
    gen_heap_type_members(ast_parser, writer, type_name)
    gen_interop_tail(writer)
    interop_cs = writer.to_string()
    if not out or out == '-':
        print(interop_cs)
    else:
        with open(out, 'w') as fh:
            fh.write(interop_cs)
if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser('Interop file generator for Python.NET')
    a.add_argument('--cc', help='C compiler to use, either clang or gcc')
    a.add_argument('--include-py', help='Include path of Python')
    a.add_argument('--version', help='Python version')
    a.add_argument('--out', help='Output path', default='-')
    args = a.parse_args()
    sys.exit(main(cc=args.cc, include_py=args.include_py, out=args.out, version=args.version))