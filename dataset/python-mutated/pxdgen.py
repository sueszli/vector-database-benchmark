"""
Auto-generates PXD files from annotated C++ headers.

Invoked via cmake during the regular build process.
"""
import argparse
import os
from pathlib import Path
import re
import sys
from pygments.token import Token
from pygments.lexers import get_lexer_for_filename
CWD = os.getcwd()

class ParserError(Exception):
    """
    Represents a fatal parsing error in PXDGenerator.
    """

    def __init__(self, filename, lineno, message):
        if False:
            i = 10
            return i + 15
        super().__init__(f'{filename}:{lineno} {message}')

class PXDGenerator:
    """
    Represents, and performs, a single conversion of a C++ header file to a
    PXD file.

    @param filename:
        input (C++ header) file name. is opened and read.
        the output filename is the same, but with .pxd instead of .h.
    """

    def __init__(self, filename):
        if False:
            while True:
                i = 10
        self.filename = filename
        self.warnings = []
        (self.stack, self.lineno, self.annotations) = (None, None, None)

    def parser_error(self, message, lineno=None):
        if False:
            print('Hello World!')
        '\n        Returns a ParserError object for this generator, at the current line.\n        '
        if lineno is None:
            lineno = self.lineno
        return ParserError(self.filename, lineno, message)

    def tokenize(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Tokenizes the input file.\n\n        Yields (tokentype, val) pairs, where val is a string.\n\n        The concatenation of all val strings is equal to the input file's\n        content.\n        "
        self.stack = []
        self.lineno = 1
        lexer = get_lexer_for_filename('.cpp')
        with open(self.filename, encoding='utf8') as infile:
            code = infile.read()
        for (token, val) in lexer.get_tokens(code):
            yield (token, val)
            self.lineno += val.count('\n')

    def handle_singleline_comment(self, val):
        if False:
            print('Hello World!')
        "\n        Breaks down a '//'-style single-line comment, and passes the result\n        to handle_comment()\n\n        @param val:\n            the comment text, as string, including the '//'\n        "
        try:
            val = re.match('^// (.*)$', val).group(1)
        except AttributeError as ex:
            raise self.parser_error('invalid single-line comment') from ex
        self.handle_comment(val)

    def handle_multiline_comment(self, val):
        if False:
            i = 10
            return i + 15
        "\n        Breaks down a '/* */'-style multi-line comment, and passes the result\n        to handle_comment()\n\n        @param val:\n            the comment text, as string, including the '/*' and '*/'\n        "
        try:
            val = re.match('^/\\*(.*)\\*/$', val, re.DOTALL).group(1)
        except AttributeError as ex:
            raise self.parser_error('invalid multi-line comment') from ex
        val = ' * ' + val.rstrip()
        lines = val.split('\n')
        comment_lines = []
        for (idx, line) in enumerate(lines):
            try:
                line = re.match('^ \\*( (.*))?$', line).group(2) or ''
            except AttributeError as ex:
                raise self.parser_error('invalid multi-line comment line', idx + self.lineno) from ex
            if comment_lines or line.strip() != '':
                comment_lines.append(line)
        self.handle_comment('\n'.join(comment_lines).rstrip())

    def handle_comment(self, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        Handles any comment, with its format characters removed,\n        extracting the pxd annotation\n        '
        annotations = re.findall('pxd:\\s(.*?)(:pxd|$)', val, re.DOTALL)
        annotations = [annotation[0] for annotation in annotations]
        if not annotations:
            raise self.parser_error('comment contains no valid pxd annotation')
        for annotation in annotations:
            annotation = annotation.rstrip()
            annotation_lines = annotation.split('\n')
            for (idx, line) in enumerate(annotation_lines):
                if line.strip() != '':
                    self.add_annotation(annotation_lines[idx:])
                    break
            else:
                raise self.parser_error('pxd annotation is empty:\n' + val)

    def add_annotation(self, annotation_lines):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a (current namespace, pxd annotation) tuple to self.annotations.\n        '
        if '{' in self.stack:
            raise self.parser_error('PXD annotation is brace-enclosed')
        if not self.stack:
            namespace = None
        else:
            namespace = '::'.join(self.stack)
        self.annotations.append((namespace, annotation_lines))

    def handle_token(self, token, val):
        if False:
            return 10
        '\n        Handles one token while the parser is in its regular state.\n\n        Returns the new state integer.\n        '
        if token == Token.Keyword and val == 'namespace':
            return 1
        if (token, val) == (Token.Punctuation, '{'):
            self.stack.append('{')
        elif (token, val) == (Token.Punctuation, '}'):
            try:
                self.stack.pop()
            except IndexError as ex:
                raise self.parser_error("unmatched '}'") from ex
        elif token == Token.Comment.Single and 'pxd:' in val:
            self.handle_singleline_comment(val)
        elif token == Token.Comment.Multiline and 'pxd:' in val:
            self.handle_multiline_comment(val)
        else:
            pass
        return 0

    def parse(self):
        if False:
            i = 10
            return i + 15
        '\n        Parses the input file.\n\n        Internally calls self.tokenize().\n\n        Adds all found PXD annotations to self.annotations,\n        together with info about the namespace in which they were encountered.\n        '

        def handle_state_0(self, token, val, namespace_parts):
            if False:
                for i in range(10):
                    print('nop')
            del namespace_parts
            return self.handle_token(token, val)

        def handle_state_1(self, token, val, namespace_parts):
            if False:
                i = 10
                return i + 15
            if token not in Token.Name:
                raise self.parser_error("expected identifier after 'namespace'")
            namespace_parts.append(val)
            return 2

        def handle_state_2(self, token, val, namespace_parts):
            if False:
                i = 10
                return i + 15
            if (token, val) == (Token.Operator, ':'):
                return 3
            if (token, val) != (Token.Punctuation, '{'):
                raise self.parser_error(f"expected '{{' or '::' after 'namespace {self.stack[-1]}'")
            self.stack.append('::'.join(namespace_parts))
            namespace_parts.clear()
            return 0

        def handle_state_3(self, token, val, namespace_parts):
            if False:
                for i in range(10):
                    print('nop')
            del namespace_parts
            if (token, val) == (Token.Operator, ':'):
                return 1
            raise self.parser_error("nested namespaces are separated with '::'")
        transitions = {0: handle_state_0, 1: handle_state_1, 2: handle_state_2, 3: handle_state_3}
        self.annotations = []
        state = 0
        namespace_parts = []
        for (token, val) in self.tokenize():
            if token in Token.Text and (not val.strip()):
                continue
            try:
                state = transitions[state](self, token, val, namespace_parts)
            except KeyError as exp:
                raise ValueError('reached invalid state in pxdgen') from exp
        if self.stack:
            raise self.parser_error("expected '}', but found EOF")

    def get_pxd_lines(self):
        if False:
            while True:
                i = 10
        '\n        calls self.parse() and processes the pxd annotations to pxd code lines.\n        '
        from datetime import datetime
        year = datetime.now().year
        yield f'# Copyright 2013-{year} the openage authors. See copying.md for legal info.'
        yield ''
        yield ('# Auto-generated from annotations in ' + self.filename.name)
        yield ('# ' + str(self.filename))
        self.parse()
        previous_namespace = None
        for (namespace, annotation_lines) in self.annotations:
            yield ''
            if namespace != previous_namespace:
                yield ''
            if namespace:
                prefix = '    '
                if namespace != previous_namespace:
                    yield ('cdef extern from r"' + self.filename.as_posix() + '" namespace "' + namespace + '" nogil:')
            else:
                prefix = ''
            for annotation in annotation_lines:
                annotation = self.postprocess_annotation_line(annotation)
                if annotation:
                    yield (prefix + annotation)
                else:
                    yield ''
            previous_namespace = namespace
        yield ''

    def postprocess_annotation_line(self, annotation):
        if False:
            for i in range(10):
                print('nop')
        '\n        Post-processes each individual annotation line, applying hacks and\n        testing it, etc.\n\n        See libopenage/pyinterface/hacks.h for documentation on the individual\n        hacks.\n        '
        annotation = annotation.rstrip()
        if annotation.endswith(';'):
            self.warnings.append("cython declaration ends in ';', what have you done?")
        if annotation.endswith(')'):
            self.warnings.append("mark the function as 'except +' or 'noexcept':\n" + annotation)
        elif annotation.endswith('noexcept'):
            annotation = annotation[:-8].rstrip()
        if 'cdef ' in annotation:
            self.warnings.append("there's no need to use 'cdef' in PXD annotations:\n" + annotation)
        return annotation

    def generate(self, pxdfile, ignore_timestamps=False, print_warnings=True):
        if False:
            while True:
                i = 10
        '\n        reads the input file and writes the output file.\n        the output file is updated only if its content will change.\n\n        on parsing failure, raises ParserError.\n        '
        if not ignore_timestamps and os.path.exists(pxdfile):
            if os.path.getmtime(self.filename) <= os.path.getmtime(pxdfile):
                return False
        result = '\n'.join(self.get_pxd_lines())
        if os.path.exists(pxdfile):
            with open(pxdfile, encoding='utf8') as outfile:
                if outfile.read() == result:
                    return False
        if not pxdfile.parent.is_dir():
            pxdfile.parent.mkdir()
        with pxdfile.open('w', encoding='utf8') as outfile:
            if pxdfile.is_absolute():
                printpath = pxdfile
            else:
                printpath = os.path.relpath(pxdfile, CWD)
            print(f'\x1b[36mpxdgen: generate {printpath}\x1b[0m')
            outfile.write(result)
        if print_warnings and self.warnings:
            print(f'\x1b[33;1mWARNING\x1b[m pxdgen[{self.filename}]:')
            for warning in self.warnings:
                print(warning)
        return True

def parse_args():
    if False:
        i = 10
        return i + 15
    '\n    pxdgen command-line interface.\n\n    designed to allow both manual and automatic (via CMake) usage.\n    '
    cli = argparse.ArgumentParser()
    cli.add_argument('files', nargs='*', metavar='HEADERFILE', help='input files (usually cpp .h files).')
    cli.add_argument('--file-list', help='a file containing a semicolon-separated list of input files.')
    cli.add_argument('--ignore-timestamps', action='store_true', help='force generating even if the output file is already up to date')
    cli.add_argument('--output-dir', help='build directory corresponding to the CWD to write the generated file(s) in.')
    cli.add_argument('-v', '--verbose', action='store_true', help='increase logging verbosity')
    args = cli.parse_args()
    if args.file_list:
        with open(args.file_list, encoding='utf8') as flist:
            file_list = flist.read().strip().split(';')
    else:
        file_list = []
    from itertools import chain
    args.all_files = list(chain(args.files, file_list))
    return args

def main():
    if False:
        i = 10
        return i + 15
    ' CLI entry point '
    args = parse_args()
    cppname = 'libopenage'
    cppdir = Path(cppname).absolute()
    out_cppdir = Path(args.output_dir) / cppname
    if args.verbose:
        hdr_count = len(args.all_files)
        plural = 's' if hdr_count > 1 else ''
        print(f'extracting pxd information from {hdr_count} header{plural}...')
    for filename in args.all_files:
        filename = Path(filename).resolve()
        if cppdir not in filename.parents:
            print(f'pxdgen source file is not in {cppdir!r}: {filename!r}')
            sys.exit(1)
        pxdfile_relpath = filename.with_suffix('.pxd').relative_to(cppdir)
        pxdfile = out_cppdir / pxdfile_relpath
        if args.verbose:
            print(f"creating '{pxdfile}' for '{filename}':")
        generator = PXDGenerator(filename)
        result = generator.generate(pxdfile, ignore_timestamps=args.ignore_timestamps, print_warnings=True)
        if args.verbose and (not result):
            print('nothing done.')
        for dirname in pxdfile_relpath.parents:
            template = out_cppdir / dirname / '__init__'
            for extension in ('py', 'pxd'):
                initfile = template.with_suffix('.' + extension)
                if not initfile.exists():
                    print(f'\x1b[36mpxdgen: create package index {initfile.relative_to(args.output_dir)}\x1b[0m')
                    initfile.touch()
if __name__ == '__main__':
    main()