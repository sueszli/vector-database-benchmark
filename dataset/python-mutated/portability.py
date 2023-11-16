"""Tools for cross-OS portability."""
from __future__ import annotations
import os
import re
import struct
import subprocess
from logging import getLogger
from os.path import basename, realpath
from ..auxlib.ish import dals
from ..base.constants import PREFIX_PLACEHOLDER
from ..base.context import context
from ..common.compat import on_linux, on_mac, on_win
from ..exceptions import BinaryPrefixReplacementError, CondaIOError
from ..gateways.disk.update import CancelOperation, update_file_in_place_as_binary
from ..models.enums import FileMode
log = getLogger(__name__)
SHEBANG_REGEX = b'^(#!(?:[ ]*)(/(?:\\\\ |[^ \\n\\r\\t])*)(.*))$'
MAX_SHEBANG_LENGTH = 127 if on_linux else 512
POPULAR_ENCODINGS = ('utf-8', 'utf-16-le', 'utf-16-be', 'utf-32-le', 'utf-32-be')

class _PaddingError(Exception):
    pass

def update_prefix(path, new_prefix, placeholder=PREFIX_PLACEHOLDER, mode=FileMode.text, subdir=context.subdir):
    if False:
        for i in range(10):
            print('nop')
    if on_win and mode == FileMode.text:
        new_prefix = new_prefix.replace('\\', '/')

    def _update_prefix(original_data):
        if False:
            while True:
                i = 10
        data = replace_prefix(mode, original_data, placeholder, new_prefix)
        if not on_win:
            data = replace_long_shebang(mode, data)
        if data == original_data:
            raise CancelOperation()
        if mode == FileMode.binary and len(data) != len(original_data):
            raise BinaryPrefixReplacementError(path, placeholder, new_prefix, len(original_data), len(data))
        return data
    updated = update_file_in_place_as_binary(realpath(path), _update_prefix)
    if updated and mode == FileMode.binary and (subdir == 'osx-arm64') and on_mac:
        subprocess.run(['/usr/bin/codesign', '-s', '-', '-f', realpath(path)], capture_output=True)

def replace_prefix(mode: FileMode, data: bytes, placeholder: str, new_prefix: str) -> bytes:
    if False:
        i = 10
        return i + 15
    '\n    Replaces `placeholder` text with the `new_prefix` provided. The `mode` provided can\n    either be text or binary.\n\n    We use the `POPULAR_ENCODINGS` module level constant defined above to make several\n    passes at replacing the placeholder. We do this to account for as many encodings as\n    possible. If this causes any performance problems in the future, it could potentially\n    be removed (i.e. just using the most popular "utf-8" encoding").\n\n    More information/discussion available here: https://github.com/conda/conda/pull/9946\n    '
    for encoding in POPULAR_ENCODINGS:
        if mode == FileMode.text:
            if not on_win:
                newline_pos = data.find(b'\n')
                if newline_pos > -1:
                    (shebang_line, rest_of_data) = (data[:newline_pos], data[newline_pos:])
                    shebang_placeholder = f'#!{placeholder}'.encode(encoding)
                    if shebang_placeholder in shebang_line:
                        escaped_shebang = f'#!{new_prefix}'.replace(' ', '\\ ').encode(encoding)
                        shebang_line = shebang_line.replace(shebang_placeholder, escaped_shebang)
                        data = shebang_line + rest_of_data
            data = data.replace(placeholder.encode(encoding), new_prefix.encode(encoding))
        elif mode == FileMode.binary:
            data = binary_replace(data, placeholder.encode(encoding), new_prefix.encode(encoding), encoding=encoding)
        else:
            raise CondaIOError('Invalid mode: %r' % mode)
    return data

def binary_replace(data: bytes, search: bytes, replacement: bytes, encoding: str='utf-8') -> bytes:
    if False:
        return 10
    '\n    Perform a binary replacement of `data`, where the placeholder `search` is\n    replaced with `replacement` and the remaining string is padded with null characters.\n    All input arguments are expected to be bytes objects.\n\n    Parameters\n    ----------\n    data:\n        The bytes object that will be searched and replaced\n    search:\n        The bytes object to find\n    replacement:\n        The bytes object that will replace `search`\n    encoding: str\n        The encoding of the expected string in the binary.\n    '
    zeros = '\x00'.encode(encoding)
    if on_win:
        if has_pyzzer_entry_point(data):
            return replace_pyzzer_entry_point_shebang(data, search, replacement)
        else:
            return data

    def replace(match):
        if False:
            for i in range(10):
                print('nop')
        occurrences = match.group().count(search)
        padding = (len(search) - len(replacement)) * occurrences
        if padding < 0:
            raise _PaddingError
        return match.group().replace(search, replacement) + b'\x00' * padding
    original_data_len = len(data)
    pat = re.compile(re.escape(search) + b'(?:(?!(?:' + zeros + b')).)*' + zeros)
    data = pat.sub(replace, data)
    assert len(data) == original_data_len
    return data

def has_pyzzer_entry_point(data):
    if False:
        return 10
    pos = data.rfind(b'PK\x05\x06')
    return pos >= 0

def replace_pyzzer_entry_point_shebang(all_data, placeholder, new_prefix):
    if False:
        while True:
            i = 10
    "Code adapted from pyzzer.  This is meant to deal with entry point exe's created by distlib,\n    which consist of a launcher, then a shebang, then a zip archive of the entry point code to run.\n    We need to change the shebang.\n    https://bitbucket.org/vinay.sajip/pyzzer/src/5d5740cb04308f067d5844a56fbe91e7a27efccc/pyzzer/__init__.py?at=default&fileviewer=file-view-default#__init__.py-112  # NOQA\n    "
    launcher = shebang = None
    pos = all_data.rfind(b'PK\x05\x06')
    if pos >= 0:
        end_cdr = all_data[pos + 12:pos + 20]
        (cdr_size, cdr_offset) = struct.unpack('<LL', end_cdr)
        arc_pos = pos - cdr_size - cdr_offset
        data = all_data[arc_pos:]
        if arc_pos > 0:
            pos = all_data.rfind(b'#!', 0, arc_pos)
            if pos >= 0:
                shebang = all_data[pos:arc_pos]
                if pos > 0:
                    launcher = all_data[:pos]
        if data and shebang and launcher:
            if hasattr(placeholder, 'encode'):
                placeholder = placeholder.encode('utf-8')
            if hasattr(new_prefix, 'encode'):
                new_prefix = new_prefix.encode('utf-8')
            shebang = shebang.replace(placeholder, new_prefix)
            all_data = b''.join([launcher, shebang, data])
    return all_data

def replace_long_shebang(mode, data):
    if False:
        i = 10
        return i + 15
    if mode == FileMode.text:
        if not isinstance(data, bytes):
            try:
                data = bytes(data, encoding='utf-8')
            except:
                data = data.encode('utf-8')
        shebang_match = re.match(SHEBANG_REGEX, data, re.MULTILINE)
        if shebang_match:
            (whole_shebang, executable, options) = shebang_match.groups()
            (prefix, executable_name) = executable.decode('utf-8').rsplit('/', 1)
            if len(whole_shebang) > MAX_SHEBANG_LENGTH or '\\ ' in prefix:
                new_shebang = f"#!/usr/bin/env {executable_name}{options.decode('utf-8')}"
                data = data.replace(whole_shebang, new_shebang.encode('utf-8'))
    else:
        pass
    return data

def generate_shebang_for_entry_point(executable, with_usr_bin_env=False):
    if False:
        return 10
    '\n    This function can be used to generate a shebang line for Python entry points.\n\n    Use cases:\n    - At install/link time, to generate the `noarch: python` entry points.\n    - conda init uses it to create its own entry point during conda-build\n    '
    shebang = f'#!{executable}\n'
    if os.environ.get('CONDA_BUILD') == '1' and '/_h_env_placehold' in executable:
        return shebang
    if len(shebang) > MAX_SHEBANG_LENGTH or ' ' in executable:
        if with_usr_bin_env:
            shebang = f'#!/usr/bin/env {basename(executable)}\n'
        else:
            shebang = dals(f"""\n                #!/bin/sh\n                '''exec' \"{executable}" "$0" "$@" #'''\n                """)
    return shebang