"""This module contains pure-Python classes and functions for replacing
paths inside text files and binaries."""
import re
from collections import OrderedDict
from typing import Dict, Union
import spack.error
Prefix = Union[str, bytes]

def encode_path(p: Prefix) -> bytes:
    if False:
        i = 10
        return i + 15
    return p if isinstance(p, bytes) else p.encode('utf-8')

def _prefix_to_prefix_as_bytes(prefix_to_prefix) -> Dict[bytes, bytes]:
    if False:
        i = 10
        return i + 15
    return OrderedDict(((encode_path(k), encode_path(v)) for (k, v) in prefix_to_prefix.items()))

def utf8_path_to_binary_regex(prefix: str):
    if False:
        return 10
    'Create a binary regex that matches the input path in utf8'
    prefix_bytes = re.escape(prefix).encode('utf-8')
    return re.compile(b'(?<![\\w\\-_/])([\\w\\-_]*?)%s([\\w\\-_/]*)' % prefix_bytes)

def _byte_strings_to_single_binary_regex(prefixes):
    if False:
        for i in range(10):
            print('nop')
    all_prefixes = b'|'.join((re.escape(p) for p in prefixes))
    return re.compile(b'(?<![\\w\\-_/])([\\w\\-_]*?)(%s)([\\w\\-_/]*)' % all_prefixes)

def utf8_paths_to_single_binary_regex(prefixes):
    if False:
        while True:
            i = 10
    'Create a (binary) regex that matches any input path in utf8'
    return _byte_strings_to_single_binary_regex((p.encode('utf-8') for p in prefixes))

def filter_identity_mappings(prefix_to_prefix):
    if False:
        i = 10
        return i + 15
    'Drop mappings that are not changed.'
    return OrderedDict(((k, v) for (k, v) in prefix_to_prefix.items() if k != v))

class PrefixReplacer:
    """Base class for applying a prefix to prefix map
    to a list of binaries or text files.
    Child classes implement _apply_to_file to do the
    actual work, which is different when it comes to
    binaries and text files."""

    def __init__(self, prefix_to_prefix: Dict[bytes, bytes]):
        if False:
            print('Hello World!')
        '\n        Arguments:\n\n            prefix_to_prefix (OrderedDict):\n\n                A ordered mapping from prefix to prefix. The order is\n                relevant to support substring fallbacks, for example\n                [("/first/sub", "/x"), ("/first", "/y")] will ensure\n                /first/sub is matched and replaced before /first.\n        '
        self.prefix_to_prefix = filter_identity_mappings(prefix_to_prefix)

    @property
    def is_noop(self) -> bool:
        if False:
            print('Hello World!')
        'Returns true when the prefix to prefix map\n        is mapping everything to the same location (identity)\n        or there are no prefixes to replace.'
        return not self.prefix_to_prefix

    def apply(self, filenames: list):
        if False:
            while True:
                i = 10
        'Returns a list of files that were modified'
        changed_files = []
        if self.is_noop:
            return []
        for filename in filenames:
            if self.apply_to_filename(filename):
                changed_files.append(filename)
        return changed_files

    def apply_to_filename(self, filename):
        if False:
            for i in range(10):
                print('nop')
        if self.is_noop:
            return False
        with open(filename, 'rb+') as f:
            return self.apply_to_file(f)

    def apply_to_file(self, f):
        if False:
            print('Hello World!')
        if self.is_noop:
            return False
        return self._apply_to_file(f)

class TextFilePrefixReplacer(PrefixReplacer):
    """This class applies prefix to prefix mappings for relocation
    on text files.

    Note that UTF-8 encoding is assumed."""

    def __init__(self, prefix_to_prefix: Dict[bytes, bytes]):
        if False:
            while True:
                i = 10
        '\n        prefix_to_prefix (OrderedDict): OrderedDictionary where the keys are\n            bytes representing the old prefixes and the values are the new.\n        '
        super().__init__(prefix_to_prefix)
        self.regex = _byte_strings_to_single_binary_regex(self.prefix_to_prefix.keys())

    @classmethod
    def from_strings_or_bytes(cls, prefix_to_prefix: Dict[Prefix, Prefix]) -> 'TextFilePrefixReplacer':
        if False:
            i = 10
            return i + 15
        'Create a TextFilePrefixReplacer from an ordered prefix to prefix map.'
        return cls(_prefix_to_prefix_as_bytes(prefix_to_prefix))

    def _apply_to_file(self, f):
        if False:
            print('Hello World!')
        'Text replacement implementation simply reads the entire file\n        in memory and applies the combined regex.'
        replacement = lambda m: m.group(1) + self.prefix_to_prefix[m.group(2)] + m.group(3)
        data = f.read()
        new_data = re.sub(self.regex, replacement, data)
        if id(data) == id(new_data):
            return False
        f.seek(0)
        f.write(new_data)
        f.truncate()
        return True

class BinaryFilePrefixReplacer(PrefixReplacer):

    def __init__(self, prefix_to_prefix, suffix_safety_size=7):
        if False:
            while True:
                i = 10
        '\n        prefix_to_prefix (OrderedDict): OrderedDictionary where the keys are\n            bytes representing the old prefixes and the values are the new\n        suffix_safety_size (int): in case of null terminated strings, what size\n            of the suffix should remain to avoid aliasing issues?\n        '
        assert suffix_safety_size >= 0
        super().__init__(prefix_to_prefix)
        self.suffix_safety_size = suffix_safety_size
        self.regex = self.binary_text_regex(self.prefix_to_prefix.keys(), suffix_safety_size)

    @classmethod
    def binary_text_regex(cls, binary_prefixes, suffix_safety_size=7):
        if False:
            i = 10
            return i + 15
        '\n        Create a regex that looks for exact matches of prefixes, and also tries to\n        match a C-string type null terminator in a small lookahead window.\n\n        Arguments:\n            binary_prefixes (list): List of byte strings of prefixes to match\n            suffix_safety_size (int): Sizeof the lookahed for null-terminated string.\n\n        Returns: compiled regex\n        '
        return re.compile(b'(' + b'|'.join((re.escape(p) for p in binary_prefixes)) + b')([^\x00]{0,%d}\x00)?' % suffix_safety_size)

    @classmethod
    def from_strings_or_bytes(cls, prefix_to_prefix: Dict[Prefix, Prefix], suffix_safety_size: int=7) -> 'BinaryFilePrefixReplacer':
        if False:
            for i in range(10):
                print('nop')
        'Create a BinaryFilePrefixReplacer from an ordered prefix to prefix map.\n\n        Arguments:\n            prefix_to_prefix (OrderedDict): Ordered mapping of prefix to prefix.\n            suffix_safety_size (int): Number of bytes to retain at the end of a C-string\n                to avoid binary string-aliasing issues.\n        '
        return cls(_prefix_to_prefix_as_bytes(prefix_to_prefix), suffix_safety_size)

    def _apply_to_file(self, f):
        if False:
            i = 10
            return i + 15
        "\n        Given a file opened in rb+ mode, apply the string replacements as\n        specified by an ordered dictionary of prefix to prefix mappings. This\n        method takes special care of null-terminated C-strings. C-string constants\n        are problematic because compilers and linkers optimize readonly strings for\n        space by aliasing those that share a common suffix (only suffix since all\n        of them are null terminated). See https://github.com/spack/spack/pull/31739\n        and https://github.com/spack/spack/pull/32253 for details. Our logic matches\n        the original prefix with a ``suffix_safety_size + 1`` lookahead for null bytes.\n        If no null terminator is found, we simply pad with leading /, assuming that\n        it's a long C-string; the full C-string after replacement has a large suffix\n        in common with its original value.\n        If there *is* a null terminator we can do the same as long as the replacement\n        has a sufficiently long common suffix with the original prefix.\n        As a last resort when the replacement does not have a long enough common suffix,\n        we can try to shorten the string, but this only works if the new length is\n        sufficiently short (typically the case when going from large padding -> normal path)\n        If the replacement string is longer, or all of the above fails, we error out.\n\n        Arguments:\n            f: file opened in rb+ mode\n\n        Returns:\n            bool: True if file was modified\n        "
        assert f.tell() == 0
        modified = True
        for match in self.regex.finditer(f.read()):
            old = match.group(1)
            new = self.prefix_to_prefix[old]
            null_terminated = match.end(0) > match.end(1)
            suffix_strlen = match.end(0) - match.end(1) - 1
            bytes_shorter = len(old) - len(new)
            if bytes_shorter < 0:
                raise CannotGrowString(old, new)
            elif not null_terminated or suffix_strlen >= self.suffix_safety_size or old[-self.suffix_safety_size + suffix_strlen:] == new[-self.suffix_safety_size + suffix_strlen:]:
                replacement = b'/' * bytes_shorter + new
            elif bytes_shorter > self.suffix_safety_size:
                replacement = new + match.group(2)
            else:
                raise CannotShrinkCString(old, new, match.group()[:-1])
            f.seek(match.start())
            f.write(replacement)
            modified = True
        return modified

class BinaryStringReplacementError(spack.error.SpackError):

    def __init__(self, file_path, old_len, new_len):
        if False:
            return 10
        'The size of the file changed after binary path substitution\n\n        Args:\n            file_path (str): file with changing size\n            old_len (str): original length of the file\n            new_len (str): length of the file after substitution\n        '
        super().__init__('Doing a binary string replacement in %s failed.\nThe size of the file changed from %s to %s\nwhen it should have remanined the same.' % (file_path, old_len, new_len))

class BinaryTextReplaceError(spack.error.SpackError):

    def __init__(self, msg):
        if False:
            print('Hello World!')
        msg += ' To fix this, compile with more padding (config:install_tree:padded_length), or install to a shorter prefix.'
        super().__init__(msg)

class CannotGrowString(BinaryTextReplaceError):

    def __init__(self, old, new):
        if False:
            return 10
        msg = 'Cannot replace {!r} with {!r} because the new prefix is longer.'.format(old, new)
        super().__init__(msg)

class CannotShrinkCString(BinaryTextReplaceError):

    def __init__(self, old, new, full_old_string):
        if False:
            for i in range(10):
                print('nop')
        msg = 'Cannot replace {!r} with {!r} in the C-string {!r}.'.format(old, new, full_old_string)
        super().__init__(msg)