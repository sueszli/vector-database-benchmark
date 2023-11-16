import functools
import re
import os
import sys
(cythonpath, _) = os.path.split(os.path.realpath(__file__))
(cythonpath, _) = os.path.split(cythonpath)
if os.path.exists(os.path.join(cythonpath, 'Cython')):
    sys.path.insert(0, cythonpath)
    print('Found (and using) local cython directory')
from Cython.Compiler import Lexicon

def main():
    if False:
        i = 10
        return i + 15
    arg = '--overwrite'
    if len(sys.argv) == 2:
        arg = sys.argv[1]
    if len(sys.argv) > 2 or arg not in ['--overwrite', '--here']:
        print('Call the script with either:\n  --overwrite    to update the existing Lexicon.py file (default)\n  --here         to create an version of Lexicon.py in the current directory\n')
        return
    generated_code = f"# Generated with 'cython-generate-lexicon.py' based on:\n# {sys.implementation.name} {sys.version.splitlines()[0].strip()}\n\n{generate_character_sets()}\n"
    print('Reading file', Lexicon.__file__)
    with open(Lexicon.__file__, 'r') as f:
        parts = re.split('(# (?:BEGIN|END) GENERATED CODE\\n?)', f.read())
    if len(parts) not in (4, 5) or ' GENERATED CODE' not in parts[1] or ' GENERATED CODE' not in parts[3]:
        print('Warning: generated code section not found - code not inserted')
        return
    parts[2] = generated_code
    output = ''.join(parts)
    if arg == '--here':
        outfile = 'Lexicon.py'
    else:
        assert arg == '--overwrite'
        outfile = Lexicon.__file__
    print('Writing to file', outfile)
    with open(outfile, 'w') as f:
        f.write(output)

@functools.lru_cache()
def get_start_characters_as_number():
    if False:
        while True:
            i = 10
    return [i for i in range(sys.maxunicode) if str.isidentifier(chr(i))]

def get_continue_characters_as_number():
    if False:
        while True:
            i = 10
    return [i for i in range(sys.maxunicode) if str.isidentifier('a' + chr(i))]

def get_continue_not_start_as_number():
    if False:
        i = 10
        return i + 15
    start = get_start_characters_as_number()
    cont = get_continue_characters_as_number()
    assert set(start) <= set(cont), 'We assume that all identifier start characters are also continuation characters.'
    return sorted(set(cont).difference(start))

def to_ranges(char_num_list):
    if False:
        for i in range(10):
            print('nop')
    char_num_list = sorted(char_num_list)
    first_good_val = char_num_list[0]
    single_chars = []
    ranges = []
    for n in range(1, len(char_num_list)):
        if char_num_list[n] - 1 != char_num_list[n - 1]:
            if first_good_val == char_num_list[n - 1]:
                single_chars.append(chr(char_num_list[n - 1]))
            else:
                ranges.append(chr(first_good_val) + chr(char_num_list[n - 1]))
            first_good_val = char_num_list[n]
    return (''.join(single_chars), ''.join(ranges))

def escape_chars(chars):
    if False:
        i = 10
        return i + 15
    escapes = []
    for char in chars:
        charval = ord(char)
        escape = f'\\U{charval:08x}' if charval > 65535 else f'\\u{charval:04x}'
        escapes.append(escape)
    return ''.join(escapes)

def make_split_strings(chars, splitby=113, indent='    '):
    if False:
        i = 10
        return i + 15
    splitby //= 10
    lines = [f'u"{escape_chars(chars[i:i + splitby])}"' for i in range(0, len(chars), splitby)]
    return indent + f'\n{indent}'.join(lines)

def generate_character_sets():
    if False:
        for i in range(10):
            print('nop')
    declarations = []
    for (char_type, char_generator) in [('unicode_start_ch', get_start_characters_as_number), ('unicode_continuation_ch', get_continue_not_start_as_number)]:
        for (set_type, chars) in zip(('any', 'range'), to_ranges(char_generator())):
            declarations.append(f'{char_type}_{set_type} = (\n{make_split_strings(chars)}\n)\n')
    return ''.join(declarations)
if __name__ == '__main__':
    main()