import os.path
import urllib.request
from itertools import zip_longest
from .utils import toolkit
from ..utils.cells import split_graphemes
from ..utils.colors import GREEN, ORANGE, RED
CACHE = '.unicode_cache'

def validate_unicode_breaks(uver=None, show_all=False, cache=True):
    if False:
        return 10
    latest = f'{CACHE}/latest'
    if not uver and cache and os.path.exists(latest):
        with open(latest) as f:
            uver = f.read()
        print('using version "latest" as:', uver)
    file = f'{CACHE}/emoji-test_{uver}.txt'
    if cache and os.path.exists(file):
        print('loading cached:', file)
        with open(file) as f:
            data = f.read()
    else:
        url = f"https://www.unicode.org/Public/emoji/{uver or 'latest'}/emoji-test.txt"
        print('downloading:', url)
        try:
            req = urllib.request.urlopen(url)
        except OSError as e:
            print(RED('Download error:'), e)
            return
        os.makedirs(os.path.dirname(file), exist_ok=True)
        if not uver:
            new_url = req.geturl()
            uver = new_url.split('/')[5]
            print('saving version "latest" as:', uver)
            with open(latest, 'w') as f:
                f.write(uver)
            file = f'{CACHE}/emoji-test_{uver}.txt'
        print('saving:', file)
        data = req.read().decode('utf8')
        with open(file, 'w') as f:
            f.write(data)

    def where():
        if False:
            for i in range(10):
                print('nop')
        nonlocal groups
        if any(groups):
            print('\n'.join((g for g in groups if g)))
        groups = [None, None]

    def expect(*chars):
        if False:
            print('Hello World!')
        nonlocal errors, total
        text = ''.join(chars)
        actual = split_graphemes(text)
        error = actual != chars
        total += 1
        errors += error
        if error or show_all:
            where()
            codes = '|'.join(((GREEN if a == c else RED)(' '.join((hex(ord(c)).replace('0x', '') for c in a)) if a else '-') for (a, c) in zip_longest(actual, chars)))
            small_name = name.replace(' skin tone', '').replace(' hair', '')
            small_status = ''.join((x[0] for x in status.split('-')))
            (a_len, c_len) = (len(actual), len(chars))
            size = f'{GREEN(a_len)} ==' if a_len == c_len else f'{RED(a_len)} !='
            print(f" {char}   {text.replace(char, 'X'):>3}: {size} {c_len} -> |{codes}| {ORANGE(small_status)} {small_name}")
    (groups, total, errors) = ([None, None], 0, 0)
    for line in filter(None, data.splitlines()):
        if line.startswith('#'):
            if line.startswith('# group:'):
                groups[0] = line.split()[-1]
            elif line.startswith('# subgroup:'):
                groups[1] = f'  - {line.split()[-1]}'
            continue
        (p1, p2) = (p.split() for p in line.split(';'))
        (status, name) = (p2[0], ' '.join(p2[4:]))
        char = ''.join((chr(int(x, 16)) for x in p1))
        expect(char)
        expect(char, char)
        expect('a', char, 'a')
        expect('a', 'a', char)
        expect(char, 'a', 'a')
    print(f'\nerrors   : {errors / total:6.2%} [{errors}/{total}]')
    print(f'successes: {1 - errors / total:6.2%} [{total - errors}/{total}]')

def find_groups(data, max_diff):
    if False:
        for i in range(10):
            print('nop')
    "Group some numbers with a maximum difference between them.\n    I've used to try to fix the current grapheme break error.\n\n    Using version unicode 13.1:\n        Component\n          - skin-tone\n         🏻    XX: 1 != 2 -> |1f3fb 1f3fb|-| c light\n         🏻   aXa: 2 != 3 -> |61 1f3fb|61|-| c light\n         🏻   aaX: 2 != 3 -> |61|61 1f3fb|-| c light\n         🏼    XX: 1 != 2 -> |1f3fc 1f3fc|-| c medium-light\n         🏼   aXa: 2 != 3 -> |61 1f3fc|61|-| c medium-light\n         🏼   aaX: 2 != 3 -> |61|61 1f3fc|-| c medium-light\n         🏽    XX: 1 != 2 -> |1f3fd 1f3fd|-| c medium\n         🏽   aXa: 2 != 3 -> |61 1f3fd|61|-| c medium\n         🏽   aaX: 2 != 3 -> |61|61 1f3fd|-| c medium\n         🏾    XX: 1 != 2 -> |1f3fe 1f3fe|-| c medium-dark\n         🏾   aXa: 2 != 3 -> |61 1f3fe|61|-| c medium-dark\n         🏾   aaX: 2 != 3 -> |61|61 1f3fe|-| c medium-dark\n         🏿    XX: 1 != 2 -> |1f3ff 1f3ff|-| c dark\n         🏿   aXa: 2 != 3 -> |61 1f3ff|61|-| c dark\n         🏿   aaX: 2 != 3 -> |61|61 1f3ff|-| c dark\n\n    The codepoints that do accept a skin tone are:\n    0x0261D, 0x026F9, 0x0270A, 0x0270B, 0x0270C, 0x0270D, 0x1F385, 0x1F3C2, 0x1F3C3, 0x1F3C4,\n    0x1F3C7, 0x1F3CA, 0x1F3CB, 0x1F3CC, 0x1F442, 0x1F443, 0x1F446, 0x1F447, 0x1F448, 0x1F449,\n    0x1F44A, 0x1F44B, 0x1F44C, 0x1F44D, 0x1F44E, 0x1F44F, 0x1F450, 0x1F466, 0x1F467, 0x1F468,\n    0x1F469, 0x1F46B, 0x1F46C, 0x1F46D, 0x1F46E, 0x1F470, 0x1F471, 0x1F472, 0x1F473, 0x1F474,\n    0x1F475, 0x1F476, 0x1F477, 0x1F478, 0x1F47C, 0x1F481, 0x1F482, 0x1F483, 0x1F485, 0x1F486,\n    0x1F487, 0x1F4AA, 0x1F574, 0x1F575, 0x1F57A, 0x1F590, 0x1F595, 0x1F596, 0x1F645, 0x1F646,\n    0x1F647, 0x1F64B, 0x1F64C, 0x1F64D, 0x1F64E, 0x1F64F, 0x1F6A3, 0x1F6B4, 0x1F6B5, 0x1F6B6,\n    0x1F6C0, 0x1F6CC, 0x1F90C, 0x1F90F, 0x1F918, 0x1F919, 0x1F91A, 0x1F91B, 0x1F91C, 0x1F91E,\n    0x1F91F, 0x1F926, 0x1F930, 0x1F931, 0x1F932, 0x1F933, 0x1F934, 0x1F935, 0x1F936, 0x1F937,\n    0x1F938, 0x1F939, 0x1F93D, 0x1F93E, 0x1F977, 0x1F9B5, 0x1F9B6, 0x1F9B8, 0x1F9B9, 0x1F9BB,\n    0x1F9CD, 0x1F9CE, 0x1F9CF, 0x1F9D1, 0x1F9D2, 0x1F9D3, 0x1F9D4, 0x1F9D5, 0x1F9D6, 0x1F9D7,\n    0x1F9D8, 0x1F9D9, 0x1F9DA, 0x1F9DB, 0x1F9DC, 0x1F9DD\n\n    "
    it = iter(sorted(data))
    last_item = next(it)
    current_group = [last_item]
    result = [current_group]
    for i in it:
        if i - last_item > max_diff:
            current_group = []
            result.append(current_group)
        current_group.append(i)
        last_item = i
    print('\n'.join((f'{len(g)}:|' + ' '.join((hex(x).replace('0x', '') for x in g)) + '|' for g in result)))
if __name__ == '__main__':
    (parser, run) = toolkit('Tests the grapheme break implementation against some unicode version.')
    parser.add_argument('uver', type=float, nargs='?', help='the unicode version to be used')
    parser.add_argument('--all', dest='show_all', action='store_true', help='shows the correct cases, in addition to the wrong ones')
    parser.add_argument('--no-cache', dest='cache', action='store_false', help='ignores the cache and re-downloads the spec')
    run(validate_unicode_breaks)