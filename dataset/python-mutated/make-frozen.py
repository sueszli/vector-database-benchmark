from __future__ import print_function
import sys
import os

def module_name(f):
    if False:
        while True:
            i = 10
    return f
modules = []
if len(sys.argv) > 1:
    root = sys.argv[1].rstrip('/')
    root_len = len(root)
for (dirpath, dirnames, filenames) in os.walk(root):
    for f in filenames:
        fullpath = dirpath + '/' + f
        st = os.stat(fullpath)
        modules.append((fullpath[root_len + 1:], st))
print('#include <stdint.h>')
print('const char mp_frozen_str_names[] = {')
for (f, st) in modules:
    m = module_name(f)
    print('"%s\\0"' % m)
print('"\\0"};')
print('const uint32_t mp_frozen_str_sizes[] = {')
for (f, st) in modules:
    print('%d,' % st.st_size)
print('0};')
print('const char mp_frozen_str_content[] = {')
for (f, st) in modules:
    data = open(sys.argv[1] + '/' + f, 'rb').read()
    data = bytearray(data)
    esc_dict = {ord('\n'): '\\n', ord('\r'): '\\r', ord('"'): '\\"', ord('\\'): '\\\\'}
    chrs = ['"']
    break_str = False
    for c in data:
        try:
            chrs.append(esc_dict[c])
        except KeyError:
            if 32 <= c <= 126:
                if break_str:
                    chrs.append('" "')
                    break_str = False
                chrs.append(chr(c))
            else:
                chrs.append('\\x%02x' % c)
                break_str = True
    chrs.append('\\0"')
    print(''.join(chrs))
print('"\\0"};')