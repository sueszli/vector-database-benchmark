import os
import string
import subprocess
import sys
from pprint import pformat
from typing import Any, Dict, List, Union
if __name__ == '__main__' and (not __package__):
    import __main__
    __main__.__package__ = 'gen'
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
functional_key_defs = '# {{{\n# kitty                     XKB                         macVK  macU\nescape                      Escape                      0x35   -\nenter                       Return                      0x24   NSCarriageReturnCharacter\ntab                         Tab                         0x30   NSTabCharacter\nbackspace                   BackSpace                   0x33   NSBackspaceCharacter\ninsert                      Insert                      0x72   Insert\ndelete                      Delete                      0x75   Delete\nleft                        Left                        0x7B   LeftArrow\nright                       Right                       0x7C   RightArrow\nup                          Up                          0x7E   UpArrow\ndown                        Down                        0x7D   DownArrow\npage_up                     Page_Up                     0x74   PageUp\npage_down                   Page_Down                   0x79   PageDown\nhome                        Home                        0x73   Home\nend                         End                         0x77   End\ncaps_lock                   Caps_Lock                   0x39   -\nscroll_lock                 Scroll_Lock                 -      ScrollLock\nnum_lock                    Num_Lock                    0x47   ClearLine\nprint_screen                Print                       -      PrintScreen\npause                       Pause                       -      Pause\nmenu                        Menu                        0x6E   Menu\nf1                          F1                          0x7A   F1\nf2                          F2                          0x78   F2\nf3                          F3                          0x63   F3\nf4                          F4                          0x76   F4\nf5                          F5                          0x60   F5\nf6                          F6                          0x61   F6\nf7                          F7                          0x62   F7\nf8                          F8                          0x64   F8\nf9                          F9                          0x65   F9\nf10                         F10                         0x6D   F10\nf11                         F11                         0x67   F11\nf12                         F12                         0x6F   F12\nf13                         F13                         0x69   F13\nf14                         F14                         0x6B   F14\nf15                         F15                         0x71   F15\nf16                         F16                         0x6A   F16\nf17                         F17                         0x40   F17\nf18                         F18                         0x4F   F18\nf19                         F19                         0x50   F19\nf20                         F20                         0x5A   F20\nf21                         F21                         -      F21\nf22                         F22                         -      F22\nf23                         F23                         -      F23\nf24                         F24                         -      F24\nf25                         F25                         -      F25\nf26                         F26                         -      F26\nf27                         F27                         -      F27\nf28                         F28                         -      F28\nf29                         F29                         -      F29\nf30                         F30                         -      F30\nf31                         F31                         -      F31\nf32                         F32                         -      F32\nf33                         F33                         -      F33\nf34                         F34                         -      F34\nf35                         F35                         -      F35\nkp_0                        KP_0                        0x52   -\nkp_1                        KP_1                        0x53   -\nkp_2                        KP_2                        0x54   -\nkp_3                        KP_3                        0x55   -\nkp_4                        KP_4                        0x56   -\nkp_5                        KP_5                        0x57   -\nkp_6                        KP_6                        0x58   -\nkp_7                        KP_7                        0x59   -\nkp_8                        KP_8                        0x5B   -\nkp_9                        KP_9                        0x5C   -\nkp_decimal                  KP_Decimal                  0x41   -\nkp_divide                   KP_Divide                   0x4B   -\nkp_multiply                 KP_Multiply                 0x43   -\nkp_subtract                 KP_Subtract                 0x4E   -\nkp_add                      KP_Add                      0x45   -\nkp_enter                    KP_Enter                    0x4C   NSEnterCharacter\nkp_equal                    KP_Equal                    0x51   -\nkp_separator                KP_Separator                -      -\nkp_left                     KP_Left                     -      -\nkp_right                    KP_Right                    -      -\nkp_up                       KP_Up                       -      -\nkp_down                     KP_Down                     -      -\nkp_page_up                  KP_Page_Up                  -      -\nkp_page_down                KP_Page_Down                -      -\nkp_home                     KP_Home                     -      -\nkp_end                      KP_End                      -      -\nkp_insert                   KP_Insert                   -      -\nkp_delete                   KP_Delete                   -      -\nkp_begin                    KP_Begin                    -      -\nmedia_play                  XF86AudioPlay               -      -\nmedia_pause                 XF86AudioPause              -      -\nmedia_play_pause            -                           -      -\nmedia_reverse               -                           -      -\nmedia_stop                  XF86AudioStop               -      -\nmedia_fast_forward          XF86AudioForward            -      -\nmedia_rewind                XF86AudioRewind             -      -\nmedia_track_next            XF86AudioNext               -      -\nmedia_track_previous        XF86AudioPrev               -      -\nmedia_record                XF86AudioRecord             -      -\nlower_volume                XF86AudioLowerVolume        -      -\nraise_volume                XF86AudioRaiseVolume        -      -\nmute_volume                 XF86AudioMute               -      -\nleft_shift                  Shift_L                     0x38   -\nleft_control                Control_L                   0x3B   -\nleft_alt                    Alt_L                       0x3A   -\nleft_super                  Super_L                     0x37   -\nleft_hyper                  Hyper_L                     -      -\nleft_meta                   Meta_L                      -      -\nright_shift                 Shift_R                     0x3C   -\nright_control               Control_R                   0x3E   -\nright_alt                   Alt_R                       0x3D   -\nright_super                 Super_R                     0x36   -\nright_hyper                 Hyper_R                     -      -\nright_meta                  Meta_R                      -      -\niso_level3_shift            ISO_Level3_Shift            -      -\niso_level5_shift            ISO_Level5_Shift            -      -\n'
shift_map = {x[0]: x[1] for x in '`~ 1! 2@ 3# 4$ 5% 6^ 7& 8* 9( 0) -_ =+ [{ ]} \\| ;: \'" ,< .> /?'.split()}
shift_map.update({x: x.upper() for x in string.ascii_lowercase})
functional_encoding_overrides = {'insert': 2, 'delete': 3, 'page_up': 5, 'page_down': 6, 'home': 7, 'end': 8, 'tab': 9, 'f1': 11, 'f2': 12, 'f3': 13, 'enter': 13, 'f4': 14, 'f5': 15, 'f6': 17, 'f7': 18, 'f8': 19, 'f9': 20, 'f10': 21, 'f11': 23, 'f12': 24, 'escape': 27, 'backspace': 127}
different_trailer_functionals = {'up': 'A', 'down': 'B', 'right': 'C', 'left': 'D', 'kp_begin': 'E', 'end': 'F', 'home': 'H', 'f1': 'P', 'f2': 'Q', 'f3': '~', 'f4': 'S', 'enter': 'u', 'tab': 'u', 'backspace': 'u', 'escape': 'u'}
macos_ansi_key_codes = {29: ord('0'), 18: ord('1'), 19: ord('2'), 20: ord('3'), 21: ord('4'), 23: ord('5'), 22: ord('6'), 26: ord('7'), 28: ord('8'), 25: ord('9'), 0: ord('a'), 11: ord('b'), 8: ord('c'), 2: ord('d'), 14: ord('e'), 3: ord('f'), 5: ord('g'), 4: ord('h'), 34: ord('i'), 38: ord('j'), 40: ord('k'), 37: ord('l'), 46: ord('m'), 45: ord('n'), 31: ord('o'), 35: ord('p'), 12: ord('q'), 15: ord('r'), 1: ord('s'), 17: ord('t'), 32: ord('u'), 9: ord('v'), 13: ord('w'), 7: ord('x'), 16: ord('y'), 6: ord('z'), 39: ord("'"), 42: ord('\\'), 43: ord(','), 24: ord('='), 50: ord('`'), 33: ord('['), 27: ord('-'), 47: ord('.'), 30: ord(']'), 41: ord(';'), 44: ord('/'), 49: ord(' ')}
functional_key_names: List[str] = []
name_to_code: Dict[str, int] = {}
name_to_xkb: Dict[str, str] = {}
name_to_vk: Dict[str, int] = {}
name_to_macu: Dict[str, str] = {}
start_code = 57344
for line in functional_key_defs.splitlines():
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    parts = line.split()
    name = parts[0]
    functional_key_names.append(name)
    name_to_code[name] = len(name_to_code) + start_code
    if parts[1] != '-':
        name_to_xkb[name] = parts[1]
    if parts[2] != '-':
        name_to_vk[name] = int(parts[2], 16)
    if parts[3] != '-':
        val = parts[3]
        if not val.startswith('NS'):
            val = f'NS{val}FunctionKey'
        name_to_macu[name] = val
last_code = start_code + len(functional_key_names) - 1
ctrl_mapping = {' ': 0, '@': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26, '[': 27, '\\': 28, ']': 29, '^': 30, '~': 30, '/': 31, '_': 31, '?': 127, '0': 48, '1': 49, '2': 0, '3': 27, '4': 28, '5': 29, '6': 30, '7': 31, '8': 127, '9': 57}

def patch_file(path: str, what: str, text: str, start_marker: str='/* ', end_marker: str=' */') -> None:
    if False:
        for i in range(10):
            print('nop')
    simple_start_q = f'{start_marker}start {what}{end_marker}'
    start_q = f'{start_marker}start {what} (auto generated by gen-key-constants.py do not edit){end_marker}'
    end_q = f'{start_marker}end {what}{end_marker}'
    with open(path, 'r+') as f:
        raw = f.read()
        try:
            start = raw.index(start_q)
        except ValueError:
            try:
                start = raw.index(simple_start_q)
            except ValueError:
                raise SystemExit(f'Failed to find "{simple_start_q}" in {path}')
        try:
            end = raw.index(end_q)
        except ValueError:
            raise SystemExit(f'Failed to find "{end_q}" in {path}')
        raw = f'{raw[:start]}{start_q}\n{text}\n{raw[end:]}'
        f.seek(0)
        f.truncate(0)
        f.write(raw)
    if path.endswith('.go'):
        subprocess.check_call(['go', 'fmt', path])

def serialize_dict(x: Dict[Any, Any]) -> str:
    if False:
        i = 10
        return i + 15
    return pformat(x, indent=4).replace('{', '{\n ', 1)

def serialize_go_dict(x: Union[Dict[str, int], Dict[int, str], Dict[int, int]]) -> str:
    if False:
        i = 10
        return i + 15
    ans = []

    def s(x: Union[int, str]) -> str:
        if False:
            print('Hello World!')
        if isinstance(x, int):
            return str(x)
        return f'"{x}"'
    for (k, v) in x.items():
        ans.append(f'{s(k)}: {s(v)}')
    return '{' + ', '.join(ans) + '}'

def generate_glfw_header() -> None:
    if False:
        for i in range(10):
            print('nop')
    lines = ['typedef enum {', f'  GLFW_FKEY_FIRST = 0x{start_code:x}u,']
    (klines, pyi, names, knames) = ([], [], [], [])
    for (name, code) in name_to_code.items():
        lines.append(f'  GLFW_FKEY_{name.upper()} = 0x{code:x}u,')
        klines.append(f'    ADDC(GLFW_FKEY_{name.upper()});')
        pyi.append(f'GLFW_FKEY_{name.upper()}: int')
        names.append(f'    case GLFW_FKEY_{name.upper()}: return "{name.upper()}";')
        knames.append(f'            case GLFW_FKEY_{name.upper()}: return PyUnicode_FromString("{name}");')
    lines.append(f'  GLFW_FKEY_LAST = 0x{last_code:x}u')
    lines.append('} GLFWFunctionKey;')
    patch_file('glfw/glfw3.h', 'functional key names', '\n'.join(lines))
    patch_file('kitty/glfw.c', 'glfw functional keys', '\n'.join(klines))
    patch_file('kitty/fast_data_types.pyi', 'glfw functional keys', '\n'.join(pyi), start_marker='# ', end_marker='')
    patch_file('glfw/input.c', 'functional key names', '\n'.join(names))
    patch_file('kitty/glfw.c', 'glfw functional key names', '\n'.join(knames))

def generate_xkb_mapping() -> None:
    if False:
        for i in range(10):
            print('nop')
    (lines, rlines) = ([], [])
    for (name, xkb) in name_to_xkb.items():
        lines.append(f'        case XKB_KEY_{xkb}: return GLFW_FKEY_{name.upper()};')
        rlines.append(f'        case GLFW_FKEY_{name.upper()}: return XKB_KEY_{xkb};')
    patch_file('glfw/xkb_glfw.c', 'xkb to glfw', '\n'.join(lines))
    patch_file('glfw/xkb_glfw.c', 'glfw to xkb', '\n'.join(rlines))

def generate_functional_table() -> None:
    if False:
        for i in range(10):
            print('nop')
    lines = ['', '.. csv-table:: Functional key codes', '   :header: "Name", "CSI", "Name", "CSI"', '']
    line_items = []
    enc_lines = []
    tilde_trailers = set()
    for (name, code) in name_to_code.items():
        if name in functional_encoding_overrides or name in different_trailer_functionals:
            trailer = different_trailer_functionals.get(name, '~')
            if trailer == '~':
                tilde_trailers.add(code)
            code = oc = functional_encoding_overrides.get(name, code)
            code = code if trailer in '~u' else 1
            enc_lines.append(' ' * 8 + f"case GLFW_FKEY_{name.upper()}: S({code}, '{trailer}');")
            if code == 1 and name not in ('up', 'down', 'left', 'right'):
                trailer += f' or {oc} ~'
        else:
            trailer = 'u'
        line_items.append(name.upper())
        line_items.append(f'``{code}\xa0{trailer}``')
    for li in chunks(line_items, 4):
        lines.append('   ' + ', '.join((f'"{x}"' for x in li)))
    lines.append('')
    patch_file('docs/keyboard-protocol.rst', 'functional key table', '\n'.join(lines), start_marker='.. ', end_marker='')
    patch_file('kitty/key_encoding.c', 'special numbers', '\n'.join(enc_lines))
    code_to_name = {v: k.upper() for (k, v) in name_to_code.items()}
    csi_map = {v: name_to_code[k] for (k, v) in functional_encoding_overrides.items()}
    letter_trailer_codes: Dict[str, int] = {v: functional_encoding_overrides.get(k, name_to_code.get(k, 0)) for (k, v) in different_trailer_functionals.items() if v in 'ABCDEHFPQRSZ'}
    text = f'functional_key_number_to_name_map = {serialize_dict(code_to_name)}'
    text += f'\ncsi_number_to_functional_number_map = {serialize_dict(csi_map)}'
    text += f'\nletter_trailer_to_csi_number_map = {letter_trailer_codes!r}'
    text += f'\ntilde_trailers = {tilde_trailers!r}'
    patch_file('kitty/key_encoding.py', 'csi mapping', text, start_marker='# ', end_marker='')
    text = f'var functional_key_number_to_name_map = map[int]string{serialize_go_dict(code_to_name)}\n'
    text += f'\nvar csi_number_to_functional_number_map = map[int]int{serialize_go_dict(csi_map)}\n'
    text += f'\nvar letter_trailer_to_csi_number_map = map[string]int{serialize_go_dict(letter_trailer_codes)}\n'
    tt = ', '.join((f'{x}: true' for x in tilde_trailers))
    text += '\nvar tilde_trailers = map[int]bool{' + f'{tt}' + '}\n'
    patch_file('tools/tui/loop/key-encoding.go', 'csi mapping', text, start_marker='// ', end_marker='')

def generate_legacy_text_key_maps() -> None:
    if False:
        i = 10
        return i + 15
    tests = []
    tp = ' ' * 8
    (shift, alt, ctrl) = (1, 2, 4)

    def simple(c: str) -> None:
        if False:
            while True:
                i = 10
        shifted = shift_map.get(c, c)
        ctrled = chr(ctrl_mapping.get(c, ord(c)))
        call = f'enc(ord({c!r}), shifted_key=ord({shifted!r})'
        for m in range(16):
            if m == 0:
                tests.append(f'{tp}ae({call}), {c!r})')
            elif m == shift:
                tests.append(f'{tp}ae({call}, mods=shift), {shifted!r})')
            elif m == alt:
                tests.append(f'{tp}ae({call}, mods=alt), "\\x1b" + {c!r})')
            elif m == ctrl:
                tests.append(f'{tp}ae({call}, mods=ctrl), {ctrled!r})')
            elif m == shift | alt:
                tests.append(f'{tp}ae({call}, mods=shift | alt), "\\x1b" + {shifted!r})')
            elif m == ctrl | alt:
                tests.append(f'{tp}ae({call}, mods=ctrl | alt), "\\x1b" + {ctrled!r})')
    for k in shift_map:
        simple(k)
    patch_file('kitty_tests/keys.py', 'legacy letter tests', '\n'.join(tests), start_marker='# ', end_marker='')

def chunks(lst: List[Any], n: int) -> Any:
    if False:
        print('Hello World!')
    'Yield successive n-sized chunks from lst.'
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def generate_ctrl_mapping() -> None:
    if False:
        i = 10
        return i + 15
    lines = ['.. csv-table:: Emitted bytes when :kbd:`ctrl` is held down and a key is pressed', '   :header: "Key", "Byte", "Key", "Byte", "Key", "Byte"', '']
    items = []
    mi = []
    for k in sorted(ctrl_mapping):
        prefix = '\\' if k == '\\' else 'SPC' if k == ' ' else ''
        items.append(prefix + k)
        val = str(ctrl_mapping[k])
        items.append(val)
        if k in "\\'":
            k = f'\\{k}'
        mi.append(f"        case '{k}': return {val};")
    for line_items in chunks(items, 6):
        lines.append('   ' + ', '.join((f'"{x}"' for x in line_items)))
    lines.append('')
    patch_file('docs/keyboard-protocol.rst', 'ctrl mapping', '\n'.join(lines), start_marker='.. ', end_marker='')
    patch_file('kitty/key_encoding.c', 'ctrl mapping', '\n'.join(mi))

def generate_macos_mapping() -> None:
    if False:
        return 10
    lines = []
    for k in sorted(macos_ansi_key_codes):
        v = macos_ansi_key_codes[k]
        lines.append(f'        case 0x{k:x}: return 0x{v:x};')
    patch_file('glfw/cocoa_window.m', 'vk to unicode', '\n'.join(lines))
    lines = []
    for (name, vk) in name_to_vk.items():
        lines.append(f'        case 0x{vk:x}: return GLFW_FKEY_{name.upper()};')
    patch_file('glfw/cocoa_window.m', 'vk to functional', '\n'.join(lines))
    lines = []
    for (name, mac) in name_to_macu.items():
        lines.append(f'        case {mac}: return GLFW_FKEY_{name.upper()};')
    patch_file('glfw/cocoa_window.m', 'macu to functional', '\n'.join(lines))
    lines = []
    for (name, mac) in name_to_macu.items():
        lines.append(f'        case GLFW_FKEY_{name.upper()}: return {mac};')
    patch_file('glfw/cocoa_window.m', 'functional to macu', '\n'.join(lines))

def main(args: List[str]=sys.argv) -> None:
    if False:
        i = 10
        return i + 15
    generate_glfw_header()
    generate_xkb_mapping()
    generate_functional_table()
    generate_legacy_text_key_maps()
    generate_ctrl_mapping()
    generate_macos_mapping()
if __name__ == '__main__':
    import runpy
    m = runpy.run_path(os.path.dirname(os.path.abspath(__file__)))
    m['main']([sys.executable, 'key-constants'])