import os
import subprocess
import sys
from collections import defaultdict
from typing import Any, DefaultDict, Dict, FrozenSet, List, Tuple, Union
if __name__ == '__main__' and (not __package__):
    import __main__
    __main__.__package__ = 'gen'
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KeymapType = Dict[str, Tuple[str, Union[FrozenSet[str], str]]]

def resolve_keys(keymap: KeymapType) -> DefaultDict[str, List[str]]:
    if False:
        for i in range(10):
            print('nop')
    ans: DefaultDict[str, List[str]] = defaultdict(list)
    for (ch, (attr, atype)) in keymap.items():
        if isinstance(atype, str) and atype in ('int', 'uint'):
            q = atype
        else:
            q = 'flag'
        ans[q].append(ch)
    return ans

def enum(keymap: KeymapType) -> str:
    if False:
        return 10
    lines = []
    for (ch, (attr, atype)) in keymap.items():
        lines.append(f"{attr}='{ch}'")
    return '\n    enum KEYS {{\n        {}\n    }};\n    '.format(',\n'.join(lines))

def parse_key(keymap: KeymapType) -> str:
    if False:
        i = 10
        return i + 15
    lines = []
    for (attr, atype) in keymap.values():
        vs = atype.upper() if isinstance(atype, str) and atype in ('uint', 'int') else 'FLAG'
        lines.append(f'case {attr}: value_state = {vs}; break;')
    return '        \n'.join(lines)

def parse_flag(keymap: KeymapType, type_map: Dict[str, Any], command_class: str) -> str:
    if False:
        print('Hello World!')
    lines = []
    for ch in type_map['flag']:
        (attr, allowed_values) = keymap[ch]
        q = ' && '.join((f"g.{attr} != '{x}'" for x in sorted(allowed_values)))
        lines.append(f'\n            case {attr}: {{\n                g.{attr} = screen->parser_buf[pos++] & 0xff;\n                if ({q}) {{\n                    REPORT_ERROR("Malformed {command_class} control block, unknown flag value for {attr}: 0x%x", g.{attr});\n                    return;\n                }};\n            }}\n            break;\n        ')
    return '        \n'.join(lines)

def parse_number(keymap: KeymapType) -> Tuple[str, str]:
    if False:
        return 10
    int_keys = [f'I({attr})' for (attr, atype) in keymap.values() if atype == 'int']
    uint_keys = [f'U({attr})' for (attr, atype) in keymap.values() if atype == 'uint']
    return ('; '.join(int_keys), '; '.join(uint_keys))

def cmd_for_report(report_name: str, keymap: KeymapType, type_map: Dict[str, Any], payload_allowed: bool) -> str:
    if False:
        for i in range(10):
            print('nop')

    def group(atype: str, conv: str) -> Tuple[str, str]:
        if False:
            print('Hello World!')
        (flag_fmt, flag_attrs) = ([], [])
        cv = {'flag': 'c', 'int': 'i', 'uint': 'I'}[atype]
        for ch in type_map[atype]:
            flag_fmt.append(f's{cv}')
            attr = keymap[ch][0]
            flag_attrs.append(f'"{attr}", {conv}g.{attr}')
        return (' '.join(flag_fmt), ', '.join(flag_attrs))
    (flag_fmt, flag_attrs) = group('flag', '')
    (int_fmt, int_attrs) = group('int', '(int)')
    (uint_fmt, uint_attrs) = group('uint', '(unsigned int)')
    fmt = f'{flag_fmt} {uint_fmt} {int_fmt}'
    if payload_allowed:
        ans = [f'REPORT_VA_COMMAND("s {{{fmt} sI}} y#", "{report_name}",']
    else:
        ans = [f'REPORT_VA_COMMAND("s {{{fmt}}}", "{report_name}",']
    ans.append(',\n     '.join((flag_attrs, uint_attrs, int_attrs)))
    if payload_allowed:
        ans.append(', "payload_sz", g.payload_sz, payload, g.payload_sz')
    ans.append(');')
    return '\n'.join(ans)

def generate(function_name: str, callback_name: str, report_name: str, keymap: KeymapType, command_class: str, initial_key: str='a', payload_allowed: bool=True) -> str:
    if False:
        return 10
    type_map = resolve_keys(keymap)
    keys_enum = enum(keymap)
    handle_key = parse_key(keymap)
    flag_keys = parse_flag(keymap, type_map, command_class)
    (int_keys, uint_keys) = parse_number(keymap)
    report_cmd = cmd_for_report(report_name, keymap, type_map, payload_allowed)
    if payload_allowed:
        payload_after_value = "case ';': state = PAYLOAD; break;"
        payload = ', PAYLOAD'
        parr = 'static uint8_t payload[4096];'
        payload_case = f'\n            case PAYLOAD: {{\n                sz = screen->parser_buf_pos - pos;\n                g.payload_sz = sizeof(payload);\n                if (!base64_decode32(screen->parser_buf + pos, sz, payload, &g.payload_sz)) {{\n                    REPORT_ERROR("Failed to parse {command_class} command payload with error: payload size (%zu) too large", sz); return; }}\n                pos = screen->parser_buf_pos;\n                }}\n                break;\n        '
        callback = f'{callback_name}(screen, &g, payload)'
    else:
        payload_after_value = payload = parr = payload_case = ''
        callback = f'{callback_name}(screen, &g)'
    return f"""\nstatic inline void\n{function_name}(Screen *screen, PyObject UNUSED *dump_callback) {{\n    unsigned int pos = 1;\n    enum PARSER_STATES {{ KEY, EQUAL, UINT, INT, FLAG, AFTER_VALUE {payload} }};\n    enum PARSER_STATES state = KEY, value_state = FLAG;\n    static {command_class} g;\n    unsigned int i, code;\n    uint64_t lcode;\n    bool is_negative;\n    memset(&g, 0, sizeof(g));\n    size_t sz;\n    {parr}\n    {keys_enum}\n    enum KEYS key = '{initial_key}';\n    if (screen->parser_buf[pos] == ';') state = AFTER_VALUE;\n\n    while (pos < screen->parser_buf_pos) {{\n        switch(state) {{\n            case KEY:\n                key = screen->parser_buf[pos++];\n                state = EQUAL;\n                switch(key) {{\n                    {handle_key}\n                    default:\n                        REPORT_ERROR("Malformed {command_class} control block, invalid key character: 0x%x", key);\n                        return;\n                }}\n                break;\n\n            case EQUAL:\n                if (screen->parser_buf[pos++] != '=') {{\n                    REPORT_ERROR("Malformed {command_class} control block, no = after key, found: 0x%x instead", screen->parser_buf[pos-1]);\n                    return;\n                }}\n                state = value_state;\n                break;\n\n            case FLAG:\n                switch(key) {{\n                    {flag_keys}\n                    default:\n                        break;\n                }}\n                state = AFTER_VALUE;\n                break;\n\n            case INT:\n#define READ_UINT \\\n                for (i = pos; i < MIN(screen->parser_buf_pos, pos + 10); i++) {{ \\\n                    if (screen->parser_buf[i] < '0' || screen->parser_buf[i] > '9') break; \\\n                }} \\\n                if (i == pos) {{ REPORT_ERROR("Malformed {command_class} control block, expecting an integer value for key: %c", key & 0xFF); return; }} \\\n                lcode = utoi(screen->parser_buf + pos, i - pos); pos = i; \\\n                if (lcode > UINT32_MAX) {{ REPORT_ERROR("Malformed {command_class} control block, number is too large"); return; }} \\\n                code = lcode;\n\n                is_negative = false;\n                if(screen->parser_buf[pos] == '-') {{ is_negative = true; pos++; }}\n#define I(x) case x: g.x = is_negative ? 0 - (int32_t)code : (int32_t)code; break\n                READ_UINT;\n                switch(key) {{\n                    {int_keys};\n                    default: break;\n                }}\n                state = AFTER_VALUE;\n                break;\n#undef I\n            case UINT:\n                READ_UINT;\n#define U(x) case x: g.x = code; break\n                switch(key) {{\n                    {uint_keys};\n                    default: break;\n                }}\n                state = AFTER_VALUE;\n                break;\n#undef U\n#undef READ_UINT\n\n            case AFTER_VALUE:\n                switch (screen->parser_buf[pos++]) {{\n                    default:\n                        REPORT_ERROR("Malformed {command_class} control block, expecting a comma or semi-colon after a value, found: 0x%x",\n                                     screen->parser_buf[pos - 1]);\n                        return;\n                    case ',':\n                        state = KEY;\n                        break;\n                    {payload_after_value}\n                }}\n                break;\n\n            {payload_case}\n\n        }} // end switch\n    }} // end while\n\n    switch(state) {{\n        case EQUAL:\n            REPORT_ERROR("Malformed {command_class} control block, no = after key"); return;\n        case INT:\n        case UINT:\n            REPORT_ERROR("Malformed {command_class} control block, expecting an integer value"); return;\n        case FLAG:\n            REPORT_ERROR("Malformed {command_class} control block, expecting a flag value"); return;\n        default:\n            break;\n    }}\n\n    {report_cmd}\n\n    {callback};\n}}\n    """

def write_header(text: str, path: str) -> None:
    if False:
        return 10
    with open(path, 'w') as f:
        print(f'// This file is generated by {os.path.basename(__file__)} do not edit!', file=f, end='\n\n')
        print('#pragma once', file=f)
        print(text, file=f)
    subprocess.check_call(['clang-format', '-i', path])

def graphics_parser() -> None:
    if False:
        i = 10
        return i + 15
    flag = frozenset
    keymap: KeymapType = {'a': ('action', flag('tTqpdfac')), 'd': ('delete_action', flag('aAiIcCfFnNpPqQxXyYzZ')), 't': ('transmission_type', flag('dfts')), 'o': ('compressed', flag('z')), 'f': ('format', 'uint'), 'm': ('more', 'uint'), 'i': ('id', 'uint'), 'I': ('image_number', 'uint'), 'p': ('placement_id', 'uint'), 'q': ('quiet', 'uint'), 'w': ('width', 'uint'), 'h': ('height', 'uint'), 'x': ('x_offset', 'uint'), 'y': ('y_offset', 'uint'), 'v': ('data_height', 'uint'), 's': ('data_width', 'uint'), 'S': ('data_sz', 'uint'), 'O': ('data_offset', 'uint'), 'c': ('num_cells', 'uint'), 'r': ('num_lines', 'uint'), 'X': ('cell_x_offset', 'uint'), 'Y': ('cell_y_offset', 'uint'), 'z': ('z_index', 'int'), 'C': ('cursor_movement', 'uint'), 'U': ('unicode_placement', 'uint'), 'P': ('parent_id', 'uint'), 'Q': ('parent_placement_id', 'uint'), 'H': ('offset_from_parent_x', 'int'), 'V': ('offset_from_parent_y', 'int')}
    text = generate('parse_graphics_code', 'screen_handle_graphics_command', 'graphics_command', keymap, 'GraphicsCommand')
    write_header(text, 'kitty/parse-graphics-command.h')

def main(args: List[str]=sys.argv) -> None:
    if False:
        return 10
    graphics_parser()
if __name__ == '__main__':
    import runpy
    m = runpy.run_path(os.path.dirname(os.path.abspath(__file__)))
    m['main']([sys.executable, 'apc-parsers'])