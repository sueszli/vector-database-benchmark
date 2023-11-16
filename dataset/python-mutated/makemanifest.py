from __future__ import print_function
import sys
import os
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../mpy-cross'))
import mpy_cross
import manifestfile
VARS = {}

class FreezeError(Exception):
    pass

def system(cmd):
    if False:
        print('Hello World!')
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        return (0, output)
    except subprocess.CalledProcessError as er:
        return (-1, er.output)

def get_timestamp(path, default=None):
    if False:
        for i in range(10):
            print('nop')
    try:
        stat = os.stat(path)
        return stat.st_mtime
    except OSError:
        if default is None:
            raise FreezeError('cannot stat {}'.format(path))
        return default

def mkdir(filename):
    if False:
        return 10
    path = os.path.dirname(filename)
    if not os.path.isdir(path):
        os.makedirs(path)

def generate_frozen_str_content(modules):
    if False:
        i = 10
        return i + 15
    output = [b'#include <stdint.h>\n', b'#define MP_FROZEN_STR_NAMES \\\n']
    for (_, target_path) in modules:
        print('STR', target_path)
        output.append(b'"%s\\0" \\\n' % target_path.encode())
    output.append(b'\n')
    output.append(b'const uint32_t mp_frozen_str_sizes[] = { ')
    for (full_path, _) in modules:
        st = os.stat(full_path)
        output.append(b'%d, ' % st.st_size)
    output.append(b'0 };\n')
    output.append(b'const char mp_frozen_str_content[] = {\n')
    for (full_path, _) in modules:
        with open(full_path, 'rb') as f:
            data = f.read()
            data = bytearray(data)
            esc_dict = {ord('\n'): b'\\n', ord('\r'): b'\\r', ord('"'): b'\\"', ord('\\'): b'\\\\'}
            output.append(b'"')
            break_str = False
            for c in data:
                try:
                    output.append(esc_dict[c])
                except KeyError:
                    if 32 <= c <= 126:
                        if break_str:
                            output.append(b'" "')
                            break_str = False
                        output.append(chr(c).encode())
                    else:
                        output.append(b'\\x%02x' % c)
                        break_str = True
            output.append(b'\\0"\n')
    output.append(b'"\\0"\n};\n\n')
    return b''.join(output)

def main():
    if False:
        i = 10
        return i + 15
    import argparse
    cmd_parser = argparse.ArgumentParser(description='A tool to generate frozen content in MicroPython firmware images.')
    cmd_parser.add_argument('-o', '--output', help='output path')
    cmd_parser.add_argument('-b', '--build-dir', help='output path')
    cmd_parser.add_argument('-f', '--mpy-cross-flags', default='', help='flags to pass to mpy-cross')
    cmd_parser.add_argument('-v', '--var', action='append', help='variables to substitute')
    cmd_parser.add_argument('--mpy-tool-flags', default='', help='flags to pass to mpy-tool')
    cmd_parser.add_argument('files', nargs='+', help='input manifest list')
    args = cmd_parser.parse_args()
    for var in args.var:
        (name, value) = var.split('=', 1)
        if os.path.exists(value):
            value = os.path.abspath(value)
        VARS[name] = value
    if 'MPY_DIR' not in VARS or 'PORT_DIR' not in VARS:
        print('MPY_DIR and PORT_DIR variables must be specified')
        sys.exit(1)
    MPY_CROSS = VARS['MPY_DIR'] + '/mpy-cross/build/mpy-cross'
    if sys.platform == 'win32':
        MPY_CROSS += '.exe'
    MPY_CROSS = os.getenv('MICROPY_MPYCROSS', MPY_CROSS)
    MPY_TOOL = VARS['MPY_DIR'] + '/tools/mpy-tool.py'
    if not os.path.exists(MPY_CROSS):
        print('mpy-cross not found at {}, please build it first'.format(MPY_CROSS))
        sys.exit(1)
    manifest = manifestfile.ManifestFile(manifestfile.MODE_FREEZE, VARS)
    for input_manifest in args.files:
        try:
            manifest.execute(input_manifest)
        except manifestfile.ManifestFileError as er:
            print('freeze error executing "{}": {}'.format(input_manifest, er.args[0]))
            sys.exit(1)
    str_paths = []
    mpy_files = []
    ts_newest = 0
    for result in manifest.files():
        if result.kind == manifestfile.KIND_FREEZE_AS_STR:
            str_paths.append((result.full_path, result.target_path))
            ts_outfile = result.timestamp
        elif result.kind == manifestfile.KIND_FREEZE_AS_MPY:
            outfile = '{}/frozen_mpy/{}.mpy'.format(args.build_dir, result.target_path[:-3])
            ts_outfile = get_timestamp(outfile, 0)
            if result.timestamp >= ts_outfile:
                print('MPY', result.target_path)
                mkdir(outfile)
                with manifestfile.tagged_py_file(result.full_path, result.metadata) as tagged_path:
                    try:
                        mpy_cross.compile(tagged_path, dest=outfile, src_path=result.target_path, opt=result.opt, mpy_cross=MPY_CROSS, extra_args=args.mpy_cross_flags.split())
                    except mpy_cross.CrossCompileError as ex:
                        print('error compiling {}:'.format(result.target_path))
                        print(ex.args[0])
                        raise SystemExit(1)
                ts_outfile = get_timestamp(outfile)
            mpy_files.append(outfile)
        else:
            assert result.kind == manifestfile.KIND_FREEZE_MPY
            mpy_files.append(result.full_path)
            ts_outfile = result.timestamp
        ts_newest = max(ts_newest, ts_outfile)
    if ts_newest < get_timestamp(args.output, 0):
        return
    output_str = generate_frozen_str_content(str_paths)
    if mpy_files:
        (res, output_mpy) = system([sys.executable, MPY_TOOL, '-f', '-q', args.build_dir + '/genhdr/qstrdefs.preprocessed.h'] + args.mpy_tool_flags.split() + mpy_files)
        if res != 0:
            print('error freezing mpy {}:'.format(mpy_files))
            print(output_mpy.decode())
            sys.exit(1)
    else:
        output_mpy = b'#include "py/emitglue.h"\nextern const qstr_pool_t mp_qstr_const_pool;\nconst qstr_pool_t mp_qstr_frozen_const_pool = {\n    (qstr_pool_t*)&mp_qstr_const_pool, MP_QSTRnumber_of, 0, 0\n};\nconst char mp_frozen_names[] = { MP_FROZEN_STR_NAMES "\\0"};\nconst mp_raw_code_t *const mp_frozen_mpy_content[] = {NULL};\n'
    print('GEN', args.output)
    mkdir(args.output)
    with open(args.output, 'wb') as f:
        f.write(b'//\n// Content for MICROPY_MODULE_FROZEN_STR\n//\n')
        f.write(output_str)
        f.write(b'//\n// Content for MICROPY_MODULE_FROZEN_MPY\n//\n')
        f.write(output_mpy)
if __name__ == '__main__':
    main()