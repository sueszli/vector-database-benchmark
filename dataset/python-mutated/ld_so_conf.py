import glob
import os
import re
import sys
from llnl.util.lang import dedupe
import spack.util.elf as elf_utils

def parse_ld_so_conf(conf_file='/etc/ld.so.conf'):
    if False:
        for i in range(10):
            print('nop')
    'Parse glibc style ld.so.conf file, which specifies default search paths for the\n    dynamic linker. This can in principle also be used for musl libc.\n\n    Arguments:\n        conf_file (str or bytes): Path to config file\n\n    Returns:\n        list: List of absolute search paths\n    '
    is_bytes = isinstance(conf_file, bytes)
    if not is_bytes:
        conf_file = conf_file.encode('utf-8')
    cwd = os.getcwd()
    try:
        paths = _process_ld_so_conf_queue([conf_file])
    finally:
        os.chdir(cwd)
    return list(paths) if is_bytes else [p.decode('utf-8') for p in paths]

def _process_ld_so_conf_queue(queue):
    if False:
        for i in range(10):
            print('nop')
    include_regex = re.compile(b'include\\s')
    paths = []
    while queue:
        p = queue.pop(0)
        try:
            with open(p, 'rb') as f:
                lines = f.readlines()
        except (IOError, OSError):
            continue
        for line in lines:
            comment = line.find(b'#')
            if comment != -1:
                line = line[:comment]
            line = line.strip()
            if not line:
                continue
            is_include = include_regex.match(line) is not None
            if not is_include:
                if os.path.isabs(line):
                    paths.append(line)
                continue
            include_path = line[8:].strip()
            if not include_path:
                continue
            cwd = os.path.dirname(p)
            os.chdir(cwd)
            queue.extend((os.path.join(cwd, p) for p in glob.glob(include_path)))
    return dedupe(paths)

def get_conf_file_from_dynamic_linker(dynamic_linker_name):
    if False:
        for i in range(10):
            print('nop')
    if 'ld-musl-' not in dynamic_linker_name:
        return 'ld.so.conf'
    idx = dynamic_linker_name.find('.')
    if idx != -1:
        return dynamic_linker_name[:idx] + '.path'

def host_dynamic_linker_search_paths():
    if False:
        while True:
            i = 10
    "Retrieve the current host runtime search paths for shared libraries;\n    for GNU and musl Linux we try to retrieve the dynamic linker from the\n    current Python interpreter and then find the corresponding config file\n    (e.g. ld.so.conf or ld-musl-<arch>.path). Similar can be done for\n    BSD and others, but this is not implemented yet. The default paths\n    are always returned. We don't check if the listed directories exist."
    default_paths = ['/usr/lib', '/usr/lib64', '/lib', '/lib64']
    if not sys.platform.startswith('linux'):
        return default_paths
    conf_file = '/etc/ld.so.conf'
    try:
        with open(sys.executable, 'rb') as f:
            elf = elf_utils.parse_elf(f, dynamic_section=False, interpreter=True)
        if elf.has_pt_interp:
            dynamic_linker = elf.pt_interp_str.decode('utf-8')
            dynamic_linker_name = os.path.basename(dynamic_linker)
            conf_name = get_conf_file_from_dynamic_linker(dynamic_linker_name)
            possible_prefix = os.path.dirname(os.path.dirname(dynamic_linker))
            possible_conf = os.path.join(possible_prefix, 'etc', conf_name)
            if os.path.exists(possible_conf):
                conf_file = possible_conf
    except (IOError, OSError, elf_utils.ElfParsingError):
        pass
    return list(dedupe(parse_ld_so_conf(conf_file) + default_paths))