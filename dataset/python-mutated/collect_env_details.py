"""Diagnose your system and show basic information.

This server mainly to get detail info for better bug reporting.

"""
import os
import platform
import sys
import pkg_resources
import torch
sys.path += [os.path.abspath('..'), os.path.abspath('')]
LEVEL_OFFSET = '\t'
KEY_PADDING = 20

def info_system() -> dict:
    if False:
        i = 10
        return i + 15
    return {'OS': platform.system(), 'architecture': platform.architecture(), 'version': platform.version(), 'release': platform.release(), 'processor': platform.processor(), 'python': platform.python_version()}

def info_cuda() -> dict:
    if False:
        print('Hello World!')
    return {'GPU': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] or None, 'available': torch.cuda.is_available(), 'version': torch.version.cuda}

def info_packages() -> dict:
    if False:
        while True:
            i = 10
    'Get name and version of all installed packages.'
    packages = {}
    for dist in pkg_resources.working_set:
        package = dist.as_requirement()
        packages[package.key] = package.specs[0][1]
    return packages

def nice_print(details: dict, level: int=0) -> list:
    if False:
        return 10
    lines = []
    for k in sorted(details):
        key = f'* {k}:' if level == 0 else f'- {k}:'
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + '- ' + v for v in details[k]]
        else:
            template = '{:%is} {}' % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [level * LEVEL_OFFSET + key_val]
    return lines

def main() -> None:
    if False:
        return 10
    details = {'System': info_system(), 'CUDA': info_cuda(), 'Packages': info_packages()}
    details['Lightning'] = {k: v for (k, v) in details['Packages'].items() if 'torch' in k or 'lightning' in k}
    lines = nice_print(details)
    text = os.linesep.join(lines)
    print(f'<details>\n  <summary>Current environment</summary>\n\n{text}\n\n</details>')
if __name__ == '__main__':
    main()