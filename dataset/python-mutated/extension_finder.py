import os

def find_extensions(ext_dir):
    if False:
        for i in range(10):
            print('nop')
    '\n    Yields `(extension_id, extension_path)` tuples found in a given extensions dir\n    '
    if os.path.exists(ext_dir):
        for entry in os.scandir(ext_dir):
            if entry.is_dir() and os.path.isfile(f'{ext_dir}/{entry.name}/manifest.json'):
                yield (entry.name, f'{ext_dir}/{entry.name}')