"""
The `pillar_roots` wheel module is used to manage files under the pillar roots
directories on the master server.
"""
import os
import salt.utils.files
import salt.utils.path
import salt.utils.verify

def find(path, saltenv='base'):
    if False:
        return 10
    '\n    Return a dict of the files located with the given path and environment\n    '
    ret = []
    if saltenv not in __opts__['pillar_roots']:
        return ret
    for root in __opts__['pillar_roots'][saltenv]:
        full = os.path.join(root, path)
        if os.path.isfile(full):
            with salt.utils.files.fopen(full, 'rb') as fp_:
                if salt.utils.files.is_text(fp_):
                    ret.append({full: 'txt'})
                else:
                    ret.append({full: 'bin'})
    return ret

def list_env(saltenv='base'):
    if False:
        while True:
            i = 10
    '\n    Return all of the file paths found in an environment\n    '
    ret = {}
    if saltenv not in __opts__['pillar_roots']:
        return ret
    for f_root in __opts__['pillar_roots'][saltenv]:
        ret[f_root] = {}
        for (root, dirs, files) in salt.utils.path.os_walk(f_root):
            sub = ret[f_root]
            if root != f_root:
                sroot = root
                above = []
                while not os.path.samefile(sroot, f_root):
                    base = os.path.basename(sroot)
                    if base:
                        above.insert(0, base)
                    sroot = os.path.dirname(sroot)
                for aroot in above:
                    sub = sub[aroot]
            for dir_ in dirs:
                sub[dir_] = {}
            for fn_ in files:
                sub[fn_] = 'f'
    return ret

def list_roots():
    if False:
        for i in range(10):
            print('nop')
    '\n    Return all of the files names in all available environments\n    '
    ret = {}
    for saltenv in __opts__['pillar_roots']:
        ret[saltenv] = []
        ret[saltenv].append(list_env(saltenv))
    return ret

def read(path, saltenv='base'):
    if False:
        print('Hello World!')
    '\n    Read the contents of a text file, if the file is binary then\n    '
    ret = []
    files = find(path, saltenv)
    for fn_ in files:
        full = next(iter(fn_.keys()))
        form = fn_[full]
        if form == 'txt':
            with salt.utils.files.fopen(full, 'rb') as fp_:
                ret.append({full: salt.utils.stringutils.to_unicode(fp_.read())})
    return ret

def write(data, path, saltenv='base', index=0):
    if False:
        return 10
    '\n    Write the named file, by default the first file found is written, but the\n    index of the file can be specified to write to a lower priority file root\n    '
    if saltenv not in __opts__['pillar_roots']:
        return 'Named environment {} is not present'.format(saltenv)
    if len(__opts__['pillar_roots'][saltenv]) <= index:
        return 'Specified index {} in environment {} is not present'.format(index, saltenv)
    if os.path.isabs(path):
        return 'The path passed in {} is not relative to the environment {}'.format(path, saltenv)
    roots_dir = __opts__['pillar_roots'][saltenv][index]
    dest = os.path.join(roots_dir, path)
    if not salt.utils.verify.clean_path(roots_dir, dest, subdir=True):
        return 'Invalid path'
    dest = os.path.join(__opts__['pillar_roots'][saltenv][index], path)
    dest_dir = os.path.dirname(dest)
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)
    with salt.utils.files.fopen(dest, 'w+') as fp_:
        fp_.write(salt.utils.stringutils.to_str(data))
    return 'Wrote data to file {}'.format(dest)