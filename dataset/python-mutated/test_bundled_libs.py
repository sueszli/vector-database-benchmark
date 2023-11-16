import subprocess
from sys import argv

def get_list_elm_match(value, elms):
    if False:
        return 10
    ' Check if any element in the elms list matches the value '
    return any((e in value for e in elms))

def check_ldd_out(lib, linked_lib, bundled_lib_names, allowed_libs):
    if False:
        i = 10
        return i + 15
    allowed_libs_to_check = []
    for k in allowed_libs.keys():
        if k in lib:
            allowed_libs_to_check += allowed_libs[k]
    return linked_lib in bundled_lib_names or get_list_elm_match(linked_lib, allowed_libs_to_check)

def main():
    if False:
        print('Hello World!')
    allowed_libs = {'': ['linux-vdso.so.1', 'libm.so.6', 'libpthread.so.0', 'libc.so.6', '/lib64/ld-linux', '/lib/ld-linux', 'libdl.so.2', 'librt.so.1', 'libstdc++.so.6', 'libgcc_s.so.1', 'libasan.so', 'liblsan.so', 'libubsan.so', 'libtsan.so']}
    bundled_libs = argv[1:]
    bundled_lib_names = []
    for lib in bundled_libs:
        beg = lib.rfind('/')
        bundled_lib_names.append(lib[beg + 1:])
    print('Checking bundled libs linkage:')
    for (lib_path, lib_name) in zip(bundled_libs, bundled_lib_names):
        print(f'- {lib_name}')
        ldd = subprocess.Popen(['ldd', lib_path], stdout=subprocess.PIPE)
        for lib in ldd.stdout:
            lib = lib.decode().strip('\t').strip('\n')
            linked_lib = lib.split()[0]
            if not check_ldd_out(lib_name, linked_lib, bundled_lib_names, allowed_libs):
                print(f"Library: '{linked_lib}' should be bundled in whl or removed from the dynamic link dependency")
                exit(1)
    print('-> OK')
if __name__ == '__main__':
    main()