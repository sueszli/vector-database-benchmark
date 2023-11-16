"""Generate MANIFEST.in file."""
import os
import shlex
import subprocess
SKIP_EXTS = ('.png', '.jpg', '.jpeg', '.svg')
SKIP_FILES = 'appveyor.yml'
SKIP_PREFIXES = ('.ci/', '.github/')

def sh(cmd):
    if False:
        for i in range(10):
            print('nop')
    return subprocess.check_output(shlex.split(cmd), universal_newlines=True).strip()

def main():
    if False:
        return 10
    files = set()
    for file in sh('git ls-files').split('\n'):
        if file.startswith(SKIP_PREFIXES) or os.path.splitext(file)[1].lower() in SKIP_EXTS or file in SKIP_FILES:
            continue
        files.add(file)
    for file in sorted(files):
        print('include ' + file)
if __name__ == '__main__':
    main()