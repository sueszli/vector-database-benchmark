import os
import subprocess
import sys
import tempfile
SAMPLES_TO_BUILD = ['asteroids']
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'samples')

def main():
    if False:
        i = 10
        return i + 15
    build_base = tempfile.TemporaryDirectory()
    dist_dir = tempfile.TemporaryDirectory()
    for sample in SAMPLES_TO_BUILD:
        sampledir = os.path.join(SAMPLES_DIR, sample)
        os.chdir(sampledir)
        args = [sys.executable, 'setup.py', 'bdist_apps', '--build-base', build_base.name, '--dist-dir', dist_dir.name]
        subprocess.check_call(args)
if __name__ == '__main__':
    main()