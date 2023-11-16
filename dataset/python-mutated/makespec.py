"""
Automatically build a spec file containing the description of the project.
"""
import argparse
import os
import PyInstaller.building.makespec
import PyInstaller.log
try:
    from argcomplete import autocomplete
except ImportError:

    def autocomplete(parser):
        if False:
            i = 10
            return i + 15
        return None

def generate_parser():
    if False:
        print('Hello World!')
    p = argparse.ArgumentParser()
    PyInstaller.building.makespec.__add_options(p)
    PyInstaller.log.__add_options(p)
    p.add_argument('scriptname', nargs='+')
    return p

def run():
    if False:
        return 10
    p = generate_parser()
    autocomplete(p)
    args = p.parse_args()
    PyInstaller.log.__process_options(p, args)
    temppaths = args.pathex[:]
    args.pathex = []
    for p in temppaths:
        args.pathex.extend(p.split(os.pathsep))
    try:
        name = PyInstaller.building.makespec.main(args.scriptname, **vars(args))
        print('Wrote %s.' % name)
        print('Now run pyinstaller.py to build the executable.')
    except KeyboardInterrupt:
        raise SystemExit('Aborted by user request.')
if __name__ == '__main__':
    run()