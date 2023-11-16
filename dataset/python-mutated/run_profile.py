"""Profile qutebrowser."""
import sys
import cProfile
import os.path
import os
import tempfile
import subprocess
import shutil
import argparse
import shlex
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
import qutebrowser.qutebrowser

def parse_args():
    if False:
        return 10
    'Parse commandline arguments.\n\n    Return:\n        A (namespace, remaining_args) tuple from argparse.\n    '
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-tool', metavar='TOOL', action='store', choices=['kcachegrind', 'snakeviz', 'gprof2dot', 'tuna', 'none'], default='snakeviz', help='The tool to use to view the profiling data')
    parser.add_argument('--profile-file', metavar='FILE', action='store', default='profile_data', help='The filename to use with --profile-tool=none')
    parser.add_argument('--profile-test', action='store_true', help='Run pytest instead of qutebrowser')
    return parser.parse_known_args()

def main():
    if False:
        i = 10
        return i + 15
    (args, remaining) = parse_args()
    tempdir = tempfile.mkdtemp()
    if args.profile_tool == 'none':
        profilefile = os.path.join(os.getcwd(), args.profile_file)
    else:
        profilefile = os.path.join(tempdir, 'profile')
    sys.argv = [sys.argv[0]] + remaining
    profiler = cProfile.Profile()
    if args.profile_test:
        import pytest
        profiler.runcall(pytest.main)
    else:
        profiler.runcall(qutebrowser.qutebrowser.main)
    sys.excepthook = sys.__excepthook__
    profiler.dump_stats(profilefile)
    if args.profile_tool == 'none':
        print('Profile data written to {}'.format(profilefile))
    elif args.profile_tool == 'gprof2dot':
        subprocess.run('gprof2dot -f pstats {} | dot -Tpng | feh -F -'.format(shlex.quote(profilefile)), shell=True, check=True)
    elif args.profile_tool == 'kcachegrind':
        callgraphfile = os.path.join(tempdir, 'callgraph')
        subprocess.run(['pyprof2calltree', '-k', '-i', profilefile, '-o', callgraphfile], check=True)
    elif args.profile_tool == 'snakeviz':
        subprocess.run(['snakeviz', profilefile], check=True)
    elif args.profile_tool == 'tuna':
        subprocess.run(['tuna', profilefile], check=True)
    shutil.rmtree(tempdir)
if __name__ == '__main__':
    main()