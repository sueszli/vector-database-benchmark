from subprocess import check_call
import argparse
import os
import logging
from ci_tools.functions import find_whl
from ci_tools.parsing import ParsedSetup
logging.getLogger().setLevel(logging.INFO)
root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))

def get_package_wheel_path(pkg_root):
    if False:
        i = 10
        return i + 15
    pkg_details = ParsedSetup.from_path(pkg_root)
    prebuilt_dir = os.getenv('PREBUILT_WHEEL_DIR')
    if prebuilt_dir:
        prebuilt_package_path = find_whl(prebuilt_dir, pkg_details.name, pkg_details.version)
    else:
        return None
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run apistubgen against target folder. ')
    parser.add_argument('-t', '--target', dest='target_package', help='The target package directory on disk.', required=True)
    parser.add_argument('-w', '--work-dir', dest='work_dir', help='Working directory to run apistubgen', required=True)
    parser.add_argument('-o', '--out-path', dest='out_path', help='Output directory to generate json token file')
    args = parser.parse_args()
    pkg_path = get_package_wheel_path(args.target_package)
    if not pkg_path:
        pkg_path = args.target_package
    cmds = ['apistubgen', '--pkg-path', pkg_path]
    if args.out_path:
        cmds.extend(['--out-path', os.path.join(args.out_path, os.path.basename(pkg_path))])
    logging.info('Running apistubgen {}.'.format(cmds))
    check_call(cmds, cwd=args.work_dir)