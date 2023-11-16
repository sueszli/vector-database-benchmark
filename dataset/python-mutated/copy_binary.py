"""Copy binaries of TensorFlow for different python versions."""
import argparse
import os
import re
import shutil
import tempfile
import zipfile
TF_NIGHTLY_REGEX = '(.+)(tf_nightly.*)-(\\d\\.[\\d]{1,2}\\.\\d.dev[\\d]{0,8})-(.+)\\.whl'
BINARY_STRING_TEMPLATE = '%s-%s-%s.whl'

def check_existence(filename):
    if False:
        print('Hello World!')
    'Check the existence of file or dir.'
    if not os.path.exists(filename):
        raise RuntimeError('%s not found.' % filename)

def copy_binary(directory, origin_tag, new_tag, version, package):
    if False:
        while True:
            i = 10
    'Rename and copy binaries for different python versions.\n\n  Args:\n    directory: string of directory\n    origin_tag: str of the old python version tag\n    new_tag: str of the new tag\n    version: the version of the package\n    package: str, name of the package\n\n  '
    print('Rename and copy binaries with %s to %s.' % (origin_tag, new_tag))
    origin_binary = BINARY_STRING_TEMPLATE % (package, version, origin_tag)
    new_binary = BINARY_STRING_TEMPLATE % (package, version, new_tag)
    zip_ref = zipfile.ZipFile(os.path.join(directory, origin_binary), 'r')
    try:
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        zip_ref.extractall()
        zip_ref.close()
        old_py_ver = re.search('(cp\\d\\d-cp\\d\\d)', origin_tag).group(1)
        new_py_ver = re.search('(cp\\d\\d-cp\\d\\d)', new_tag).group(1)
        wheel_file = os.path.join(tmpdir, '%s-%s.dist-info' % (package, version), 'WHEEL')
        with open(wheel_file, 'r') as f:
            content = f.read()
        with open(wheel_file, 'w') as f:
            f.write(content.replace(old_py_ver, new_py_ver))
        zout = zipfile.ZipFile(directory + new_binary, 'w', zipfile.ZIP_DEFLATED)
        zip_these_files = ['%s-%s.dist-info' % (package, version), '%s-%s.data' % (package, version), 'tensorflow', 'tensorflow_core']
        for dirname in zip_these_files:
            for (root, _, files) in os.walk(dirname):
                for filename in files:
                    zout.write(os.path.join(root, filename))
        zout.close()
    finally:
        shutil.rmtree(tmpdir)

def main():
    if False:
        i = 10
        return i + 15
    'This script copies binaries.\n\n  Requirements:\n    filename: The path to the whl file\n    AND\n    new_py_ver: Create a nightly tag with current date\n\n  Raises:\n    RuntimeError: If the whl file was not found\n  '
    parser = argparse.ArgumentParser(description='Cherry picking automation.')
    parser.add_argument('--filename', help='path to whl file we are copying', required=True)
    parser.add_argument('--new_py_ver', help='two digit py version eg. 27 or 33', required=True)
    args = parser.parse_args()
    args.filename = os.path.abspath(args.filename)
    check_existence(args.filename)
    regex_groups = re.search(TF_NIGHTLY_REGEX, args.filename)
    directory = regex_groups.group(1)
    package = regex_groups.group(2)
    version = regex_groups.group(3)
    origin_tag = regex_groups.group(4)
    old_py_ver = re.search('(cp\\d\\d)', origin_tag).group(1)
    new_tag = origin_tag.replace(old_py_ver, 'cp' + args.new_py_ver)
    copy_binary(directory, origin_tag, new_tag, version, package)
if __name__ == '__main__':
    main()