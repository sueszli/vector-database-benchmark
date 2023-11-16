"""Installation script for Oppia third-party libraries."""
from __future__ import annotations
import argparse
import os
import pathlib
import shutil
import subprocess
import zipfile
from core import feconf
from scripts import install_python_dev_dependencies
from typing import Final, List
if not feconf.OPPIA_IS_DOCKERIZED:
    install_python_dev_dependencies.main(['--assert_compiled'])
    from . import install_third_party
    from . import pre_commit_hook
    from . import pre_push_hook
    from . import setup
    from . import setup_gae
from . import common
from core import utils
_PARSER: Final = argparse.ArgumentParser(description='\nInstallation script for Oppia third-party libraries.\n')
BUF_BASE_URL: Final = 'https://github.com/bufbuild/buf/releases/download/v0.29.0/'
BUF_LINUX_FILES: Final = ['buf-Linux-x86_64', 'protoc-gen-buf-check-lint-Linux-x86_64', 'protoc-gen-buf-check-breaking-Linux-x86_64']
BUF_DARWIN_FILES: Final = ['buf-Darwin-x86_64', 'protoc-gen-buf-check-lint-Darwin-x86_64', 'protoc-gen-buf-check-breaking-Darwin-x86_64']
PROTOC_URL: Final = 'https://github.com/protocolbuffers/protobuf/releases/download/v%s' % common.PROTOC_VERSION
PROTOC_LINUX_FILE: Final = 'protoc-%s-linux-x86_64.zip' % common.PROTOC_VERSION
PROTOC_DARWIN_FILE: Final = 'protoc-%s-osx-x86_64.zip' % common.PROTOC_VERSION
BUF_DIR: Final = os.path.join(common.OPPIA_TOOLS_DIR, 'buf-%s' % common.BUF_VERSION)
PROTOC_DIR: Final = os.path.join(BUF_DIR, 'protoc')
PROTO_FILES_PATHS: Final = [os.path.join(common.THIRD_PARTY_DIR, 'oppia-ml-proto-0.0.0')]
PROTOC_GEN_TS_PATH: Final = os.path.join(common.NODE_MODULES_PATH, 'protoc-gen-ts')

def tweak_yarn_executable() -> None:
    if False:
        print('Hello World!')
    "When yarn is run on Windows, the file yarn will be executed by default.\n    However, this file is a bash script, and can't be executed directly on\n    Windows. So, to prevent Windows automatically executing it by default\n    (while preserving the behavior on other systems), we rename it to yarn.sh\n    here.\n    "
    origin_file_path = os.path.join(common.YARN_PATH, 'bin', 'yarn')
    if os.path.isfile(origin_file_path):
        renamed_file_path = os.path.join(common.YARN_PATH, 'bin', 'yarn.sh')
        os.rename(origin_file_path, renamed_file_path)

def get_yarn_command() -> str:
    if False:
        return 10
    'Get the executable file for yarn.'
    if common.is_windows_os():
        return 'yarn.cmd'
    return 'yarn'

def install_buf_and_protoc() -> None:
    if False:
        print('Hello World!')
    'Installs buf and protoc for Linux or Darwin, depending upon the\n    platform.\n    '
    buf_files = BUF_DARWIN_FILES if common.is_mac_os() else BUF_LINUX_FILES
    protoc_file = PROTOC_DARWIN_FILE if common.is_mac_os() else PROTOC_LINUX_FILE
    buf_path = os.path.join(BUF_DIR, buf_files[0])
    protoc_path = os.path.join(PROTOC_DIR, 'bin', 'protoc')
    if os.path.isfile(buf_path) and os.path.isfile(protoc_path):
        return
    common.ensure_directory_exists(BUF_DIR)
    for bin_file in buf_files:
        common.url_retrieve('%s/%s' % (BUF_BASE_URL, bin_file), os.path.join(BUF_DIR, bin_file))
    common.url_retrieve('%s/%s' % (PROTOC_URL, protoc_file), os.path.join(BUF_DIR, protoc_file))
    try:
        with zipfile.ZipFile(os.path.join(BUF_DIR, protoc_file), 'r') as zfile:
            zfile.extractall(path=PROTOC_DIR)
        os.remove(os.path.join(BUF_DIR, protoc_file))
    except Exception as e:
        raise Exception('Error installing protoc binary') from e
    common.recursive_chmod(buf_path, 484)
    common.recursive_chmod(protoc_path, 484)

def compile_protobuf_files(proto_files_paths: List[str]) -> None:
    if False:
        while True:
            i = 10
    'Compiles protobuf files using buf.\n\n    Raises:\n        Exception. If there is any error in compiling the proto files.\n    '
    proto_env = os.environ.copy()
    proto_env['PATH'] += '%s%s/bin' % (os.pathsep, PROTOC_DIR)
    proto_env['PATH'] += '%s%s/bin' % (os.pathsep, PROTOC_GEN_TS_PATH)
    buf_path = os.path.join(BUF_DIR, BUF_DARWIN_FILES[0] if common.is_mac_os() else BUF_LINUX_FILES[0])
    for path in proto_files_paths:
        command = [buf_path, 'generate', path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proto_env)
        (stdout, stderr) = process.communicate()
        if process.returncode == 0:
            print(stdout)
        else:
            print(stderr)
            raise Exception('Error compiling proto files at %s' % path)
    compiled_protobuf_dir = pathlib.Path(os.path.join(common.CURR_DIR, 'proto_files'))
    for p in compiled_protobuf_dir.iterdir():
        if p.suffix == '.py':
            common.inplace_replace_file(p.absolute().as_posix(), '^import (\\w*_pb2 as)', 'from proto_files import \\1')

def main() -> None:
    if False:
        print('Hello World!')
    'Install third-party libraries for Oppia.'
    if feconf.OPPIA_IS_DOCKERIZED:
        return
    setup.main(args=[])
    setup_gae.main(args=[])
    print('Installing third-party JS libraries and zip files.')
    install_third_party.main(args=[])
    print('Copying Google Cloud SDK modules to third_party/python_libs...')
    correct_google_path = os.path.join(common.THIRD_PARTY_PYTHON_LIBS_DIR, 'google')
    if not os.path.isdir(correct_google_path):
        os.mkdir(correct_google_path)
    if not os.path.isdir(os.path.join(correct_google_path, 'appengine')):
        shutil.copytree(os.path.join(common.GOOGLE_APP_ENGINE_SDK_HOME, 'google', 'appengine'), os.path.join(correct_google_path, 'appengine'))
    if not os.path.isdir(os.path.join(correct_google_path, 'net')):
        shutil.copytree(os.path.join(common.GOOGLE_APP_ENGINE_SDK_HOME, 'google', 'net'), os.path.join(correct_google_path, 'net'))
    if not os.path.isdir(os.path.join(correct_google_path, 'pyglib')):
        shutil.copytree(os.path.join(common.GOOGLE_APP_ENGINE_SDK_HOME, 'google', 'pyglib'), os.path.join(correct_google_path, 'pyglib'))
    print('Checking that all google library modules contain __init__.py files...')
    for path_list in os.walk(correct_google_path):
        root_path = path_list[0]
        if not root_path.endswith('__pycache__'):
            with utils.open_file(os.path.join(root_path, '__init__.py'), 'a'):
                pass
    if common.is_windows_os():
        tweak_yarn_executable()
    subprocess.check_call([get_yarn_command(), 'install', '--pure-lockfile'])
    print('Installing buf and protoc binary.')
    install_buf_and_protoc()
    print('Compiling protobuf files.')
    compile_protobuf_files(PROTO_FILES_PATHS)
    print('Installing pre-commit hook for git')
    pre_commit_hook.main(args=['--install'])
    if not common.is_windows_os():
        print('Installing pre-push hook for git')
        pre_push_hook.main(args=['--install'])
if __name__ == '__main__':
    main()