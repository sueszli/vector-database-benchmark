import argparse
import logging
import os
import glob
import shutil
from tox_helper_tasks import unzip_file_to_directory
from ci_tools.parsing import ParsedSetup
logging.getLogger().setLevel(logging.INFO)
EXCLUDED_PACKAGES = ['azure', 'azure-mgmt', 'azure-common', 'azure-applicationinsights', 'azure-loganalytics']

def extract_whl(dist_dir, version):
    if False:
        i = 10
        return i + 15
    path_to_whl = glob.glob(os.path.join(dist_dir, '*{}*.whl'.format(version)))[0]
    zip_file = path_to_whl.replace('.whl', '.zip')
    cleanup(zip_file)
    os.rename(path_to_whl, zip_file)
    extract_location = os.path.join(dist_dir, 'unzipped')
    cleanup(extract_location)
    unzip_file_to_directory(zip_file, extract_location)
    return extract_location

def verify_whl_root_directory(dist_dir, expected_top_level_module, version):
    if False:
        for i in range(10):
            print('nop')
    extract_location = extract_whl(dist_dir, version)
    root_folders = os.listdir(extract_location)
    non_azure_folders = [d for d in root_folders if d != expected_top_level_module and (not d.endswith('.dist-info'))]
    if non_azure_folders:
        logging.error('whl has following incorrect directory at root level [%s]', non_azure_folders)
        return False
    else:
        return True

def cleanup(path):
    if False:
        for i in range(10):
            print('nop')
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def should_verify_package(package_name):
    if False:
        while True:
            i = 10
    return package_name not in EXCLUDED_PACKAGES and 'nspkg' not in package_name and ('-mgmt' not in package_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify directories included in whl and contents in manifest file')
    parser.add_argument('-t', '--target', dest='target_package', help='The target package directory on disk.', required=True)
    parser.add_argument('-d', '--dist_dir', dest='dist_dir', help='The dist location on disk. Usually /tox/dist.', required=True)
    args = parser.parse_args()
    pkg_dir = os.path.abspath(args.target_package)
    pkg_details = ParsedSetup.from_path(pkg_dir)
    top_level_module = pkg_details.namespace.split('.')[0]
    if should_verify_package(pkg_details.name):
        logging.info('Verifying root directory in whl for package: [%s]', pkg_details.name)
        if verify_whl_root_directory(args.dist_dir, top_level_module, pkg_details.version):
            logging.info('Verified root directory in whl for package: [%s]', pkg_details.name)
        else:
            logging.info('Failed to verify root directory in whl for package: [%s]', pkg_details.name)
            exit(1)