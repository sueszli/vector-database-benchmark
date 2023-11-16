import glob
import logging
import shutil
import argparse
from pkg_resources import Requirement
import ast
import os
import textwrap
import io
from tox_helper_tasks import unzip_sdist_to_directory, move_and_rename
from ci_tools.parsing import ParsedSetup
logging.getLogger().setLevel(logging.INFO)
RST_EXTENSION_FOR_INDEX = '\n\n## Indices and tables\n\n- {{ref}}`genindex`\n- {{ref}}`modindex`\n- {{ref}}`search`\n\n```{{toctree}}\n:caption: Developer Documentation\n:glob: true\n:maxdepth: 5\n\n{}\n\n```\n\n'
root_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', '..'))
sphinx_conf = os.path.join(root_dir, 'doc', 'sphinx', 'individual_build_conf.py')

def should_build_docs(package_name):
    if False:
        for i in range(10):
            print('nop')
    return not ('nspkg' in package_name or package_name in ['azure', 'azure-mgmt', 'azure-keyvault', 'azure-documentdb', 'azure-mgmt-documentdb', 'azure-servicemanagement-legacy', 'azure-core-tracing-opencensus'])

def create_index_file(readme_location, package_rst):
    if False:
        for i in range(10):
            print('nop')
    readme_ext = os.path.splitext(readme_location)[1]
    if readme_ext == '.md':
        with open(readme_location, 'r') as file:
            output = file.read()
    else:
        logging.error('{} is not a valid readme type. Expecting RST or MD.'.format(readme_location))
    output += RST_EXTENSION_FOR_INDEX.format(package_rst)
    return output

def copy_conf(doc_folder):
    if False:
        i = 10
        return i + 15
    if not os.path.exists(doc_folder):
        os.mkdir(doc_folder)
    shutil.copy(sphinx_conf, os.path.join(doc_folder, 'conf.py'))

def create_index(doc_folder, source_location, namespace):
    if False:
        i = 10
        return i + 15
    index_content = ''
    package_rst = '{}.rst'.format(namespace)
    content_destination = os.path.join(doc_folder, 'index.md')
    if not os.path.exists(doc_folder):
        os.mkdir(doc_folder)
    markdown_readmes = glob.glob(os.path.join(source_location, 'README.md'))
    if markdown_readmes:
        index_content = create_index_file(markdown_readmes[0], package_rst)
    else:
        logging.warning('No readmes detected for this namespace {}'.format(namespace))
        index_content = RST_EXTENSION_FOR_INDEX.format(package_rst)
    with open(content_destination, 'w+', encoding='utf-8') as f:
        f.write(index_content)

def write_version(site_folder, version):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.isdir(site_folder):
        os.mkdir(site_folder)
    with open(os.path.join(site_folder, 'version.txt'), 'w') as f:
        f.write(version)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prep a doc folder for consumption by sphinx.')
    parser.add_argument('-d', '--dist_dir', dest='dist_dir', help='The dist location on disk. Usually /tox/dist.', required=True)
    parser.add_argument('-t', '--target', dest='target_package', help='The target package directory on disk. The target module passed to pylint will be <target_package>/azure.', required=True)
    args = parser.parse_args()
    package_path = os.path.abspath(args.target_package)
    pkg_details = ParsedSetup.from_path(package_path)
    if should_build_docs(pkg_details.name):
        source_location = move_and_rename(unzip_sdist_to_directory(args.dist_dir))
        doc_folder = os.path.join(source_location, 'docgen')
        create_index(doc_folder, source_location, pkg_details.namespace)
        site_folder = os.path.join(args.dist_dir, 'site')
        write_version(site_folder, pkg_details.version)
    else:
        logging.info('Skipping sphinx prep for {}'.format(pkg_details.name))