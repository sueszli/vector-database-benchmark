"""
A script to pull licenses for Python.
The script is executed within Docker.
"""
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import yaml
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from urllib.parse import urljoin
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential
LICENSE_DIR = '/opt/apache/beam/third_party_licenses'

def run_bash_command(command):
    if False:
        for i in range(10):
            print('nop')
    return subprocess.check_output(command.split()).decode('utf-8')

def run_pip_licenses():
    if False:
        for i in range(10):
            print('nop')
    command = 'pip-licenses --with-license-file --with-urls --from=mixed --ignore apache-beam --format=json'
    dependencies = run_bash_command(command)
    return json.loads(dependencies)

@retry(stop=stop_after_attempt(3))
def copy_license_files(dep):
    if False:
        return 10
    source_license_file = dep['LicenseFile']
    if source_license_file.lower() == 'unknown':
        return False
    name = dep['Name'].lower()
    dest_dir = os.path.join(LICENSE_DIR, name)
    try:
        os.mkdir(dest_dir)
        shutil.copy(source_license_file, dest_dir + '/LICENSE')
        logging.debug('Successfully pulled license for {dep} with pip-licenses.'.format(dep=name))
        return True
    except Exception as e:
        logging.error('Failed to copy from {source} to {dest}'.format(source=source_license_file, dest=dest_dir + '/LICENSE'))
        traceback.print_exc()
        raise

@retry(reraise=True, wait=wait_exponential(multiplier=2), stop=stop_after_attempt(5))
def pull_from_url(dep, configs):
    if False:
        print('Hello World!')
    '\n  :param dep: name of a dependency\n  :param configs: a dict from dep_urls_py.yaml\n  :return: boolean\n\n  It downloads files form urls to a temp directory first in order to avoid\n  to deal with any temp files. It helps keep clean final directory.\n  '
    if dep in configs:
        config = configs[dep]
        dest_dir = os.path.join(LICENSE_DIR, dep)
        cur_temp_dir = tempfile.mkdtemp()
        try:
            if config['license'] == 'skip':
                print('Skip pulling license for ', dep)
            else:
                url_read = urlopen(Request(config['license'], headers={'User-Agent': 'Apache Beam'}))
                with open(cur_temp_dir + '/LICENSE', 'wb') as temp_write:
                    shutil.copyfileobj(url_read, temp_write)
                logging.debug('Successfully pulled license for {dep} from {url}.'.format(dep=dep, url=config['license']))
            if 'notice' in config:
                url_read = urlopen(config['notice'])
                with open(cur_temp_dir + '/NOTICE', 'wb') as temp_write:
                    shutil.copyfileobj(url_read, temp_write)
            shutil.copytree(cur_temp_dir, dest_dir)
            return True
        except Exception as e:
            logging.error('Error occurred when pull license for {dep} from {url}.'.format(dep=dep, url=config))
            traceback.print_exc()
            raise
        finally:
            shutil.rmtree(cur_temp_dir)

def license_url(name, project_url, dep_config):
    if False:
        for i in range(10):
            print('nop')
    '\n  Gets the license URL for a dependency, either from the parsed yaml or,\n  if it is github, by looking for a license file in the repo.\n  '
    configs = dep_config['pip_dependencies']
    if name.lower() in configs:
        return configs[name.lower()]['license']
    p = urlparse(project_url)
    if p.netloc != 'github.com':
        return project_url
    raw = 'https://raw.githubusercontent.com'
    path = p.path
    if not path.endswith('/'):
        path = path + '/'
    for license in ('LICENSE', 'LICENSE.txt', 'LICENSE.md', 'LICENSE.rst', 'COPYING'):
        try:
            url = raw + urljoin(path, 'master/' + license)
            with urlopen(url) as a:
                if a.getcode() == 200:
                    return url
        except:
            pass
    return project_url

def save_license_list(csv_filename, dependencies, dep_config):
    if False:
        return 10
    '\n  Save the names, URLs, and license type for python dependency licenses in a CSV file.\n  '
    with open(csv_filename, mode='w') as f:
        writer = csv.writer(f)
        for dep in dependencies:
            url = license_url(dep['Name'], dep['URL'], dep_config)
            writer.writerow([dep['Name'], url, dep['License']])
if __name__ == '__main__':
    no_licenses = []
    logging.getLogger().setLevel(logging.INFO)
    with open('/tmp/license_scripts/dep_urls_py.yaml') as file:
        dep_config = yaml.full_load(file)
    dependencies = run_pip_licenses()
    csv_filename = os.path.join(LICENSE_DIR, 'python-licenses.csv')
    save_license_list(csv_filename, dependencies, dep_config)
    for dep in dependencies:
        if not (copy_license_files(dep) or pull_from_url(dep['Name'].lower(), dep_config['pip_dependencies'])):
            no_licenses.append(dep['Name'].lower())
    if no_licenses:
        py_ver = '%d.%d' % (sys.version_info[0], sys.version_info[1])
        how_to = 'These licenses were not able to be pulled automatically. Please search code source of the dependencies on the internet and add urls to RAW license file at sdks/python/container/license_scripts/dep_urls_py.yaml for each missing license and rerun the test. If no such urls can be found, you need to manually add LICENSE and NOTICE (if available) files at sdks/python/container/license_scripts/manual_licenses/{dep}/ and add entries to sdks/python/container/license_scripts/dep_urls_py.yaml.'
        raise RuntimeError('Could not retrieve licences for packages {license_list} in Python{py_ver} environment. \n {how_to}'.format(py_ver=py_ver, license_list=sorted(no_licenses), how_to=how_to))
    else:
        logging.info('Successfully pulled licenses for {n} dependencies'.format(n=len(dependencies)))