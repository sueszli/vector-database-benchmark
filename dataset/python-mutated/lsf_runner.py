"""
.. Copyright 2012-2015 Spotify AB
   Copyright 2018
   Copyright 2018 EMBL-European Bioinformatics Institute

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import os
import sys
try:
    import dill as pickle
except ImportError:
    import pickle
import logging
import tarfile

def do_work_on_compute_node(work_dir):
    if False:
        return 10
    extract_packages_archive(work_dir)
    os.chdir(work_dir)
    with open('job-instance.pickle', 'r') as pickle_file_handle:
        job = pickle.load(pickle_file_handle)
    job.work()

def extract_packages_archive(work_dir):
    if False:
        for i in range(10):
            print('nop')
    package_file = os.path.join(work_dir, 'packages.tar')
    if not os.path.exists(package_file):
        return
    curdir = os.path.abspath(os.curdir)
    os.chdir(work_dir)
    tar = tarfile.open(package_file)
    for tarinfo in tar:
        tar.extract(tarinfo)
    tar.close()
    if '' not in sys.path:
        sys.path.insert(0, '')
    os.chdir(curdir)

def main(args=sys.argv):
    if False:
        while True:
            i = 10
    'Run the work() method from the class instance in the file "job-instance.pickle".\n    '
    try:
        logging.basicConfig(level=logging.WARN)
        work_dir = args[1]
        assert os.path.exists(work_dir), 'First argument to lsf_runner.py must be a directory that exists'
        do_work_on_compute_node(work_dir)
    except Exception as exc:
        print(exc)
        raise
if __name__ == '__main__':
    main()