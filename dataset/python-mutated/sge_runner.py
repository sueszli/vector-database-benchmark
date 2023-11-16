"""
The SunGrid Engine runner

The main() function of this module will be executed on the
compute node by the submitted job. It accepts as a single
argument the shared temp folder containing the package archive
and pickled task to run, and carries out these steps:

- extract tarfile of package dependencies and place on the path
- unpickle SGETask instance created on the master node
- run SGETask.work()

On completion, SGETask on the master node will detect that
the job has left the queue, delete the temporary folder, and
return from SGETask.run()
"""
import os
import sys
import pickle
import logging
import tarfile

def _do_work_on_compute_node(work_dir, tarball=True):
    if False:
        for i in range(10):
            print('nop')
    if tarball:
        _extract_packages_archive(work_dir)
    os.chdir(work_dir)
    with open('job-instance.pickle', 'r') as f:
        job = pickle.load(f)
    job.work()

def _extract_packages_archive(work_dir):
    if False:
        i = 10
        return i + 15
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
        i = 10
        return i + 15
    'Run the work() method from the class instance in the file "job-instance.pickle".\n    '
    try:
        tarball = '--no-tarball' not in args
        logging.basicConfig(level=logging.WARN)
        work_dir = args[1]
        assert os.path.exists(work_dir), 'First argument to sge_runner.py must be a directory that exists'
        project_dir = args[2]
        sys.path.append(project_dir)
        _do_work_on_compute_node(work_dir, tarball)
    except Exception as e:
        print(e)
        raise
if __name__ == '__main__':
    main()