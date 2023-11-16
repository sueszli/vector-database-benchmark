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
import subprocess
import time
import sys
import logging
import random
import shutil
try:
    import dill as pickle
except ImportError:
    import pickle
import luigi
import luigi.configuration
from luigi.contrib.hadoop import create_packages_archive
from luigi.contrib import lsf_runner
from luigi.task_status import PENDING, FAILED, DONE, RUNNING, UNKNOWN
"\nLSF batch system Tasks.\n=======================\n\nWhat's LSF? see http://en.wikipedia.org/wiki/Platform_LSF\nand https://wiki.med.harvard.edu/Orchestra/IntroductionToLSF\n\nSee: https://github.com/spotify/luigi/issues/1936\n\nThis extension is modeled after the hadoop.py approach.\nI'll be making a few assumptions, and will try to note them.\n\nGoing into it, the assumptions are:\n\n- You schedule your jobs on an LSF submission node.\n- The 'bjobs' command on an LSF batch submission system returns a standardized format.\n- All nodes have access to the code you're running.\n- The sysadmin won't get pissed if we run a 'bjobs' check every thirty\n  seconds or so per job (there are ways of coalescing the bjobs calls if that's not cool).\n\nThe procedure:\n\n- Pickle the class\n- Construct a bsub argument that runs a generic runner function with the path to the pickled class\n- Runner function loads the class from pickle\n- Runner function hits the work button on it\n\n"
LOGGER = logging.getLogger('luigi-interface')

def track_job(job_id):
    if False:
        print('Hello World!')
    '\n    Tracking is done by requesting each job and then searching for whether the job\n    has one of the following states:\n    - "RUN",\n    - "PEND",\n    - "SSUSP",\n    - "EXIT"\n    based on the LSF documentation\n    '
    cmd = 'bjobs -noheader -o stat {}'.format(job_id)
    track_job_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    status = track_job_proc.communicate()[0].strip('\n')
    return status

def kill_job(job_id):
    if False:
        print('Hello World!')
    '\n    Kill a running LSF job\n    '
    subprocess.call(['bkill', job_id])

class LSFJobTask(luigi.Task):
    """
    Takes care of uploading and executing an LSF job
    """
    n_cpu_flag = luigi.IntParameter(default=2, significant=False)
    shared_tmp_dir = luigi.Parameter(default='/tmp', significant=False)
    resource_flag = luigi.Parameter(default='mem=8192', significant=False)
    memory_flag = luigi.Parameter(default='8192', significant=False)
    queue_flag = luigi.Parameter(default='queue_name', significant=False)
    runtime_flag = luigi.IntParameter(default=60)
    job_name_flag = luigi.Parameter(default='')
    poll_time = luigi.FloatParameter(significant=False, default=5, description='specify the wait time to poll bjobs for the job status')
    save_job_info = luigi.BoolParameter(default=False)
    output = luigi.Parameter(default='')
    extra_bsub_args = luigi.Parameter(default='')
    job_status = None

    def fetch_task_failures(self):
        if False:
            while True:
                i = 10
        '\n        Read in the error file from bsub\n        '
        error_file = os.path.join(self.tmp_dir, 'job.err')
        if os.path.isfile(error_file):
            with open(error_file, 'r') as f_err:
                errors = f_err.readlines()
        else:
            errors = ''
        return errors

    def fetch_task_output(self):
        if False:
            i = 10
            return i + 15
        '\n        Read in the output file\n        '
        if os.path.isfile(os.path.join(self.tmp_dir, 'job.out')):
            with open(os.path.join(self.tmp_dir, 'job.out'), 'r') as f_out:
                outputs = f_out.readlines()
        else:
            outputs = ''
        return outputs

    def _init_local(self):
        if False:
            for i in range(10):
                print('nop')
        base_tmp_dir = self.shared_tmp_dir
        random_id = '%016x' % random.getrandbits(64)
        task_name = random_id + self.task_id
        task_name = task_name.replace('/', '::')
        max_filename_length = os.fstatvfs(0).f_namemax
        self.tmp_dir = os.path.join(base_tmp_dir, task_name[:max_filename_length])
        LOGGER.info('Tmp dir: %s', self.tmp_dir)
        os.makedirs(self.tmp_dir)
        LOGGER.debug('Dumping pickled class')
        self._dump(self.tmp_dir)
        LOGGER.debug('Tarballing dependencies')
        packages = [luigi, __import__(self.__module__, None, None, 'dummy')]
        create_packages_archive(packages, os.path.join(self.tmp_dir, 'packages.tar'))
        self.init_local()

    def init_local(self):
        if False:
            return 10
        '\n        Implement any work to setup any internal datastructure etc here.\n        You can add extra input using the requires_local/input_local methods.\n        Anything you set on the object will be pickled and available on the compute nodes.\n        '
        pass

    def run(self):
        if False:
            return 10
        "\n        The procedure:\n        - Pickle the class\n        - Tarball the dependencies\n        - Construct a bsub argument that runs a generic runner function with the path to the pickled class\n        - Runner function loads the class from pickle\n        - Runner class untars the dependencies\n        - Runner function hits the button on the class's work() method\n        "
        self._init_local()
        self._run_job()

    def work(self):
        if False:
            while True:
                i = 10
        "\n        Subclass this for where you're doing your actual work.\n\n        Why not run(), like other tasks? Because we need run to always be\n        something that the Worker can call, and that's the real logical place to\n        do LSF scheduling.\n        So, the work will happen in work().\n        "
        pass

    def _dump(self, out_dir=''):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dump instance to file.\n        '
        self.job_file = os.path.join(out_dir, 'job-instance.pickle')
        if self.__module__ == '__main__':
            dump_inst = pickle.dumps(self)
            module_name = os.path.basename(sys.argv[0]).rsplit('.', 1)[0]
            dump_inst = dump_inst.replace('(c__main__', '(c' + module_name)
            open(self.job_file, 'w').write(dump_inst)
        else:
            pickle.dump(self, open(self.job_file, 'w'))

    def _run_job(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Build a bsub argument that will run lsf_runner.py on the directory we've specified.\n        "
        args = []
        if isinstance(self.output(), list):
            log_output = os.path.split(self.output()[0].path)
        else:
            log_output = os.path.split(self.output().path)
        args += ['bsub', '-q', self.queue_flag]
        args += ['-n', str(self.n_cpu_flag)]
        args += ['-M', str(self.memory_flag)]
        args += ['-R', 'rusage[%s]' % self.resource_flag]
        args += ['-W', str(self.runtime_flag)]
        if self.job_name_flag:
            args += ['-J', str(self.job_name_flag)]
        args += ['-o', os.path.join(log_output[0], 'job.out')]
        args += ['-e', os.path.join(log_output[0], 'job.err')]
        if self.extra_bsub_args:
            args += self.extra_bsub_args.split()
        runner_path = os.path.abspath(lsf_runner.__file__)
        args += [runner_path]
        args += [self.tmp_dir]
        LOGGER.info('### LSF SUBMISSION ARGS: %s', ' '.join([str(a) for a in args]))
        run_job_proc = subprocess.Popen([str(a) for a in args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.tmp_dir)
        output = run_job_proc.communicate()[0]
        LOGGER.info('### JOB SUBMISSION OUTPUT: %s', str(output))
        self.job_id = int(output.split('<')[1].split('>')[0])
        LOGGER.info('Job %ssubmitted as job %s', self.job_name_flag + ' ', str(self.job_id))
        self._track_job()
        if self.save_job_info:
            LOGGER.info('Saving up temporary bits')
            shutil.move(self.tmp_dir, '/'.join(log_output[0:-1]))
        self._finish()

    def _track_job(self):
        if False:
            print('Hello World!')
        time0 = 0
        while True:
            time.sleep(self.poll_time)
            lsf_status = track_job(self.job_id)
            if lsf_status == 'RUN':
                self.job_status = RUNNING
                LOGGER.info('Job is running...')
                if time0 == 0:
                    time0 = int(round(time.time()))
            elif lsf_status == 'PEND':
                self.job_status = PENDING
                LOGGER.info('Job is pending...')
            elif lsf_status == 'DONE' or lsf_status == 'EXIT':
                errors = self.fetch_task_failures()
                if not errors:
                    self.job_status = DONE
                    LOGGER.info('Job is done')
                    time1 = int(round(time.time()))
                    job_name = str(self.job_id)
                    if self.job_name_flag:
                        job_name = '%s %s' % (self.job_name_flag, job_name)
                    LOGGER.info('### JOB COMPLETED: %s in %s seconds', job_name, str(time1 - time0))
                else:
                    self.job_status = FAILED
                    LOGGER.error('Job has FAILED')
                    LOGGER.error('\n\n')
                    LOGGER.error('Traceback: ')
                    for error in errors:
                        LOGGER.error(error)
                break
            elif lsf_status == 'SSUSP':
                self.job_status = PENDING
                LOGGER.info('Job is suspended (basically, pending)...')
            else:
                self.job_status = UNKNOWN
                LOGGER.info('Job status is UNKNOWN!')
                LOGGER.info('Status is : %s', lsf_status)
                break

    def _finish(self):
        if False:
            i = 10
            return i + 15
        LOGGER.info('Cleaning up temporary bits')
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            LOGGER.info('Removing directory %s', self.tmp_dir)
            shutil.rmtree(self.tmp_dir)

    def __del__(self):
        if False:
            print('Hello World!')
        pass

class LocalLSFJobTask(LSFJobTask):
    """
    A local version of JobTask, for easier debugging.
    """

    def run(self):
        if False:
            print('Hello World!')
        self.init_local()
        self.work()