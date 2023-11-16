"""
Provides functionality to run a Hadoop job using a Jar
"""
import logging
import os
import pipes
import random
import warnings
import luigi.contrib.hadoop
import luigi.contrib.hdfs
logger = logging.getLogger('luigi-interface')

def fix_paths(job):
    if False:
        print('Hello World!')
    '\n    Coerce input arguments to use temporary files when used for output.\n\n    Return a list of temporary file pairs (tmpfile, destination path) and\n    a list of arguments.\n\n    Converts each HdfsTarget to a string for the path.\n    '
    tmp_files = []
    args = []
    for x in job.args():
        if isinstance(x, luigi.contrib.hdfs.HdfsTarget):
            if x.exists() or not job.atomic_output():
                args.append(x.path)
            else:
                x_path_no_slash = x.path[:-1] if x.path[-1] == '/' else x.path
                y = luigi.contrib.hdfs.HdfsTarget(x_path_no_slash + '-luigi-tmp-%09d' % random.randrange(0, 10000000000))
                tmp_files.append((y, x_path_no_slash))
                logger.info('Using temp path: %s for path %s', y.path, x.path)
                args.append(y.path)
        else:
            try:
                args.append(x.path)
            except AttributeError:
                args.append(str(x))
    return (tmp_files, args)

class HadoopJarJobError(Exception):
    pass

class HadoopJarJobRunner(luigi.contrib.hadoop.JobRunner):
    """
    JobRunner for `hadoop jar` commands. Used to run a HadoopJarJobTask.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        pass

    def run_job(self, job, tracking_url_callback=None):
        if False:
            return 10
        if tracking_url_callback is not None:
            warnings.warn('tracking_url_callback argument is deprecated, task.set_tracking_url is used instead.', DeprecationWarning)
        if not job.jar():
            raise HadoopJarJobError('Jar not defined')
        hadoop_arglist = luigi.contrib.hdfs.load_hadoop_cmd() + ['jar', job.jar()]
        if job.main():
            hadoop_arglist.append(job.main())
        jobconfs = job.jobconfs()
        for jc in jobconfs:
            hadoop_arglist += ['-D' + jc]
        (tmp_files, job_args) = fix_paths(job)
        hadoop_arglist += job_args
        ssh_config = job.ssh()
        if ssh_config:
            host = ssh_config.get('host', None)
            key_file = ssh_config.get('key_file', None)
            username = ssh_config.get('username', None)
            if not host or not key_file or (not username):
                raise HadoopJarJobError('missing some config for HadoopRemoteJarJobRunner')
            arglist = ['ssh', '-i', key_file, '-o', 'BatchMode=yes']
            if ssh_config.get('no_host_key_check', False):
                arglist += ['-o', 'UserKnownHostsFile=/dev/null', '-o', 'StrictHostKeyChecking=no']
            arglist.append('{}@{}'.format(username, host))
            hadoop_arglist = [pipes.quote(arg) for arg in hadoop_arglist]
            arglist.append(' '.join(hadoop_arglist))
        else:
            if not os.path.exists(job.jar()):
                logger.error("Can't find jar: %s, full path %s", job.jar(), os.path.abspath(job.jar()))
                raise HadoopJarJobError('job jar does not exist')
            arglist = hadoop_arglist
        luigi.contrib.hadoop.run_and_track_hadoop_job(arglist, job.set_tracking_url)
        for (a, b) in tmp_files:
            a.move(b)

class HadoopJarJobTask(luigi.contrib.hadoop.BaseHadoopJobTask):
    """
    A job task for `hadoop jar` commands that define a jar and (optional) main method.
    """

    def jar(self):
        if False:
            while True:
                i = 10
        '\n        Path to the jar for this Hadoop Job.\n        '
        return None

    def main(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        optional main method for this Hadoop Job.\n        '
        return None

    def job_runner(self):
        if False:
            return 10
        return HadoopJarJobRunner()

    def atomic_output(self):
        if False:
            i = 10
            return i + 15
        '\n        If True, then rewrite output arguments to be temp locations and\n        atomically move them into place after the job finishes.\n        '
        return True

    def ssh(self):
        if False:
            while True:
                i = 10
        '\n        Set this to run hadoop command remotely via ssh. It needs to be a dict that looks like\n        {"host": "myhost", "key_file": None, "username": None, ["no_host_key_check": False]}\n        '
        return None

    def args(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns an array of args to pass to the job (after hadoop jar <jar> <main>).\n        '
        return []