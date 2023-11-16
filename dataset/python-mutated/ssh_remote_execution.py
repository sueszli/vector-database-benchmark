from collections import defaultdict
import luigi
from luigi.contrib.ssh import RemoteContext, RemoteTarget
from luigi.mock import MockTarget
SSH_HOST = 'some.accessible.host'

class CreateRemoteData(luigi.Task):
    """
    Dump info on running processes on remote host.
    Data is still stored on the remote host
    """

    def output(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on a remote server using SSH.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return RemoteTarget('/tmp/stuff', SSH_HOST)

    def run(self):
        if False:
            i = 10
            return i + 15
        remote = RemoteContext(SSH_HOST)
        print(remote.check_output(['ps aux > {0}'.format(self.output().path)]))

class ProcessRemoteData(luigi.Task):
    """
    Create a toplist of users based on how many running processes they have on a remote machine.

    In this example the processed data is stored in a MockTarget.
    """

    def requires(self):
        if False:
            return 10
        "\n        This task's dependencies:\n\n        * :py:class:`~.CreateRemoteData`\n\n        :return: object (:py:class:`luigi.task.Task`)\n        "
        return CreateRemoteData()

    def run(self):
        if False:
            i = 10
            return i + 15
        processes_per_user = defaultdict(int)
        with self.input().open('r') as infile:
            for line in infile:
                username = line.split()[0]
                processes_per_user[username] += 1
        toplist = sorted(processes_per_user.items(), key=lambda x: x[1], reverse=True)
        with self.output().open('w') as outfile:
            for (user, n_processes) in toplist:
                print(n_processes, user, file=outfile)

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will simulate the creation of a file in a filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`~luigi.target.Target`)\n        '
        return MockTarget('output', mirror_on_stderr=True)