"""
    salt.utils.vt_helper
    ~~~~~~~~~~~~~~~~~~~~

    VT Helper

    This module provides the SSHConnection to expose an SSH connection object
    allowing users to programmatically execute commands on a remote server using
    Salt VT.
"""
import logging
import os
import re
from salt.utils.vt import Terminal, TerminalException
SSH_PASSWORD_PROMPT_RE = re.compile('(?:.*)[Pp]assword(?: for .*)?:', re.M)
KEY_VALID_RE = re.compile('.*\\(yes\\/no\\).*')
log = logging.getLogger(__name__)

class SSHConnection:
    """
    SSH Connection to a remote server.
    """

    def __init__(self, username='salt', password='password', host='localhost', key_accept=False, prompt='(Cmd)', passwd_retries=3, linesep=os.linesep, ssh_args=''):
        if False:
            while True:
                i = 10
        "\n        Establishes a connection to the remote server.\n\n        The format for parameters is:\n\n        username (string): The username to use for this\n            ssh connection. Defaults to root.\n        password (string): The password to use for this\n            ssh connection. Defaults to password.\n        host (string): The host to connect to.\n            Defaults to localhost.\n        key_accept (boolean): Should we accept this host's key\n            and add it to the known_hosts file? Defaults to False.\n        prompt (string): The shell prompt (regex) on the server.\n            Prompt is compiled into a regular expression.\n            Defaults to (Cmd)\n        passwd_retries (int): How many times should I try to send the password?\n            Defaults to 3.\n        linesep (string): The line separator to use when sending\n            commands to the server. Defaults to os.linesep.\n        ssh_args (string): Extra ssh args to use with ssh.\n             Example: '-o PubkeyAuthentication=no'\n        "
        self.conn = Terminal('ssh {} -l {} {}'.format(ssh_args, username, host), shell=True, log_stdout=True, log_stdout_level='trace', log_stderr=True, log_stderr_level='trace', stream_stdout=False, stream_stderr=False)
        sent_passwd = 0
        self.prompt_re = re.compile(prompt)
        self.linesep = linesep
        while self.conn.has_unread_data:
            (stdout, stderr) = self.conn.recv()
            if stdout and SSH_PASSWORD_PROMPT_RE.search(stdout):
                if not password:
                    log.error('Failure while authentication.')
                    raise TerminalException('Permission denied, no authentication information')
                if sent_passwd < passwd_retries:
                    self.conn.sendline(password, self.linesep)
                    sent_passwd += 1
                    continue
                else:
                    raise TerminalException('Password authentication failed')
            elif stdout and KEY_VALID_RE.search(stdout):
                if key_accept:
                    log.info('Adding %s to known_hosts', host)
                    self.conn.sendline('yes')
                    continue
                else:
                    self.conn.sendline('no')
            elif stdout and self.prompt_re.search(stdout):
                break

    def sendline(self, cmd):
        if False:
            while True:
                i = 10
        '\n        Send this command to the server and\n        return a tuple of the output and the stderr.\n\n        The format for parameters is:\n\n        cmd (string): The command to send to the sever.\n        '
        self.conn.sendline(cmd, self.linesep)
        ret_stdout = []
        ret_stderr = []
        while self.conn.has_unread_data:
            (stdout, stderr) = self.conn.recv()
            if stdout:
                ret_stdout.append(stdout)
            if stderr:
                log.debug('Error while executing command.')
                ret_stderr.append(stderr)
            if stdout and self.prompt_re.search(stdout):
                break
        return (''.join(ret_stdout), ''.join(ret_stderr))

    def close_connection(self):
        if False:
            print('Hello World!')
        '\n        Close the server connection\n        '
        self.conn.close(terminate=True, kill=True)