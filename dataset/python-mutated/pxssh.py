"""This class extends pexpect.spawn to specialize setting up SSH connections.
This adds methods for login, logout, and expecting the shell prompt.

PEXPECT LICENSE

    This license is approved by the OSI and FSF as GPL-compatible.
        http://opensource.org/licenses/isc-license.txt

    Copyright (c) 2012, Noah Spurrier <noah@noah.org>
    PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE THIS SOFTWARE FOR ANY
    PURPOSE WITH OR WITHOUT FEE IS HEREBY GRANTED, PROVIDED THAT THE ABOVE
    COPYRIGHT NOTICE AND THIS PERMISSION NOTICE APPEAR IN ALL COPIES.
    THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
from pipenv.vendor.pexpect import ExceptionPexpect, TIMEOUT, EOF, spawn
import time
import os
import sys
import re
__all__ = ['ExceptionPxssh', 'pxssh']

class ExceptionPxssh(ExceptionPexpect):
    """Raised for pxssh exceptions.
    """
if sys.version_info > (3, 0):
    from shlex import quote
else:
    _find_unsafe = re.compile('[^\\w@%+=:,./-]').search

    def quote(s):
        if False:
            return 10
        'Return a shell-escaped version of the string *s*.'
        if not s:
            return "''"
        if _find_unsafe(s) is None:
            return s
        return "'" + s.replace("'", '\'"\'"\'') + "'"

class pxssh(spawn):
    """This class extends pexpect.spawn to specialize setting up SSH
    connections. This adds methods for login, logout, and expecting the shell
    prompt. It does various tricky things to handle many situations in the SSH
    login process. For example, if the session is your first login, then pxssh
    automatically accepts the remote certificate; or if you have public key
    authentication setup then pxssh won't wait for the password prompt.

    pxssh uses the shell prompt to synchronize output from the remote host. In
    order to make this more robust it sets the shell prompt to something more
    unique than just $ or #. This should work on most Borne/Bash or Csh style
    shells.

    Example that runs a few commands on a remote server and prints the result::

        from pipenv.vendor.pexpect import pxssh
        import getpass
        try:
            s = pxssh.pxssh()
            hostname = raw_input('hostname: ')
            username = raw_input('username: ')
            password = getpass.getpass('password: ')
            s.login(hostname, username, password)
            s.sendline('uptime')   # run a command
            s.prompt()             # match the prompt
            print(s.before)        # print everything before the prompt.
            s.sendline('ls -l')
            s.prompt()
            print(s.before)
            s.sendline('df')
            s.prompt()
            print(s.before)
            s.logout()
        except pxssh.ExceptionPxssh as e:
            print("pxssh failed on login.")
            print(e)

    Example showing how to specify SSH options::

        from pipenv.vendor.pexpect import pxssh
        s = pxssh.pxssh(options={
                            "StrictHostKeyChecking": "no",
                            "UserKnownHostsFile": "/dev/null"})
        ...

    Note that if you have ssh-agent running while doing development with pxssh
    then this can lead to a lot of confusion. Many X display managers (xdm,
    gdm, kdm, etc.) will automatically start a GUI agent. You may see a GUI
    dialog box popup asking for a password during development. You should turn
    off any key agents during testing. The 'force_password' attribute will turn
    off public key authentication. This will only work if the remote SSH server
    is configured to allow password logins. Example of using 'force_password'
    attribute::

            s = pxssh.pxssh()
            s.force_password = True
            hostname = raw_input('hostname: ')
            username = raw_input('username: ')
            password = getpass.getpass('password: ')
            s.login (hostname, username, password)

    `debug_command_string` is only for the test suite to confirm that the string
    generated for SSH is correct, using this will not allow you to do
    anything other than get a string back from `pxssh.pxssh.login()`.
    """

    def __init__(self, timeout=30, maxread=2000, searchwindowsize=None, logfile=None, cwd=None, env=None, ignore_sighup=True, echo=True, options={}, encoding=None, codec_errors='strict', debug_command_string=False, use_poll=False):
        if False:
            while True:
                i = 10
        spawn.__init__(self, None, timeout=timeout, maxread=maxread, searchwindowsize=searchwindowsize, logfile=logfile, cwd=cwd, env=env, ignore_sighup=ignore_sighup, echo=echo, encoding=encoding, codec_errors=codec_errors, use_poll=use_poll)
        self.name = '<pxssh>'
        self.UNIQUE_PROMPT = '\\[PEXPECT\\][\\$\\#] '
        self.PROMPT = self.UNIQUE_PROMPT
        self.PROMPT_SET_SH = "PS1='[PEXPECT]\\$ '"
        self.PROMPT_SET_CSH = "set prompt='[PEXPECT]\\$ '"
        self.SSH_OPTS = "-o'RSAAuthentication=no'" + " -o 'PubkeyAuthentication=no'"
        self.force_password = False
        self.debug_command_string = debug_command_string
        self.options = options

    def levenshtein_distance(self, a, b):
        if False:
            return 10
        'This calculates the Levenshtein distance between a and b.\n        '
        (n, m) = (len(a), len(b))
        if n > m:
            (a, b) = (b, a)
            (n, m) = (m, n)
        current = range(n + 1)
        for i in range(1, m + 1):
            (previous, current) = (current, [i] + [0] * n)
            for j in range(1, n + 1):
                (add, delete) = (previous[j] + 1, current[j - 1] + 1)
                change = previous[j - 1]
                if a[j - 1] != b[i - 1]:
                    change = change + 1
                current[j] = min(add, delete, change)
        return current[n]

    def try_read_prompt(self, timeout_multiplier):
        if False:
            i = 10
            return i + 15
        'This facilitates using communication timeouts to perform\n        synchronization as quickly as possible, while supporting high latency\n        connections with a tunable worst case performance. Fast connections\n        should be read almost immediately. Worst case performance for this\n        method is timeout_multiplier * 3 seconds.\n        '
        first_char_timeout = timeout_multiplier * 0.5
        inter_char_timeout = timeout_multiplier * 0.1
        total_timeout = timeout_multiplier * 3.0
        prompt = self.string_type()
        begin = time.time()
        expired = 0.0
        timeout = first_char_timeout
        while expired < total_timeout:
            try:
                prompt += self.read_nonblocking(size=1, timeout=timeout)
                expired = time.time() - begin
                timeout = inter_char_timeout
            except TIMEOUT:
                break
        return prompt

    def sync_original_prompt(self, sync_multiplier=1.0):
        if False:
            i = 10
            return i + 15
        'This attempts to find the prompt. Basically, press enter and record\n        the response; press enter again and record the response; if the two\n        responses are similar then assume we are at the original prompt.\n        This can be a slow function. Worst case with the default sync_multiplier\n        can take 12 seconds. Low latency connections are more likely to fail\n        with a low sync_multiplier. Best case sync time gets worse with a\n        high sync multiplier (500 ms with default). '
        self.sendline()
        time.sleep(0.1)
        try:
            self.try_read_prompt(sync_multiplier)
        except TIMEOUT:
            pass
        self.sendline()
        x = self.try_read_prompt(sync_multiplier)
        self.sendline()
        a = self.try_read_prompt(sync_multiplier)
        self.sendline()
        b = self.try_read_prompt(sync_multiplier)
        ld = self.levenshtein_distance(a, b)
        len_a = len(a)
        if len_a == 0:
            return False
        if float(ld) / len_a < 0.4:
            return True
        return False

    def login(self, server, username=None, password='', terminal_type='ansi', original_prompt='[#$]', login_timeout=10, port=None, auto_prompt_reset=True, ssh_key=None, quiet=True, sync_multiplier=1, check_local_ip=True, password_regex='(?i)(?:password:)|(?:passphrase for key)', ssh_tunnels={}, spawn_local_ssh=True, sync_original_prompt=True, ssh_config=None, cmd='ssh'):
        if False:
            print('Hello World!')
        'This logs the user into the given server.\n\n        It uses \'original_prompt\' to try to find the prompt right after login.\n        When it finds the prompt it immediately tries to reset the prompt to\n        something more easily matched. The default \'original_prompt\' is very\n        optimistic and is easily fooled. It\'s more reliable to try to match the original\n        prompt as exactly as possible to prevent false matches by server\n        strings such as the "Message Of The Day". On many systems you can\n        disable the MOTD on the remote server by creating a zero-length file\n        called :file:`~/.hushlogin` on the remote server. If a prompt cannot be found\n        then this will not necessarily cause the login to fail. In the case of\n        a timeout when looking for the prompt we assume that the original\n        prompt was so weird that we could not match it, so we use a few tricks\n        to guess when we have reached the prompt. Then we hope for the best and\n        blindly try to reset the prompt to something more unique. If that fails\n        then login() raises an :class:`ExceptionPxssh` exception.\n\n        In some situations it is not possible or desirable to reset the\n        original prompt. In this case, pass ``auto_prompt_reset=False`` to\n        inhibit setting the prompt to the UNIQUE_PROMPT. Remember that pxssh\n        uses a unique prompt in the :meth:`prompt` method. If the original prompt is\n        not reset then this will disable the :meth:`prompt` method unless you\n        manually set the :attr:`PROMPT` attribute.\n\n        Set ``password_regex`` if there is a MOTD message with `password` in it.\n        Changing this is like playing in traffic, don\'t (p)expect it to match straight\n        away.\n\n        If you require to connect to another SSH server from the your original SSH\n        connection set ``spawn_local_ssh`` to `False` and this will use your current\n        session to do so. Setting this option to `False` and not having an active session\n        will trigger an error.\n\n        Set ``ssh_key`` to a file path to an SSH private key to use that SSH key\n        for the session authentication.\n        Set ``ssh_key`` to `True` to force passing the current SSH authentication socket\n        to the desired ``hostname``.\n\n        Set ``ssh_config`` to a file path string of an SSH client config file to pass that\n        file to the client to handle itself. You may set any options you wish in here, however\n        doing so will require you to post extra information that you may not want to if you\n        run into issues.\n\n        Alter the ``cmd`` to change the ssh client used, or to prepend it with network\n        namespaces. For example ```cmd="ip netns exec vlan2 ssh"``` to execute the ssh in\n        network namespace named ```vlan```.\n        '
        session_regex_array = ['(?i)are you sure you want to continue connecting', original_prompt, password_regex, '(?i)permission denied', '(?i)terminal type', TIMEOUT]
        session_init_regex_array = []
        session_init_regex_array.extend(session_regex_array)
        session_init_regex_array.extend(['(?i)connection closed by remote host', EOF])
        ssh_options = ''.join([" -o '%s=%s'" % (o, v) for (o, v) in self.options.items()])
        if quiet:
            ssh_options = ssh_options + ' -q'
        if not check_local_ip:
            ssh_options = ssh_options + " -o'NoHostAuthenticationForLocalhost=yes'"
        if self.force_password:
            ssh_options = ssh_options + ' ' + self.SSH_OPTS
        if ssh_config is not None:
            if spawn_local_ssh and (not os.path.isfile(ssh_config)):
                raise ExceptionPxssh('SSH config does not exist or is not a file.')
            ssh_options = ssh_options + ' -F ' + ssh_config
        if port is not None:
            ssh_options = ssh_options + ' -p %s' % str(port)
        if ssh_key is not None:
            if ssh_key == True:
                ssh_options = ssh_options + ' -A'
            else:
                if spawn_local_ssh and (not os.path.isfile(ssh_key)):
                    raise ExceptionPxssh('private ssh key does not exist or is not a file.')
                ssh_options = ssh_options + ' -i %s' % ssh_key
        if ssh_tunnels != {} and isinstance({}, type(ssh_tunnels)):
            tunnel_types = {'local': 'L', 'remote': 'R', 'dynamic': 'D'}
            for tunnel_type in tunnel_types:
                cmd_type = tunnel_types[tunnel_type]
                if tunnel_type in ssh_tunnels:
                    tunnels = ssh_tunnels[tunnel_type]
                    for tunnel in tunnels:
                        if spawn_local_ssh == False:
                            tunnel = quote(str(tunnel))
                        ssh_options = ssh_options + ' -' + cmd_type + ' ' + str(tunnel)
        if username is not None:
            ssh_options = ssh_options + ' -l ' + username
        elif ssh_config is None:
            raise TypeError('login() needs either a username or an ssh_config')
        else:
            with open(ssh_config, 'rt') as f:
                lines = [l.strip() for l in f.readlines()]
            server_regex = '^Host\\s+%s\\s*$' % server
            user_regex = '^User\\s+\\w+\\s*$'
            config_has_server = False
            server_has_username = False
            for line in lines:
                if not config_has_server and re.match(server_regex, line, re.IGNORECASE):
                    config_has_server = True
                elif config_has_server and 'hostname' in line.lower():
                    pass
                elif config_has_server and 'host' in line.lower():
                    server_has_username = False
                    break
                elif config_has_server and re.match(user_regex, line, re.IGNORECASE):
                    server_has_username = True
                    break
            if lines:
                del line
            del lines
            if not config_has_server:
                raise TypeError('login() ssh_config has no Host entry for %s' % server)
            elif not server_has_username:
                raise TypeError('login() ssh_config has no user entry for %s' % server)
        cmd += ' %s %s' % (ssh_options, server)
        if self.debug_command_string:
            return cmd
        if spawn_local_ssh:
            spawn._spawn(self, cmd)
        else:
            self.sendline(cmd)
        i = self.expect(session_init_regex_array, timeout=login_timeout)
        if i == 0:
            self.sendline('yes')
            i = self.expect(session_regex_array)
        if i == 2:
            self.sendline(password)
            i = self.expect(session_regex_array)
        if i == 4:
            self.sendline(terminal_type)
            i = self.expect(session_regex_array)
        if i == 7:
            self.close()
            raise ExceptionPxssh('Could not establish connection to host')
        if i == 0:
            self.close()
            raise ExceptionPxssh('Weird error. Got "are you sure" prompt twice.')
        elif i == 1:
            pass
        elif i == 2:
            self.close()
            raise ExceptionPxssh('password refused')
        elif i == 3:
            self.close()
            raise ExceptionPxssh('permission denied')
        elif i == 4:
            self.close()
            raise ExceptionPxssh('Weird error. Got "terminal type" prompt twice.')
        elif i == 5:
            pass
        elif i == 6:
            self.close()
            raise ExceptionPxssh('connection closed')
        else:
            self.close()
            raise ExceptionPxssh('unexpected login response')
        if sync_original_prompt:
            if not self.sync_original_prompt(sync_multiplier):
                self.close()
                raise ExceptionPxssh('could not synchronize with original prompt')
        if auto_prompt_reset:
            if not self.set_unique_prompt():
                self.close()
                raise ExceptionPxssh('could not set shell prompt (received: %r, expected: %r).' % (self.before, self.PROMPT))
        return True

    def logout(self):
        if False:
            print('Hello World!')
        'Sends exit to the remote shell.\n\n        If there are stopped jobs then this automatically sends exit twice.\n        '
        self.sendline('exit')
        index = self.expect([EOF, '(?i)there are stopped jobs'])
        if index == 1:
            self.sendline('exit')
            self.expect(EOF)
        self.close()

    def prompt(self, timeout=-1):
        if False:
            for i in range(10):
                print('nop')
        'Match the next shell prompt.\n\n        This is little more than a short-cut to the :meth:`~pexpect.spawn.expect`\n        method. Note that if you called :meth:`login` with\n        ``auto_prompt_reset=False``, then before calling :meth:`prompt` you must\n        set the :attr:`PROMPT` attribute to a regex that it will use for\n        matching the prompt.\n\n        Calling :meth:`prompt` will erase the contents of the :attr:`before`\n        attribute even if no prompt is ever matched. If timeout is not given or\n        it is set to -1 then self.timeout is used.\n\n        :return: True if the shell prompt was matched, False if the timeout was\n                 reached.\n        '
        if timeout == -1:
            timeout = self.timeout
        i = self.expect([self.PROMPT, TIMEOUT], timeout=timeout)
        if i == 1:
            return False
        return True

    def set_unique_prompt(self):
        if False:
            print('Hello World!')
        "This sets the remote prompt to something more unique than ``#`` or ``$``.\n        This makes it easier for the :meth:`prompt` method to match the shell prompt\n        unambiguously. This method is called automatically by the :meth:`login`\n        method, but you may want to call it manually if you somehow reset the\n        shell prompt. For example, if you 'su' to a different user then you\n        will need to manually reset the prompt. This sends shell commands to\n        the remote host to set the prompt, so this assumes the remote host is\n        ready to receive commands.\n\n        Alternatively, you may use your own prompt pattern. In this case you\n        should call :meth:`login` with ``auto_prompt_reset=False``; then set the\n        :attr:`PROMPT` attribute to a regular expression. After that, the\n        :meth:`prompt` method will try to match your prompt pattern.\n        "
        self.sendline('unset PROMPT_COMMAND')
        self.sendline(self.PROMPT_SET_SH)
        i = self.expect([TIMEOUT, self.PROMPT], timeout=10)
        if i == 0:
            self.sendline(self.PROMPT_SET_CSH)
            i = self.expect([TIMEOUT, self.PROMPT], timeout=10)
            if i == 0:
                return False
        return True