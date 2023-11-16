__author__ = 'Cyril Jaquier, Yaroslav Halchenko'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier, 2013- Yaroslav Halchenko'
__license__ = 'GPL'
from ..exceptions import UnknownJailException, DuplicateJailException
from ..helpers import getLogger, logging
logSys = getLogger(__name__)

class Beautifier:

    def __init__(self, cmd=None):
        if False:
            i = 10
            return i + 15
        self.__inputCmd = cmd

    def setInputCmd(self, cmd):
        if False:
            for i in range(10):
                print('nop')
        self.__inputCmd = cmd

    def getInputCmd(self):
        if False:
            i = 10
            return i + 15
        return self.__inputCmd

    def beautify(self, response):
        if False:
            while True:
                i = 10
        logSys.log(5, 'Beautify ' + repr(response) + ' with ' + repr(self.__inputCmd))
        inC = self.__inputCmd
        msg = response
        try:
            if inC[0] == 'ping':
                msg = 'Server replied: ' + response
            elif inC[0] == 'version':
                msg = response
            elif inC[0] == 'start':
                msg = 'Jail started'
            elif inC[0] == 'stop':
                if len(inC) == 1:
                    if response is None:
                        msg = 'Shutdown successful'
                elif response is None:
                    msg = 'Jail stopped'
            elif inC[0] == 'add':
                msg = 'Added jail ' + response
            elif inC[0] == 'flushlogs':
                msg = 'logs: ' + response
            elif inC[0] == 'echo':
                msg = ' '.join(msg)
            elif inC[0:1] == ['status']:
                if len(inC) > 1:
                    msg = ['Status for the jail: %s' % inC[1]]
                    for (n, res1) in enumerate(response):
                        prefix1 = '`-' if n == len(response) - 1 else '|-'
                        msg.append('%s %s' % (prefix1, res1[0]))
                        prefix1 = '   ' if n == len(response) - 1 else '|  '
                        for (m, res2) in enumerate(res1[1]):
                            prefix2 = prefix1 + ('`-' if m == len(res1[1]) - 1 else '|-')
                            val = ' '.join(map(str, res2[1])) if isinstance(res2[1], list) else res2[1]
                            msg.append('%s %s:\t%s' % (prefix2, res2[0], val))
                else:
                    msg = ['Status']
                    for (n, res1) in enumerate(response):
                        prefix1 = '`-' if n == len(response) - 1 else '|-'
                        val = ' '.join(map(str, res1[1])) if isinstance(res1[1], list) else res1[1]
                        msg.append('%s %s:\t%s' % (prefix1, res1[0], val))
                msg = '\n'.join(msg)
            elif len(inC) < 2:
                pass
            elif inC[1] == 'syslogsocket':
                msg = 'Current syslog socket is:\n'
                msg += '`- ' + response
            elif inC[1] == 'logtarget':
                msg = 'Current logging target is:\n'
                msg += '`- ' + response
            elif inC[1:2] == ['loglevel']:
                msg = 'Current logging level is '
                msg += repr(logging.getLevelName(response) if isinstance(response, int) else response)
            elif inC[1] == 'dbfile':
                if response is None:
                    msg = 'Database currently disabled'
                else:
                    msg = 'Current database file is:\n'
                    msg += '`- ' + response
            elif inC[1] == 'dbpurgeage':
                if response is None:
                    msg = 'Database currently disabled'
                else:
                    msg = 'Current database purge age is:\n'
                    msg += '`- %iseconds' % response
            elif len(inC) < 3:
                pass
            elif inC[2] in ('logpath', 'addlogpath', 'dellogpath'):
                if len(response) == 0:
                    msg = 'No file is currently monitored'
                else:
                    msg = 'Current monitored log file(s):\n'
                    for path in response[:-1]:
                        msg += '|- ' + path + '\n'
                    msg += '`- ' + response[-1]
            elif inC[2] == 'logencoding':
                msg = 'Current log encoding is set to:\n'
                msg += response
            elif inC[2] in ('journalmatch', 'addjournalmatch', 'deljournalmatch'):
                if len(response) == 0:
                    msg = 'No journal match filter set'
                else:
                    msg = 'Current match filter:\n'
                    msg += ' + '.join((' '.join(res) for res in response))
            elif inC[2] == 'datepattern':
                msg = 'Current date pattern set to: '
                if response is None:
                    msg += 'Not set/required'
                elif response[0] is None:
                    msg += '%s' % response[1]
                else:
                    msg += '%s (%s)' % response
            elif inC[2] in ('ignoreip', 'addignoreip', 'delignoreip'):
                if len(response) == 0:
                    msg = 'No IP address/network is ignored'
                else:
                    msg = 'These IP addresses/networks are ignored:\n'
                    for ip in response[:-1]:
                        msg += '|- ' + ip + '\n'
                    msg += '`- ' + response[-1]
            elif inC[2] in ('failregex', 'addfailregex', 'delfailregex', 'ignoreregex', 'addignoreregex', 'delignoreregex'):
                if len(response) == 0:
                    msg = 'No regular expression is defined'
                else:
                    msg = 'The following regular expression are defined:\n'
                    c = 0
                    for l in response[:-1]:
                        msg += '|- [' + str(c) + ']: ' + l + '\n'
                        c += 1
                    msg += '`- [' + str(c) + ']: ' + response[-1]
            elif inC[2] == 'actions':
                if len(response) == 0:
                    msg = 'No actions for jail %s' % inC[1]
                else:
                    msg = 'The jail %s has the following actions:\n' % inC[1]
                    msg += ', '.join(response)
            elif inC[2] == 'actionproperties':
                if len(response) == 0:
                    msg = 'No properties for jail %s action %s' % (inC[1], inC[3])
                else:
                    msg = 'The jail %s action %s has the following properties:\n' % (inC[1], inC[3])
                    msg += ', '.join(response)
            elif inC[2] == 'actionmethods':
                if len(response) == 0:
                    msg = 'No methods for jail %s action %s' % (inC[1], inC[3])
                else:
                    msg = 'The jail %s action %s has the following methods:\n' % (inC[1], inC[3])
                    msg += ', '.join(response)
            elif inC[2] == 'banip' and inC[0] == 'get':
                if isinstance(response, list):
                    sep = ' ' if len(inC) <= 3 else inC[3]
                    if sep == '--with-time':
                        sep = '\n'
                    msg = sep.join(response)
        except Exception:
            logSys.warning('Beautifier error. Please report the error')
            logSys.error('Beautify %r with %r failed', response, self.__inputCmd, exc_info=logSys.getEffectiveLevel() <= logging.DEBUG)
            msg = repr(msg) + repr(response)
        return msg

    def beautifyError(self, response):
        if False:
            for i in range(10):
                print('nop')
        logSys.debug('Beautify (error) %r with %r', response, self.__inputCmd)
        msg = response
        if isinstance(response, UnknownJailException):
            msg = "Sorry but the jail '" + response.args[0] + "' does not exist"
        elif isinstance(response, IndexError):
            msg = 'Sorry but the command is invalid'
        elif isinstance(response, DuplicateJailException):
            msg = "The jail '" + response.args[0] + "' already exists"
        return msg