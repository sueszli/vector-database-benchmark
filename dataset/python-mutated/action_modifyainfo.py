from fail2ban.server.action import ActionBase

class TestAction(ActionBase):

    def ban(self, aInfo):
        if False:
            return 10
        del aInfo['ip']
        self._logSys.info('%s ban deleted aInfo IP', self._name)

    def unban(self, aInfo):
        if False:
            for i in range(10):
                print('nop')
        del aInfo['ip']
        self._logSys.info('%s unban deleted aInfo IP', self._name)

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError('intended error')
Action = TestAction