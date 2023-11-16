import time
import uuid
import base64
from cme.helpers.misc import gen_random_string
from impacket.dcerpc.v5.dtypes import NULL
from impacket.dcerpc.v5.dcomrt import DCOMConnection
from impacket.dcerpc.v5.dcom.wmi import CLSID_WbemLevel1Login, IID_IWbemLevel1Login, WBEM_FLAG_FORWARD_ONLY, IWbemLevel1Login

class WMIEXEC:

    def __init__(self, host, username, password, domain, lmhash, nthash, doKerberos, kdcHost, aesKey, logger, interval_time, codec):
        if False:
            for i in range(10):
                print('nop')
        self.__host = host
        self.__username = username
        self.__password = password
        self.__domain = domain
        self.__lmhash = lmhash
        self.__nthash = nthash
        self.__doKerberos = doKerberos
        self.__kdcHost = kdcHost
        self.__aesKey = aesKey
        self.logger = logger
        self.__interval_time = interval_time
        self.__registry_Path = ''
        self.__outputBuffer = ''
        self.__retOutput = True
        self.__shell = 'cmd.exe /Q /c '
        self.__pwd = str('C:\\')
        self.__codec = codec
        self.__dcom = DCOMConnection(self.__host, self.__username, self.__password, self.__domain, self.__lmhash, self.__nthash, oxidResolver=True, doKerberos=self.__doKerberos, kdcHost=self.__kdcHost, aesKey=self.__aesKey)
        iInterface = self.__dcom.CoCreateInstanceEx(CLSID_WbemLevel1Login, IID_IWbemLevel1Login)
        iWbemLevel1Login = IWbemLevel1Login(iInterface)
        self.__iWbemServices = iWbemLevel1Login.NTLMLogin('//./root/cimv2', NULL, NULL)
        iWbemLevel1Login.RemRelease()
        (self.__win32Process, _) = self.__iWbemServices.GetObject('Win32_Process')

    def execute(self, command, output=False):
        if False:
            while True:
                i = 10
        self.__retOutput = output
        if self.__retOutput:
            self.execute_WithOutput(command)
        else:
            command = self.__shell + command
            self.execute_remote(command)
        self.__dcom.disconnect()
        return self.__outputBuffer

    def execute_remote(self, command):
        if False:
            i = 10
            return i + 15
        self.logger.info(f'Executing command: {command}')
        try:
            self.__win32Process.Create(command, self.__pwd, None)
        except Exception as e:
            self.logger.error(str(e))

    def execute_WithOutput(self, command):
        if False:
            print('Hello World!')
        result_output = f'C:\\windows\\temp\\{str(uuid.uuid4())}.txt'
        result_output_b64 = f'C:\\windows\\temp\\{str(uuid.uuid4())}.txt'
        keyName = str(uuid.uuid4())
        self.__registry_Path = f'Software\\Classes\\{gen_random_string(6)}'
        command = f'{self.__shell} {command} 1> {result_output} 2>&1 && certutil -encodehex -f {result_output} {result_output_b64} 0x40000001 && for /F "usebackq" %G in ("{result_output_b64}") do reg add HKLM\\{self.__registry_Path} /v {keyName} /t REG_SZ /d "%G" /f && del /q /f /s {result_output} {result_output_b64}'
        self.execute_remote(command)
        self.logger.info('Waiting {}s for command completely executed.'.format(self.__interval_time))
        time.sleep(self.__interval_time)
        self.queryRegistry(keyName)

    def queryRegistry(self, keyName):
        if False:
            i = 10
            return i + 15
        try:
            self.logger.debug(f'Querying registry key: HKLM\\{self.__registry_Path}')
            (descriptor, _) = self.__iWbemServices.GetObject('StdRegProv')
            descriptor = descriptor.SpawnInstance()
            retVal = descriptor.GetStringValue(2147483650, self.__registry_Path, keyName)
            self.__outputBuffer = base64.b64decode(retVal.sValue).decode(self.__codec, errors='replace').rstrip('\r\n')
        except Exception as e:
            self.logger.fail(f"""WMIEXEC: Get output file error, maybe command not executed successfully or got detected by AV software, please increase the interval time of command execution with "--interval-time" option. If it's still failing maybe something is blocking the schedule job in vbscript, try another exec method""")
        try:
            self.logger.debug(f'Removing temporary registry path: HKLM\\{self.__registry_Path}')
            retVal = descriptor.DeleteKey(2147483650, self.__registry_Path)
        except Exception as e:
            self.logger.debug(f'Target: {self.__host} removing temporary registry path error: {str(e)}')