from __future__ import division
from __future__ import print_function
import pytest
import unittest
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import tsch, atsvc, sasec
from impacket.dcerpc.v5.atsvc import AT_INFO
from impacket.dcerpc.v5.dtypes import NULL
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_LEVEL_PKT_PRIVACY
from impacket.system_errors import ERROR_NOT_SUPPORTED

class ATSVCTests(DCERPCTests):
    iface_uuid = atsvc.MSRPC_UUID_ATSVC
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\atsvc]'
    authn = True
    authn_level = RPC_C_AUTHN_LEVEL_PKT_PRIVACY

    def test_NetrJobEnum(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        request = atsvc.NetrJobEnum()
        request['ServerName'] = NULL
        request['pEnumContainer']['Buffer'] = NULL
        request['PreferedMaximumLength'] = 4294967295
        try:
            resp = dce.request(request)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return

    def test_hNetrJobEnum(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        try:
            resp = atsvc.hNetrJobEnum(dce, NULL, NULL, 4294967295)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return

    def test_hNetrJobAdd_hNetrJobEnum_hNetrJobDel(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\BTO\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        resp = atsvc.hNetrJobEnum(dce)
        resp.dump()
        for job in resp['pEnumContainer']['Buffer']:
            resp = atsvc.hNetrJobDel(dce, NULL, job['JobId'], job['JobId'])
            resp.dump()

    def test_NetrJobAdd_NetrJobEnum_NetrJobDel(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        request = atsvc.NetrJobAdd()
        request['ServerName'] = NULL
        request['pAtInfo']['JobTime'] = NULL
        request['pAtInfo']['DaysOfMonth'] = 0
        request['pAtInfo']['DaysOfWeek'] = 0
        request['pAtInfo']['Flags'] = 0
        request['pAtInfo']['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\BTO\x00'
        try:
            resp = dce.request(request)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        request = atsvc.NetrJobEnum()
        request['ServerName'] = NULL
        request['pEnumContainer']['Buffer'] = NULL
        request['PreferedMaximumLength'] = 4294967295
        resp = dce.request(request)
        resp.dump()
        for job in resp['pEnumContainer']['Buffer']:
            request = atsvc.NetrJobDel()
            request['ServerName'] = NULL
            request['MinJobId'] = job['JobId']
            request['MaxJobId'] = job['JobId']
            resp = dce.request(request)
            resp.dump()

    def test_NetrJobAdd_NetrJobGetInfo_NetrJobDel(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        request = atsvc.NetrJobAdd()
        request['ServerName'] = NULL
        request['pAtInfo']['JobTime'] = NULL
        request['pAtInfo']['DaysOfMonth'] = 0
        request['pAtInfo']['DaysOfWeek'] = 0
        request['pAtInfo']['Flags'] = 0
        request['pAtInfo']['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\BTO\x00'
        try:
            resp = dce.request(request)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        request = atsvc.NetrJobGetInfo()
        request['ServerName'] = NULL
        request['JobId'] = resp['pJobId']
        resp2 = dce.request(request)
        resp2.dump()
        request = atsvc.NetrJobDel()
        request['ServerName'] = NULL
        request['MinJobId'] = resp['pJobId']
        request['MaxJobId'] = resp['pJobId']
        resp = dce.request(request)
        resp.dump()

    def test_hNetrJobAdd_hNetrJobGetInfo_hNetrJobDel(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\BTO\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        resp2 = atsvc.hNetrJobGetInfo(dce, NULL, resp['pJobId'])
        resp2.dump()
        resp = atsvc.hNetrJobDel(dce, NULL, resp['pJobId'], resp['pJobId'])
        resp.dump()

class SASECTests(DCERPCTests):
    iface_uuid = sasec.MSRPC_UUID_SASEC
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\atsvc]'
    authn = True
    authn_level = RPC_C_AUTHN_LEVEL_PKT_PRIVACY

    def test_SASetAccountInformation(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        request = sasec.SASetAccountInformation()
        request['Handle'] = NULL
        request['pwszJobName'] = 'MyJob.job\x00'
        request['pwszAccount'] = self.username + '\x00'
        request['pwszPassword'] = self.password + '\x00'
        request['dwJobFlags'] = sasec.TASK_FLAG_RUN_ONLY_IF_LOGGED_ON
        try:
            resp = dce.request(request)
            resp.dump()
        except sasec.DCERPCSessionError as e:
            if e.get_error_code() != 2147942402:
                raise

    def test_hSASetAccountInformation(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        try:
            resp = sasec.hSASetAccountInformation(dce, NULL, 'MyJob.job', self.username, self.password, 0)
            resp.dump()
        except sasec.DCERPCSessionError as e:
            if e.get_error_code() != 2147942402:
                raise

    def test_SASetNSAccountInformation(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        request = sasec.SASetNSAccountInformation()
        request['Handle'] = NULL
        request['pwszAccount'] = self.username + '\x00'
        request['pwszPassword'] = self.password + '\x00'
        resp = dce.request(request)
        resp.dump()

    def test_hSASetNSAccountInformation(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        resp = sasec.hSASetNSAccountInformation(dce, NULL, self.username, self.password)
        resp.dump()

    def test_SAGetNSAccountInformation(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        request = sasec.SAGetNSAccountInformation()
        request['Handle'] = NULL
        request['ccBufferSize'] = 25
        for i in range(request['ccBufferSize']):
            request['wszBuffer'].append(0)
        resp = dce.request(request)
        resp.dump()

    def test_hSAGetNSAccountInformation(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        resp = sasec.hSAGetNSAccountInformation(dce, NULL, 25)
        resp.dump()

    def test_SAGetAccountInformation(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        request = sasec.SAGetAccountInformation()
        request['Handle'] = NULL
        request['pwszJobName'] = 'MyJob.job\x00'
        request['ccBufferSize'] = 15
        for i in range(request['ccBufferSize']):
            request['wszBuffer'].append(0)
        try:
            resp = dce.request(request)
            resp.dump()
        except sasec.DCERPCSessionError as e:
            if e.get_error_code() != 2147942402:
                raise

    def test_hSAGetAccountInformation(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        try:
            resp = sasec.hSAGetAccountInformation(dce, NULL, 'MyJob.job', 15)
            resp.dump()
        except sasec.DCERPCSessionError as e:
            if e.get_error_code() != 2147942402:
                raise

class TSCHTests(DCERPCTests):
    iface_uuid = tsch.MSRPC_UUID_TSCHS
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\atsvc]'
    authn = True
    authn_level = RPC_C_AUTHN_LEVEL_PKT_PRIVACY

    def test_SchRpcHighestVersion(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcHighestVersion()
        resp = dce.request(request)
        resp.dump()

    def test_hSchRpcHighestVersion(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        resp = tsch.hSchRpcHighestVersion(dce)
        resp.dump()

    @pytest.mark.skip(reason='Disabled test')
    def test_SchRpcRegisterTask(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        xml = '\n<!-- Task -->\n<xs:complexType name="taskType">\n<xs:all>\n<xs:element name="RegistrationInfo" type="registrationInfoType" minOccurs="0"/>\n<xs:element name="Triggers" type="triggersType" minOccurs="0"/>\n<xs:element name="Settings" type="settingsType" minOccurs="0"/>\n<xs:element name="Data" type="dataType" minOccurs="0"/>\n<xs:element name="Principals" type="principalsType" minOccurs="0"/>\n<xs:element name="Actions" type="actionsType"/>\n</xs:all>\n<xs:attribute name="version" type="versionType" use="optional"/> </xs:complexType>\x00\n'
        request = tsch.SchRpcRegisterTask()
        request['path'] = NULL
        request['xml'] = xml
        request['flags'] = 1
        request['sddl'] = NULL
        request['logonType'] = tsch.TASK_LOGON_NONE
        request['cCreds'] = 0
        request['pCreds'] = NULL
        resp = dce.request(request)
        resp.dump()

    def test_SchRpcRetrieveTask(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\BTO\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        request = tsch.SchRpcRetrieveTask()
        request['path'] = '\\At%d.job\x00' % jobId
        request['lpcwszLanguagesBuffer'] = '\x00'
        request['pulNumLanguages'] = 0
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if e.get_error_code() != 2147942402:
                raise
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_hSchRpcRetrieveTask(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        try:
            resp = tsch.hSchRpcRetrieveTask(dce, '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag\x00')
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass

    def test_SchRpcCreateFolder_SchRpcEnumFolders_SchRpcDelete(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcCreateFolder()
        request['path'] = '\\Beto\x00'
        request['sddl'] = NULL
        request['flags'] = 0
        resp = dce.request(request)
        resp.dump()
        request = tsch.SchRpcEnumFolders()
        request['path'] = '\\\x00'
        request['flags'] = tsch.TASK_ENUM_HIDDEN
        request['startIndex'] = 0
        request['cRequested'] = 10
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass
        request = tsch.SchRpcDelete()
        request['path'] = '\\Beto\x00'
        request['flags'] = 0
        resp = dce.request(request)
        resp.dump()

    def test_hSchRpcCreateFolder_hSchRpcEnumFolders_hSchRpcDelete(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        resp = tsch.hSchRpcCreateFolder(dce, '\\Beto')
        resp.dump()
        resp = tsch.hSchRpcEnumFolders(dce, '\\')
        resp.dump()
        resp = tsch.hSchRpcDelete(dce, '\\Beto')
        resp.dump()

    def test_SchRpcEnumTasks(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\BTO\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        request = tsch.SchRpcEnumTasks()
        request['path'] = '\\\x00'
        request['flags'] = tsch.TASK_ENUM_HIDDEN
        request['startIndex'] = 0
        request['cRequested'] = 10
        resp = dce.request(request)
        resp.dump()
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_hSchRpcEnumTasks(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\BTO\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        resp = tsch.hSchRpcEnumTasks(dce, '\\')
        resp.dump()
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_SchRpcEnumInstances(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcEnumInstances()
        request['path'] = '\\\x00'
        request['flags'] = tsch.TASK_ENUM_HIDDEN
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if e.get_error_code() != 2147942402:
                raise

    def test_hSchRpcEnumInstances(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        try:
            resp = tsch.hSchRpcEnumInstances(dce, '\\')
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if e.get_error_code() != 2147942402:
                raise

    def test_SchRpcRun(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        request = tsch.SchRpcRun()
        request['path'] = '\\At%d\x00' % jobId
        request['cArgs'] = 0
        request['pArgs'] = NULL
        request['flags'] = tsch.TASK_RUN_AS_SELF
        request['sessionId'] = 0
        request['user'] = NULL
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_hSchRpcRun(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C dir > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        try:
            resp = tsch.hSchRpcRun(dce, '\\At%d\x00' % jobId, ('arg0', 'arg1'))
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_SchRpcGetInstanceInfo(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C vssadmin > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        try:
            resp = tsch.hSchRpcRun(dce, '\\At%d\x00' % jobId, ('arg0', 'arg1'))
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass
        request = tsch.SchRpcGetInstanceInfo()
        request['guid'] = resp['pGuid']
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('SCHED_E_TASK_NOT_RUNNING') <= 0:
                raise
            pass
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_hSchRpcGetInstanceInfo(self):
        if False:
            return 10
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C vssadmin > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        try:
            resp = tsch.hSchRpcRun(dce, '\\At%d\x00' % jobId, ('arg0', 'arg1'))
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass
        try:
            resp = tsch.hSchRpcGetInstanceInfo(dce, resp['pGuid'])
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('SCHED_E_TASK_NOT_RUNNING') <= 0:
                raise
            pass
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_SchRpcStopInstance(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C vssadmin > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        try:
            resp = tsch.hSchRpcRun(dce, '\\At%d\x00' % jobId, ('arg0', 'arg1'))
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass
        request = tsch.SchRpcStopInstance()
        request['guid'] = resp['pGuid']
        request['flags'] = 0
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('SCHED_E_TASK_NOT_RUNNING') <= 0:
                raise
            pass
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_hSchRpcStopInstance(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C vssadmin > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        try:
            resp = tsch.hSchRpcRun(dce, '\\At%d\x00' % jobId, ('arg0', 'arg1'))
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass
        try:
            resp = tsch.hSchRpcStopInstance(dce, resp['pGuid'])
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('SCHED_E_TASK_NOT_RUNNING') <= 0:
                raise
            pass
        try:
            resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return

    def test_SchRpcStop(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C vssadmin > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        request = tsch.SchRpcStop()
        request['path'] = '\\At%d\x00' % jobId
        request['flags'] = 0
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('ERROR_INVALID_FUNCTION') <= 0:
                raise
            pass
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_hSchRpcStop(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        (dce_2, rpc_transport_2) = self.connect(iface_uuid=atsvc.MSRPC_UUID_ATSVC)
        atInfo = AT_INFO()
        atInfo['JobTime'] = NULL
        atInfo['DaysOfMonth'] = 0
        atInfo['DaysOfWeek'] = 0
        atInfo['Flags'] = 0
        atInfo['Command'] = '%%COMSPEC%% /C vssadmin > %%SYSTEMROOT%%\\Temp\\ANI 2>&1\x00'
        try:
            resp = atsvc.hNetrJobAdd(dce_2, NULL, atInfo)
            resp.dump()
        except atsvc.DCERPCSessionError as e:
            if e.get_error_code() != ERROR_NOT_SUPPORTED:
                raise
            else:
                return
        jobId = resp['pJobId']
        try:
            resp = tsch.hSchRpcStop(dce, '\\At%d\x00' % jobId)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('ERROR_INVALID_FUNCTION') <= 0:
                raise
            pass
        resp = atsvc.hNetrJobDel(dce_2, NULL, jobId, jobId)
        resp.dump()

    def test_SchRpcRename(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        resp = tsch.hSchRpcCreateFolder(dce, '\\Beto')
        resp.dump()
        request = tsch.SchRpcRename()
        request['path'] = '\\Beto\x00'
        request['newName'] = '\\Anita\x00'
        request['flags'] = 0
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('E_NOTIMPL') <= 0:
                raise
            pass
        resp = tsch.hSchRpcDelete(dce, '\\Beto')
        resp.dump()

    def test_hSchRpcRename(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        resp = tsch.hSchRpcCreateFolder(dce, '\\Beto')
        resp.dump()
        try:
            resp = tsch.hSchRpcRename(dce, '\\Beto', '\\Anita')
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('E_NOTIMPL') <= 0:
                raise
            pass
        resp = tsch.hSchRpcDelete(dce, '\\Beto')
        resp.dump()

    def test_SchRpcScheduledRuntimes(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcScheduledRuntimes()
        request['path'] = '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag\x00'
        request['start'] = NULL
        request['end'] = NULL
        request['flags'] = 0
        request['cRequested'] = 10
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('ERROR_INVALID_FUNCTIO') <= 0 and str(e).find('SCHED_S_TASK_NOT_SCHEDULED') < 0:
                raise
            e.get_packet().dump()
            pass

    def test_hSchRpcScheduledRuntimes(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcScheduledRuntimes()
        request['path'] = '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag\x00'
        request['start'] = NULL
        request['end'] = NULL
        request['flags'] = 0
        request['cRequested'] = 10
        try:
            resp = tsch.hSchRpcScheduledRuntimes(dce, '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag', NULL, NULL, 0, 10)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('ERROR_INVALID_FUNCTIO') <= 0 and str(e).find('SCHED_S_TASK_NOT_SCHEDULED') < 0:
                raise
            e.get_packet().dump()
            pass

    def test_SchRpcGetLastRunInfo(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcGetLastRunInfo()
        request['path'] = '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag\x00'
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('SCHED_S_TASK_HAS_NOT_RUN') <= 0:
                raise
            pass

    def test_hSchRpcGetLastRunInfo(self):
        if False:
            while True:
                i = 10
        (dce, rpc_transport) = self.connect()
        try:
            resp = tsch.hSchRpcGetLastRunInfo(dce, '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag')
            resp.dump()
        except tsch.DCERPCSessionError as e:
            if str(e).find('SCHED_S_TASK_HAS_NOT_RUN') <= 0:
                raise
            pass

    def test_SchRpcGetTaskInfo(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcGetTaskInfo()
        request['path'] = '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag\x00'
        request['flags'] = tsch.SCH_FLAG_STATE
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass

    def test_hSchRpcGetTaskInfo(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        try:
            resp = tsch.hSchRpcGetTaskInfo(dce, '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag', tsch.SCH_FLAG_STATE)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass

    def test_SchRpcGetNumberOfMissedRuns(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcGetNumberOfMissedRuns()
        request['path'] = '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag\x00'
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass

    def test_hSchRpcGetNumberOfMissedRuns(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpc_transport) = self.connect()
        try:
            resp = tsch.hSchRpcGetNumberOfMissedRuns(dce, '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag')
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass

    def test_SchRpcEnableTask(self):
        if False:
            print('Hello World!')
        (dce, rpc_transport) = self.connect()
        request = tsch.SchRpcEnableTask()
        request['path'] = '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag\x00'
        request['enabled'] = 1
        try:
            resp = dce.request(request)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass

    def test_hSchRpcEnableTask(self):
        if False:
            i = 10
            return i + 15
        (dce, rpc_transport) = self.connect()
        try:
            resp = tsch.hSchRpcEnableTask(dce, '\\Microsoft\\Windows\\Defrag\\ScheduledDefrag', True)
            resp.dump()
        except tsch.DCERPCSessionError as e:
            print(e)
            pass

@pytest.mark.remote
class ATSVCTestsSMBTransport(ATSVCTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class ATSVCTestsSMBTransport64(ATSVCTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64

@pytest.mark.remote
class SASECTestsSMBTransport(SASECTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class SASECTestsSMBTransport64(SASECTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64

@pytest.mark.remote
class TSCHTestsSMBTransport(TSCHTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class TSCHTestsSMBTransport64(TSCHTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)