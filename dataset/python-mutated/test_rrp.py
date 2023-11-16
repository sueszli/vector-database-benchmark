from __future__ import division
from __future__ import print_function
import pytest
import unittest
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import transport
from impacket.dcerpc.v5 import rrp, scmr
from impacket.dcerpc.v5.dtypes import NULL, MAXIMUM_ALLOWED, OWNER_SECURITY_INFORMATION

class RRPTests(DCERPCTests):
    iface_uuid = rrp.MSRPC_UUID_RRP
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\winreg]'
    authn = True
    test_key = 'BETO\x00'
    test_value_name = 'BETO2\x00'
    test_value_data = 'HOLA COMO TE VA\x00'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(RRPTests, self).setUp()
        self.rrp_started = False

    def connect_scmr(self):
        if False:
            while True:
                i = 10
        rpctransport = transport.DCERPCTransportFactory('ncacn_np:%s[\\pipe\\svcctl]' % self.machine)
        if hasattr(rpctransport, 'set_credentials'):
            rpctransport.set_credentials(self.username, self.password, self.domain, self.lmhash, self.nthash)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(scmr.MSRPC_UUID_SCMR)
        return (dce, rpctransport)

    def open_scmanager(self, dce):
        if False:
            while True:
                i = 10
        lpMachineName = 'DUMMY\x00'
        lpDatabaseName = 'ServicesActive\x00'
        desiredAccess = scmr.SERVICE_START | scmr.SERVICE_STOP | scmr.SERVICE_CHANGE_CONFIG | scmr.SERVICE_QUERY_CONFIG | scmr.SERVICE_QUERY_STATUS | scmr.SERVICE_ENUMERATE_DEPENDENTS | scmr.SC_MANAGER_ENUMERATE_SERVICE
        resp = scmr.hROpenSCManagerW(dce, lpMachineName, lpDatabaseName, desiredAccess)
        sc_handle = resp['lpScHandle']
        return sc_handle

    def start_rrp_service(self, dce, sc_handle):
        if False:
            return 10
        desiredAccess = scmr.SERVICE_START | scmr.SERVICE_STOP | scmr.SERVICE_CHANGE_CONFIG | scmr.SERVICE_QUERY_CONFIG | scmr.SERVICE_QUERY_STATUS | scmr.SERVICE_ENUMERATE_DEPENDENTS
        resp = scmr.hROpenServiceW(dce, sc_handle, 'RemoteRegistry\x00', desiredAccess)
        serviceHandle = resp['lpServiceHandle']
        try:
            scmr.hRStartServiceW(dce, serviceHandle)
        except Exception as e:
            if str(e).find('ERROR_SERVICE_ALREADY_RUNNING') >= 0:
                pass
            else:
                raise
        scmr.hRCloseServiceHandle(dce, sc_handle)
        self.rrp_started = True

    def connect(self):
        if False:
            print('Hello World!')
        if not self.rrp_started:
            (dce, rpctransport) = self.connect_scmr()
            sc_handle = self.open_scmanager(dce)
            self.start_rrp_service(dce, sc_handle)
        return super(RRPTests, self).connect()

    def open_local_machine(self, dce):
        if False:
            return 10
        resp = rrp.hOpenLocalMachine(dce, MAXIMUM_ALLOWED | rrp.KEY_WOW64_32KEY | rrp.KEY_ENUMERATE_SUB_KEYS)
        return resp['phKey']

    def test_OpenClassesRoot(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = rrp.OpenClassesRoot()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_OpenCurrentUser(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = rrp.OpenCurrentUser()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_OpenLocalMachine(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = rrp.OpenLocalMachine()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_OpenPerformanceData(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = rrp.OpenPerformanceData()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_OpenUsers(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = rrp.OpenUsers()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_BaseRegCloseKey(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegCloseKey()
        request['hKey'] = phKey
        resp = dce.request(request)
        resp.dump()

    def test_hBaseRegCreateKey_hBaseRegSetValue_hBaseRegDeleteKey(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        resp = rrp.hOpenClassesRoot(dce)
        resp.dump()
        regHandle = resp['phKey']
        resp = rrp.hBaseRegCreateKey(dce, regHandle, self.test_key)
        resp.dump()
        phKey = resp['phkResult']
        try:
            resp = rrp.hBaseRegSetValue(dce, phKey, self.test_value_name, rrp.REG_SZ, self.test_value_data)
            resp.dump()
        except Exception as e:
            print(e)
        (type, data) = rrp.hBaseRegQueryValue(dce, phKey, self.test_value_name)
        resp = rrp.hBaseRegDeleteValue(dce, phKey, self.test_value_name)
        resp.dump()
        resp = rrp.hBaseRegDeleteKey(dce, regHandle, self.test_key)
        resp.dump()
        self.assertEqual(self.test_value_data, data)

    def test_BaseRegCreateKey_BaseRegSetValue_BaseRegDeleteKey(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = rrp.OpenClassesRoot()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        regHandle = resp['phKey']
        request = rrp.BaseRegCreateKey()
        request['hKey'] = regHandle
        request['lpSubKey'] = self.test_key
        request['lpClass'] = NULL
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED
        request['lpSecurityAttributes']['RpcSecurityDescriptor']['lpSecurityDescriptor'] = NULL
        request['lpdwDisposition'] = rrp.REG_CREATED_NEW_KEY
        resp = dce.request(request)
        resp.dump()
        phKey = resp['phkResult']
        request = rrp.BaseRegSetValue()
        request['hKey'] = phKey
        request['lpValueName'] = self.test_value_name
        request['dwType'] = rrp.REG_SZ
        request['lpData'] = self.test_value_data.encode('utf-16le')
        request['cbData'] = len(self.test_value_data) * 2
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            print(e)
        request = rrp.BaseRegQueryValue()
        request['hKey'] = phKey
        request['lpValueName'] = self.test_value_name
        request['lpData'] = b' ' * 100
        request['lpcbData'] = 100
        request['lpcbLen'] = 100
        resp = dce.request(request)
        resp.dump()
        resData = resp['lpData']
        request = rrp.BaseRegDeleteKey()
        request['hKey'] = regHandle
        request['lpSubKey'] = self.test_key
        resp = dce.request(request)
        resp.dump()
        print(b''.join(resData).decode('utf-16le'))
        self.assertEqual(self.test_value_data, b''.join(resData).decode('utf-16le'))

    def test_BaseRegEnumKey(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED | rrp.KEY_ENUMERATE_SUB_KEYS
        resp = dce.request(request)
        request = rrp.BaseRegEnumKey()
        request['hKey'] = resp['phkResult']
        request['dwIndex'] = 1
        request.fields['lpNameIn'].fields['MaximumLength'] = 510
        request.fields['lpNameIn'].fields['Data'].fields['Data'].fields['MaximumCount'] = 255
        request['lpClassIn'] = ' ' * 100
        request['lpftLastWriteTime'] = NULL
        resp = dce.request(request)
        resp.dump()

    def test_hBaseRegEnumKey(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED | rrp.KEY_ENUMERATE_SUB_KEYS
        resp = dce.request(request)
        resp = rrp.hBaseRegEnumKey(dce, resp['phkResult'], 1)
        resp.dump()

    def test_BaseRegEnumValue(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        request = rrp.BaseRegEnumValue()
        request['hKey'] = resp['phkResult']
        request['dwIndex'] = 6
        request['lpValueNameIn'] = ' ' * 100
        request['lpData'] = b' ' * 100
        request['lpcbData'] = 100
        request['lpcbLen'] = 100
        resp = dce.request(request)
        resp.dump()

    def test_hBaseRegEnumValue(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp = rrp.hBaseRegEnumValue(dce, resp['phkResult'], 6, 100)
        resp.dump()

    def test_BaseRegFlushKey(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        resp = rrp.hBaseRegFlushKey(dce, phKey)
        resp.dump()

    def test_BaseRegGetKeySecurity(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        resp = rrp.hBaseRegGetKeySecurity(dce, phKey, OWNER_SECURITY_INFORMATION)
        resp.dump()

    def test_BaseRegOpenKey(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_hBaseRegQueryInfoKey(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        resp = rrp.hBaseRegOpenKey(dce, phKey, 'SYSTEM\\CurrentControlSet\\Control\\Lsa\\JD\x00')
        resp = rrp.hBaseRegQueryInfoKey(dce, resp['phkResult'])
        resp.dump()

    def test_BaseRegQueryValue(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegQueryValue()
        request['hKey'] = resp['phkResult']
        request['lpValueName'] = 'ProductName\x00'
        request['lpData'] = b' ' * 100
        request['lpcbData'] = 100
        request['lpcbLen'] = 100
        resp = dce.request(request)
        resp.dump()

    def test_hBaseRegQueryValue(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        resp = rrp.hBaseRegOpenKey(dce, phKey, 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00')
        resp.dump()
        rrp.hBaseRegQueryValue(dce, resp['phkResult'], 'ProductName\x00')

    def test_BaseRegReplaceKey(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegReplaceKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\x00'
        request['lpNewFile'] = 'SOFTWARE\x00'
        request['lpOldFile'] = 'SOFTWARE\x00'
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('ERROR_FILE_NOT_FOUND') < 0:
                raise

    def test_hBaseRegReplaceKey(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        try:
            resp = rrp.hBaseRegReplaceKey(dce, phKey, 'SOFTWARE\x00', 'SOFTWARE\x00', 'SOFTWARE\x00')
            resp.dump()
        except Exception as e:
            if str(e).find('ERROR_FILE_NOT_FOUND') < 0:
                raise

    def test_BaseRegRestoreKey(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegRestoreKey()
        request['hKey'] = phKey
        request['lpFile'] = 'SOFTWARE\x00'
        request['Flags'] = rrp.REG_REFRESH_HIVE
        try:
            resp = dce.request(request)
            resp.dump()
        except Exception as e:
            if str(e).find('ERROR_FILE_NOT_FOUND') < 0:
                raise

    def test_hBaseRegRestoreKey(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        try:
            resp = rrp.hBaseRegRestoreKey(dce, phKey, 'SOFTWARE\x00')
            resp.dump()
        except Exception as e:
            if str(e).find('ERROR_FILE_NOT_FOUND') < 0:
                raise

    def test_BaseRegSaveKey(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = rrp.OpenCurrentUser()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegSaveKey()
        request['hKey'] = resp['phKey']
        request['lpFile'] = 'BETUSFILE2\x00'
        request['pSecurityAttributes'] = NULL
        resp = dce.request(request)
        resp.dump()
        smb = rpctransport.get_smb_connection()
        smb.deleteFile('ADMIN$', 'System32\\BETUSFILE2')

    def test_hBaseRegSaveKey(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        resp = rrp.hOpenCurrentUser(dce)
        resp.dump()
        resp = rrp.hBaseRegSaveKey(dce, resp['phKey'], 'BETUSFILE2\x00')
        resp.dump()
        smb = rpctransport.get_smb_connection()
        smb.deleteFile('ADMIN$', 'System32\\BETUSFILE2')

    def test_BaseRegGetVersion(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegGetVersion()
        request['hKey'] = phKey
        resp = dce.request(request)
        resp.dump()

    def test_hBaseRegGetVersion(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        resp = rrp.hBaseRegGetVersion(dce, phKey)
        resp.dump()

    def test_OpenCurrentConfig(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = rrp.OpenCurrentConfig()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_hOpenCurrentConfig(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        resp = rrp.hOpenCurrentConfig(dce)
        resp.dump()

    def test_BaseRegQueryMultipleValues(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED | rrp.KEY_QUERY_VALUE
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegQueryMultipleValues()
        item1 = rrp.RVALENT()
        item1['ve_valuename'] = 'ProductName\x00'
        item1['ve_valuelen'] = len('ProductName\x00')
        item1['ve_valueptr'] = NULL
        item1['ve_type'] = rrp.REG_SZ
        item2 = rrp.RVALENT()
        item2['ve_valuename'] = 'SystemRoot\x00'
        item2['ve_valuelen'] = len('SystemRoot\x00')
        item1['ve_valueptr'] = NULL
        item2['ve_type'] = rrp.REG_SZ
        item3 = rrp.RVALENT()
        item3['ve_valuename'] = 'EditionID\x00'
        item3['ve_valuelen'] = len('EditionID\x00')
        item3['ve_valueptr'] = NULL
        item3['ve_type'] = rrp.REG_SZ
        request['hKey'] = resp['phkResult']
        request['val_listIn'].append(item1)
        request['val_listIn'].append(item2)
        request['val_listIn'].append(item3)
        request['num_vals'] = len(request['val_listIn'])
        request['lpvalueBuf'] = list(b' ' * 128)
        request['ldwTotsize'] = 128
        resp = dce.request(request)
        resp.dump()

    def test_hBaseRegQueryMultipleValues(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        resp = rrp.hBaseRegOpenKey(dce, phKey, 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00')
        resp.dump()
        valueIn = list()
        item1 = {}
        item1['ValueName'] = 'ProductName\x00'
        item1['ValueType'] = rrp.REG_SZ
        valueIn.append(item1)
        item2 = {}
        item2['ValueName'] = 'InstallDate\x00'
        item2['ValueType'] = rrp.REG_DWORD
        valueIn.append(item2)
        item3 = {}
        item3['ValueName'] = 'DigitalProductId\x00'
        item3['ValueType'] = rrp.REG_BINARY
        rrp.hBaseRegQueryMultipleValues(dce, resp['phkResult'], valueIn)

    def test_BaseRegSaveKeyEx(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        request = rrp.OpenCurrentUser()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegSaveKeyEx()
        request['hKey'] = resp['phKey']
        request['lpFile'] = 'BETUSFILE2\x00'
        request['pSecurityAttributes'] = NULL
        request['Flags'] = 4
        resp = dce.request(request)
        resp.dump()
        smb = rpctransport.get_smb_connection()
        smb.deleteFile('ADMIN$', 'System32\\BETUSFILE2')

    def test_hBaseRegSaveKeyEx(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        resp = rrp.hOpenCurrentUser(dce)
        resp.dump()
        resp = rrp.hBaseRegSaveKeyEx(dce, resp['phKey'], 'BETUSFILE2\x00')
        resp.dump()
        smb = rpctransport.get_smb_connection()
        smb.deleteFile('ADMIN$', 'System32\\BETUSFILE2')

    def test_OpenPerformanceText(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = rrp.OpenPerformanceText()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_hOpenPerformanceText(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        resp = rrp.hOpenPerformanceText(dce)
        resp.dump()

    def test_OpenPerformanceNlsText(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = rrp.OpenPerformanceNlsText()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()

    def test_hOpenPerformanceNlsText(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        resp = rrp.hOpenPerformanceNlsText(dce)
        resp.dump()

    def test_BaseRegQueryMultipleValues2(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED | rrp.KEY_QUERY_VALUE
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegQueryMultipleValues2()
        item1 = rrp.RVALENT()
        item1['ve_valuename'] = 'ProductName\x00'
        item1['ve_valuelen'] = len('ProductName\x00')
        item1['ve_valueptr'] = NULL
        item1['ve_type'] = rrp.REG_SZ
        item2 = rrp.RVALENT()
        item2['ve_valuename'] = 'SystemRoot\x00'
        item2['ve_valuelen'] = len('SystemRoot\x00')
        item1['ve_valueptr'] = NULL
        item2['ve_type'] = rrp.REG_SZ
        item3 = rrp.RVALENT()
        item3['ve_valuename'] = 'EditionID\x00'
        item3['ve_valuelen'] = len('EditionID\x00')
        item3['ve_valueptr'] = NULL
        item3['ve_type'] = rrp.REG_SZ
        request['hKey'] = resp['phkResult']
        request['val_listIn'].append(item1)
        request['val_listIn'].append(item2)
        request['val_listIn'].append(item3)
        request['num_vals'] = len(request['val_listIn'])
        request['lpvalueBuf'] = list(b' ' * 128)
        request['ldwTotsize'] = 128
        resp = dce.request(request)
        resp.dump()

    def test_BaseRegDeleteKeyEx(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = rrp.OpenClassesRoot()
        request['ServerName'] = NULL
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        regHandle = resp['phKey']
        request = rrp.BaseRegCreateKey()
        request['hKey'] = regHandle
        request['lpSubKey'] = self.test_key
        request['lpClass'] = NULL
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED
        request['lpSecurityAttributes']['RpcSecurityDescriptor']['lpSecurityDescriptor'] = NULL
        request['lpdwDisposition'] = rrp.REG_CREATED_NEW_KEY
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegDeleteKeyEx()
        request['hKey'] = regHandle
        request['lpSubKey'] = self.test_key
        request['AccessMask'] = rrp.KEY_WOW64_32KEY
        request['Reserved'] = 0
        resp = dce.request(request)
        resp.dump()

    def test_BaseRegLoadKey_BaseRegUnLoadKey(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        request = rrp.BaseRegOpenKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'SECURITY\x00'
        request['dwOptions'] = 1
        request['samDesired'] = MAXIMUM_ALLOWED
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegSaveKey()
        request['hKey'] = resp['phkResult']
        request['lpFile'] = 'SEC\x00'
        request['pSecurityAttributes'] = NULL
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegLoadKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'BETUS\x00'
        request['lpFile'] = 'SEC\x00'
        resp = dce.request(request)
        resp.dump()
        request = rrp.BaseRegUnLoadKey()
        request['hKey'] = phKey
        request['lpSubKey'] = 'BETUS\x00'
        resp = dce.request(request)
        resp.dump()
        smb = rpctransport.get_smb_connection()
        smb.deleteFile('ADMIN$', 'System32\\SEC')

    def test_hBaseRegLoadKey_hBaseRegUnLoadKey(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        phKey = self.open_local_machine(dce)
        resp = rrp.hBaseRegOpenKey(dce, phKey, 'SECURITY\x00')
        resp.dump()
        request = rrp.BaseRegSaveKey()
        request['hKey'] = resp['phkResult']
        request['lpFile'] = 'SEC\x00'
        request['pSecurityAttributes'] = NULL
        resp = dce.request(request)
        resp.dump()
        resp = rrp.hBaseRegLoadKey(dce, phKey, 'BETUS\x00', 'SEC\x00')
        resp.dump()
        resp = rrp.hBaseRegUnLoadKey(dce, phKey, 'BETUS\x00')
        resp.dump()
        smb = rpctransport.get_smb_connection()
        smb.deleteFile('ADMIN$', 'System32\\SEC')

@pytest.mark.remote
class RRPTestsSMBTransport(RRPTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class RRPTestsSMBTransport64(RRPTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)