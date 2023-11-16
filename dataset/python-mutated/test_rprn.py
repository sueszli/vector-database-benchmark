from __future__ import division
from __future__ import print_function
import pytest
import unittest
from six import assertRaisesRegex
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import rprn
from impacket.dcerpc.v5.dtypes import NULL
from impacket.structure import hexdump

class RPRNTests(DCERPCTests):
    iface_uuid = rprn.MSRPC_UUID_RPRN
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\spoolss]'
    authn = True

    def test_RpcEnumPrinters(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = rprn.RpcEnumPrinters()
        request['Flags'] = rprn.PRINTER_ENUM_LOCAL
        request['Name'] = NULL
        request['pPrinterEnum'] = NULL
        request['Level'] = 1
        request.dump()
        with assertRaisesRegex(self, rprn.DCERPCSessionError, 'ERROR_INSUFFICIENT_BUFFER') as cm:
            dce.request(request)
        bytesNeeded = cm.exception.get_packet()['pcbNeeded']
        request = rprn.RpcEnumPrinters()
        request['Flags'] = rprn.PRINTER_ENUM_LOCAL
        request['Name'] = NULL
        request['Level'] = 1
        request['cbBuf'] = bytesNeeded
        request['pPrinterEnum'] = b'a' * bytesNeeded
        request.dump()
        resp = dce.request(request)
        resp.dump()
        hexdump(b''.join(resp['pPrinterEnum']))

    def test_hRpcEnumPrinters(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        resp = rprn.hRpcEnumPrinters(dce, rprn.PRINTER_ENUM_LOCAL, NULL, 1)
        hexdump(b''.join(resp['pPrinterEnum']))

    def test_RpcOpenPrinter(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        request = rprn.RpcOpenPrinter()
        request['pPrinterName'] = '\\\\%s\x00' % self.machine
        request['pDatatype'] = NULL
        request['pDevModeContainer']['pDevMode'] = NULL
        request['AccessRequired'] = rprn.SERVER_READ
        request.dump()
        resp = dce.request(request)
        resp.dump()

    def test_hRpcOpenPrinter(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        resp = rprn.hRpcOpenPrinter(dce, '\\\\%s\x00' % self.machine)
        resp.dump()

    def test_RpcGetPrinterDriverDirectory(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = rprn.RpcGetPrinterDriverDirectory()
        request['pName'] = NULL
        request['pEnvironment'] = NULL
        request['Level'] = 1
        request['pDriverDirectory'] = NULL
        request['cbBuf'] = 0
        request.dump()
        with assertRaisesRegex(self, rprn.DCERPCSessionError, 'ERROR_INSUFFICIENT_BUFFER'):
            dce.request(request)

    def test_hRpcGetPrinterDriverDirectory(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        resp = rprn.hRpcGetPrinterDriverDirectory(dce, NULL, NULL, 1)
        resp.dump()

    def test_RpcClosePrinter(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        request = rprn.RpcOpenPrinter()
        request['pPrinterName'] = '\\\\%s\x00' % self.machine
        request['pDatatype'] = NULL
        request['pDevModeContainer']['pDevMode'] = NULL
        request['AccessRequired'] = rprn.SERVER_READ
        request.dump()
        resp = dce.request(request)
        resp.dump()
        request = rprn.RpcClosePrinter()
        request['phPrinter'] = resp['pHandle']
        request.dump()
        resp = dce.request(request)
        resp.dump()

    def test_hRpcClosePrinter(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        resp = rprn.hRpcOpenPrinter(dce, '\\\\%s\x00' % self.machine)
        resp.dump()
        resp = rprn.hRpcClosePrinter(dce, resp['pHandle'])
        resp.dump()

    def test_RpcOpenPrinterEx(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        request = rprn.RpcOpenPrinterEx()
        request['pPrinterName'] = '\\\\%s\x00' % self.machine
        request['pDatatype'] = NULL
        request['AccessRequired'] = rprn.SERVER_READ
        request['pDevModeContainer']['pDevMode'] = NULL
        request['pClientInfo']['Level'] = 1
        request['pClientInfo']['ClientInfo']['tag'] = 1
        request['pClientInfo']['ClientInfo']['pClientInfo1']['dwSize'] = 28
        request['pClientInfo']['ClientInfo']['pClientInfo1']['pMachineName'] = '%s\x00' % self.machine
        request['pClientInfo']['ClientInfo']['pClientInfo1']['pUserName'] = '%s\\%s\x00' % (self.domain, self.username)
        request['pClientInfo']['ClientInfo']['pClientInfo1']['dwBuildNum'] = 0
        request['pClientInfo']['ClientInfo']['pClientInfo1']['dwMajorVersion'] = 0
        request['pClientInfo']['ClientInfo']['pClientInfo1']['dwMinorVersion'] = 0
        request['pClientInfo']['ClientInfo']['pClientInfo1']['wProcessorArchitecture'] = 9
        request.dump()
        resp = dce.request(request)
        resp.dump()

    def test_hRpcOpenPrinterEx(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        clientInfo = rprn.SPLCLIENT_CONTAINER()
        clientInfo['Level'] = 1
        clientInfo['ClientInfo']['tag'] = 1
        clientInfo['ClientInfo']['pClientInfo1']['dwSize'] = 28
        clientInfo['ClientInfo']['pClientInfo1']['pMachineName'] = '%s\x00' % self.machine
        clientInfo['ClientInfo']['pClientInfo1']['pUserName'] = '%s\\%s\x00' % (self.domain, self.username)
        clientInfo['ClientInfo']['pClientInfo1']['dwBuildNum'] = 0
        clientInfo['ClientInfo']['pClientInfo1']['dwMajorVersion'] = 0
        clientInfo['ClientInfo']['pClientInfo1']['dwMinorVersion'] = 0
        clientInfo['ClientInfo']['pClientInfo1']['wProcessorArchitecture'] = 9
        resp = rprn.hRpcOpenPrinterEx(dce, '\\\\%s\x00' % self.machine, pClientInfo=clientInfo)
        resp.dump()

    def test_RpcRemoteFindFirstPrinterChangeNotificationEx(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        request = rprn.RpcOpenPrinter()
        request['pPrinterName'] = '\\\\%s\x00' % self.machine
        request['pDatatype'] = NULL
        request['pDevModeContainer']['pDevMode'] = NULL
        request['AccessRequired'] = rprn.SERVER_READ | rprn.SERVER_ALL_ACCESS | rprn.SERVER_ACCESS_ADMINISTER
        request.dump()
        resp = dce.request(request)
        resp.dump()
        request = rprn.RpcRemoteFindFirstPrinterChangeNotificationEx()
        request['hPrinter'] = resp['pHandle']
        request['fdwFlags'] = rprn.PRINTER_CHANGE_ADD_JOB
        request['pszLocalMachine'] = '\\\\%s\x00' % self.machine
        request['pOptions'] = NULL
        request.dump()
        with assertRaisesRegex(self, rprn.DCERPCSessionError, 'ERROR_INVALID_HANDLE'):
            dce.request(request)

    def test_hRpcRemoteFindFirstPrinterChangeNotificationEx(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        resp = rprn.hRpcOpenPrinter(dce, '\\\\%s\x00' % self.machine)
        with assertRaisesRegex(self, rprn.DCERPCSessionError, 'ERROR_INVALID_HANDLE'):
            rprn.hRpcRemoteFindFirstPrinterChangeNotificationEx(dce, resp['pHandle'], rprn.PRINTER_CHANGE_ADD_JOB, pszLocalMachine='\\\\%s\x00' % self.machine)

@pytest.mark.remote
class RPRNTestsSMBTransport(RPRNTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class RPRNTestsSMBTransport64(RPRNTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)