from __future__ import division
from __future__ import print_function
import pytest
import unittest
from tests import RemoteTestCase
from tests.dcerpc import DCERPCTests
from impacket import ntlm
from impacket.uuid import string_to_bin, uuidtup_to_bin
from impacket.dcerpc.v5 import dcomrt
from impacket.dcerpc.v5.dcom import scmp, vds, oaut, comev

class DCOMTests(DCERPCTests):
    string_binding = 'ncacn_ip_tcp:{0.machine}'
    authn = True
    authn_level = ntlm.NTLM_AUTH_PKT_INTEGRITY

    def test_ServerAlive(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        objExporter = dcomrt.IObjectExporter(dce)
        objExporter.ServerAlive()

    def test_ServerAlive2(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        objExporter = dcomrt.IObjectExporter(dce)
        objExporter.ServerAlive2()

    def test_ComplexPing_SimplePing(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect()
        objExporter = dcomrt.IObjectExporter(dce)
        resp = objExporter.ComplexPing()
        objExporter.SimplePing(resp['pSetId'])

    def test_ResolveOxid(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        scm = dcomrt.IRemoteSCMActivator(dce)
        iInterface = scm.RemoteCreateInstance(comev.CLSID_EventSystem, comev.IID_IEventSystem)
        objExporter = dcomrt.IObjectExporter(dce)
        objExporter.ResolveOxid(iInterface.get_oxid(), (7,))

    def test_ResolveOxid2(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        scm = dcomrt.IActivation(dce)
        iInterface = scm.RemoteActivation(comev.CLSID_EventSystem, comev.IID_IEventSystem)
        objExporter = dcomrt.IObjectExporter(dce)
        objExporter.ResolveOxid2(iInterface.get_oxid(), (7,))

    def test_RemoteActivation(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        scm = dcomrt.IActivation(dce)
        scm.RemoteActivation(comev.CLSID_EventSystem, comev.IID_IEventSystem)

    def test_RemoteGetClassObject(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect()
        IID_IClassFactory = uuidtup_to_bin(('00000001-0000-0000-C000-000000000046', '0.0'))
        scm = dcomrt.IRemoteSCMActivator(dce)
        iInterface = scm.RemoteGetClassObject(comev.CLSID_EventSystem, IID_IClassFactory)
        iInterface.RemRelease()

    def test_RemoteCreateInstance(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        scm = dcomrt.IRemoteSCMActivator(dce)
        scm.RemoteCreateInstance(comev.CLSID_EventSystem, comev.IID_IEventSystem)

    @pytest.mark.skip
    def test_scmp(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect()
        scm = dcomrt.IRemoteSCMActivator(dce)
        iInterface = scm.RemoteCreateInstance(scmp.CLSID_ShadowCopyProvider, scmp.IID_IVssSnapshotMgmt)
        iVssSnapshotMgmt = scmp.IVssSnapshotMgmt(iInterface)
        iVssEnumMgmtObject = iVssSnapshotMgmt.QueryVolumesSupportedForSnapshots(scmp.IID_ShadowCopyProvider, 31)
        iVssEnumMgmtObject.Next(10)

    @pytest.mark.skip
    def test_vds(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect()
        scm = dcomrt.IRemoteSCMActivator(dce)
        iInterface = scm.RemoteCreateInstance(vds.CLSID_VirtualDiskService, vds.IID_IVdsServiceInitialization)
        serviceInitialization = vds.IVdsServiceInitialization(iInterface)
        serviceInitialization.Initialize()
        iInterface = serviceInitialization.RemQueryInterface(1, (vds.IID_IVdsService,))
        vdsService = vds.IVdsService(iInterface)
        resp = vdsService.IsServiceReady()
        while resp['ErrorCode'] == 1:
            print('Waiting.. ')
            resp = vdsService.IsServiceReady()
        vdsService.WaitForServiceReady()
        vdsService.GetProperties()
        enumObject = vdsService.QueryProviders(1)
        interfaces = enumObject.Next(1)
        iii = interfaces[0].RemQueryInterface(1, (vds.IID_IVdsProvider,))
        provider = vds.IVdsProvider(iii)
        resp = provider.GetProperties()
        resp.dump()

    @pytest.mark.skip
    def test_oaut(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        IID_IDispatch = string_to_bin('00020400-0000-0000-C000-000000000046')
        scm = dcomrt.IRemoteSCMActivator(dce)
        iInterface = scm.RemoteCreateInstance(string_to_bin('4E14FBA2-2E22-11D1-9964-00C04FBBB345'), IID_IDispatch)
        iDispatch = oaut.IDispatch(iInterface)
        kk = iDispatch.GetTypeInfoCount()
        kk.dump()
        iTypeInfo = iDispatch.GetTypeInfo()
        iTypeInfo.GetTypeAttr()

    @pytest.mark.skip
    def test_ie(self):
        if False:
            print('Hello World!')
        (dce, rpctransport) = self.connect()
        scm = dcomrt.IRemoteSCMActivator(dce)
        iInterface = scm.RemoteCreateInstance(string_to_bin('72C24DD5-D70A-438B-8A42-98424B88AFB8'), dcomrt.IID_IRemUnknown)

@pytest.mark.remote
class DCOMConnectionTests(RemoteTestCase, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.set_transport_config()

    def test_RemQueryInterface(self):
        if False:
            while True:
                i = 10
        dcom = dcomrt.DCOMConnection(self.machine, self.username, self.password, self.domain)
        iInterface = dcom.CoCreateInstanceEx(comev.CLSID_EventSystem, comev.IID_IEventSystem)
        iEventSystem = comev.IEventSystem(iInterface)
        iEventSystem.RemQueryInterface(1, (comev.IID_IEventSystem,))
        dcom.disconnect()

    def test_RemRelease(self):
        if False:
            for i in range(10):
                print('nop')
        dcom = dcomrt.DCOMConnection(self.machine, self.username, self.password, self.domain)
        iInterface = dcom.CoCreateInstanceEx(comev.CLSID_EventSystem, comev.IID_IEventSystem)
        iEventSystem = comev.IEventSystem(iInterface)
        iEventSystem.RemRelease()
        dcom.disconnect()

    @pytest.mark.remote
    def test_comev(self):
        if False:
            while True:
                i = 10
        dcom = dcomrt.DCOMConnection(self.machine, self.username, self.password, self.domain, self.lmhash, self.nthash)
        iInterface = dcom.CoCreateInstanceEx(comev.CLSID_EventSystem, comev.IID_IEventSystem)
        iDispatch = oaut.IDispatch(iInterface)
        iEventSystem = comev.IEventSystem(iInterface)
        iTypeInfo = iEventSystem.GetTypeInfo()
        resp = iTypeInfo.GetTypeAttr()
        for i in range(1, resp['ppTypeAttr']['cFuncs']):
            resp = iTypeInfo.GetFuncDesc(i)
            iTypeInfo.GetNames(resp['ppFuncDesc']['memid'])
            iTypeInfo.GetDocumentation(resp['ppFuncDesc']['memid'])
        iEventSystem.RemRelease()
        iTypeInfo.RemRelease()
        objCollection = iEventSystem.Query('EventSystem.EventSubscriptionCollection', 'ALL')
        objCollection.get_Count()
        evnObj = objCollection.get_NewEnum()
        for i in range(3):
            iUnknown = evnObj.Next(1)[0]
            es = iUnknown.RemQueryInterface(1, (comev.IID_IEventSubscription3,))
            es = comev.IEventSubscription3(es)
            print(es.get_SubscriptionName()['pbstrSubscriptionName']['asData'])
            es.RemRelease()
        objCollection = iEventSystem.Query('EventSystem.EventClassCollection', 'ALL')
        objCollection.get_Count()
        evnObj = objCollection.get_NewEnum()
        for i in range(3):
            iUnknown = evnObj.Next(1)[0]
            ev = iUnknown.RemQueryInterface(1, (comev.IID_IEventClass2,))
            ev = comev.IEventClass2(ev)
            ev.get_EventClassID()
            ev.RemRelease()
        print('=' * 80)
        dcom.disconnect()

@pytest.mark.remote
class DCOMTestsTCPTransport(DCOMTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class DCOMTestsTCPTransport(DCOMTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64
if __name__ == '__main__':
    unittest.main(verbosity=1)