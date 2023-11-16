import unittest
import netbios
import win32api
import win32wnet
RESOURCE_CONNECTED = 1
RESOURCE_GLOBALNET = 2
RESOURCE_REMEMBERED = 3
RESOURCE_RECENT = 4
RESOURCE_CONTEXT = 5
RESOURCETYPE_ANY = 0
RESOURCETYPE_DISK = 1
RESOURCETYPE_PRINT = 2
RESOURCETYPE_RESERVED = 8
RESOURCETYPE_UNKNOWN = 4294967295
RESOURCEUSAGE_CONNECTABLE = 1
RESOURCEUSAGE_CONTAINER = 2
RESOURCEDISPLAYTYPE_GENERIC = 0
RESOURCEDISPLAYTYPE_DOMAIN = 1
RESOURCEDISPLAYTYPE_SERVER = 2
RESOURCEDISPLAYTYPE_SHARE = 3
NETRESOURCE_attributes = [('dwScope', int), ('dwType', int), ('dwDisplayType', int), ('dwUsage', int), ('lpLocalName', str), ('lpRemoteName', str), ('lpComment', str), ('lpProvider', str)]
NCB_attributes = [('Command', int), ('Retcode', int), ('Lsn', int), ('Num', int), ('Callname', str), ('Name', str), ('Rto', int), ('Sto', int), ('Lana_num', int), ('Cmd_cplt', int), ('Event', int), ('Post', int)]

class TestCase(unittest.TestCase):

    def testGetUser(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(win32api.GetUserName(), win32wnet.WNetGetUser())

    def _checkItemAttributes(self, item, attrs):
        if False:
            return 10
        for (attr, typ) in attrs:
            val = getattr(item, attr)
            if typ is int:
                self.assertTrue(isinstance(val, int), f'Attr {attr!r} has value {val!r}')
                new_val = val + 1
            elif typ is str:
                if val is not None:
                    self.assertTrue(isinstance(val, str), f'Attr {attr!r} has value {val!r}')
                    new_val = val + ' new value'
                else:
                    new_val = 'new value'
            else:
                self.fail(f"Don't know what {typ} is")
            setattr(item, attr, new_val)

    def testNETRESOURCE(self):
        if False:
            print('Hello World!')
        nr = win32wnet.NETRESOURCE()
        self._checkItemAttributes(nr, NETRESOURCE_attributes)

    def testWNetEnumResource(self):
        if False:
            while True:
                i = 10
        handle = win32wnet.WNetOpenEnum(RESOURCE_GLOBALNET, RESOURCETYPE_ANY, 0, None)
        try:
            while 1:
                items = win32wnet.WNetEnumResource(handle, 0)
                if len(items) == 0:
                    break
                for item in items:
                    self._checkItemAttributes(item, NETRESOURCE_attributes)
        finally:
            handle.Close()

    def testNCB(self):
        if False:
            i = 10
            return i + 15
        ncb = win32wnet.NCB()
        self._checkItemAttributes(ncb, NCB_attributes)

    def testNetbios(self):
        if False:
            i = 10
            return i + 15
        ncb = win32wnet.NCB()
        ncb.Command = netbios.NCBENUM
        la_enum = netbios.LANA_ENUM()
        ncb.Buffer = la_enum
        rc = win32wnet.Netbios(ncb)
        self.assertEqual(rc, 0)
        for i in range(la_enum.length):
            ncb.Reset()
            ncb.Command = netbios.NCBRESET
            ncb.Lana_num = la_enum.lana[i]
            rc = Netbios(ncb)
            self.assertEqual(rc, 0)
            ncb.Reset()
            ncb.Command = netbios.NCBASTAT
            ncb.Lana_num = la_enum.lana[i]
            ncb.Callname = b'*               '
            adapter = netbios.ADAPTER_STATUS()
            ncb.Buffer = adapter
            Netbios(ncb)
            self.assertTrue(len(adapter.adapter_address), 6)

    def iterConnectableShares(self):
        if False:
            return 10
        nr = win32wnet.NETRESOURCE()
        nr.dwScope = RESOURCE_GLOBALNET
        nr.dwUsage = RESOURCEUSAGE_CONTAINER
        nr.lpRemoteName = '\\\\' + win32api.GetComputerName()
        handle = win32wnet.WNetOpenEnum(RESOURCE_GLOBALNET, RESOURCETYPE_ANY, 0, nr)
        while 1:
            items = win32wnet.WNetEnumResource(handle, 0)
            if len(items) == 0:
                break
            for item in items:
                if item.dwDisplayType == RESOURCEDISPLAYTYPE_SHARE:
                    yield item

    def findUnusedDriveLetter(self):
        if False:
            print('Hello World!')
        existing = [x[0].lower() for x in win32api.GetLogicalDriveStrings().split('\x00') if x]
        handle = win32wnet.WNetOpenEnum(RESOURCE_REMEMBERED, RESOURCETYPE_DISK, 0, None)
        try:
            while 1:
                items = win32wnet.WNetEnumResource(handle, 0)
                if len(items) == 0:
                    break
                xtra = [i.lpLocalName[0].lower() for i in items if i.lpLocalName]
                existing.extend(xtra)
        finally:
            handle.Close()
        for maybe in 'defghijklmnopqrstuvwxyz':
            if maybe not in existing:
                return maybe
        self.fail('All drive mappings are taken?')

    def testAddConnection(self):
        if False:
            while True:
                i = 10
        localName = self.findUnusedDriveLetter() + ':'
        for share in self.iterConnectableShares():
            share.lpLocalName = localName
            win32wnet.WNetAddConnection2(share)
            win32wnet.WNetCancelConnection2(localName, 0, 0)
            break

    def testAddConnectionOld(self):
        if False:
            while True:
                i = 10
        localName = self.findUnusedDriveLetter() + ':'
        for share in self.iterConnectableShares():
            win32wnet.WNetAddConnection2(share.dwType, localName, share.lpRemoteName)
            win32wnet.WNetCancelConnection2(localName, 0, 0)
            break
if __name__ == '__main__':
    unittest.main()