from __future__ import division
from __future__ import print_function
from struct import unpack
import pytest
import unittest
from tests import RemoteTestCase
from impacket.dcerpc.v5 import transport, epm, rpch
from impacket.dcerpc.v5.ndr import NULL

@pytest.mark.remote
class RPCHTest(RemoteTestCase, unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(RPCHTest, self).setUp()
        self.set_transport_config()

    def test_1(self):
        if False:
            while True:
                i = 10
        stringbinding = 'ncacn_http:%s' % self.machine
        rpctransport = transport.DCERPCTransportFactory(stringbinding)
        dce = rpctransport.get_dce_rpc()
        dce.connect()
        dce.bind(epm.MSRPC_UUID_PORTMAP)
        request = epm.ept_lookup()
        request['inquiry_type'] = epm.RPC_C_EP_ALL_ELTS
        request['object'] = NULL
        request['Ifid'] = NULL
        request['vers_option'] = epm.RPC_C_VERS_ALL
        request['max_ents'] = 10
        dce.request(request)
        dce.disconnect()
        dce.connect()
        dce.bind(epm.MSRPC_UUID_PORTMAP)
        dce.request(request)
        dce.disconnect()

class RPCHLocalTest(unittest.TestCase):

    def test_2(self):
        if False:
            print('Hello World!')
        resp = b'\x05\x00\x14\x03\x10\x00\x00\x00L\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x04\x00\x06\x00\x00\x00\x01\x00\x00\x00' + b'\x03\x00\x00\x00\xb0\xf6\xaf=wb\x98\x07\x9b!' + b'Tn\xec\xf4"S\x03\x00\x00\x00:$z7' + b'm\xc1\xed,h]45\x13FC%\x00\x00' + b'\x00\x00\x00\x00\x04\x00'
        packet = rpch.RTSHeader(resp)
        packet.dump()
        pduData = packet['pduData']
        numberOfCommands = packet['NumberOfCommands']
        self.assertEqual(numberOfCommands, 4)
        self.assertEqual(packet['Flags'], rpch.RTS_FLAG_NONE)
        self.assertEqual(packet['frag_len'], 76)
        self.assertEqual(len(pduData), 56)
        server_cmds = []
        while numberOfCommands > 0:
            numberOfCommands -= 1
            cmd_type = unpack('<L', pduData[:4])[0]
            cmd = rpch.COMMANDS[cmd_type](pduData)
            server_cmds.append(cmd)
            pduData = pduData[len(cmd):]
        for cmd in server_cmds:
            cmd.dump()
        self.assertEqual(server_cmds[0].getData(), rpch.Version().getData())
        receiveWindowSize = rpch.ReceiveWindowSize()
        receiveWindowSize['ReceiveWindowSize'] = 262144
        self.assertEqual(server_cmds[3].getData(), receiveWindowSize.getData())
        cookie = rpch.Cookie()
        cookie['Cookie'] = b'\xb0\xf6\xaf=wb\x98\x07\x9b!Tn\xec\xf4"S'
        self.assertEqual(server_cmds[1].getData(), cookie.getData())

    def test_3(self):
        if False:
            i = 10
            return i + 15
        resp = b'\x05\x00\x14\x03\x10\x00\x00\x00\x1c\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x01\x00\x02\x00\x00\x00\xc0\xd4\x01\x00'
        packet = rpch.RTSHeader(resp)
        packet.dump()
        pduData = packet['pduData']
        numberOfCommands = packet['NumberOfCommands']
        server_cmds = []
        while numberOfCommands > 0:
            numberOfCommands -= 1
            cmd_type = unpack('<L', pduData[:4])[0]
            cmd = rpch.COMMANDS[cmd_type](pduData)
            server_cmds.append(cmd)
            pduData = pduData[len(cmd):]
        for cmd in server_cmds:
            cmd.dump()
        connectionTimeout = rpch.ConnectionTimeout()
        connectionTimeout['ConnectionTimeout'] = 120000
        self.assertEqual(server_cmds[0].getData(), connectionTimeout.getData())

    def test_4(self):
        if False:
            print('Hello World!')
        resp = b'\x05\x00\x14\x03\x10\x00\x00\x00\x14\x00\x00\x00\x00\x00' + b'\x00\x00\x01\x00\x00\x00'
        packet = rpch.RTSHeader(resp)
        packet.dump()
        pduData = packet['pduData']
        numberOfCommands = packet['NumberOfCommands']
        server_cmds = []
        while numberOfCommands > 0:
            numberOfCommands -= 1
            cmd_type = unpack('<L', pduData[:4])[0]
            cmd = rpch.COMMANDS[cmd_type](pduData)
            server_cmds.append(cmd)
            pduData = pduData[len(cmd):]
        for cmd in server_cmds:
            cmd.dump()
        self.assertEqual(packet['Flags'], rpch.RTS_FLAG_PING)

    def test_5(self):
        if False:
            for i in range(10):
                print('nop')
        resp = b'\x05\x00\x14\x03\x10\x00\x00\x00,\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x03\x00\x06\x00\x00\x00\x01\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x01\x00\x02\x00\x00\x00\xc0\xd4' + b'\x01\x00'
        packet = rpch.RTSHeader(resp)
        packet.dump()
        pduData = packet['pduData']
        numberOfCommands = packet['NumberOfCommands']
        server_cmds = []
        while numberOfCommands > 0:
            numberOfCommands -= 1
            cmd_type = unpack('<L', pduData[:4])[0]
            cmd = rpch.COMMANDS[cmd_type](pduData)
            server_cmds.append(cmd)
            pduData = pduData[len(cmd):]
        for cmd in server_cmds:
            cmd.dump()
        connectionTimeout = rpch.ConnectionTimeout()
        connectionTimeout['ConnectionTimeout'] = 120000
        self.assertEqual(server_cmds[2].getData(), connectionTimeout.getData())
        receiveWindowSize = rpch.ReceiveWindowSize()
        receiveWindowSize['ReceiveWindowSize'] = 65536
        self.assertEqual(server_cmds[1].getData(), receiveWindowSize.getData())
        self.assertEqual(server_cmds[0].getData(), rpch.Version().getData())

    def test_6(self):
        if False:
            while True:
                i = 10
        resp = b'\x05\x00\x14\x03\x10\x00\x00\x008\x00\x00\x00\x00\x00' + b'\x00\x00\x02\x00\x02\x00\r\x00\x00\x00\x00\x00\x00\x00' + b'\x01\x00\x00\x00\x92\x80\x00\x00\x00\x00\x01\x00\xe3y' + b'n|\xbch\xa9M\xab\x8d\x82@\xa0\x05r2'
        packet = rpch.RTSHeader(resp)
        packet.dump()
        pduData = packet['pduData']
        numberOfCommands = packet['NumberOfCommands']
        server_cmds = []
        while numberOfCommands > 0:
            numberOfCommands -= 1
            cmd_type = unpack('<L', pduData[:4])[0]
            cmd = rpch.COMMANDS[cmd_type](pduData)
            server_cmds.append(cmd)
            pduData = pduData[len(cmd):]
        for cmd in server_cmds:
            cmd.dump()
        self.assertEqual(packet['Flags'], rpch.RTS_FLAG_OTHER_CMD)
        ack = rpch.Ack()
        ack['BytesReceived'] = 32914
        ack['AvailableWindow'] = 65536
        ack['ChannelCookie'] = rpch.RTSCookie()
        ack['ChannelCookie']['Cookie'] = b'\xe3yn|\xbch\xa9M\xab\x8d\x82@\xa0\x05r2'
        self.assertEqual(server_cmds[1]['Ack'].getData(), ack.getData())

    def test_7(self):
        if False:
            i = 10
            return i + 15
        resp = b'\x05\x00\x14\x03\x10\x00\x00\x00\x80\x00\x00\x00\x00\x00' + b'\x00\x00\x08\x00\x07\x00\x06\x00\x00\x00\x01\x00\x00\x00' + b'\x03\x00\x00\x00a\xec\x8b\xb3@(\xa8F\xba\xfd' + b'\x90\xcfm1\xdc)\x03\x00\x00\x00 \xce\x94"' + b'0\x83\x1bE\x94\xea\r~\x05\xd2\xa8Z\x00\x00' + b'\x00\x00\x00\x00\x01\x00\x02\x00\x00\x00\xc0\xd4\x01\x00' + b'\x0c\x00\x00\x00\xdf(\xb4 w\xa4pB\xb1\xd1' + b'J\x03I_k{\x0b\x00\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\xc0\xa8\x02\xfe\x00\x00\x00\x00\x00\x00' + b'\x00\x00'
        packet = rpch.RTSHeader(resp)
        packet.dump()
        pduData = packet['pduData']
        numberOfCommands = packet['NumberOfCommands']
        self.assertEqual(packet['Flags'], rpch.RTS_FLAG_IN_CHANNEL)
        server_cmds = []
        while numberOfCommands > 0:
            numberOfCommands -= 1
            cmd_type = unpack('<L', pduData[:4])[0]
            cmd = rpch.COMMANDS[cmd_type](pduData)
            server_cmds.append(cmd)
            pduData = pduData[len(cmd):]
        for cmd in server_cmds:
            cmd.dump()

    def test_8(self):
        if False:
            return 10
        resp = b'\x05\x00\x14\x03\x10\x00\x00\x00T\x00\x00\x00\x00\x00' + b'\x00\x00\x10\x00\x05\x00\x06\x00\x00\x00\x01\x00\x00\x00' + b'\x03\x00\x00\x00a\xec\x8b\xb3@(\xa8F\xba\xfd' + b'\x90\xcfm1\xdc)\x03\x00\x00\x00\xbc8\x105' + b'\xa7\xf0=C\x9c?D\x85n\xf1\xc3\xb0\x04\x00' + b'\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x01\x00'
        packet = rpch.RTSHeader(resp)
        packet.dump()
        pduData = packet['pduData']
        numberOfCommands = packet['NumberOfCommands']
        self.assertEqual(packet['Flags'], rpch.RTS_FLAG_OUT_CHANNEL)
        server_cmds = []
        while numberOfCommands > 0:
            numberOfCommands -= 1
            cmd_type = unpack('<L', pduData[:4])[0]
            cmd = rpch.COMMANDS[cmd_type](pduData)
            server_cmds.append(cmd)
            pduData = pduData[len(cmd):]
        for cmd in server_cmds:
            cmd.dump()
        channelLifetime = rpch.ChannelLifetime()
        channelLifetime['ChannelLifetime'] = 1073741824
        self.assertEqual(server_cmds[-2].getData(), channelLifetime.getData())
if __name__ == '__main__':
    unittest.main(verbosity=1)