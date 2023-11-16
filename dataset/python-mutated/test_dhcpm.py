from __future__ import division
from __future__ import print_function
import socket
import struct
import pytest
import unittest
from six import assertRaisesRegex
from tests.dcerpc import DCERPCTests
from impacket.dcerpc.v5 import dhcpm
from impacket.dcerpc.v5.dtypes import NULL
from impacket.dcerpc.v5.rpcrt import RPC_C_AUTHN_LEVEL_PKT_PRIVACY, DCERPCException

class DHCPMTests(DCERPCTests):
    iface_uuid_v1 = dhcpm.MSRPC_UUID_DHCPSRV
    iface_uuid_v2 = dhcpm.MSRPC_UUID_DHCPSRV2
    string_binding = 'ncacn_np:{0.machine}[\\PIPE\\dhcpserver]'
    authn = True
    authn_level = RPC_C_AUTHN_LEVEL_PKT_PRIVACY

    def test_DhcpGetClientInfoV4(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect(iface_uuid=self.iface_uuid_v1)
        request = dhcpm.DhcpGetClientInfoV4()
        request['ServerIpAddress'] = NULL
        request['SearchInfo']['SearchType'] = dhcpm.DHCP_SEARCH_INFO_TYPE.DhcpClientName
        request['SearchInfo']['SearchInfo']['tag'] = dhcpm.DHCP_SEARCH_INFO_TYPE.DhcpClientName
        request['SearchInfo']['SearchInfo']['ClientName'] = self.serverName + '\x00'
        request.dump()
        with assertRaisesRegex(self, DCERPCException, 'ERROR_DHCP_JET_ERROR'):
            dce.request(request)

    def test_hDhcpGetClientInfoV4(self):
        if False:
            return 10
        (dce, rpctransport) = self.connect(iface_uuid=self.iface_uuid_v1)
        with assertRaisesRegex(self, DCERPCException, 'ERROR_DHCP_JET_ERROR'):
            dhcpm.hDhcpGetClientInfoV4(dce, dhcpm.DHCP_SEARCH_INFO_TYPE.DhcpClientName, self.serverName + '\x00')

    def test_DhcpV4GetClientInfo(self):
        if False:
            i = 10
            return i + 15
        (dce, rpctransport) = self.connect(iface_uuid=self.iface_uuid_v2)
        request = dhcpm.DhcpV4GetClientInfo()
        request['ServerIpAddress'] = NULL
        request['SearchInfo']['SearchType'] = dhcpm.DHCP_SEARCH_INFO_TYPE.DhcpClientName
        request['SearchInfo']['SearchInfo']['tag'] = dhcpm.DHCP_SEARCH_INFO_TYPE.DhcpClientName
        request['SearchInfo']['SearchInfo']['ClientName'] = self.serverName + '\x00'
        request.dump()
        with assertRaisesRegex(self, DCERPCException, 'ERROR_DHCP_INVALID_DHCP_CLIENT'):
            dce.request(request)

    def test_hDhcpEnumSubnetClientsV5(self):
        if False:
            for i in range(10):
                print('nop')
        (dce, rpctransport) = self.connect(iface_uuid=self.iface_uuid_v2)
        with assertRaisesRegex(self, DCERPCException, 'ERROR_NO_MORE_ITEMS'):
            dhcpm.hDhcpEnumSubnetClientsV5(dce)

    def test_hDhcpGetOptionValueV5(self):
        if False:
            while True:
                i = 10
        (dce, rpctransport) = self.connect(iface_uuid=self.iface_uuid_v2)
        netId = self.machine.split('.')[:-1]
        netId.append('0')
        subnet_id = struct.unpack('!I', socket.inet_aton('.'.join(netId)))[0]
        with assertRaisesRegex(self, DCERPCException, 'ERROR_DHCP_SUBNET_NOT_PRESENT'):
            dhcpm.hDhcpGetOptionValueV5(dce, 3, dhcpm.DHCP_FLAGS_OPTION_DEFAULT, NULL, NULL, dhcpm.DHCP_OPTION_SCOPE_TYPE.DhcpSubnetOptions, subnet_id)

@pytest.mark.remote
@pytest.mark.skip(reason='Disabled in Windows Server 2008 onwards')
class DHCPMTestsSMBTransport(DHCPMTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
@pytest.mark.skip(reason='Disabled in Windows Server 2008 onwards')
class DHCPMTestsSMBTransport64(DHCPMTests, unittest.TestCase):
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64

@pytest.mark.remote
class DHCPMTestsTCPTransport(DHCPMTests, unittest.TestCase):
    protocol = 'ncacn_ip_tcp'
    iface_uuid = dhcpm.MSRPC_UUID_DHCPSRV2
    string_binding_formatting = DCERPCTests.STRING_BINDING_MAPPER
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR

@pytest.mark.remote
class DHCPMTestsTCPTransport64(DHCPMTests, unittest.TestCase):
    protocol = 'ncacn_ip_tcp'
    iface_uuid = dhcpm.MSRPC_UUID_DHCPSRV2
    string_binding_formatting = DCERPCTests.STRING_BINDING_MAPPER
    transfer_syntax = DCERPCTests.TRANSFER_SYNTAX_NDR64

    @pytest.mark.xfail(reason='NDRUNION without fields as in DhcpSubnetOptions is not implemented with NDR64')
    def test_hDhcpGetOptionValueV5(self):
        if False:
            for i in range(10):
                print('nop')
        super(DHCPMTestsTCPTransport64, self).test_hDhcpGetOptionValueV5()
if __name__ == '__main__':
    unittest.main(verbosity=1)