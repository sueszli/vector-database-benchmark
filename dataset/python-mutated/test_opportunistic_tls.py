import pytest
from sslyze.json.json_output import _ServerTlsProbingResultAsJson
from sslyze.server_connectivity import check_connectivity_to_server
from sslyze.server_setting import ServerNetworkLocation, ServerNetworkConfiguration
from sslyze.errors import ServerRejectedOpportunisticTlsNegotiation
from sslyze.connection_helpers.opportunistic_tls_helpers import ProtocolWithOpportunisticTlsEnum

class TestOpportunisticTls:

    @pytest.mark.parametrize('hostname, port, protocol', [('smtp.gmail.com', 587, ProtocolWithOpportunisticTlsEnum.SMTP), ('imap.comcast.net', 143, ProtocolWithOpportunisticTlsEnum.IMAP), ('pop.comcast.net', 110, ProtocolWithOpportunisticTlsEnum.POP3), ('ldap.uchicago.edu', 389, ProtocolWithOpportunisticTlsEnum.LDAP), ('jabber.org', 5222, ProtocolWithOpportunisticTlsEnum.XMPP_SERVER)])
    def test(self, hostname, port, protocol):
        if False:
            print('Hello World!')
        server_location = ServerNetworkLocation(hostname, port)
        network_configuration = ServerNetworkConfiguration(tls_server_name_indication=hostname, tls_opportunistic_encryption=protocol)
        tls_probing_result = check_connectivity_to_server(server_location=server_location, network_configuration=network_configuration)
        assert tls_probing_result
        assert tls_probing_result.client_auth_requirement
        assert tls_probing_result.highest_tls_version_supported
        assert tls_probing_result.cipher_suite_supported
        tls_probing_result_as_json = _ServerTlsProbingResultAsJson.from_orm(tls_probing_result)
        assert tls_probing_result_as_json.json()

    def test_xmpp_but_server_rejected_opportunistic_tls(self):
        if False:
            while True:
                i = 10
        hostname = 'jabber.org'
        server_location = ServerNetworkLocation(hostname=hostname, port=5222)
        network_configuration = ServerNetworkConfiguration(xmpp_to_hostname='lol.lol', tls_server_name_indication=hostname, tls_opportunistic_encryption=ProtocolWithOpportunisticTlsEnum.XMPP)
        with pytest.raises(ServerRejectedOpportunisticTlsNegotiation):
            check_connectivity_to_server(server_location=server_location, network_configuration=network_configuration)