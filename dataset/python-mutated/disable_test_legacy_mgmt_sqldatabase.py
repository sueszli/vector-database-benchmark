import unittest
from azure.servicemanagement import EventLog, ServerQuota, Server, Servers, ServiceObjective, Database, FirewallRule, SqlDatabaseManagementService
from testutils.common_recordingtestcase import TestMode, record
from tests.legacy_mgmt_testcase import LegacyMgmtTestCase

class LegacyMgmtSqlDatabaseTest(LegacyMgmtTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(LegacyMgmtSqlDatabaseTest, self).setUp()
        self.sqlms = self.create_service_management(SqlDatabaseManagementService)
        self.created_server = None

    def tearDown(self):
        if False:
            print('Hello World!')
        if not self.is_playback():
            if self.created_server:
                try:
                    self.sqlms.delete_server(self.created_server)
                except:
                    pass
        return super(LegacyMgmtSqlDatabaseTest, self).tearDown()

    def _create_server(self):
        if False:
            print('Hello World!')
        result = self.sqlms.create_server('azuredb', 'T5ii-B48x', 'West US')
        self.created_server = result.server_name

    def _server_exists(self, server_name):
        if False:
            return 10
        result = self.sqlms.list_servers()
        match = [s for s in result if s.name == server_name]
        return len(match) == 1

    def _create_database(self, name):
        if False:
            for i in range(10):
                print('nop')
        result = self.sqlms.create_database(self.created_server, name, 'dd6d99bb-f193-4ec1-86f2-43d3bccbc49c', edition='Basic')

    @record
    def test_create_server(self):
        if False:
            print('Hello World!')
        result = self.sqlms.create_server('azuredb', 'T5ii-B48x', 'West US')
        self.created_server = result.server_name
        self.assertGreater(len(result.server_name), 0)
        self.assertGreater(len(result.fully_qualified_domain_name), 0)
        self.assertTrue(self._server_exists(self.created_server))

    @record
    def test_set_server_admin_password(self):
        if False:
            return 10
        self._create_server()
        result = self.sqlms.set_server_admin_password(self.created_server, 'U6jj-C59y')
        self.assertIsNone(result)

    @record
    def test_delete_server(self):
        if False:
            return 10
        self._create_server()
        result = self.sqlms.delete_server(self.created_server)
        self.assertIsNone(result)
        self.assertFalse(self._server_exists(self.created_server))

    @record
    def test_list_servers(self):
        if False:
            print('Hello World!')
        self._create_server()
        result = self.sqlms.list_servers()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Servers)
        for server in result:
            self.assertIsInstance(server, Server)
        match = [s for s in result if s.name == self.created_server][0]
        self.assertEqual(match.name, self.created_server)
        self.assertEqual(match.administrator_login, 'azuredb')
        self.assertEqual(match.location, 'West US')
        self.assertEqual(match.geo_paired_region, '')
        self.assertTrue(match.fully_qualified_domain_name.startswith(self.created_server))
        self.assertGreater(len(match.version), 0)

    @record
    def test_list_quotas(self):
        if False:
            print('Hello World!')
        self._create_server()
        result = self.sqlms.list_quotas(self.created_server)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        for quota in result:
            self.assertIsInstance(quota, ServerQuota)
            self.assertGreater(len(quota.name), 0)
            self.assertGreater(quota.value, 0)

    @record
    def test_create_firewall_rule(self):
        if False:
            i = 10
            return i + 15
        self._create_server()
        result = self.sqlms.create_firewall_rule(self.created_server, 'AllowAll', '192.168.144.0', '192.168.144.255')
        self.assertIsNone(result)

    @record
    def test_delete_firewall_rule(self):
        if False:
            print('Hello World!')
        self._create_server()
        result = self.sqlms.create_firewall_rule(self.created_server, 'AllowAll', '192.168.144.0', '192.168.144.255')
        result = self.sqlms.delete_firewall_rule(self.created_server, 'AllowAll')
        self.assertIsNone(result)

    @record
    def test_update_firewall_rule(self):
        if False:
            print('Hello World!')
        self._create_server()
        result = self.sqlms.create_firewall_rule(self.created_server, 'AllowAll', '192.168.144.0', '192.168.144.255')
        result = self.sqlms.update_firewall_rule(self.created_server, 'AllowAll', '192.168.116.0', '192.168.116.255')
        self.assertIsNone(result)

    @record
    def test_list_firewall_rules(self):
        if False:
            for i in range(10):
                print('nop')
        self._create_server()
        result = self.sqlms.create_firewall_rule(self.created_server, 'AllowAll', '192.168.144.0', '192.168.144.255')
        result = self.sqlms.list_firewall_rules(self.created_server)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        for rule in result:
            self.assertIsInstance(rule, FirewallRule)

    @record
    def test_list_service_level_objectives(self):
        if False:
            print('Hello World!')
        self._create_server()
        result = self.sqlms.list_service_level_objectives(self.created_server)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        for rule in result:
            self.assertIsInstance(rule, ServiceObjective)

    @record
    def test_create_database(self):
        if False:
            return 10
        self._create_server()
        result = self.sqlms.create_database(self.created_server, 'testdb', 'dd6d99bb-f193-4ec1-86f2-43d3bccbc49c', edition='Basic')
        self.assertIsNone(result)

    @record
    def test_delete_database(self):
        if False:
            i = 10
            return i + 15
        self._create_server()
        self._create_database('temp')
        result = self.sqlms.delete_database(self.created_server, 'temp')
        result = self.sqlms.list_databases(self.created_server)
        match = [d for d in result if d.name == 'temp']
        self.assertEqual(len(match), 0)

    @record
    def test_update_database(self):
        if False:
            i = 10
            return i + 15
        self._create_server()
        self._create_database('temp')
        result = self.sqlms.update_database(self.created_server, 'temp', 'newname')
        result = self.sqlms.list_databases(self.created_server)
        match = [d for d in result if d.name == 'newname']
        self.assertEqual(len(match), 1)

    @record
    def test_list_databases(self):
        if False:
            while True:
                i = 10
        self._create_server()
        self._create_database('temp')
        result = self.sqlms.list_databases(self.created_server)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)
        for db in result:
            self.assertIsInstance(db, Database)
        match = [d for d in result if d.name == 'temp'][0]
        self.assertEqual(match.name, 'temp')
        self.assertEqual(match.state, 'Normal')
        self.assertGreater(match.max_size_bytes, 0)
        self.assertGreater(match.id, 0)
        self.assertGreater(len(match.edition), 0)
        self.assertGreater(len(match.collation_name), 0)
if __name__ == '__main__':
    unittest.main()