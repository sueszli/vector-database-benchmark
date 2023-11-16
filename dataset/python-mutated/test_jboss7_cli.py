import re
import salt.modules.jboss7_cli as jboss7_cli
from salt.exceptions import CommandExecutionError
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import patch
from tests.support.unit import TestCase

class CmdMock:
    commands = []
    command_response_func = None
    cli_commands = []
    default_response = {'retcode': 0, 'stdout': ' {\n        "outcome" => "success"\n    }', 'stderr': ''}

    def __init__(self, command_response_func=None):
        if False:
            while True:
                i = 10
        self.command_response_func = command_response_func

    def run_all(self, command):
        if False:
            return 10
        self.commands.append(command)
        if self.command_response_func is not None:
            return self.command_response_func(command)
        cli_command = self.__get_cli_command(command)
        self.cli_commands.append(cli_command)
        return self.default_response

    @staticmethod
    def __get_cli_command(command):
        if False:
            return 10
        command_re = re.compile('--command=\\"\\s*(.+?)\\s*\\"$', re.DOTALL)
        m = command_re.search(command)
        if m:
            cli_command = m.group(1)
            return cli_command
        return None

    def get_last_command(self):
        if False:
            print('Hello World!')
        if len(self.commands) > 0:
            return self.commands[-1]
        else:
            return None

    def get_last_cli_command(self):
        if False:
            print('Hello World!')
        if len(self.cli_commands) > 0:
            return self.cli_commands[-1]
        else:
            return None

    def clear(self):
        if False:
            print('Hello World!')
        self.commands = []
        self.command_response_func = None
        self.cli_commands = []

class JBoss7CliTestCase(TestCase, LoaderModuleMockMixin):
    cmd = CmdMock()
    jboss_config = {'cli_path': '/opt/jboss/jboss-eap-6.0.1/bin/jboss-cli.sh', 'controller': '123.234.345.456:9999', 'instance_name': 'Instance1', 'cli_user': 'jbossadm', 'cli_password': 'jbossadm', 'status_url': 'http://sampleapp.example.com:8080/'}

    def setup_loader_modules(self):
        if False:
            while True:
                i = 10
        self.cmd = CmdMock()
        self.addCleanup(delattr, self, 'cmd')
        return {jboss7_cli: {'__salt__': {'cmd.run_all': self.cmd.run_all}}}

    def test_controller_authentication(self):
        if False:
            for i in range(10):
                print('nop')
        jboss7_cli.run_operation(self.jboss_config, 'some cli operation')
        self.assertEqual(self.cmd.get_last_command(), '/opt/jboss/jboss-eap-6.0.1/bin/jboss-cli.sh --connect --controller="123.234.345.456:9999" --user="jbossadm" --password="jbossadm" --command="some cli operation"')

    def test_controller_without_authentication(self):
        if False:
            return 10
        jboss_config = {'cli_path': '/opt/jboss/jboss-eap-6.0.1/bin/jboss-cli.sh', 'controller': '123.234.345.456:9999'}
        jboss7_cli.run_operation(jboss_config, 'some cli operation')
        self.assertEqual(self.cmd.get_last_command(), '/opt/jboss/jboss-eap-6.0.1/bin/jboss-cli.sh --connect --controller="123.234.345.456:9999" --command="some cli operation"')

    def test_operation_execution(self):
        if False:
            i = 10
            return i + 15
        operation = 'sample_operation'
        jboss7_cli.run_operation(self.jboss_config, operation)
        self.assertEqual(self.cmd.get_last_command(), '/opt/jboss/jboss-eap-6.0.1/bin/jboss-cli.sh --connect --controller="123.234.345.456:9999" --user="jbossadm" --password="jbossadm" --command="sample_operation"')

    def test_handling_jboss_error(self):
        if False:
            for i in range(10):
                print('nop')

        def command_response(command):
            if False:
                while True:
                    i = 10
            return {'retcode': 1, 'stdout': '{\n                       "outcome" => "failed",\n                       "failure-description" => "JBAS014807: Management resource \'[\n                       (\\"subsystem\\" => \\"datasources\\"),\n                       (\\"data-source\\" => \\"non-existing\\")\n                    ]\' not found",\n                        "rolled-back" => true,\n                        "response-headers" => {"process-state" => "reload-required"}\n                    }\n                    ', 'stderr': 'some err'}
        self.cmd.command_response_func = command_response
        result = jboss7_cli.run_operation(self.jboss_config, 'some cli command')
        self.assertFalse(result['success'])
        self.assertEqual(result['err_code'], 'JBAS014807')

    def test_handling_cmd_not_exists(self):
        if False:
            for i in range(10):
                print('nop')

        def command_response(command):
            if False:
                for i in range(10):
                    print('nop')
            return {'retcode': 127, 'stdout': 'Command not exists', 'stderr': 'some err'}
        self.cmd.command_response_func = command_response
        try:
            jboss7_cli.run_operation(self.jboss_config, 'some cli command')
            assert False
        except CommandExecutionError as err:
            self.assertTrue(str(err).startswith('Could not execute jboss-cli.sh script'))

    def test_handling_other_cmd_error(self):
        if False:
            print('Hello World!')

        def command_response(command):
            if False:
                print('Hello World!')
            return {'retcode': 1, 'stdout': 'Command not exists', 'stderr': 'some err'}
        self.cmd.command_response_func = command_response
        try:
            jboss7_cli.run_command(self.jboss_config, 'some cli command')
            self.fail('An exception should be thrown')
        except CommandExecutionError as err:
            self.assertTrue(str(err).startswith('Command execution failed'))

    def test_matches_cli_output(self):
        if False:
            print('Hello World!')
        text = '{\n            "key1" => "value1"\n            "key2" => "value2"\n            }\n            '
        self.assertTrue(jboss7_cli._is_cli_output(text))

    def test_not_matches_cli_output(self):
        if False:
            print('Hello World!')
        text = 'Some error '
        self.assertFalse(jboss7_cli._is_cli_output(text))

    def test_parse_flat_dictionary(self):
        if False:
            i = 10
            return i + 15
        text = '{\n            "key1" => "value1"\n            "key2" => "value2"\n            }'
        result = jboss7_cli._parse(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['key1'], 'value1')
        self.assertEqual(result['key2'], 'value2')

    def test_parse_nested_dictionary(self):
        if False:
            return 10
        text = '{\n            "key1" => "value1",\n            "key2" => {\n                "nested_key1" => "nested_value1"\n            }\n        }'
        result = jboss7_cli._parse(text)
        self.assertEqual(len(result), 2)
        self.assertEqual(result['key1'], 'value1')
        self.assertEqual(len(result['key2']), 1)
        self.assertEqual(result['key2']['nested_key1'], 'nested_value1')

    def test_parse_string_after_dict(self):
        if False:
            while True:
                i = 10
        text = '{\n            "result" => {\n                "jta" => true\n            },\n            "response-headers" => {"process-state" => "reload-required"}\n        }'
        result = jboss7_cli._parse(text)
        self.assertTrue(result['result']['jta'])
        self.assertEqual(result['response-headers']['process-state'], 'reload-required')

    def test_parse_all_datatypes(self):
        if False:
            for i in range(10):
                print('nop')
        text = '{\n            "outcome" => "success",\n            "result" => {\n                "allocation-retry" => undefined,\n                "connection-url" => "jdbc:mysql://localhost:3306/appdb",\n                "driver-name" => "mysql",\n                "enabled" => false,\n                "jta" => true\n            },\n            "response-headers" => {"process-state" => "reload-required"}\n        }'
        result = jboss7_cli._parse(text)
        self.assertEqual(result['outcome'], 'success')
        self.assertIsNone(result['result']['allocation-retry'])
        self.assertEqual(result['result']['connection-url'], 'jdbc:mysql://localhost:3306/appdb')
        self.assertEqual(result['result']['driver-name'], 'mysql')
        self.assertEqual(result['result']['enabled'], False)
        self.assertTrue(result['result']['jta'])
        self.assertEqual(result['response-headers']['process-state'], 'reload-required')

    def test_multiline_strings_with_escaped_quotes(self):
        if False:
            print('Hello World!')
        text = '{\n            "outcome" => "failed",\n            "failure-description" => "JBAS014807: Management resource \'[\n            (\\"subsystem\\" => \\"datasources\\"),\n            (\\"data-source\\" => \\"asc\\")\n        ]\' not found",\n            "rolled-back" => true,\n            "response-headers" => {"process-state" => "reload-required"}\n        }'
        result = jboss7_cli._parse(text)
        self.assertEqual(result['outcome'], 'failed')
        self.assertTrue(result['rolled-back'])
        self.assertEqual(result['response-headers']['process-state'], 'reload-required')
        self.assertEqual(result['failure-description'], 'JBAS014807: Management resource \'[\n            (\\"subsystem\\" => \\"datasources\\"),\n            (\\"data-source\\" => \\"asc\\")\n        ]\' not found')

    def test_handling_double_backslash_in_return_values(self):
        if False:
            while True:
                i = 10
        text = '{\n                 "outcome" => "success",\n                 "result" => {\n                    "binding-type" => "simple",\n                    "value" => "DOMAIN\\\\foo"\n                   }\n                }'
        result = jboss7_cli._parse(text)
        self.assertEqual(result['outcome'], 'success')
        self.assertEqual(result['result']['binding-type'], 'simple')
        self.assertEqual(result['result']['value'], 'DOMAIN\\foo')

    def test_numbers_without_quotes(self):
        if False:
            while True:
                i = 10
        text = '{\n                "outcome" => "success",\n                "result" => {\n                    "min-pool-size" => 1233,\n                    "new-connection-sql" => undefined\n                }\n            }'
        result = jboss7_cli._parse(text)
        self.assertEqual(result['outcome'], 'success')
        self.assertEqual(result['result']['min-pool-size'], 1233)
        self.assertIsNone(result['result']['new-connection-sql'])

    def test_all_datasource_properties(self):
        if False:
            while True:
                i = 10
        text = '{\n            "outcome" => "success",\n            "result" => {\n                "allocation-retry" => undefined,\n                "allocation-retry-wait-millis" => undefined,\n                "allow-multiple-users" => undefined,\n                "background-validation" => undefined,\n                "background-validation-millis" => undefined,\n                "blocking-timeout-wait-millis" => undefined,\n                "check-valid-connection-sql" => undefined,\n                "connection-properties" => undefined,\n                "connection-url" => "jdbc:mysql:thin:@db.example.com",\n                "datasource-class" => undefined,\n                "driver-class" => undefined,\n                "driver-name" => "mysql",\n                "enabled" => true,\n                "exception-sorter-class-name" => undefined,\n                "exception-sorter-properties" => undefined,\n                "flush-strategy" => "FailingConnectionOnly",\n                "idle-timeout-minutes" => undefined,\n                "jndi-name" => "java:/appDS",\n                "jta" => true,\n                "max-pool-size" => 20,\n                "min-pool-size" => 3,\n                "new-connection-sql" => undefined,\n                "password" => "Password4321",\n                "pool-prefill" => undefined,\n                "pool-use-strict-min" => undefined,\n                "prepared-statements-cache-size" => undefined,\n                "query-timeout" => undefined,\n                "reauth-plugin-class-name" => undefined,\n                "reauth-plugin-properties" => undefined,\n                "security-domain" => undefined,\n                "set-tx-query-timeout" => false,\n                "share-prepared-statements" => false,\n                "spy" => false,\n                "stale-connection-checker-class-name" => undefined,\n                "stale-connection-checker-properties" => undefined,\n                "track-statements" => "NOWARN",\n                "transaction-isolation" => undefined,\n                "url-delimiter" => undefined,\n                "url-selector-strategy-class-name" => undefined,\n                "use-ccm" => "true",\n                "use-fast-fail" => false,\n                "use-java-context" => "false",\n                "use-try-lock" => undefined,\n                "user-name" => "user1",\n                "valid-connection-checker-class-name" => undefined,\n                "valid-connection-checker-properties" => undefined,\n                "validate-on-match" => false,\n                "statistics" => {\n                    "jdbc" => undefined,\n                    "pool" => undefined\n                }\n            },\n            "response-headers" => {"process-state" => "reload-required"}\n        }'
        result = jboss7_cli._parse(text)
        self.assertEqual(result['outcome'], 'success')
        self.assertEqual(result['result']['max-pool-size'], 20)
        self.assertIsNone(result['result']['new-connection-sql'])
        self.assertIsNone(result['result']['url-delimiter'])
        self.assertFalse(result['result']['validate-on-match'])

    def test_datasource_resource_one_attribute_description(self):
        if False:
            return 10
        cli_output = '{\n            "outcome" => "success",\n            "result" => {\n                "description" => "A JDBC data-source configuration",\n                "head-comment-allowed" => true,\n                "tail-comment-allowed" => true,\n                "attributes" => {\n                    "connection-url" => {\n                        "type" => STRING,\n                        "description" => "The JDBC driver connection URL",\n                        "expressions-allowed" => true,\n                        "nillable" => false,\n                        "min-length" => 1L,\n                        "max-length" => 2147483647L,\n                        "access-type" => "read-write",\n                        "storage" => "configuration",\n                        "restart-required" => "no-services"\n                    }\n                },\n                "children" => {"connection-properties" => {"description" => "The connection-properties element allows you to pass in arbitrary connection properties to the Driver.connect(url, props) method"}}\n            }\n        }\n        '
        result = jboss7_cli._parse(cli_output)
        self.assertEqual(result['outcome'], 'success')
        conn_url_attributes = result['result']['attributes']['connection-url']
        self.assertEqual(conn_url_attributes['type'], 'STRING')
        self.assertEqual(conn_url_attributes['description'], 'The JDBC driver connection URL')
        self.assertTrue(conn_url_attributes['expressions-allowed'])
        self.assertFalse(conn_url_attributes['nillable'])
        self.assertEqual(conn_url_attributes['min-length'], 1)
        self.assertEqual(conn_url_attributes['max-length'], 2147483647)
        self.assertEqual(conn_url_attributes['access-type'], 'read-write')
        self.assertEqual(conn_url_attributes['storage'], 'configuration')
        self.assertEqual(conn_url_attributes['restart-required'], 'no-services')

    def test_datasource_complete_resource_description(self):
        if False:
            i = 10
            return i + 15
        cli_output = '{\n            "outcome" => "success",\n            "result" => {\n                "description" => "A JDBC data-source configuration",\n                "head-comment-allowed" => true,\n                "tail-comment-allowed" => true,\n                "attributes" => {\n                    "connection-url" => {\n                        "type" => STRING,\n                        "description" => "The JDBC driver connection URL",\n                        "expressions-allowed" => true,\n                        "nillable" => false,\n                        "min-length" => 1L,\n                        "max-length" => 2147483647L,\n                        "access-type" => "read-write",\n                        "storage" => "configuration",\n                        "restart-required" => "no-services"\n                    }\n                },\n                "children" => {"connection-properties" => {"description" => "The connection-properties element allows you to pass in arbitrary connection properties to the Driver.connect(url, props) method"}}\n            }\n        }\n        '
        result = jboss7_cli._parse(cli_output)
        self.assertEqual(result['outcome'], 'success')
        conn_url_attributes = result['result']['attributes']['connection-url']
        self.assertEqual(conn_url_attributes['type'], 'STRING')
        self.assertEqual(conn_url_attributes['description'], 'The JDBC driver connection URL')
        self.assertTrue(conn_url_attributes['expressions-allowed'])
        self.assertFalse(conn_url_attributes['nillable'])
        self.assertEqual(conn_url_attributes['min-length'], 1)
        self.assertEqual(conn_url_attributes['max-length'], 2147483647)
        self.assertEqual(conn_url_attributes['access-type'], 'read-write')
        self.assertEqual(conn_url_attributes['storage'], 'configuration')
        self.assertEqual(conn_url_attributes['restart-required'], 'no-services')

    def test_escaping_operation_with_backslashes_and_quotes(self):
        if False:
            i = 10
            return i + 15
        operation = '/subsystem=naming/binding="java:/sampleapp/web-module/ldap/username":add(binding-type=simple, value="DOMAIN\\\\\\\\user")'
        jboss7_cli.run_operation(self.jboss_config, operation)
        self.assertEqual(self.cmd.get_last_command(), '/opt/jboss/jboss-eap-6.0.1/bin/jboss-cli.sh --connect --controller="123.234.345.456:9999" --user="jbossadm" --password="jbossadm" --command="/subsystem=naming/binding=\\"java:/sampleapp/web-module/ldap/username\\":add(binding-type=simple, value=\\"DOMAIN\\\\\\\\\\\\\\\\user\\")"')

    def test_run_operation_wflyctl_error(self):
        if False:
            return 10
        call_cli_ret = {'retcode': 1, 'stdout': '{"failure-description" => "WFLYCTL0234523: ops"}'}
        with patch('salt.modules.jboss7_cli._call_cli', return_value=call_cli_ret) as _call_cli:
            ret = jboss7_cli.run_operation(None, 'ls', False)
            self.assertEqual(ret['err_code'], 'WFLYCTL0234523')

    def test_run_operation_no_code_error(self):
        if False:
            for i in range(10):
                print('nop')
        call_cli_ret = {'retcode': 1, 'stdout': '{"failure-description" => "ERROR234523: ops"}'}
        with patch('salt.modules.jboss7_cli._call_cli', return_value=call_cli_ret) as _call_cli:
            ret = jboss7_cli.run_operation(None, 'ls', False)
            self.assertEqual(ret['err_code'], '-1')