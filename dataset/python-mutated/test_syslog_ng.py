"""
Test module for syslog_ng state
"""
import os
import re
import tempfile
import salt.modules.syslog_ng as syslog_ng_module
import salt.states.syslog_ng as syslog_ng
import salt.utils.files
import salt.utils.yaml
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import MagicMock, patch
from tests.support.unit import TestCase
SOURCE_1_CONFIG = {'id': 's_tail', 'config': '\n        source:\n            - file:\n              - \'"/var/log/apache/access.log"\'\n              - follow_freq : 1\n              - flags:\n                - no-parse\n                - validate-utf8\n        '}
SOURCE_1_EXPECTED = '\nsource s_tail {\n   file(\n         "/var/log/apache/access.log",\n         follow_freq(1),\n         flags(no-parse, validate-utf8)\n   );\n};\n'
SOURCE_2_CONFIG = {'id': 's_gsoc2014', 'config': '\n        source:\n          - tcp:\n            - ip: \'"0.0.0.0"\'\n            - port: 1234\n            - flags: no-parse\n        '}
SOURCE_2_EXPECTED = '\nsource s_gsoc2014 {\n   tcp(\n         ip("0.0.0.0"),\n         port(1234),\n         flags(no-parse)\n   );\n};'
FILTER_1_CONFIG = {'id': 'f_json', 'config': '\n        filter:\n          - match:\n            - \'"@json:"\'\n        '}
FILTER_1_EXPECTED = '\n    filter f_json {\n       match(\n             "@json:"\n       );\n    };\n    '
TEMPLATE_1_CONFIG = {'id': 't_demo_filetemplate', 'config': '\n        template:\n          - template:\n            - \'"$ISODATE $HOST $MSG\n"\'\n          - template_escape:\n            - "no"\n        '}
TEMPLATE_1_EXPECTED = '\n    template t_demo_filetemplate {\n       template(\n             "$ISODATE $HOST $MSG "\n       );\n       template_escape(\n             no\n       );\n    };\n    '
REWRITE_1_CONFIG = {'id': 'r_set_message_to_MESSAGE', 'config': '\n        rewrite:\n          - set:\n            - \'"${.json.message}"\'\n            - value : \'"$MESSAGE"\'\n        '}
REWRITE_1_EXPECTED = '\n    rewrite r_set_message_to_MESSAGE {\n       set(\n             "${.json.message}",\n             value("$MESSAGE")\n       );\n    };\n    '
LOG_1_CONFIG = {'id': 'l_gsoc2014', 'config': '\n        log:\n          - source: s_gsoc2014\n          - junction:\n            - channel:\n              - filter: f_json\n              - parser: p_json\n              - rewrite: r_set_json_tag\n              - rewrite: r_set_message_to_MESSAGE\n              - destination:\n                - file:\n                  - \'"/tmp/json-input.log"\'\n                  - template: t_gsoc2014\n              - flags: final\n            - channel:\n              - filter: f_not_json\n              - parser:\n                - syslog-parser: []\n              - rewrite: r_set_syslog_tag\n              - flags: final\n          - destination:\n            - file:\n              - \'"/tmp/all.log"\'\n              - template: t_gsoc2014\n        '}
LOG_1_EXPECTED = '\n    log {\n       source(s_gsoc2014);\n       junction {\n          channel {\n             filter(f_json);\n             parser(p_json);\n             rewrite(r_set_json_tag);\n             rewrite(r_set_message_to_MESSAGE);\n             destination {\n                file(\n                      "/tmp/json-input.log",\n                      template(t_gsoc2014)\n                );\n             };\n             flags(final);\n          };\n          channel {\n             filter(f_not_json);\n             parser {\n                syslog-parser(\n\n                );\n             };\n             rewrite(r_set_syslog_tag);\n             flags(final);\n          };\n       };\n       destination {\n          file(\n                "/tmp/all.log",\n                template(t_gsoc2014)\n          );\n       };\n    };\n    '
OPTIONS_1_CONFIG = {'id': 'global_options', 'config': '\n        options:\n          - time_reap: 30\n          - mark_freq: 10\n          - keep_hostname: "yes"\n        '}
OPTIONS_1_EXPECTED = '\n    options {\n        time_reap(30);\n        mark_freq(10);\n        keep_hostname(yes);\n    };\n    '
SHORT_FORM_CONFIG = {'id': 'source.s_gsoc', 'config': '\n          - tcp:\n            - ip: \'"0.0.0.0"\'\n            - port: 1234\n            - flags: no-parse\n        '}
SHORT_FORM_EXPECTED = '\n    source s_gsoc {\n        tcp(\n            ip(\n                "0.0.0.0"\n            ),\n            port(\n                1234\n            ),\n            flags(\n              no-parse\n            )\n        );\n    };\n    '
GIVEN_CONFIG = {'id': 'config.some_name', 'config': '\n               source s_gsoc {\n                  tcp(\n                      ip(\n                          "0.0.0.0"\n                      ),\n                      port(\n                          1234\n                      ),\n                      flags(\n                        no-parse\n                      )\n                  );\n               };\n        '}
_SALT_VAR_WITH_MODULE_METHODS = {'syslog_ng.config': syslog_ng_module.config, 'syslog_ng.start': syslog_ng_module.start, 'syslog_ng.reload': syslog_ng_module.reload_, 'syslog_ng.stop': syslog_ng_module.stop, 'syslog_ng.write_version': syslog_ng_module.write_version, 'syslog_ng.write_config': syslog_ng_module.write_config}

def remove_whitespaces(source):
    if False:
        return 10
    return re.sub('\\s+', '', source.strip())

class SyslogNGTestCase(TestCase, LoaderModuleMockMixin):

    def setup_loader_modules(self):
        if False:
            while True:
                i = 10
        return {syslog_ng: {}, syslog_ng_module: {'__opts__': {'test': False}}}

    def test_generate_source_config(self):
        if False:
            while True:
                i = 10
        self._config_generator_template(SOURCE_1_CONFIG, SOURCE_1_EXPECTED)

    def test_generate_log_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._config_generator_template(LOG_1_CONFIG, LOG_1_EXPECTED)

    def test_generate_tcp_source_config(self):
        if False:
            i = 10
            return i + 15
        self._config_generator_template(SOURCE_2_CONFIG, SOURCE_2_EXPECTED)

    def test_generate_filter_config(self):
        if False:
            print('Hello World!')
        self._config_generator_template(FILTER_1_CONFIG, FILTER_1_EXPECTED)

    def test_generate_template_config(self):
        if False:
            print('Hello World!')
        self._config_generator_template(TEMPLATE_1_CONFIG, TEMPLATE_1_EXPECTED)

    def test_generate_rewrite_config(self):
        if False:
            while True:
                i = 10
        self._config_generator_template(REWRITE_1_CONFIG, REWRITE_1_EXPECTED)

    def test_generate_global_options_config(self):
        if False:
            for i in range(10):
                print('nop')
        self._config_generator_template(OPTIONS_1_CONFIG, OPTIONS_1_EXPECTED)

    def test_generate_short_form_statement(self):
        if False:
            print('Hello World!')
        self._config_generator_template(SHORT_FORM_CONFIG, SHORT_FORM_EXPECTED)

    def test_generate_given_config(self):
        if False:
            return 10
        self._config_generator_template(GIVEN_CONFIG, SHORT_FORM_EXPECTED)

    def _config_generator_template(self, yaml_input, expected):
        if False:
            i = 10
            return i + 15
        parsed_yaml_config = salt.utils.data.decode(salt.utils.yaml.safe_load(yaml_input['config']))
        id = yaml_input['id']
        with patch.dict(syslog_ng.__salt__, _SALT_VAR_WITH_MODULE_METHODS):
            got = syslog_ng.config(id, config=parsed_yaml_config, write=False)
            config = got['changes']['new']
            self.assertEqual(remove_whitespaces(expected), remove_whitespaces(config))
            self.assertEqual(False, got['result'])

    def test_write_config(self):
        if False:
            return 10
        yaml_inputs = (SOURCE_2_CONFIG, SOURCE_1_CONFIG, FILTER_1_CONFIG, TEMPLATE_1_CONFIG, REWRITE_1_CONFIG, LOG_1_CONFIG)
        expected_outputs = (SOURCE_2_EXPECTED, SOURCE_1_EXPECTED, FILTER_1_EXPECTED, TEMPLATE_1_EXPECTED, REWRITE_1_EXPECTED, LOG_1_EXPECTED)
        (config_file_fd, config_file_name) = tempfile.mkstemp()
        os.close(config_file_fd)
        with patch.dict(syslog_ng.__salt__, _SALT_VAR_WITH_MODULE_METHODS):
            syslog_ng_module.set_config_file(config_file_name)
            syslog_ng_module.write_version('3.6')
            syslog_ng_module.write_config(config='@include "scl.conf"')
            for i in yaml_inputs:
                parsed_yaml_config = salt.utils.data.decode(salt.utils.yaml.safe_load(i['config']))
                id = i['id']
                got = syslog_ng.config(id, config=parsed_yaml_config, write=True)
            written_config = ''
            with salt.utils.files.fopen(config_file_name, 'r') as f:
                written_config = salt.utils.stringutils.to_unicode(f.read())
            config_without_whitespaces = remove_whitespaces(written_config)
            for i in expected_outputs:
                without_whitespaces = remove_whitespaces(i)
                self.assertIn(without_whitespaces, config_without_whitespaces)
            syslog_ng_module.set_config_file('')
            os.remove(config_file_name)

    def test_started_state_generate_valid_cli_command(self):
        if False:
            return 10
        mock_func = MagicMock(return_value={'retcode': 0, 'stdout': '', 'pid': 1000})
        with patch.dict(syslog_ng.__salt__, _SALT_VAR_WITH_MODULE_METHODS):
            with patch.dict(syslog_ng_module.__salt__, {'cmd.run_all': mock_func}):
                got = syslog_ng.started(user='joe', group='users', enable_core=True)
                command = got['changes']['new']
                self.assertTrue(command.endswith('syslog-ng --user=joe --group=users --enable-core --cfgfile=/etc/syslog-ng.conf'))