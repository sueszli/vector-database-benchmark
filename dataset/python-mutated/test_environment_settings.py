from pyflink.java_gateway import get_gateway
from pyflink.common import Configuration
from pyflink.table import EnvironmentSettings
from pyflink.testing.test_case_utils import PyFlinkTestCase

class EnvironmentSettingsTests(PyFlinkTestCase):

    def test_mode_selection(self):
        if False:
            print('Hello World!')
        builder = EnvironmentSettings.new_instance()
        environment_settings = builder.build()
        self.assertTrue(environment_settings.is_streaming_mode())
        environment_settings = builder.in_streaming_mode().build()
        self.assertTrue(environment_settings.is_streaming_mode())
        environment_settings = EnvironmentSettings.in_streaming_mode()
        self.assertTrue(environment_settings.is_streaming_mode())
        environment_settings = builder.in_batch_mode().build()
        self.assertFalse(environment_settings.is_streaming_mode())
        environment_settings = EnvironmentSettings.in_batch_mode()
        self.assertFalse(environment_settings.is_streaming_mode())

    def test_with_built_in_catalog_name(self):
        if False:
            i = 10
            return i + 15
        gateway = get_gateway()
        DEFAULT_BUILTIN_CATALOG = gateway.jvm.TableConfigOptions.TABLE_CATALOG_NAME.defaultValue()
        builder = EnvironmentSettings.new_instance()
        environment_settings = builder.build()
        self.assertEqual(environment_settings.get_built_in_catalog_name(), DEFAULT_BUILTIN_CATALOG)
        environment_settings = builder.with_built_in_catalog_name('my_catalog').build()
        self.assertEqual(environment_settings.get_built_in_catalog_name(), 'my_catalog')

    def test_with_built_in_database_name(self):
        if False:
            while True:
                i = 10
        gateway = get_gateway()
        DEFAULT_BUILTIN_DATABASE = gateway.jvm.TableConfigOptions.TABLE_DATABASE_NAME.defaultValue()
        builder = EnvironmentSettings.new_instance()
        environment_settings = builder.build()
        self.assertEqual(environment_settings.get_built_in_database_name(), DEFAULT_BUILTIN_DATABASE)
        environment_settings = builder.with_built_in_database_name('my_database').build()
        self.assertEqual(environment_settings.get_built_in_database_name(), 'my_database')

    def test_to_configuration(self):
        if False:
            for i in range(10):
                print('nop')
        expected_settings = EnvironmentSettings.new_instance().in_batch_mode().build()
        config = expected_settings.get_configuration()
        self.assertEqual('BATCH', config.get_string('execution.runtime-mode', 'stream'))

    def test_from_configuration(self):
        if False:
            return 10
        config = Configuration()
        config.set_string('execution.runtime-mode', 'batch')
        actual_setting = EnvironmentSettings.new_instance().with_configuration(config).build()
        self.assertFalse(actual_setting.is_streaming_mode(), 'Use batch mode.')