from pathlib import Path
from typing import Generator
import pytest
from sentry import options
from sentry.options.manager import FLAG_AUTOMATOR_MODIFIABLE, FLAG_IMMUTABLE, UpdateChannel
from sentry.runner.commands.configoptions import configoptions
from sentry.runner.commands.presenters.consolepresenter import ConsolePresenter
from sentry.testutils.cases import CliTestCase

class ConfigOptionsTest(CliTestCase):
    command = configoptions

    @pytest.fixture(autouse=True, scope='class')
    def register_options(self) -> Generator[None, None, None]:
        if False:
            return 10
        options.register('readonly_option', default=10, flags=FLAG_IMMUTABLE)
        options.register('int_option', default=20, flags=FLAG_AUTOMATOR_MODIFIABLE)
        options.register('str_option', default='blabla', flags=FLAG_AUTOMATOR_MODIFIABLE)
        options.register('map_option', default={}, flags=FLAG_AUTOMATOR_MODIFIABLE)
        options.register('list_option', default=[1, 2], flags=FLAG_AUTOMATOR_MODIFIABLE)
        options.register('drifted_option', default=[], flags=FLAG_AUTOMATOR_MODIFIABLE)
        options.register('change_channel_option', default=[], flags=FLAG_AUTOMATOR_MODIFIABLE)
        options.register('to_unset_option', default=[], flags=FLAG_AUTOMATOR_MODIFIABLE)
        options.register('invalid_type', default=15, flags=FLAG_AUTOMATOR_MODIFIABLE)
        yield
        options.unregister('readonly_option')
        options.unregister('int_option')
        options.unregister('str_option')
        options.unregister('map_option')
        options.unregister('list_option')
        options.unregister('drifted_option')
        options.unregister('change_channel_option')
        options.unregister('to_unset_option')
        options.unregister('invalid_type')

    @pytest.fixture(autouse=True)
    def set_options(self) -> None:
        if False:
            i = 10
            return i + 15
        options.delete('int_option')
        options.delete('map_option')
        options.delete('list_option')
        options.set('str_option', 'old value', channel=UpdateChannel.AUTOMATOR)
        options.set('drifted_option', [1, 2, 3], channel=UpdateChannel.CLI)
        options.set('change_channel_option', [5, 6, 7], channel=UpdateChannel.CLI)
        options.set('to_unset_option', [7, 8, 9], channel=UpdateChannel.AUTOMATOR)

    def _clean_cache(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The isset method returns true even if the option is not set\n        in the DB but still present in cache after a call to `get`.\n        Till we fix that behavior, we need to clean up the cache\n        when we run this test.\n        '
        options.default_store.flush_local_cache()
        options.default_store.delete_cache(options.lookup_key('int_option'))
        options.default_store.delete_cache(options.lookup_key('str_option'))
        options.default_store.delete_cache(options.lookup_key('map_option'))
        options.default_store.delete_cache(options.lookup_key('list_option'))
        options.default_store.delete_cache(options.lookup_key('drifted_option'))
        options.default_store.delete_cache(options.lookup_key('change_channel_option'))
        options.default_store.delete_cache(options.lookup_key('invalid_type'))

    def test_patch(self):
        if False:
            i = 10
            return i + 15

        def assert_not_set() -> None:
            if False:
                while True:
                    i = 10
            self._clean_cache()
            assert not options.isset('int_option')
            assert not options.isset('map_option')
            assert not options.isset('list_option')

        def assert_output(rv):
            if False:
                return 10
            assert rv.exit_code == 2, rv.output
            expected_output = '\n'.join([ConsolePresenter.DRIFT_MSG % 'drifted_option', ConsolePresenter.DB_VALUE % 'drifted_option', '- 1', '- 2', '- 3', '', ConsolePresenter.CHANNEL_UPDATE_MSG % 'change_channel_option', ConsolePresenter.UPDATE_MSG % ('str_option', 'old value', 'new value'), ConsolePresenter.SET_MSG % ('int_option', 40), ConsolePresenter.SET_MSG % ('map_option', {'a': 1, 'b': 2}), ConsolePresenter.SET_MSG % ('list_option', [1, 2])])
            assert expected_output in rv.output
        assert_not_set()
        rv = self.invoke('--dry-run', '--file=tests/sentry/runner/commands/valid_patch.yaml', 'patch')
        assert_output(rv)
        assert_not_set()
        rv = self.invoke('--file=tests/sentry/runner/commands/valid_patch.yaml', 'patch')
        assert_output(rv)
        assert options.get('int_option') == 40
        assert options.get('str_option') == 'new value'
        assert options.get('map_option') == {'a': 1, 'b': 2}
        assert options.get('list_option') == [1, 2]
        assert options.get('drifted_option') == [1, 2, 3]

    def test_stdin(self):
        if False:
            print('Hello World!')
        rv = self.invoke('patch', input=Path('tests/sentry/runner/commands/valid_patch.yaml').read_text())
        assert rv.exit_code == 2
        assert options.get('int_option') == 40
        assert options.get('str_option') == 'new value'
        assert options.get('map_option') == {'a': 1, 'b': 2}
        assert options.get('list_option') == [1, 2]
        assert options.get('drifted_option') == [1, 2, 3]

    def test_sync(self):
        if False:
            i = 10
            return i + 15
        rv = self.invoke('-f', 'tests/sentry/runner/commands/valid_patch.yaml', 'sync')
        assert rv.exit_code == 2, rv.output
        expected_output = '\n'.join([ConsolePresenter.DRIFT_MSG % 'drifted_option', ConsolePresenter.DB_VALUE % 'drifted_option', '- 1', '- 2', '- 3', '', ConsolePresenter.CHANNEL_UPDATE_MSG % 'change_channel_option', ConsolePresenter.UPDATE_MSG % ('str_option', 'old value', 'new value'), ConsolePresenter.SET_MSG % ('int_option', 40), ConsolePresenter.SET_MSG % ('map_option', {'a': 1, 'b': 2}), ConsolePresenter.SET_MSG % ('list_option', [1, 2]), ConsolePresenter.UNSET_MSG % 'to_unset_option'])
        assert expected_output in rv.output
        assert options.get('int_option') == 40
        assert options.get('str_option') == 'new value'
        assert options.get('map_option') == {'a': 1, 'b': 2}
        assert options.get('list_option') == [1, 2]
        assert options.get('drifted_option') == [1, 2, 3]
        assert not options.isset('to_unset_option')

    def test_bad_patch(self):
        if False:
            return 10
        rv = self.invoke('--file=tests/sentry/runner/commands/badpatch.yaml', 'patch')
        assert rv.exit_code == 2, rv.output
        assert ConsolePresenter.SET_MSG % ('int_option', 50) in rv.output
        assert ConsolePresenter.INVALID_TYPE_ERROR % ('invalid_type', "<class 'list'>", 'integer') in rv.output
        assert ConsolePresenter.UNREGISTERED_OPTION_ERROR % 'inexistent_option' in rv.output
        assert not options.isset('readonly_option')
        assert not options.isset('invalid_type')
        assert options.get('int_option') == 50