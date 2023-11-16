from allennlp.commands import Subcommand
from allennlp.common.plugins import discover_plugins, import_plugins
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import pushd

class TestPlugins(AllenNlpTestCase):

    def setup_method(self):
        if False:
            i = 10
            return i + 15
        super().setup_method()
        self.plugins_root = self.FIXTURES_ROOT / 'plugins'

    def test_no_plugins(self):
        if False:
            return 10
        available_plugins = set(discover_plugins())
        assert available_plugins == set()

    def test_file_plugin(self):
        if False:
            for i in range(10):
                print('nop')
        available_plugins = set(discover_plugins())
        assert available_plugins == set()
        with pushd(self.plugins_root):
            available_plugins = set(discover_plugins())
            assert available_plugins == {'d'}
            import_plugins()
            subcommands_available = Subcommand.list_available()
            assert 'd' in subcommands_available