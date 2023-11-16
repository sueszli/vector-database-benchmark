from bzrlib import config, errors, osutils
from bzrlib.tests import TestCase, TestCaseWithTransport
from bzrlib.tests.features import ModuleAvailableFeature
launchpadlib_feature = ModuleAvailableFeature('launchpadlib')

class TestDependencyManagement(TestCase):
    """Tests for managing the dependency on launchpadlib."""
    _test_needs_features = [launchpadlib_feature]

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestDependencyManagement, self).setUp()
        from bzrlib.plugins.launchpad import lp_api
        self.lp_api = lp_api

    def patch(self, obj, name, value):
        if False:
            while True:
                i = 10
        "Temporarily set the 'name' attribute of 'obj' to 'value'."
        self.overrideAttr(obj, name, value)

    def test_get_launchpadlib_version(self):
        if False:
            while True:
                i = 10
        version_info = self.lp_api.parse_launchpadlib_version('1.5.1')
        self.assertEqual((1, 5, 1), version_info)

    def test_supported_launchpadlib_version(self):
        if False:
            for i in range(10):
                print('nop')
        launchpadlib = launchpadlib_feature.module
        self.patch(launchpadlib, '__version__', '1.5.1')
        self.lp_api.MINIMUM_LAUNCHPADLIB_VERSION = (1, 5, 1)
        self.lp_api.check_launchpadlib_compatibility()

    def test_unsupported_launchpadlib_version(self):
        if False:
            while True:
                i = 10
        launchpadlib = launchpadlib_feature.module
        self.patch(launchpadlib, '__version__', '1.5.0')
        self.lp_api.MINIMUM_LAUNCHPADLIB_VERSION = (1, 5, 1)
        self.assertRaises(errors.IncompatibleAPI, self.lp_api.check_launchpadlib_compatibility)

class TestCacheDirectory(TestCase):
    """Tests for get_cache_directory."""
    _test_needs_features = [launchpadlib_feature]

    def test_get_cache_directory(self):
        if False:
            while True:
                i = 10
        from bzrlib.plugins.launchpad import lp_api
        expected_path = osutils.pathjoin(config.config_dir(), 'launchpad')
        self.assertEqual(expected_path, lp_api.get_cache_directory())

class TestLaunchpadMirror(TestCaseWithTransport):
    """Tests for the 'bzr lp-mirror' command."""
    _test_needs_features = [launchpadlib_feature]

    def test_command_exists(self):
        if False:
            while True:
                i = 10
        (out, err) = self.run_bzr(['launchpad-mirror', '--help'], retcode=0)
        self.assertEqual('', err)

    def test_alias_exists(self):
        if False:
            for i in range(10):
                print('nop')
        (out, err) = self.run_bzr(['lp-mirror', '--help'], retcode=0)
        self.assertEqual('', err)