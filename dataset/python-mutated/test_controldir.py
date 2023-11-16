"""Tests for the ControlDir facility.

For interface contract tests, see tests/per_control_dir.
"""
from bzrlib import controldir, errors, tests, ui
from bzrlib.tests.scenarios import load_tests_apply_scenarios
load_tests = load_tests_apply_scenarios

class SampleComponentFormat(controldir.ControlComponentFormat):

    def get_format_string(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Example component format.'

class SampleExtraComponentFormat(controldir.ControlComponentFormat):
    """Extra format, no format string."""

class TestMetaComponentFormatRegistry(tests.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestMetaComponentFormatRegistry, self).setUp()
        self.registry = controldir.ControlComponentFormatRegistry()

    def test_register_unregister_format(self):
        if False:
            for i in range(10):
                print('nop')
        format = SampleComponentFormat()
        self.registry.register(format)
        self.assertEqual(format, self.registry.get('Example component format.'))
        self.registry.remove(format)
        self.assertRaises(KeyError, self.registry.get, 'Example component format.')

    def test_get_all(self):
        if False:
            i = 10
            return i + 15
        format = SampleComponentFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register(format)
        self.assertEqual([format], self.registry._get_all())

    def test_get_all_modules(self):
        if False:
            i = 10
            return i + 15
        format = SampleComponentFormat()
        self.assertEqual(set(), self.registry._get_all_modules())
        self.registry.register(format)
        self.assertEqual(set(['bzrlib.tests.test_controldir']), self.registry._get_all_modules())

    def test_register_extra(self):
        if False:
            print('Hello World!')
        format = SampleExtraComponentFormat()
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra(format)
        self.assertEqual([format], self.registry._get_all())

    def test_register_extra_lazy(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual([], self.registry._get_all())
        self.registry.register_extra_lazy('bzrlib.tests.test_controldir', 'SampleExtraComponentFormat')
        formats = self.registry._get_all()
        self.assertEqual(1, len(formats))
        self.assertIsInstance(formats[0], SampleExtraComponentFormat)

class TestControlDirFormatDeprecated(tests.TestCaseWithTransport):
    """Tests for removed registration method in the ControlDirFormat facility."""

    def test_register_format(self):
        if False:
            print('Hello World!')
        self.assertRaises(errors.BzrError, controldir.ControlDirFormat.register_format, object())

class TestProber(tests.TestCaseWithTransport):
    """Per-prober tests."""
    scenarios = [(prober_cls.__name__, {'prober_cls': prober_cls}) for prober_cls in controldir.ControlDirFormat._probers]

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestProber, self).setUp()
        self.prober = self.prober_cls()

    def test_probe_transport_empty(self):
        if False:
            i = 10
            return i + 15
        transport = self.get_transport('.')
        self.assertRaises(errors.NotBranchError, self.prober.probe_transport, transport)

    def test_known_formats(self):
        if False:
            print('Hello World!')
        known_formats = self.prober_cls.known_formats()
        self.assertIsInstance(known_formats, set)
        for format in known_formats:
            self.assertIsInstance(format, controldir.ControlDirFormat, repr(format))

class NotBzrDir(controldir.ControlDir):
    """A non .bzr based control directory."""

    def __init__(self, transport, format):
        if False:
            for i in range(10):
                print('nop')
        self._format = format
        self.root_transport = transport
        self.transport = transport.clone('.not')

class NotBzrDirFormat(controldir.ControlDirFormat):
    """A test class representing any non-.bzr based disk format."""

    def initialize_on_transport(self, transport):
        if False:
            return 10
        'Initialize a new .not dir in the base directory of a Transport.'
        transport.mkdir('.not')
        return self.open(transport)

    def open(self, transport):
        if False:
            for i in range(10):
                print('nop')
        'Open this directory.'
        return NotBzrDir(transport, self)

class NotBzrDirProber(controldir.Prober):

    def probe_transport(self, transport):
        if False:
            while True:
                i = 10
        "Our format is present if the transport ends in '.not/'."
        if transport.has('.not'):
            return NotBzrDirFormat()

    @classmethod
    def known_formats(cls):
        if False:
            i = 10
            return i + 15
        return set([NotBzrDirFormat()])

class TestNotBzrDir(tests.TestCaseWithTransport):
    """Tests for using the controldir api with a non .bzr based disk format.

    If/when one of these is in the core, we can let the implementation tests
    verify this works.
    """

    def test_create_and_find_format(self):
        if False:
            while True:
                i = 10
        format = NotBzrDirFormat()
        dir = format.initialize(self.get_url())
        self.assertIsInstance(dir, NotBzrDir)
        controldir.ControlDirFormat.register_prober(NotBzrDirProber)
        try:
            found = controldir.ControlDirFormat.find_format(self.get_transport())
            self.assertIsInstance(found, NotBzrDirFormat)
        finally:
            controldir.ControlDirFormat.unregister_prober(NotBzrDirProber)

    def test_included_in_known_formats(self):
        if False:
            for i in range(10):
                print('nop')
        controldir.ControlDirFormat.register_prober(NotBzrDirProber)
        self.addCleanup(controldir.ControlDirFormat.unregister_prober, NotBzrDirProber)
        formats = controldir.ControlDirFormat.known_formats()
        self.assertIsInstance(formats, set)
        for format in formats:
            if isinstance(format, NotBzrDirFormat):
                break
        else:
            self.fail('No NotBzrDirFormat in %s' % formats)

class UnsupportedControlComponentFormat(controldir.ControlComponentFormat):

    def is_supported(self):
        if False:
            return 10
        return False

class OldControlComponentFormat(controldir.ControlComponentFormat):

    def get_format_description(self):
        if False:
            i = 10
            return i + 15
        return 'An old format that is slow'
    upgrade_recommended = True

class DefaultControlComponentFormatTests(tests.TestCase):
    """Tests for default ControlComponentFormat implementation."""

    def test_check_support_status_unsupported(self):
        if False:
            print('Hello World!')
        self.assertRaises(errors.UnsupportedFormatError, UnsupportedControlComponentFormat().check_support_status, allow_unsupported=False)
        UnsupportedControlComponentFormat().check_support_status(allow_unsupported=True)

    def test_check_support_status_supported(self):
        if False:
            while True:
                i = 10
        controldir.ControlComponentFormat().check_support_status(allow_unsupported=False)
        controldir.ControlComponentFormat().check_support_status(allow_unsupported=True)

    def test_recommend_upgrade_current_format(self):
        if False:
            return 10
        stderr = tests.StringIOWrapper()
        ui.ui_factory = tests.TestUIFactory(stderr=stderr)
        format = controldir.ControlComponentFormat()
        format.check_support_status(allow_unsupported=False, recommend_upgrade=True)
        self.assertEqual('', stderr.getvalue())

    def test_recommend_upgrade_old_format(self):
        if False:
            i = 10
            return i + 15
        stderr = tests.StringIOWrapper()
        ui.ui_factory = tests.TestUIFactory(stderr=stderr)
        format = OldControlComponentFormat()
        format.check_support_status(allow_unsupported=False, recommend_upgrade=False)
        self.assertEqual('', stderr.getvalue())
        format.check_support_status(allow_unsupported=False, recommend_upgrade=True, basedir='apath')
        self.assertEqual('An old format that is slow is deprecated and a better format is available.\nIt is recommended that you upgrade by running the command\n  bzr upgrade apath\n', stderr.getvalue())