"""
GObject Introspection reactor tests; i.e. `gireactor` module for gio/glib/gtk
integration.
"""
from __future__ import annotations
from unittest import skipIf
try:
    from gi.repository import Gio
except ImportError:
    giImported = False
    gtkVersion = None
else:
    giImported = True
    from os import environ
    from gi import get_required_version, require_version
    from twisted.internet import gireactor

    def requireEach(someVersion: str) -> str:
        if False:
            print('Hello World!')
        try:
            require_version('Gtk', someVersion)
        except ValueError as ve:
            return str(ve)
        else:
            return ''
    errorMessage = ', '.join((requireEach(version) for version in environ.get('TWISTED_TEST_GTK_VERSION', '4.0,3.0').split(',')))
    actualVersion = get_required_version('Gtk')
    gtkVersion = actualVersion if actualVersion is not None else errorMessage
from twisted.internet.error import ReactorAlreadyRunning
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.trial.unittest import SkipTest, TestCase
if not giImported:
    skip = 'GObject Introspection `gi` module not importable'
noGtkSkip = gtkVersion is None or gtkVersion not in ('3.0', '4.0')
noGtkMessage = f'Unknown GTK version: {repr(gtkVersion)}'
if not noGtkSkip:
    from gi.repository import Gtk

class GApplicationRegistrationTests(ReactorBuilder, TestCase):
    """
    GtkApplication and GApplication are supported by
    L{twisted.internet.gtk3reactor} and L{twisted.internet.gireactor}.

    We inherit from L{ReactorBuilder} in order to use some of its
    reactor-running infrastructure, but don't need its test-creation
    functionality.
    """

    def runReactor(self, app: Gio.Application, reactor: gireactor.GIReactor) -> None:
        if False:
            while True:
                i = 10
        '\n        Register the app, run the reactor, make sure app was activated, and\n        that reactor was running, and that reactor can be stopped.\n        '
        if not hasattr(app, 'quit'):
            raise SkipTest('Version of PyGObject is too old.')
        result = []

        def stop() -> None:
            if False:
                for i in range(10):
                    print('nop')
            result.append('stopped')
            reactor.stop()

        def activate(widget: object) -> None:
            if False:
                i = 10
                return i + 15
            result.append('activated')
            reactor.callLater(0, stop)
        app.connect('activate', activate)
        app.hold()
        reactor.registerGApplication(app)
        ReactorBuilder.runReactor(self, reactor)
        self.assertEqual(result, ['activated', 'stopped'])

    def test_gApplicationActivate(self) -> None:
        if False:
            return 10
        '\n        L{Gio.Application} instances can be registered with a gireactor.\n        '
        self.reactorFactory = lambda : gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.runReactor(app, reactor)

    @skipIf(noGtkSkip, noGtkMessage)
    def test_gtkAliases(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{twisted.internet.gtk3reactor} is now just a set of compatibility\n        aliases for L{twisted.internet.GIReactor}.\n        '
        from twisted.internet.gtk3reactor import Gtk3Reactor, PortableGtk3Reactor, install
        self.assertIs(Gtk3Reactor, gireactor.GIReactor)
        self.assertIs(PortableGtk3Reactor, gireactor.PortableGIReactor)
        self.assertIs(install, gireactor.install)
        warnings = self.flushWarnings()
        self.assertEqual(len(warnings), 1)
        self.assertIn('twisted.internet.gtk3reactor was deprecated', warnings[0]['message'])

    @skipIf(noGtkSkip, noGtkMessage)
    def test_gtkApplicationActivate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Gtk.Application} instances can be registered with a gtk3reactor.\n        '
        self.reactorFactory = gireactor.GIReactor
        reactor = self.buildReactor()
        app = Gtk.Application(application_id='com.twistedmatrix.trial.gtk3reactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.runReactor(app, reactor)

    def test_portable(self) -> None:
        if False:
            print('Hello World!')
        "\n        L{gireactor.PortableGIReactor} doesn't support application\n        registration at this time.\n        "
        self.reactorFactory = gireactor.PortableGIReactor
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.assertRaises(NotImplementedError, reactor.registerGApplication, app)

    def test_noQuit(self) -> None:
        if False:
            print('Hello World!')
        "\n        Older versions of PyGObject lack C{Application.quit}, and so won't\n        allow registration.\n        "
        self.reactorFactory = lambda : gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = object()
        exc = self.assertRaises(RuntimeError, reactor.registerGApplication, app)
        self.assertTrue(exc.args[0].startswith('Application registration is not'))

    def test_cantRegisterAfterRun(self) -> None:
        if False:
            while True:
                i = 10
        '\n        It is not possible to register a C{Application} after the reactor has\n        already started.\n        '
        self.reactorFactory = lambda : gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)

        def tryRegister() -> None:
            if False:
                return 10
            exc = self.assertRaises(ReactorAlreadyRunning, reactor.registerGApplication, app)
            self.assertEqual(exc.args[0], "Can't register application after reactor was started.")
            reactor.stop()
        reactor.callLater(0, tryRegister)
        ReactorBuilder.runReactor(self, reactor)

    def test_cantRegisterTwice(self) -> None:
        if False:
            return 10
        '\n        It is not possible to register more than one C{Application}.\n        '
        self.reactorFactory = lambda : gireactor.GIReactor(useGtk=False)
        reactor = self.buildReactor()
        app = Gio.Application(application_id='com.twistedmatrix.trial.gireactor', flags=Gio.ApplicationFlags.FLAGS_NONE)
        reactor.registerGApplication(app)
        app2 = Gio.Application(application_id='com.twistedmatrix.trial.gireactor2', flags=Gio.ApplicationFlags.FLAGS_NONE)
        exc = self.assertRaises(RuntimeError, reactor.registerGApplication, app2)
        self.assertEqual(exc.args[0], "Can't register more than one application instance.")