"""
Tests for twisted.internet.glibbase.
"""
import sys
from twisted.internet._glibbase import ensureNotImported
from twisted.trial.unittest import TestCase

class EnsureNotImportedTests(TestCase):
    """
    L{ensureNotImported} protects against unwanted past and future imports.
    """

    def test_ensureWhenNotImported(self):
        if False:
            i = 10
            return i + 15
        '\n        If the specified modules have never been imported, and import\n        prevention is requested, L{ensureNotImported} makes sure they will not\n        be imported in the future.\n        '
        modules = {}
        self.patch(sys, 'modules', modules)
        ensureNotImported(['m1', 'm2'], 'A message.', preventImports=['m1', 'm2', 'm3'])
        self.assertEqual(modules, {'m1': None, 'm2': None, 'm3': None})

    def test_ensureWhenNotImportedDontPrevent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the specified modules have never been imported, and import\n        prevention is not requested, L{ensureNotImported} has no effect.\n        '
        modules = {}
        self.patch(sys, 'modules', modules)
        ensureNotImported(['m1', 'm2'], 'A message.')
        self.assertEqual(modules, {})

    def test_ensureWhenFailedToImport(self):
        if False:
            print('Hello World!')
        '\n        If the specified modules have been set to L{None} in C{sys.modules},\n        L{ensureNotImported} does not complain.\n        '
        modules = {'m2': None}
        self.patch(sys, 'modules', modules)
        ensureNotImported(['m1', 'm2'], 'A message.', preventImports=['m1', 'm2'])
        self.assertEqual(modules, {'m1': None, 'm2': None})

    def test_ensureFailsWhenImported(self):
        if False:
            return 10
        '\n        If one of the specified modules has been previously imported,\n        L{ensureNotImported} raises an exception.\n        '
        module = object()
        modules = {'m2': module}
        self.patch(sys, 'modules', modules)
        e = self.assertRaises(ImportError, ensureNotImported, ['m1', 'm2'], 'A message.', preventImports=['m1', 'm2'])
        self.assertEqual(modules, {'m2': module})
        self.assertEqual(e.args, ('A message.',))
try:
    from twisted.internet import gireactor as _gireactor
except ImportError:
    gireactor = None
else:
    gireactor = _gireactor
missingGlibReactor = None
if gireactor is None:
    missingGlibReactor = 'gi reactor not available'

class GlibReactorBaseTests(TestCase):
    """
    Tests for the private C{twisted.internet._glibbase.GlibReactorBase}
    done via the public C{twisted.internet.gireactor.PortableGIReactor}
    """
    skip = missingGlibReactor

    def test_simulate(self):
        if False:
            while True:
                i = 10
        '\n        C{simulate} can be called without raising any errors when there are\n        no delayed calls for the reactor and hence there is no defined sleep\n        period.\n        '
        sut = gireactor.PortableGIReactor(useGtk=False)
        self.assertIs(None, sut.timeout())
        sut.simulate()