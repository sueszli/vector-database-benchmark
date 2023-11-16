"""
Helpers for URI and method injection tests.

@see: U{CVE-2019-12387}
"""
import string
UNPRINTABLE_ASCII = frozenset(range(0, 128)) - frozenset(bytearray(string.printable, 'ascii'))
NONASCII = frozenset(range(128, 256))

class MethodInjectionTestsMixin:
    """
    A mixin that runs HTTP method injection tests.  Define
    L{MethodInjectionTestsMixin.attemptRequestWithMaliciousMethod} in
    a L{twisted.trial.unittest.SynchronousTestCase} subclass to test
    how HTTP client code behaves when presented with malicious HTTP
    methods.

    @see: U{CVE-2019-12387}
    """

    def attemptRequestWithMaliciousMethod(self, method):
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempt to send a request with the given method.  This should\n        synchronously raise a L{ValueError} if either is invalid.\n\n        @param method: the method (e.g. C{GET\x00})\n\n        @param uri: the URI\n\n        @type method:\n        '
        raise NotImplementedError()

    def test_methodWithCLRFRejected(self):
        if False:
            while True:
                i = 10
        '\n        Issuing a request with a method that contains a carriage\n        return and line feed fails with a L{ValueError}.\n        '
        with self.assertRaises(ValueError) as cm:
            method = b'GET\r\nX-Injected-Header: value'
            self.attemptRequestWithMaliciousMethod(method)
        self.assertRegex(str(cm.exception), '^Invalid method')

    def test_methodWithUnprintableASCIIRejected(self):
        if False:
            i = 10
            return i + 15
        '\n        Issuing a request with a method that contains unprintable\n        ASCII characters fails with a L{ValueError}.\n        '
        for c in UNPRINTABLE_ASCII:
            method = b'GET%s' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousMethod(method)
            self.assertRegex(str(cm.exception), '^Invalid method')

    def test_methodWithNonASCIIRejected(self):
        if False:
            return 10
        '\n        Issuing a request with a method that contains non-ASCII\n        characters fails with a L{ValueError}.\n        '
        for c in NONASCII:
            method = b'GET%s' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousMethod(method)
            self.assertRegex(str(cm.exception), '^Invalid method')

class URIInjectionTestsMixin:
    """
    A mixin that runs HTTP URI injection tests.  Define
    L{MethodInjectionTestsMixin.attemptRequestWithMaliciousURI} in a
    L{twisted.trial.unittest.SynchronousTestCase} subclass to test how
    HTTP client code behaves when presented with malicious HTTP
    URIs.
    """

    def attemptRequestWithMaliciousURI(self, method):
        if False:
            for i in range(10):
                print('nop')
        '\n        Attempt to send a request with the given URI.  This should\n        synchronously raise a L{ValueError} if either is invalid.\n\n        @param uri: the URI.\n\n        @type method:\n        '
        raise NotImplementedError()

    def test_hostWithCRLFRejected(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Issuing a request with a URI whose host contains a carriage\n        return and line feed fails with a L{ValueError}.\n        '
        with self.assertRaises(ValueError) as cm:
            uri = b'http://twisted\r\n.invalid/path'
            self.attemptRequestWithMaliciousURI(uri)
        self.assertRegex(str(cm.exception), '^Invalid URI')

    def test_hostWithWithUnprintableASCIIRejected(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Issuing a request with a URI whose host contains unprintable\n        ASCII characters fails with a L{ValueError}.\n        '
        for c in UNPRINTABLE_ASCII:
            uri = b'http://twisted%s.invalid/OK' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousURI(uri)
            self.assertRegex(str(cm.exception), '^Invalid URI')

    def test_hostWithNonASCIIRejected(self):
        if False:
            while True:
                i = 10
        '\n        Issuing a request with a URI whose host contains non-ASCII\n        characters fails with a L{ValueError}.\n        '
        for c in NONASCII:
            uri = b'http://twisted%s.invalid/OK' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousURI(uri)
            self.assertRegex(str(cm.exception), '^Invalid URI')

    def test_pathWithCRLFRejected(self):
        if False:
            while True:
                i = 10
        '\n        Issuing a request with a URI whose path contains a carriage\n        return and line feed fails with a L{ValueError}.\n        '
        with self.assertRaises(ValueError) as cm:
            uri = b'http://twisted.invalid/\r\npath'
            self.attemptRequestWithMaliciousURI(uri)
        self.assertRegex(str(cm.exception), '^Invalid URI')

    def test_pathWithWithUnprintableASCIIRejected(self):
        if False:
            while True:
                i = 10
        '\n        Issuing a request with a URI whose path contains unprintable\n        ASCII characters fails with a L{ValueError}.\n        '
        for c in UNPRINTABLE_ASCII:
            uri = b'http://twisted.invalid/OK%s' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousURI(uri)
            self.assertRegex(str(cm.exception), '^Invalid URI')

    def test_pathWithNonASCIIRejected(self):
        if False:
            i = 10
            return i + 15
        '\n        Issuing a request with a URI whose path contains non-ASCII\n        characters fails with a L{ValueError}.\n        '
        for c in NONASCII:
            uri = b'http://twisted.invalid/OK%s' % (bytearray([c]),)
            with self.assertRaises(ValueError) as cm:
                self.attemptRequestWithMaliciousURI(uri)
            self.assertRegex(str(cm.exception), '^Invalid URI')