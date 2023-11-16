import datetime
import locale
from twisted.python import log
from twisted.trial import unittest
from buildbot.test.util import validation
from buildbot.util import UTC

class VerifyDict(unittest.TestCase):

    def doValidationTest(self, validator, good, bad):
        if False:
            for i in range(10):
                print('nop')
        for g in good:
            log.msg(f'expect {repr(g)} to be good')
            msgs = list(validator.validate('g', g))
            self.assertEqual(msgs, [], f'messages for {repr(g)}')
        for b in bad:
            log.msg(f'expect {repr(b)} to be bad')
            msgs = list(validator.validate('b', b))
            self.assertNotEqual(msgs, [], f'no messages for {repr(b)}')
            log.msg('..got messages:')
            for msg in msgs:
                log.msg('  ' + msg)

    def test_IntValidator(self):
        if False:
            i = 10
            return i + 15
        self.doValidationTest(validation.IntValidator(), good=[1, 10 ** 100], bad=[1.0, 'one', '1', None])

    def test_BooleanValidator(self):
        if False:
            print('Hello World!')
        self.doValidationTest(validation.BooleanValidator(), good=[True, False], bad=['yes', 'no', 1, 0, None])

    def test_StringValidator(self):
        if False:
            print('Hello World!')
        self.doValidationTest(validation.StringValidator(), good=['unicode only'], bad=[None, b'bytestring'])

    def test_BinaryValidator(self):
        if False:
            while True:
                i = 10
        self.doValidationTest(validation.BinaryValidator(), good=[b'bytestring'], bad=[None, 'no unicode'])

    def test_DateTimeValidator(self):
        if False:
            for i in range(10):
                print('nop')
        self.doValidationTest(validation.DateTimeValidator(), good=[datetime.datetime(1980, 6, 15, 12, 31, 15, tzinfo=UTC)], bad=[None, 198847493, datetime.datetime(1980, 6, 15, 12, 31, 15)])

    def test_IdentifierValidator(self):
        if False:
            print('Hello World!')
        os_encoding = locale.getpreferredencoding()
        try:
            '☃'.encode(os_encoding)
        except UnicodeEncodeError as e:
            raise unittest.SkipTest(f'Cannot encode weird unicode on this platform with {os_encoding}') from e
        self.doValidationTest(validation.IdentifierValidator(50), good=['linux', 'Linux', 'abc123', 'a' * 50, '☃'], bad=[None, '', b'linux', 'a/b', 'a.b.c.d', 'a-b_c.d9', 'spaces not allowed', 'a' * 51, '123 no initial digits'])

    def test_NoneOk(self):
        if False:
            i = 10
            return i + 15
        self.doValidationTest(validation.NoneOk(validation.BooleanValidator()), good=[True, False, None], bad=[1, 'yes'])

    def test_DictValidator(self):
        if False:
            for i in range(10):
                print('nop')
        self.doValidationTest(validation.DictValidator(a=validation.BooleanValidator(), b=validation.StringValidator(), optionalNames=['b']), good=[{'a': True}, {'a': True, 'b': 'xyz'}], bad=[None, 1, 'hi', {}, {'a': 1}, {'a': 1, 'b': 'xyz'}, {'a': True, 'b': 999}, {'a': True, 'b': 'xyz', 'c': 'extra'}])

    def test_DictValidator_names(self):
        if False:
            return 10
        v = validation.DictValidator(a=validation.BooleanValidator())
        self.assertEqual(list(v.validate('v', {'a': 1})), ["v['a'] (1) is not a boolean"])

    def test_ListValidator(self):
        if False:
            return 10
        self.doValidationTest(validation.ListValidator(validation.BooleanValidator()), good=[[], [True], [False, True]], bad=[None, ['a'], [True, 'a'], 1, 'hi'])

    def test_ListValidator_names(self):
        if False:
            return 10
        v = validation.ListValidator(validation.BooleanValidator())
        self.assertEqual(list(v.validate('v', ['a'])), ["v[0] ('a') is not a boolean"])

    def test_SourcedPropertiesValidator(self):
        if False:
            return 10
        self.doValidationTest(validation.SourcedPropertiesValidator(), good=[{'pname': ('{"a":"b"}', 'test')}], bad=[None, 1, b'hi', {'pname': {b'a': b'b'}}, {'pname': ({b'a': b'b'}, 'test')}, {'pname': ({b'a': b'b'}, 'test')}, {'pname': (self, 'test')}])

    def test_MessageValidator(self):
        if False:
            print('Hello World!')
        self.doValidationTest(validation.MessageValidator(events=[b'started', b'stopped'], messageValidator=validation.DictValidator(a=validation.BooleanValidator(), xid=validation.IntValidator(), yid=validation.IntValidator())), good=[(('thing', '1', '2', 'started'), {'xid': 1, 'yid': 2, 'a': True})], bad=[('thing', {}), (('thing', '1', '2', 'exploded'), {'xid': 1, 'yid': 2, 'a': True}), (('thing', 1, 2, 'started'), {'xid': 1, 'yid': 2, 'a': True}), (('thing', '1', '2', 'started'), {'xid': 1, 'a': True}), (('thing', '1', '2', 'started'), {'xid': 1, 'yid': 2, 'a': 'x'})])

    def test_Selector(self):
        if False:
            for i in range(10):
                print('nop')
        sel = validation.Selector()
        sel.add(lambda x: x == 'int', validation.IntValidator())
        sel.add(lambda x: x == 'str', validation.StringValidator())
        self.doValidationTest(sel, good=[('int', 1), ('str', 'hi')], bad=[('int', 'hi'), ('str', 1), ('float', 1.0)])