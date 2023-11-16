import codecs
import html.entities
import itertools
import sys
import unicodedata
import unittest

class PosReturn:

    def __init__(self):
        if False:
            return 10
        self.pos = 0

    def handle(self, exc):
        if False:
            for i in range(10):
                print('nop')
        oldpos = self.pos
        realpos = oldpos
        if realpos < 0:
            realpos = len(exc.object) + realpos
        if realpos <= exc.start:
            self.pos = len(exc.object)
        return ('<?>', oldpos)

class RepeatedPosReturn:

    def __init__(self, repl='<?>'):
        if False:
            print('Hello World!')
        self.repl = repl
        self.pos = 0
        self.count = 0

    def handle(self, exc):
        if False:
            for i in range(10):
                print('nop')
        if self.count > 0:
            self.count -= 1
            return (self.repl, self.pos)
        return (self.repl, exc.end)

class BadStartUnicodeEncodeError(UnicodeEncodeError):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        UnicodeEncodeError.__init__(self, 'ascii', '', 0, 1, 'bad')
        self.start = []

class BadObjectUnicodeEncodeError(UnicodeEncodeError):

    def __init__(self):
        if False:
            while True:
                i = 10
        UnicodeEncodeError.__init__(self, 'ascii', '', 0, 1, 'bad')
        self.object = []

class NoEndUnicodeDecodeError(UnicodeDecodeError):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        UnicodeDecodeError.__init__(self, 'ascii', bytearray(b''), 0, 1, 'bad')
        del self.end

class BadObjectUnicodeDecodeError(UnicodeDecodeError):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        UnicodeDecodeError.__init__(self, 'ascii', bytearray(b''), 0, 1, 'bad')
        self.object = []

class NoStartUnicodeTranslateError(UnicodeTranslateError):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        UnicodeTranslateError.__init__(self, '', 0, 1, 'bad')
        del self.start

class NoEndUnicodeTranslateError(UnicodeTranslateError):

    def __init__(self):
        if False:
            return 10
        UnicodeTranslateError.__init__(self, '', 0, 1, 'bad')
        del self.end

class NoObjectUnicodeTranslateError(UnicodeTranslateError):

    def __init__(self):
        if False:
            return 10
        UnicodeTranslateError.__init__(self, '', 0, 1, 'bad')
        del self.object

class CodecCallbackTest(unittest.TestCase):

    def test_xmlcharrefreplace(self):
        if False:
            i = 10
            return i + 15
        s = 'スパモ änd eggs'
        self.assertEqual(s.encode('ascii', 'xmlcharrefreplace'), b'&#12473;&#12497;&#12514; &#228;nd eggs')
        self.assertEqual(s.encode('latin-1', 'xmlcharrefreplace'), b'&#12473;&#12497;&#12514; \xe4nd eggs')

    def test_xmlcharnamereplace(self):
        if False:
            while True:
                i = 10

        def xmlcharnamereplace(exc):
            if False:
                print('Hello World!')
            if not isinstance(exc, UnicodeEncodeError):
                raise TypeError("don't know how to handle %r" % exc)
            l = []
            for c in exc.object[exc.start:exc.end]:
                try:
                    l.append('&%s;' % html.entities.codepoint2name[ord(c)])
                except KeyError:
                    l.append('&#%d;' % ord(c))
            return (''.join(l), exc.end)
        codecs.register_error('test.xmlcharnamereplace', xmlcharnamereplace)
        sin = '«ℜ» = 〈ሴ€〉'
        sout = b'&laquo;&real;&raquo; = &lang;&#4660;&euro;&rang;'
        self.assertEqual(sin.encode('ascii', 'test.xmlcharnamereplace'), sout)
        sout = b'\xab&real;\xbb = &lang;&#4660;&euro;&rang;'
        self.assertEqual(sin.encode('latin-1', 'test.xmlcharnamereplace'), sout)
        sout = b'\xab&real;\xbb = &lang;&#4660;\xa4&rang;'
        self.assertEqual(sin.encode('iso-8859-15', 'test.xmlcharnamereplace'), sout)

    def test_uninamereplace(self):
        if False:
            for i in range(10):
                print('nop')

        def uninamereplace(exc):
            if False:
                i = 10
                return i + 15
            if not isinstance(exc, UnicodeEncodeError):
                raise TypeError("don't know how to handle %r" % exc)
            l = []
            for c in exc.object[exc.start:exc.end]:
                l.append(unicodedata.name(c, '0x%x' % ord(c)))
            return ('\x1b[1m%s\x1b[0m' % ', '.join(l), exc.end)
        codecs.register_error('test.uninamereplace', uninamereplace)
        sin = '¬ሴ€耀'
        sout = b'\x1b[1mNOT SIGN, ETHIOPIC SYLLABLE SEE, EURO SIGN, CJK UNIFIED IDEOGRAPH-8000\x1b[0m'
        self.assertEqual(sin.encode('ascii', 'test.uninamereplace'), sout)
        sout = b'\xac\x1b[1mETHIOPIC SYLLABLE SEE, EURO SIGN, CJK UNIFIED IDEOGRAPH-8000\x1b[0m'
        self.assertEqual(sin.encode('latin-1', 'test.uninamereplace'), sout)
        sout = b'\xac\x1b[1mETHIOPIC SYLLABLE SEE\x1b[0m\xa4\x1b[1mCJK UNIFIED IDEOGRAPH-8000\x1b[0m'
        self.assertEqual(sin.encode('iso-8859-15', 'test.uninamereplace'), sout)

    def test_backslashescape(self):
        if False:
            while True:
                i = 10
        sin = 'a¬ሴ€耀\U0010ffff'
        sout = b'a\\xac\\u1234\\u20ac\\u8000\\U0010ffff'
        self.assertEqual(sin.encode('ascii', 'backslashreplace'), sout)
        sout = b'a\xac\\u1234\\u20ac\\u8000\\U0010ffff'
        self.assertEqual(sin.encode('latin-1', 'backslashreplace'), sout)
        sout = b'a\xac\\u1234\xa4\\u8000\\U0010ffff'
        self.assertEqual(sin.encode('iso-8859-15', 'backslashreplace'), sout)

    def test_nameescape(self):
        if False:
            print('Hello World!')
        sin = 'a¬ሴ€耀\U0010ffff'
        sout = b'a\\N{NOT SIGN}\\N{ETHIOPIC SYLLABLE SEE}\\N{EURO SIGN}\\N{CJK UNIFIED IDEOGRAPH-8000}\\U0010ffff'
        self.assertEqual(sin.encode('ascii', 'namereplace'), sout)
        sout = b'a\xac\\N{ETHIOPIC SYLLABLE SEE}\\N{EURO SIGN}\\N{CJK UNIFIED IDEOGRAPH-8000}\\U0010ffff'
        self.assertEqual(sin.encode('latin-1', 'namereplace'), sout)
        sout = b'a\xac\\N{ETHIOPIC SYLLABLE SEE}\xa4\\N{CJK UNIFIED IDEOGRAPH-8000}\\U0010ffff'
        self.assertEqual(sin.encode('iso-8859-15', 'namereplace'), sout)

    def test_decoding_callbacks(self):
        if False:
            print('Hello World!')

        def relaxedutf8(exc):
            if False:
                return 10
            if not isinstance(exc, UnicodeDecodeError):
                raise TypeError("don't know how to handle %r" % exc)
            if exc.object[exc.start:exc.start + 2] == b'\xc0\x80':
                return ('\x00', exc.start + 2)
            else:
                raise exc
        codecs.register_error('test.relaxedutf8', relaxedutf8)
        sin = b'a\x00b\xc0\x80c\xc3\xbc\xc0\x80\xc0\x80'
        sout = 'a\x00b\x00cü\x00\x00'
        self.assertEqual(sin.decode('utf-8', 'test.relaxedutf8'), sout)
        sin = b'\xc0\x80\xc0\x81'
        self.assertRaises(UnicodeDecodeError, sin.decode, 'utf-8', 'test.relaxedutf8')

    def test_charmapencode(self):
        if False:
            for i in range(10):
                print('nop')
        charmap = dict(((ord(c), bytes(2 * c.upper(), 'ascii')) for c in 'abcdefgh'))
        sin = 'abc'
        sout = b'AABBCC'
        self.assertEqual(codecs.charmap_encode(sin, 'strict', charmap)[0], sout)
        sin = 'abcA'
        self.assertRaises(UnicodeError, codecs.charmap_encode, sin, 'strict', charmap)
        charmap[ord('?')] = b'XYZ'
        sin = 'abcDEF'
        sout = b'AABBCCXYZXYZXYZ'
        self.assertEqual(codecs.charmap_encode(sin, 'replace', charmap)[0], sout)
        charmap[ord('?')] = 'XYZ'
        self.assertRaises(TypeError, codecs.charmap_encode, sin, 'replace', charmap)

    def test_callbacks(self):
        if False:
            i = 10
            return i + 15

        def handler1(exc):
            if False:
                i = 10
                return i + 15
            r = range(exc.start, exc.end)
            if isinstance(exc, UnicodeEncodeError):
                l = ['<%d>' % ord(exc.object[pos]) for pos in r]
            elif isinstance(exc, UnicodeDecodeError):
                l = ['<%d>' % exc.object[pos] for pos in r]
            else:
                raise TypeError("don't know how to handle %r" % exc)
            return ('[%s]' % ''.join(l), exc.end)
        codecs.register_error('test.handler1', handler1)

        def handler2(exc):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(exc, UnicodeDecodeError):
                raise TypeError("don't know how to handle %r" % exc)
            l = ['<%d>' % exc.object[pos] for pos in range(exc.start, exc.end)]
            return ('[%s]' % ''.join(l), exc.end + 1)
        codecs.register_error('test.handler2', handler2)
        s = b'\x00\x81\x7f\x80\xff'
        self.assertEqual(s.decode('ascii', 'test.handler1'), '\x00[<129>]\x7f[<128>][<255>]')
        self.assertEqual(s.decode('ascii', 'test.handler2'), '\x00[<129>][<128>]')
        self.assertEqual(b'\\u3042\\u3xxx'.decode('unicode-escape', 'test.handler1'), 'あ[<92><117><51>]xxx')
        self.assertEqual(b'\\u3042\\u3xx'.decode('unicode-escape', 'test.handler1'), 'あ[<92><117><51>]xx')
        self.assertEqual(codecs.charmap_decode(b'abc', 'test.handler1', {ord('a'): 'z'})[0], 'z[<98>][<99>]')
        self.assertEqual('güßrk'.encode('ascii', 'test.handler1'), b'g[<252><223>]rk')
        self.assertEqual('güß'.encode('ascii', 'test.handler1'), b'g[<252><223>]')

    def test_longstrings(self):
        if False:
            return 10
        errors = ['strict', 'ignore', 'replace', 'xmlcharrefreplace', 'backslashreplace', 'namereplace']
        for err in errors:
            codecs.register_error('test.' + err, codecs.lookup_error(err))
        l = 1000
        errors += ['test.' + err for err in errors]
        for uni in [s * l for s in ('x', 'あ', 'aä')]:
            for enc in ('ascii', 'latin-1', 'iso-8859-1', 'iso-8859-15', 'utf-8', 'utf-7', 'utf-16', 'utf-32'):
                for err in errors:
                    try:
                        uni.encode(enc, err)
                    except UnicodeError:
                        pass

    def check_exceptionobjectargs(self, exctype, args, msg):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, exctype, *args[:-1])
        self.assertRaises(TypeError, exctype, *args + ['too much'])
        wrongargs = ['spam', b'eggs', b'spam', 42, 1.0, None]
        for i in range(len(args)):
            for wrongarg in wrongargs:
                if type(wrongarg) is type(args[i]):
                    continue
                callargs = []
                for j in range(len(args)):
                    if i == j:
                        callargs.append(wrongarg)
                    else:
                        callargs.append(args[i])
                self.assertRaises(TypeError, exctype, *callargs)
        exc = exctype(*args)
        self.assertEqual(str(exc), msg)

    def test_unicodeencodeerror(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_exceptionobjectargs(UnicodeEncodeError, ['ascii', 'gürk', 1, 2, 'ouch'], "'ascii' codec can't encode character '\\xfc' in position 1: ouch")
        self.check_exceptionobjectargs(UnicodeEncodeError, ['ascii', 'gürk', 1, 4, 'ouch'], "'ascii' codec can't encode characters in position 1-3: ouch")
        self.check_exceptionobjectargs(UnicodeEncodeError, ['ascii', 'üx', 0, 1, 'ouch'], "'ascii' codec can't encode character '\\xfc' in position 0: ouch")
        self.check_exceptionobjectargs(UnicodeEncodeError, ['ascii', 'Āx', 0, 1, 'ouch'], "'ascii' codec can't encode character '\\u0100' in position 0: ouch")
        self.check_exceptionobjectargs(UnicodeEncodeError, ['ascii', '\uffffx', 0, 1, 'ouch'], "'ascii' codec can't encode character '\\uffff' in position 0: ouch")
        self.check_exceptionobjectargs(UnicodeEncodeError, ['ascii', '𐀀x', 0, 1, 'ouch'], "'ascii' codec can't encode character '\\U00010000' in position 0: ouch")

    def test_unicodedecodeerror(self):
        if False:
            print('Hello World!')
        self.check_exceptionobjectargs(UnicodeDecodeError, ['ascii', bytearray(b'g\xfcrk'), 1, 2, 'ouch'], "'ascii' codec can't decode byte 0xfc in position 1: ouch")
        self.check_exceptionobjectargs(UnicodeDecodeError, ['ascii', bytearray(b'g\xfcrk'), 1, 3, 'ouch'], "'ascii' codec can't decode bytes in position 1-2: ouch")

    def test_unicodetranslateerror(self):
        if False:
            while True:
                i = 10
        self.check_exceptionobjectargs(UnicodeTranslateError, ['gürk', 1, 2, 'ouch'], "can't translate character '\\xfc' in position 1: ouch")
        self.check_exceptionobjectargs(UnicodeTranslateError, ['gĀrk', 1, 2, 'ouch'], "can't translate character '\\u0100' in position 1: ouch")
        self.check_exceptionobjectargs(UnicodeTranslateError, ['g\uffffrk', 1, 2, 'ouch'], "can't translate character '\\uffff' in position 1: ouch")
        self.check_exceptionobjectargs(UnicodeTranslateError, ['g𐀀rk', 1, 2, 'ouch'], "can't translate character '\\U00010000' in position 1: ouch")
        self.check_exceptionobjectargs(UnicodeTranslateError, ['gürk', 1, 3, 'ouch'], "can't translate characters in position 1-2: ouch")

    def test_badandgoodstrictexceptions(self):
        if False:
            while True:
                i = 10
        self.assertRaises(TypeError, codecs.strict_errors, 42)
        self.assertRaises(Exception, codecs.strict_errors, Exception('ouch'))
        self.assertRaises(UnicodeEncodeError, codecs.strict_errors, UnicodeEncodeError('ascii', 'あ', 0, 1, 'ouch'))
        self.assertRaises(UnicodeDecodeError, codecs.strict_errors, UnicodeDecodeError('ascii', bytearray(b'\xff'), 0, 1, 'ouch'))
        self.assertRaises(UnicodeTranslateError, codecs.strict_errors, UnicodeTranslateError('あ', 0, 1, 'ouch'))

    def test_badandgoodignoreexceptions(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, codecs.ignore_errors, 42)
        self.assertRaises(TypeError, codecs.ignore_errors, UnicodeError('ouch'))
        self.assertEqual(codecs.ignore_errors(UnicodeEncodeError('ascii', 'aあb', 1, 2, 'ouch')), ('', 2))
        self.assertEqual(codecs.ignore_errors(UnicodeDecodeError('ascii', bytearray(b'a\xffb'), 1, 2, 'ouch')), ('', 2))
        self.assertEqual(codecs.ignore_errors(UnicodeTranslateError('aあb', 1, 2, 'ouch')), ('', 2))

    def test_badandgoodreplaceexceptions(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, codecs.replace_errors, 42)
        self.assertRaises(TypeError, codecs.replace_errors, UnicodeError('ouch'))
        self.assertRaises(TypeError, codecs.replace_errors, BadObjectUnicodeEncodeError())
        self.assertRaises(TypeError, codecs.replace_errors, BadObjectUnicodeDecodeError())
        self.assertEqual(codecs.replace_errors(UnicodeEncodeError('ascii', 'aあb', 1, 2, 'ouch')), ('?', 2))
        self.assertEqual(codecs.replace_errors(UnicodeDecodeError('ascii', bytearray(b'a\xffb'), 1, 2, 'ouch')), ('�', 2))
        self.assertEqual(codecs.replace_errors(UnicodeTranslateError('aあb', 1, 2, 'ouch')), ('�', 2))

    def test_badandgoodxmlcharrefreplaceexceptions(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, codecs.xmlcharrefreplace_errors, 42)
        self.assertRaises(TypeError, codecs.xmlcharrefreplace_errors, UnicodeError('ouch'))
        self.assertRaises(TypeError, codecs.xmlcharrefreplace_errors, UnicodeDecodeError('ascii', bytearray(b'\xff'), 0, 1, 'ouch'))
        self.assertRaises(TypeError, codecs.xmlcharrefreplace_errors, UnicodeTranslateError('あ', 0, 1, 'ouch'))
        cs = (0, 1, 9, 10, 99, 100, 999, 1000, 9999, 10000, 99999, 100000, 999999, 1000000)
        cs += (55296, 57343)
        s = ''.join((chr(c) for c in cs))
        self.assertEqual(codecs.xmlcharrefreplace_errors(UnicodeEncodeError('ascii', 'a' + s + 'b', 1, 1 + len(s), 'ouch')), (''.join(('&#%d;' % c for c in cs)), 1 + len(s)))

    def test_badandgoodbackslashreplaceexceptions(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, codecs.backslashreplace_errors, 42)
        self.assertRaises(TypeError, codecs.backslashreplace_errors, UnicodeError('ouch'))
        tests = [('あ', '\\u3042'), ('\n', '\\x0a'), ('a', '\\x61'), ('\x00', '\\x00'), ('ÿ', '\\xff'), ('Ā', '\\u0100'), ('\uffff', '\\uffff'), ('𐀀', '\\U00010000'), ('\U0010ffff', '\\U0010ffff'), ('\ud800', '\\ud800'), ('\udfff', '\\udfff'), ('\ud800\udfff', '\\ud800\\udfff')]
        for (s, r) in tests:
            with self.subTest(str=s):
                self.assertEqual(codecs.backslashreplace_errors(UnicodeEncodeError('ascii', 'a' + s + 'b', 1, 1 + len(s), 'ouch')), (r, 1 + len(s)))
                self.assertEqual(codecs.backslashreplace_errors(UnicodeTranslateError('a' + s + 'b', 1, 1 + len(s), 'ouch')), (r, 1 + len(s)))
        tests = [(b'a', '\\x61'), (b'\n', '\\x0a'), (b'\x00', '\\x00'), (b'\xff', '\\xff')]
        for (b, r) in tests:
            with self.subTest(bytes=b):
                self.assertEqual(codecs.backslashreplace_errors(UnicodeDecodeError('ascii', bytearray(b'a' + b + b'b'), 1, 2, 'ouch')), (r, 2))

    def test_badandgoodnamereplaceexceptions(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(TypeError, codecs.namereplace_errors, 42)
        self.assertRaises(TypeError, codecs.namereplace_errors, UnicodeError('ouch'))
        self.assertRaises(TypeError, codecs.namereplace_errors, UnicodeDecodeError('ascii', bytearray(b'\xff'), 0, 1, 'ouch'))
        self.assertRaises(TypeError, codecs.namereplace_errors, UnicodeTranslateError('あ', 0, 1, 'ouch'))
        tests = [('あ', '\\N{HIRAGANA LETTER A}'), ('\x00', '\\x00'), ('ﯹ', '\\N{ARABIC LIGATURE UIGHUR KIRGHIZ YEH WITH HAMZA ABOVE WITH ALEF MAKSURA ISOLATED FORM}'), ('\U000e007f', '\\N{CANCEL TAG}'), ('\U0010ffff', '\\U0010ffff'), ('\ud800', '\\ud800'), ('\udfff', '\\udfff'), ('\ud800\udfff', '\\ud800\\udfff')]
        for (s, r) in tests:
            with self.subTest(str=s):
                self.assertEqual(codecs.namereplace_errors(UnicodeEncodeError('ascii', 'a' + s + 'b', 1, 1 + len(s), 'ouch')), (r, 1 + len(s)))

    def test_badandgoodsurrogateescapeexceptions(self):
        if False:
            return 10
        surrogateescape_errors = codecs.lookup_error('surrogateescape')
        self.assertRaises(TypeError, surrogateescape_errors, 42)
        self.assertRaises(TypeError, surrogateescape_errors, UnicodeError('ouch'))
        self.assertRaises(TypeError, surrogateescape_errors, UnicodeTranslateError('\udc80', 0, 1, 'ouch'))
        for s in ('a', '\udc7f', '\udd00'):
            with self.subTest(str=s):
                self.assertRaises(UnicodeEncodeError, surrogateescape_errors, UnicodeEncodeError('ascii', s, 0, 1, 'ouch'))
        self.assertEqual(surrogateescape_errors(UnicodeEncodeError('ascii', 'a\udc80b', 1, 2, 'ouch')), (b'\x80', 2))
        self.assertRaises(UnicodeDecodeError, surrogateescape_errors, UnicodeDecodeError('ascii', bytearray(b'a'), 0, 1, 'ouch'))
        self.assertEqual(surrogateescape_errors(UnicodeDecodeError('ascii', bytearray(b'a\x80b'), 1, 2, 'ouch')), ('\udc80', 2))

    def test_badandgoodsurrogatepassexceptions(self):
        if False:
            return 10
        surrogatepass_errors = codecs.lookup_error('surrogatepass')
        self.assertRaises(TypeError, surrogatepass_errors, 42)
        self.assertRaises(TypeError, surrogatepass_errors, UnicodeError('ouch'))
        self.assertRaises(TypeError, surrogatepass_errors, UnicodeTranslateError('\ud800', 0, 1, 'ouch'))
        for enc in ('utf-8', 'utf-16le', 'utf-16be', 'utf-32le', 'utf-32be'):
            with self.subTest(encoding=enc):
                self.assertRaises(UnicodeEncodeError, surrogatepass_errors, UnicodeEncodeError(enc, 'a', 0, 1, 'ouch'))
                self.assertRaises(UnicodeDecodeError, surrogatepass_errors, UnicodeDecodeError(enc, 'a'.encode(enc), 0, 1, 'ouch'))
        for s in ('\ud800', '\udfff', '\ud800\udfff'):
            with self.subTest(str=s):
                self.assertRaises(UnicodeEncodeError, surrogatepass_errors, UnicodeEncodeError('ascii', s, 0, len(s), 'ouch'))
        tests = [('utf-8', '\ud800', b'\xed\xa0\x80', 3), ('utf-16le', '\ud800', b'\x00\xd8', 2), ('utf-16be', '\ud800', b'\xd8\x00', 2), ('utf-32le', '\ud800', b'\x00\xd8\x00\x00', 4), ('utf-32be', '\ud800', b'\x00\x00\xd8\x00', 4), ('utf-8', '\udfff', b'\xed\xbf\xbf', 3), ('utf-16le', '\udfff', b'\xff\xdf', 2), ('utf-16be', '\udfff', b'\xdf\xff', 2), ('utf-32le', '\udfff', b'\xff\xdf\x00\x00', 4), ('utf-32be', '\udfff', b'\x00\x00\xdf\xff', 4), ('utf-8', '\ud800\udfff', b'\xed\xa0\x80\xed\xbf\xbf', 3), ('utf-16le', '\ud800\udfff', b'\x00\xd8\xff\xdf', 2), ('utf-16be', '\ud800\udfff', b'\xd8\x00\xdf\xff', 2), ('utf-32le', '\ud800\udfff', b'\x00\xd8\x00\x00\xff\xdf\x00\x00', 4), ('utf-32be', '\ud800\udfff', b'\x00\x00\xd8\x00\x00\x00\xdf\xff', 4)]
        for (enc, s, b, n) in tests:
            with self.subTest(encoding=enc, str=s, bytes=b):
                self.assertEqual(surrogatepass_errors(UnicodeEncodeError(enc, 'a' + s + 'b', 1, 1 + len(s), 'ouch')), (b, 1 + len(s)))
                self.assertEqual(surrogatepass_errors(UnicodeDecodeError(enc, bytearray(b'a' + b[:n] + b'b'), 1, 1 + n, 'ouch')), (s[:1], 1 + n))

    def test_badhandlerresults(self):
        if False:
            for i in range(10):
                print('nop')
        results = (42, 'foo', (1, 2, 3), ('foo', 1, 3), ('foo', None), ('foo',), ('foo', 1, 3), ('foo', None), ('foo',))
        encs = ('ascii', 'latin-1', 'iso-8859-1', 'iso-8859-15')
        for res in results:
            codecs.register_error('test.badhandler', lambda x: res)
            for enc in encs:
                self.assertRaises(TypeError, 'あ'.encode, enc, 'test.badhandler')
            for (enc, bytes) in (('ascii', b'\xff'), ('utf-8', b'\xff'), ('utf-7', b'+x-')):
                self.assertRaises(TypeError, bytes.decode, enc, 'test.badhandler')

    def test_lookup(self):
        if False:
            while True:
                i = 10
        self.assertEqual(codecs.strict_errors, codecs.lookup_error('strict'))
        self.assertEqual(codecs.ignore_errors, codecs.lookup_error('ignore'))
        self.assertEqual(codecs.strict_errors, codecs.lookup_error('strict'))
        self.assertEqual(codecs.xmlcharrefreplace_errors, codecs.lookup_error('xmlcharrefreplace'))
        self.assertEqual(codecs.backslashreplace_errors, codecs.lookup_error('backslashreplace'))
        self.assertEqual(codecs.namereplace_errors, codecs.lookup_error('namereplace'))

    def test_encode_nonascii_replacement(self):
        if False:
            print('Hello World!')

        def handle(exc):
            if False:
                return 10
            if isinstance(exc, UnicodeEncodeError):
                return (repl, exc.end)
            raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.replacing', handle)
        for (enc, input, repl) in (('ascii', '[¤]', 'abc'), ('iso-8859-1', '[€]', '½¾'), ('iso-8859-15', '[¤]', 'œŸ')):
            res = input.encode(enc, 'test.replacing')
            self.assertEqual(res, ('[' + repl + ']').encode(enc))
        for (enc, input, repl) in (('utf-8', '[\udc80]', '🐍'), ('utf-16', '[\udc80]', '🐍'), ('utf-32', '[\udc80]', '🐍')):
            with self.subTest(encoding=enc):
                with self.assertRaises(UnicodeEncodeError) as cm:
                    input.encode(enc, 'test.replacing')
                exc = cm.exception
                self.assertEqual(exc.start, 1)
                self.assertEqual(exc.end, 2)
                self.assertEqual(exc.object, input)

    def test_encode_unencodable_replacement(self):
        if False:
            print('Hello World!')

        def unencrepl(exc):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(exc, UnicodeEncodeError):
                return (repl, exc.end)
            else:
                raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.unencreplhandler', unencrepl)
        for (enc, input, repl) in (('ascii', '[¤]', '½'), ('iso-8859-1', '[€]', 'œ'), ('iso-8859-15', '[¤]', '½'), ('utf-8', '[\udc80]', '\udcff'), ('utf-16', '[\udc80]', '\udcff'), ('utf-32', '[\udc80]', '\udcff')):
            with self.subTest(encoding=enc):
                with self.assertRaises(UnicodeEncodeError) as cm:
                    input.encode(enc, 'test.unencreplhandler')
                exc = cm.exception
                self.assertEqual(exc.start, 1)
                self.assertEqual(exc.end, 2)
                self.assertEqual(exc.object, input)

    def test_encode_bytes_replacement(self):
        if False:
            for i in range(10):
                print('nop')

        def handle(exc):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(exc, UnicodeEncodeError):
                return (repl, exc.end)
            raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.replacing', handle)
        for (enc, input, repl) in (('ascii', '[¤]', b'\xbd\xbe'), ('iso-8859-1', '[€]', b'\xbd\xbe'), ('iso-8859-15', '[¤]', b'\xbd\xbe'), ('utf-8', '[\udc80]', b'\xbd\xbe'), ('utf-16le', '[\udc80]', b'\xbd\xbe'), ('utf-16be', '[\udc80]', b'\xbd\xbe'), ('utf-32le', '[\udc80]', b'\xbc\xbd\xbe\xbf'), ('utf-32be', '[\udc80]', b'\xbc\xbd\xbe\xbf')):
            with self.subTest(encoding=enc):
                res = input.encode(enc, 'test.replacing')
                self.assertEqual(res, '['.encode(enc) + repl + ']'.encode(enc))

    def test_encode_odd_bytes_replacement(self):
        if False:
            for i in range(10):
                print('nop')

        def handle(exc):
            if False:
                i = 10
                return i + 15
            if isinstance(exc, UnicodeEncodeError):
                return (repl, exc.end)
            raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.replacing', handle)
        input = '[\udc80]'
        for (enc, repl) in (*itertools.product(('utf-16le', 'utf-16be'), [b'a', b'abc']), *itertools.product(('utf-32le', 'utf-32be'), [b'a', b'ab', b'abc', b'abcde'])):
            with self.subTest(encoding=enc, repl=repl):
                with self.assertRaises(UnicodeEncodeError) as cm:
                    input.encode(enc, 'test.replacing')
                exc = cm.exception
                self.assertEqual(exc.start, 1)
                self.assertEqual(exc.end, 2)
                self.assertEqual(exc.object, input)
                self.assertEqual(exc.reason, 'surrogates not allowed')

    def test_badregistercall(self):
        if False:
            return 10
        self.assertRaises(TypeError, codecs.register_error, 42)
        self.assertRaises(TypeError, codecs.register_error, 'test.dummy', 42)

    def test_badlookupcall(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(TypeError, codecs.lookup_error)

    def test_unknownhandler(self):
        if False:
            return 10
        self.assertRaises(LookupError, codecs.lookup_error, 'test.unknown')

    def test_xmlcharrefvalues(self):
        if False:
            return 10
        v = (1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000)
        s = ''.join([chr(x) for x in v])
        codecs.register_error('test.xmlcharrefreplace', codecs.xmlcharrefreplace_errors)
        for enc in ('ascii', 'iso-8859-15'):
            for err in ('xmlcharrefreplace', 'test.xmlcharrefreplace'):
                s.encode(enc, err)

    def test_decodehelper(self):
        if False:
            return 10
        self.assertRaises(LookupError, b'\xff'.decode, 'ascii', 'test.unknown')

        def baddecodereturn1(exc):
            if False:
                i = 10
                return i + 15
            return 42
        codecs.register_error('test.baddecodereturn1', baddecodereturn1)
        self.assertRaises(TypeError, b'\xff'.decode, 'ascii', 'test.baddecodereturn1')
        self.assertRaises(TypeError, b'\\'.decode, 'unicode-escape', 'test.baddecodereturn1')
        self.assertRaises(TypeError, b'\\x0'.decode, 'unicode-escape', 'test.baddecodereturn1')
        self.assertRaises(TypeError, b'\\x0y'.decode, 'unicode-escape', 'test.baddecodereturn1')
        self.assertRaises(TypeError, b'\\Uffffeeee'.decode, 'unicode-escape', 'test.baddecodereturn1')
        self.assertRaises(TypeError, b'\\uyyyy'.decode, 'raw-unicode-escape', 'test.baddecodereturn1')

        def baddecodereturn2(exc):
            if False:
                return 10
            return ('?', None)
        codecs.register_error('test.baddecodereturn2', baddecodereturn2)
        self.assertRaises(TypeError, b'\xff'.decode, 'ascii', 'test.baddecodereturn2')
        handler = PosReturn()
        codecs.register_error('test.posreturn', handler.handle)
        handler.pos = -1
        self.assertEqual(b'\xff0'.decode('ascii', 'test.posreturn'), '<?>0')
        handler.pos = -2
        self.assertEqual(b'\xff0'.decode('ascii', 'test.posreturn'), '<?><?>')
        handler.pos = -3
        self.assertRaises(IndexError, b'\xff0'.decode, 'ascii', 'test.posreturn')
        handler.pos = 1
        self.assertEqual(b'\xff0'.decode('ascii', 'test.posreturn'), '<?>0')
        handler.pos = 2
        self.assertEqual(b'\xff0'.decode('ascii', 'test.posreturn'), '<?>')
        handler.pos = 3
        self.assertRaises(IndexError, b'\xff0'.decode, 'ascii', 'test.posreturn')
        handler.pos = 6
        self.assertEqual(b'\\uyyyy0'.decode('raw-unicode-escape', 'test.posreturn'), '<?>0')

        class D(dict):

            def __getitem__(self, key):
                if False:
                    i = 10
                    return i + 15
                raise ValueError
        self.assertRaises(UnicodeError, codecs.charmap_decode, b'\xff', 'strict', {255: None})
        self.assertRaises(ValueError, codecs.charmap_decode, b'\xff', 'strict', D())
        self.assertRaises(TypeError, codecs.charmap_decode, b'\xff', 'strict', {255: sys.maxunicode + 1})

    def test_encodehelper(self):
        if False:
            return 10
        self.assertRaises(LookupError, 'ÿ'.encode, 'ascii', 'test.unknown')

        def badencodereturn1(exc):
            if False:
                return 10
            return 42
        codecs.register_error('test.badencodereturn1', badencodereturn1)
        self.assertRaises(TypeError, 'ÿ'.encode, 'ascii', 'test.badencodereturn1')

        def badencodereturn2(exc):
            if False:
                return 10
            return ('?', None)
        codecs.register_error('test.badencodereturn2', badencodereturn2)
        self.assertRaises(TypeError, 'ÿ'.encode, 'ascii', 'test.badencodereturn2')
        handler = PosReturn()
        codecs.register_error('test.posreturn', handler.handle)
        handler.pos = -1
        self.assertEqual('ÿ0'.encode('ascii', 'test.posreturn'), b'<?>0')
        handler.pos = -2
        self.assertEqual('ÿ0'.encode('ascii', 'test.posreturn'), b'<?><?>')
        handler.pos = -3
        self.assertRaises(IndexError, 'ÿ0'.encode, 'ascii', 'test.posreturn')
        handler.pos = 1
        self.assertEqual('ÿ0'.encode('ascii', 'test.posreturn'), b'<?>0')
        handler.pos = 2
        self.assertEqual('ÿ0'.encode('ascii', 'test.posreturn'), b'<?>')
        handler.pos = 3
        self.assertRaises(IndexError, 'ÿ0'.encode, 'ascii', 'test.posreturn')
        handler.pos = 0

        class D(dict):

            def __getitem__(self, key):
                if False:
                    while True:
                        i = 10
                raise ValueError
        for err in ('strict', 'replace', 'xmlcharrefreplace', 'backslashreplace', 'namereplace', 'test.posreturn'):
            self.assertRaises(UnicodeError, codecs.charmap_encode, 'ÿ', err, {255: None})
            self.assertRaises(ValueError, codecs.charmap_encode, 'ÿ', err, D())
            self.assertRaises(TypeError, codecs.charmap_encode, 'ÿ', err, {255: 300})

    def test_decodehelper_bug36819(self):
        if False:
            print('Hello World!')
        handler = RepeatedPosReturn('x')
        codecs.register_error('test.bug36819', handler.handle)
        testcases = [('ascii', b'\xff'), ('utf-8', b'\xff'), ('utf-16be', b'\xdc\x80'), ('utf-32be', b'\x00\x00\xdc\x80'), ('iso-8859-6', b'\xff')]
        for (enc, bad) in testcases:
            input = 'abcd'.encode(enc) + bad
            with self.subTest(encoding=enc):
                handler.count = 50
                decoded = input.decode(enc, 'test.bug36819')
                self.assertEqual(decoded, 'abcdx' * 51)

    def test_encodehelper_bug36819(self):
        if False:
            i = 10
            return i + 15
        handler = RepeatedPosReturn()
        codecs.register_error('test.bug36819', handler.handle)
        input = 'abcd\udc80'
        encodings = ['ascii', 'latin1', 'utf-8', 'utf-16', 'utf-32']
        encodings += ['iso-8859-15']
        if sys.platform == 'win32':
            encodings = ['mbcs', 'oem']
        handler.repl = '\udcff'
        for enc in encodings:
            with self.subTest(encoding=enc):
                handler.count = 50
                with self.assertRaises(UnicodeEncodeError) as cm:
                    input.encode(enc, 'test.bug36819')
                exc = cm.exception
                self.assertEqual(exc.start, 4)
                self.assertEqual(exc.end, 5)
                self.assertEqual(exc.object, input)
        if sys.platform == 'win32':
            handler.count = 50
            with self.assertRaises(UnicodeEncodeError) as cm:
                codecs.code_page_encode(437, input, 'test.bug36819')
            exc = cm.exception
            self.assertEqual(exc.start, 4)
            self.assertEqual(exc.end, 5)
            self.assertEqual(exc.object, input)
        handler.repl = 'x'
        for enc in encodings:
            with self.subTest(encoding=enc):
                handler.count = 50
                encoded = input.encode(enc, 'test.bug36819')
                self.assertEqual(encoded.decode(enc), 'abcdx' * 51)
        if sys.platform == 'win32':
            handler.count = 50
            encoded = codecs.code_page_encode(437, input, 'test.bug36819')
            self.assertEqual(encoded[0].decode(), 'abcdx' * 51)
            self.assertEqual(encoded[1], len(input))

    def test_translatehelper(self):
        if False:
            for i in range(10):
                print('nop')

        class D(dict):

            def __getitem__(self, key):
                if False:
                    while True:
                        i = 10
                raise ValueError
        self.assertRaises(ValueError, 'ÿ'.translate, {255: sys.maxunicode + 1})
        self.assertRaises(TypeError, 'ÿ'.translate, {255: ()})

    def test_bug828737(self):
        if False:
            print('Hello World!')
        charmap = {ord('&'): '&amp;', ord('<'): '&lt;', ord('>'): '&gt;', ord('"'): '&quot;'}
        for n in (1, 10, 100, 1000):
            text = 'abc<def>ghi' * n
            text.translate(charmap)

    def test_mutatingdecodehandler(self):
        if False:
            print('Hello World!')
        baddata = [('ascii', b'\xff'), ('utf-7', b'++'), ('utf-8', b'\xff'), ('utf-16', b'\xff'), ('utf-32', b'\xff'), ('unicode-escape', b'\\u123g'), ('raw-unicode-escape', b'\\u123g')]

        def replacing(exc):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(exc, UnicodeDecodeError):
                exc.object = 42
                return ('䉂', 0)
            else:
                raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.replacing', replacing)
        for (encoding, data) in baddata:
            with self.assertRaises(TypeError):
                data.decode(encoding, 'test.replacing')

        def mutating(exc):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(exc, UnicodeDecodeError):
                exc.object = b''
                return ('䉂', 0)
            else:
                raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.mutating', mutating)
        for (encoding, data) in baddata:
            self.assertEqual(data.decode(encoding, 'test.mutating'), '䉂')

    def test_crashing_decode_handler(self):
        if False:
            i = 10
            return i + 15

        def forward_shorter_than_end(exc):
            if False:
                while True:
                    i = 10
            if isinstance(exc, UnicodeDecodeError):
                return ('�', exc.start + 1)
            else:
                raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.forward_shorter_than_end', forward_shorter_than_end)
        self.assertEqual(b'\xd8\xd8\xd8\xd8\xd8\x00\x00\x00'.decode('utf-16-le', 'test.forward_shorter_than_end'), '����Ø\x00')
        self.assertEqual(b'\xd8\xd8\xd8\xd8\x00\xd8\x00\x00'.decode('utf-16-be', 'test.forward_shorter_than_end'), '����Ø\x00')
        self.assertEqual(b'\x11\x11\x11\x11\x11\x00\x00\x00\x00\x00\x00'.decode('utf-32-le', 'test.forward_shorter_than_end'), '���ᄑ\x00')
        self.assertEqual(b'\x11\x11\x11\x00\x00\x11\x11\x00\x00\x00\x00'.decode('utf-32-be', 'test.forward_shorter_than_end'), '���ᄑ\x00')

        def replace_with_long(exc):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(exc, UnicodeDecodeError):
                exc.object = b'\x00' * 8
                return ('�', exc.start)
            else:
                raise TypeError("don't know how to handle %r" % exc)
        codecs.register_error('test.replace_with_long', replace_with_long)
        self.assertEqual(b'\x00'.decode('utf-16', 'test.replace_with_long'), '�\x00\x00\x00\x00')
        self.assertEqual(b'\x00'.decode('utf-32', 'test.replace_with_long'), '�\x00\x00')

    def test_fake_error_class(self):
        if False:
            while True:
                i = 10
        handlers = [codecs.strict_errors, codecs.ignore_errors, codecs.replace_errors, codecs.backslashreplace_errors, codecs.namereplace_errors, codecs.xmlcharrefreplace_errors, codecs.lookup_error('surrogateescape'), codecs.lookup_error('surrogatepass')]
        for cls in (UnicodeEncodeError, UnicodeDecodeError, UnicodeTranslateError):

            class FakeUnicodeError(str):
                __class__ = cls
            for handler in handlers:
                with self.subTest(handler=handler, error_class=cls):
                    self.assertRaises(TypeError, handler, FakeUnicodeError())

            class FakeUnicodeError(Exception):
                __class__ = cls
            for handler in handlers:
                with self.subTest(handler=handler, error_class=cls):
                    with self.assertRaises((TypeError, FakeUnicodeError)):
                        handler(FakeUnicodeError())
if __name__ == '__main__':
    unittest.main()