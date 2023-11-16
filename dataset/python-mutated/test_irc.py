"""
Tests for L{twisted.words.protocols.irc}.
"""
import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A

class IRCTestCase(TestCase):

    def assertEqualBufferValue(self, buf, val):
        if False:
            while True:
                i = 10
        '\n        A buffer is always bytes, but sometimes\n        we need to compare it to a utf-8 unicode string\n\n        @param buf: the buffer\n        @type buf: L{bytes} or L{unicode} or L{list}\n        @param val: the value to compare\n        @type val: L{bytes} or L{unicode} or L{list}\n        '
        bufferValue = buf
        if isinstance(val, str):
            bufferValue = bufferValue.decode('utf-8')
        if isinstance(bufferValue, list):
            if isinstance(val[0], str):
                bufferValue = [b.decode('utf8') for b in bufferValue]
        self.assertEqual(bufferValue, val)

class ModeParsingTests(IRCTestCase):
    """
    Tests for L{twisted.words.protocols.irc.parseModes}.
    """
    paramModes = ('klb', 'b')

    def test_emptyModes(self):
        if False:
            return 10
        '\n        Parsing an empty mode string raises L{irc.IRCBadModes}.\n        '
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '', [])

    def test_emptyModeSequence(self):
        if False:
            print('Hello World!')
        '\n        Parsing a mode string that contains an empty sequence (either a C{+} or\n        C{-} followed directly by another C{+} or C{-}, or not followed by\n        anything at all) raises L{irc.IRCBadModes}.\n        '
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '++k', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '-+k', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '-', [])

    def test_malformedModes(self):
        if False:
            return 10
        '\n        Parsing a mode string that does not start with C{+} or C{-} raises\n        L{irc.IRCBadModes}.\n        '
        self.assertRaises(irc.IRCBadModes, irc.parseModes, 'foo', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '%', [])

    def test_nullModes(self):
        if False:
            print('Hello World!')
        '\n        Parsing a mode string that contains no mode characters raises\n        L{irc.IRCBadModes}.\n        '
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+', [])
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '-', [])

    def test_singleMode(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parsing a single mode setting with no parameters results in that mode,\n        with no parameters, in the "added" direction and no modes in the\n        "removed" direction.\n        '
        (added, removed) = irc.parseModes('+s', [])
        self.assertEqual(added, [('s', None)])
        self.assertEqual(removed, [])
        (added, removed) = irc.parseModes('-s', [])
        self.assertEqual(added, [])
        self.assertEqual(removed, [('s', None)])

    def test_singleDirection(self):
        if False:
            print('Hello World!')
        '\n        Parsing a single-direction mode setting with multiple modes and no\n        parameters, results in all modes falling into the same direction group.\n        '
        (added, removed) = irc.parseModes('+stn', [])
        self.assertEqual(added, [('s', None), ('t', None), ('n', None)])
        self.assertEqual(removed, [])
        (added, removed) = irc.parseModes('-nt', [])
        self.assertEqual(added, [])
        self.assertEqual(removed, [('n', None), ('t', None)])

    def test_multiDirection(self):
        if False:
            print('Hello World!')
        '\n        Parsing a multi-direction mode setting with no parameters.\n        '
        (added, removed) = irc.parseModes('+s-n+ti', [])
        self.assertEqual(added, [('s', None), ('t', None), ('i', None)])
        self.assertEqual(removed, [('n', None)])

    def test_consecutiveDirection(self):
        if False:
            print('Hello World!')
        '\n        Parsing a multi-direction mode setting containing two consecutive mode\n        sequences with the same direction results in the same result as if\n        there were only one mode sequence in the same direction.\n        '
        (added, removed) = irc.parseModes('+sn+ti', [])
        self.assertEqual(added, [('s', None), ('n', None), ('t', None), ('i', None)])
        self.assertEqual(removed, [])

    def test_mismatchedParams(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the number of mode parameters does not match the number of modes\n        expecting parameters, L{irc.IRCBadModes} is raised.\n        '
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+k', [], self.paramModes)
        self.assertRaises(irc.IRCBadModes, irc.parseModes, '+kl', ['foo', '10', 'lulz_extra_param'], self.paramModes)

    def test_parameters(self):
        if False:
            while True:
                i = 10
        '\n        Modes which require parameters are parsed and paired with their relevant\n        parameter, modes which do not require parameters do not consume any of\n        the parameters.\n        '
        (added, removed) = irc.parseModes('+klbb', ['somekey', '42', 'nick!user@host', 'other!*@*'], self.paramModes)
        self.assertEqual(added, [('k', 'somekey'), ('l', '42'), ('b', 'nick!user@host'), ('b', 'other!*@*')])
        self.assertEqual(removed, [])
        (added, removed) = irc.parseModes('-klbb', ['nick!user@host', 'other!*@*'], self.paramModes)
        self.assertEqual(added, [])
        self.assertEqual(removed, [('k', None), ('l', None), ('b', 'nick!user@host'), ('b', 'other!*@*')])
        (added, removed) = irc.parseModes('+knbb', ['somekey', 'nick!user@host', 'other!*@*'], self.paramModes)
        self.assertEqual(added, [('k', 'somekey'), ('n', None), ('b', 'nick!user@host'), ('b', 'other!*@*')])
        self.assertEqual(removed, [])

class MiscTests(IRCTestCase):
    """
    Tests for miscellaneous functions.
    """

    def test_foldr(self):
        if False:
            while True:
                i = 10
        '\n        Apply a function of two arguments cumulatively to the items of\n        a sequence, from right to left, so as to reduce the sequence to\n        a single value.\n        '
        self.assertEqual(irc._foldr(operator.sub, 0, [1, 2, 3, 4]), -2)

        def insertTop(l, x):
            if False:
                while True:
                    i = 10
            l.insert(0, x)
            return l
        self.assertEqual(irc._foldr(insertTop, [], [[1], [2], [3], [4]]), [[[[[], 4], 3], 2], 1])

class FormattedTextTests(IRCTestCase):
    """
    Tests for parsing and assembling formatted IRC text.
    """

    def assertAssembledEqually(self, text, expectedFormatted):
        if False:
            for i in range(10):
                print('nop')
        '\n        Assert that C{text} is parsed and assembled to the same value as what\n        C{expectedFormatted} is assembled to. This provides a way to ignore\n        meaningless differences in the formatting structure that would be\n        difficult to detect without rendering the structures.\n        '
        formatted = irc.parseFormattedText(text)
        self.assertAssemblesTo(formatted, expectedFormatted)

    def assertAssemblesTo(self, formatted, expectedFormatted):
        if False:
            while True:
                i = 10
        '\n        Assert that C{formatted} and C{expectedFormatted} assemble to the same\n        value.\n        '
        text = irc.assembleFormattedText(formatted)
        expectedText = irc.assembleFormattedText(expectedFormatted)
        self.assertEqual(irc.assembleFormattedText(formatted), expectedText, '%r (%r) is not equivalent to %r (%r)' % (text, formatted, expectedText, expectedFormatted))

    def test_parseEmpty(self):
        if False:
            while True:
                i = 10
        '\n        An empty string parses to a I{normal} attribute with no text.\n        '
        self.assertAssembledEqually('', A.normal)

    def test_assembleEmpty(self):
        if False:
            return 10
        '\n        An attribute with no text assembles to the empty string. An attribute\n        whose text is the empty string assembles to two control codes: C{off}\n        and that of the attribute.\n        '
        self.assertEqual(irc.assembleFormattedText(A.normal), '')
        self.assertEqual(irc.assembleFormattedText(A.bold['']), '\x0f\x02')

    def test_assembleNormal(self):
        if False:
            while True:
                i = 10
        '\n        A I{normal} string assembles to a string prefixed with the I{off}\n        control code.\n        '
        self.assertEqual(irc.assembleFormattedText(A.normal['hello']), '\x0fhello')

    def test_assembleBold(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A I{bold} string assembles to a string prefixed with the I{off} and\n        I{bold} control codes.\n        '
        self.assertEqual(irc.assembleFormattedText(A.bold['hello']), '\x0f\x02hello')

    def test_assembleUnderline(self):
        if False:
            return 10
        '\n        An I{underline} string assembles to a string prefixed with the I{off}\n        and I{underline} control codes.\n        '
        self.assertEqual(irc.assembleFormattedText(A.underline['hello']), '\x0f\x1fhello')

    def test_assembleReverseVideo(self):
        if False:
            while True:
                i = 10
        '\n        A I{reverse video} string assembles to a string prefixed with the I{off}\n        and I{reverse video} control codes.\n        '
        self.assertEqual(irc.assembleFormattedText(A.reverseVideo['hello']), '\x0f\x16hello')

    def test_assembleForegroundColor(self):
        if False:
            while True:
                i = 10
        '\n        A I{foreground color} string assembles to a string prefixed with the\n        I{off} and I{color} (followed by the relevant foreground color code)\n        control codes.\n        '
        self.assertEqual(irc.assembleFormattedText(A.fg.blue['hello']), '\x0f\x0302hello')

    def test_assembleBackgroundColor(self):
        if False:
            while True:
                i = 10
        '\n        A I{background color} string assembles to a string prefixed with the\n        I{off} and I{color} (followed by a I{,} to indicate the absence of a\n        foreground color, followed by the relevant background color code)\n        control codes.\n        '
        self.assertEqual(irc.assembleFormattedText(A.bg.blue['hello']), '\x0f\x03,02hello')

    def test_assembleColor(self):
        if False:
            while True:
                i = 10
        '\n        A I{foreground} and I{background} color string assembles to a string\n        prefixed with the I{off} and I{color} (followed by the relevant\n        foreground color, I{,} and the relevant background color code) control\n        codes.\n        '
        self.assertEqual(irc.assembleFormattedText(A.fg.red[A.bg.blue['hello']]), '\x0f\x0305,02hello')

    def test_assembleNested(self):
        if False:
            print('Hello World!')
        '\n        Nested attributes retain the attributes of their parents.\n        '
        self.assertEqual(irc.assembleFormattedText(A.bold['hello', A.underline[' world']]), '\x0f\x02hello\x0f\x02\x1f world')
        self.assertEqual(irc.assembleFormattedText(A.normal[A.fg.red[A.bg.green['hello'], ' world'], A.reverseVideo[' yay']]), '\x0f\x0305,03hello\x0f\x0305 world\x0f\x16 yay')

    def test_parseUnformattedText(self):
        if False:
            print('Hello World!')
        '\n        Parsing unformatted text results in text with attributes that\n        constitute a no-op.\n        '
        self.assertEqual(irc.parseFormattedText('hello'), A.normal['hello'])

    def test_colorFormatting(self):
        if False:
            while True:
                i = 10
        '\n        Correctly formatted text with colors uses 2 digits to specify\n        foreground and (optionally) background.\n        '
        self.assertEqual(irc.parseFormattedText('\x0301yay\x03'), A.fg.black['yay'])
        self.assertEqual(irc.parseFormattedText('\x0301,02yay\x03'), A.fg.black[A.bg.blue['yay']])
        self.assertEqual(irc.parseFormattedText('\x0301yay\x0302yipee\x03'), A.fg.black['yay', A.fg.blue['yipee']])

    def test_weirdColorFormatting(self):
        if False:
            while True:
                i = 10
        '\n        Formatted text with colors can use 1 digit for both foreground and\n        background, as long as the text part does not begin with a digit.\n        Foreground and background colors are only processed to a maximum of 2\n        digits per component, anything else is treated as text. Color sequences\n        must begin with a digit, otherwise processing falls back to unformatted\n        text.\n        '
        self.assertAssembledEqually('\x031kinda valid', A.fg.black['kinda valid'])
        self.assertAssembledEqually('\x03999,999kinda valid', A.fg.green['9,999kinda valid'])
        self.assertAssembledEqually('\x031,2kinda valid', A.fg.black[A.bg.blue['kinda valid']])
        self.assertAssembledEqually('\x031,999kinda valid', A.fg.black[A.bg.green['9kinda valid']])
        self.assertAssembledEqually('\x031,242 is a special number', A.fg.black[A.bg.yellow['2 is a special number']])
        self.assertAssembledEqually('\x03,02oops\x03', A.normal[',02oops'])
        self.assertAssembledEqually('\x03wrong', A.normal['wrong'])
        self.assertAssembledEqually('\x031,hello', A.fg.black['hello'])
        self.assertAssembledEqually('\x03\x03', A.normal)

    def test_clearColorFormatting(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An empty color format specifier clears foreground and background\n        colors.\n        '
        self.assertAssembledEqually('\x0301yay\x03reset', A.normal[A.fg.black['yay'], 'reset'])
        self.assertAssembledEqually('\x0301,02yay\x03reset', A.normal[A.fg.black[A.bg.blue['yay']], 'reset'])

    def test_resetFormatting(self):
        if False:
            while True:
                i = 10
        '\n        A reset format specifier clears all formatting attributes.\n        '
        self.assertAssembledEqually('\x02\x1fyay\x0freset', A.normal[A.bold[A.underline['yay']], 'reset'])
        self.assertAssembledEqually('\x0301yay\x0freset', A.normal[A.fg.black['yay'], 'reset'])
        self.assertAssembledEqually('\x0301,02yay\x0freset', A.normal[A.fg.black[A.bg.blue['yay']], 'reset'])

    def test_stripFormatting(self):
        if False:
            return 10
        '\n        Strip formatting codes from formatted text, leaving only the text parts.\n        '
        self.assertEqual(irc.stripFormatting(irc.assembleFormattedText(A.bold[A.underline[A.reverseVideo[A.fg.red[A.bg.green['hello']]], ' world']])), 'hello world')

class FormattingStateAttributeTests(IRCTestCase):
    """
    Tests for L{twisted.words.protocols.irc._FormattingState}.
    """

    def test_equality(self):
        if False:
            print('Hello World!')
        '\n        L{irc._FormattingState}s must have matching character attribute\n        values (bold, underline, etc) with the same values to be considered\n        equal.\n        '
        self.assertEqual(irc._FormattingState(), irc._FormattingState())
        self.assertEqual(irc._FormattingState(), irc._FormattingState(off=False))
        self.assertEqual(irc._FormattingState(bold=True, underline=True, off=False, reverseVideo=True, foreground=irc._IRC_COLORS['blue']), irc._FormattingState(bold=True, underline=True, off=False, reverseVideo=True, foreground=irc._IRC_COLORS['blue']))
        self.assertNotEqual(irc._FormattingState(bold=True), irc._FormattingState(bold=False))
stringSubjects = ['Hello, this is a nice string with no complications.', 'xargs{NUL}might{NUL}like{NUL}this'.format(NUL=irc.NUL), 'embedded{CR}newline{CR}{NL}FUN{NL}'.format(CR=irc.CR, NL=irc.NL), 'escape!%(X)s escape!%(M)s %(X)s%(X)sa %(M)s0' % {'X': irc.X_QUOTE, 'M': irc.M_QUOTE}]

class QuotingTests(IRCTestCase):

    def test_lowquoteSanity(self):
        if False:
            while True:
                i = 10
        '\n        Testing client-server level quote/dequote.\n        '
        for s in stringSubjects:
            self.assertEqual(s, irc.lowDequote(irc.lowQuote(s)))

    def test_ctcpquoteSanity(self):
        if False:
            return 10
        '\n        Testing CTCP message level quote/dequote.\n        '
        for s in stringSubjects:
            self.assertEqual(s, irc.ctcpDequote(irc.ctcpQuote(s)))

class Dispatcher(irc._CommandDispatcherMixin):
    """
    A dispatcher that exposes one known command and handles unknown commands.
    """
    prefix = 'disp'

    def disp_working(self, a, b):
        if False:
            i = 10
            return i + 15
        '\n        A known command that returns its input.\n        '
        return (a, b)

    def disp_unknown(self, name, a, b):
        if False:
            print('Hello World!')
        '\n        Handle unknown commands by returning their name and inputs.\n        '
        return (name, a, b)

class DispatcherTests(IRCTestCase):
    """
    Tests for L{irc._CommandDispatcherMixin}.
    """

    def test_dispatch(self):
        if False:
            return 10
        '\n        Dispatching a command invokes the correct handler.\n        '
        disp = Dispatcher()
        args = (1, 2)
        res = disp.dispatch('working', *args)
        self.assertEqual(res, args)

    def test_dispatchUnknown(self):
        if False:
            return 10
        '\n        Dispatching an unknown command invokes the default handler.\n        '
        disp = Dispatcher()
        name = 'missing'
        args = (1, 2)
        res = disp.dispatch(name, *args)
        self.assertEqual(res, (name,) + args)

    def test_dispatchMissingUnknown(self):
        if False:
            while True:
                i = 10
        '\n        Dispatching an unknown command, when no default handler is present,\n        results in an exception being raised.\n        '
        disp = Dispatcher()
        disp.disp_unknown = None
        self.assertRaises(irc.UnhandledCommand, disp.dispatch, 'bar')

class ServerSupportedFeatureTests(IRCTestCase):
    """
    Tests for L{ServerSupportedFeatures} and related functions.
    """

    def test_intOrDefault(self):
        if False:
            i = 10
            return i + 15
        '\n        L{_intOrDefault} converts values to C{int} if possible, otherwise\n        returns a default value.\n        '
        self.assertEqual(irc._intOrDefault(None), None)
        self.assertEqual(irc._intOrDefault([]), None)
        self.assertEqual(irc._intOrDefault(''), None)
        self.assertEqual(irc._intOrDefault('hello', 5), 5)
        self.assertEqual(irc._intOrDefault('123'), 123)
        self.assertEqual(irc._intOrDefault(123), 123)

    def test_splitParam(self):
        if False:
            print('Hello World!')
        '\n        L{ServerSupportedFeatures._splitParam} splits ISUPPORT parameters\n        into key and values. Parameters without a separator are split into a\n        key and a list containing only the empty string. Escaped parameters\n        are unescaped.\n        '
        params = [('FOO', ('FOO', [''])), ('FOO=', ('FOO', [''])), ('FOO=1', ('FOO', ['1'])), ('FOO=1,2,3', ('FOO', ['1', '2', '3'])), ('FOO=A\\x20B', ('FOO', ['A B'])), ('FOO=\\x5Cx', ('FOO', ['\\x'])), ('FOO=\\', ('FOO', ['\\'])), ('FOO=\\n', ('FOO', ['\\n']))]
        _splitParam = irc.ServerSupportedFeatures._splitParam
        for (param, expected) in params:
            res = _splitParam(param)
            self.assertEqual(res, expected)
        self.assertRaises(ValueError, _splitParam, 'FOO=\\x')
        self.assertRaises(ValueError, _splitParam, 'FOO=\\xNN')
        self.assertRaises(ValueError, _splitParam, 'FOO=\\xN')
        self.assertRaises(ValueError, _splitParam, 'FOO=\\x20\\x')

    def test_splitParamArgs(self):
        if False:
            i = 10
            return i + 15
        '\n        L{ServerSupportedFeatures._splitParamArgs} splits ISUPPORT parameter\n        arguments into key and value.  Arguments without a separator are\n        split into a key and an empty string.\n        '
        res = irc.ServerSupportedFeatures._splitParamArgs(['A:1', 'B:2', 'C:', 'D'])
        self.assertEqual(res, [('A', '1'), ('B', '2'), ('C', ''), ('D', '')])

    def test_splitParamArgsProcessor(self):
        if False:
            return 10
        '\n        L{ServerSupportedFeatures._splitParamArgs} uses the argument processor\n        passed to convert ISUPPORT argument values to some more suitable\n        form.\n        '
        res = irc.ServerSupportedFeatures._splitParamArgs(['A:1', 'B:2', 'C'], irc._intOrDefault)
        self.assertEqual(res, [('A', 1), ('B', 2), ('C', None)])

    def test_parsePrefixParam(self):
        if False:
            return 10
        '\n        L{ServerSupportedFeatures._parsePrefixParam} parses the ISUPPORT PREFIX\n        parameter into a mapping from modes to prefix symbols, returns\n        L{None} if there is no parseable prefix parameter or raises\n        C{ValueError} if the prefix parameter is malformed.\n        '
        _parsePrefixParam = irc.ServerSupportedFeatures._parsePrefixParam
        self.assertEqual(_parsePrefixParam(''), None)
        self.assertRaises(ValueError, _parsePrefixParam, 'hello')
        self.assertEqual(_parsePrefixParam('(ov)@+'), {'o': ('@', 0), 'v': ('+', 1)})

    def test_parseChanModesParam(self):
        if False:
            return 10
        '\n        L{ServerSupportedFeatures._parseChanModesParam} parses the ISUPPORT\n        CHANMODES parameter into a mapping from mode categories to mode\n        characters. Passing fewer than 4 parameters results in the empty string\n        for the relevant categories. Passing more than 4 parameters raises\n        C{ValueError}.\n        '
        _parseChanModesParam = irc.ServerSupportedFeatures._parseChanModesParam
        self.assertEqual(_parseChanModesParam(['', '', '', '']), {'addressModes': '', 'param': '', 'setParam': '', 'noParam': ''})
        self.assertEqual(_parseChanModesParam(['b', 'k', 'l', 'imnpst']), {'addressModes': 'b', 'param': 'k', 'setParam': 'l', 'noParam': 'imnpst'})
        self.assertEqual(_parseChanModesParam(['b', 'k', 'l', '']), {'addressModes': 'b', 'param': 'k', 'setParam': 'l', 'noParam': ''})
        self.assertRaises(ValueError, _parseChanModesParam, ['a', 'b', 'c', 'd', 'e'])

    def test_parse(self):
        if False:
            print('Hello World!')
        '\n        L{ServerSupportedFeatures.parse} changes the internal state of the\n        instance to reflect the features indicated by the parsed ISUPPORT\n        parameters, including unknown parameters and unsetting previously set\n        parameters.\n        '
        supported = irc.ServerSupportedFeatures()
        supported.parse(['MODES=4', 'CHANLIMIT=#:20,&:10', 'INVEX', 'EXCEPTS=Z', 'UNKNOWN=A,B,C'])
        self.assertEqual(supported.getFeature('MODES'), 4)
        self.assertEqual(supported.getFeature('CHANLIMIT'), [('#', 20), ('&', 10)])
        self.assertEqual(supported.getFeature('INVEX'), 'I')
        self.assertEqual(supported.getFeature('EXCEPTS'), 'Z')
        self.assertEqual(supported.getFeature('UNKNOWN'), ('A', 'B', 'C'))
        self.assertTrue(supported.hasFeature('INVEX'))
        supported.parse(['-INVEX'])
        self.assertFalse(supported.hasFeature('INVEX'))
        supported.parse(['-INVEX'])

    def _parse(self, features):
        if False:
            print('Hello World!')
        '\n        Parse all specified features according to the ISUPPORT specifications.\n\n        @type features: C{list} of C{(featureName, value)}\n        @param features: Feature names and values to parse\n\n        @rtype: L{irc.ServerSupportedFeatures}\n        '
        supported = irc.ServerSupportedFeatures()
        features = ['{}={}'.format(name, value or '') for (name, value) in features]
        supported.parse(features)
        return supported

    def _parseFeature(self, name, value=None):
        if False:
            return 10
        '\n        Parse a feature, with the given name and value, according to the\n        ISUPPORT specifications and return the parsed value.\n        '
        supported = self._parse([(name, value)])
        return supported.getFeature(name)

    def _testIntOrDefaultFeature(self, name, default=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform some common tests on a feature known to use L{_intOrDefault}.\n        '
        self.assertEqual(self._parseFeature(name, None), default)
        self.assertEqual(self._parseFeature(name, 'notanint'), default)
        self.assertEqual(self._parseFeature(name, '42'), 42)

    def _testFeatureDefault(self, name, features=None):
        if False:
            print('Hello World!')
        "\n        Features known to have default values are reported as being present by\n        L{irc.ServerSupportedFeatures.hasFeature}, and their value defaults\n        correctly, when they don't appear in an ISUPPORT message.\n        "
        default = irc.ServerSupportedFeatures()._features[name]
        if features is None:
            features = [('DEFINITELY_NOT', 'a_feature')]
        supported = self._parse(features)
        self.assertTrue(supported.hasFeature(name))
        self.assertEqual(supported.getFeature(name), default)

    def test_support_CHANMODES(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        The CHANMODES ISUPPORT parameter is parsed into a C{dict} giving the\n        four mode categories, C{'addressModes'}, C{'param'}, C{'setParam'}, and\n        C{'noParam'}.\n        "
        self._testFeatureDefault('CHANMODES')
        self._testFeatureDefault('CHANMODES', [('CHANMODES', 'b,,lk,')])
        self._testFeatureDefault('CHANMODES', [('CHANMODES', 'b,,lk,ha,ha')])
        self.assertEqual(self._parseFeature('CHANMODES', ',,,'), {'addressModes': '', 'param': '', 'setParam': '', 'noParam': ''})
        self.assertEqual(self._parseFeature('CHANMODES', ',A,,'), {'addressModes': '', 'param': 'A', 'setParam': '', 'noParam': ''})
        self.assertEqual(self._parseFeature('CHANMODES', 'A,Bc,Def,Ghij'), {'addressModes': 'A', 'param': 'Bc', 'setParam': 'Def', 'noParam': 'Ghij'})

    def test_support_IDCHAN(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The IDCHAN support parameter is parsed into a sequence of two-tuples\n        giving channel prefix and ID length pairs.\n        '
        self.assertEqual(self._parseFeature('IDCHAN', '!:5'), [('!', '5')])

    def test_support_MAXLIST(self):
        if False:
            print('Hello World!')
        '\n        The MAXLIST support parameter is parsed into a sequence of two-tuples\n        giving modes and their limits.\n        '
        self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50'), [('b', 25), ('eI', 50)])
        self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50,a:3.1415'), [('b', 25), ('eI', 50), ('a', None)])
        self.assertEqual(self._parseFeature('MAXLIST', 'b:25,eI:50,a:notanint'), [('b', 25), ('eI', 50), ('a', None)])

    def test_support_NETWORK(self):
        if False:
            while True:
                i = 10
        '\n        The NETWORK support parameter is parsed as the network name, as\n        specified by the server.\n        '
        self.assertEqual(self._parseFeature('NETWORK', 'IRCNet'), 'IRCNet')

    def test_support_SAFELIST(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The SAFELIST support parameter is parsed into a boolean indicating\n        whether the safe "list" command is supported or not.\n        '
        self.assertEqual(self._parseFeature('SAFELIST'), True)

    def test_support_STATUSMSG(self):
        if False:
            print('Hello World!')
        '\n        The STATUSMSG support parameter is parsed into a string of channel\n        status that support the exclusive channel notice method.\n        '
        self.assertEqual(self._parseFeature('STATUSMSG', '@+'), '@+')

    def test_support_TARGMAX(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The TARGMAX support parameter is parsed into a dictionary, mapping\n        strings to integers, of the maximum number of targets for a particular\n        command.\n        '
        self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3'), {'PRIVMSG': 4, 'NOTICE': 3})
        self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3,KICK:3.1415'), {'PRIVMSG': 4, 'NOTICE': 3, 'KICK': None})
        self.assertEqual(self._parseFeature('TARGMAX', 'PRIVMSG:4,NOTICE:3,KICK:notanint'), {'PRIVMSG': 4, 'NOTICE': 3, 'KICK': None})

    def test_support_NICKLEN(self):
        if False:
            return 10
        '\n        The NICKLEN support parameter is parsed into an integer value\n        indicating the maximum length of a nickname the client may use,\n        otherwise, if the parameter is missing or invalid, the default value\n        (as specified by RFC 1459) is used.\n        '
        default = irc.ServerSupportedFeatures()._features['NICKLEN']
        self._testIntOrDefaultFeature('NICKLEN', default)

    def test_support_CHANNELLEN(self):
        if False:
            print('Hello World!')
        '\n        The CHANNELLEN support parameter is parsed into an integer value\n        indicating the maximum channel name length, otherwise, if the\n        parameter is missing or invalid, the default value (as specified by\n        RFC 1459) is used.\n        '
        default = irc.ServerSupportedFeatures()._features['CHANNELLEN']
        self._testIntOrDefaultFeature('CHANNELLEN', default)

    def test_support_CHANTYPES(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The CHANTYPES support parameter is parsed into a tuple of\n        valid channel prefix characters.\n        '
        self._testFeatureDefault('CHANTYPES')
        self.assertEqual(self._parseFeature('CHANTYPES', '#&%'), ('#', '&', '%'))

    def test_support_KICKLEN(self):
        if False:
            return 10
        '\n        The KICKLEN support parameter is parsed into an integer value\n        indicating the maximum length of a kick message a client may use.\n        '
        self._testIntOrDefaultFeature('KICKLEN')

    def test_support_PREFIX(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The PREFIX support parameter is parsed into a dictionary mapping\n        modes to two-tuples of status symbol and priority.\n        '
        self._testFeatureDefault('PREFIX')
        self._testFeatureDefault('PREFIX', [('PREFIX', 'hello')])
        self.assertEqual(self._parseFeature('PREFIX', None), None)
        self.assertEqual(self._parseFeature('PREFIX', '(ohv)@%+'), {'o': ('@', 0), 'h': ('%', 1), 'v': ('+', 2)})
        self.assertEqual(self._parseFeature('PREFIX', '(hov)@%+'), {'o': ('%', 1), 'h': ('@', 0), 'v': ('+', 2)})

    def test_support_TOPICLEN(self):
        if False:
            print('Hello World!')
        '\n        The TOPICLEN support parameter is parsed into an integer value\n        indicating the maximum length of a topic a client may set.\n        '
        self._testIntOrDefaultFeature('TOPICLEN')

    def test_support_MODES(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The MODES support parameter is parsed into an integer value\n        indicating the maximum number of "variable" modes (defined as being\n        modes from C{addressModes}, C{param} or C{setParam} categories for\n        the C{CHANMODES} ISUPPORT parameter) which may by set on a channel\n        by a single MODE command from a client.\n        '
        self._testIntOrDefaultFeature('MODES')

    def test_support_EXCEPTS(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The EXCEPTS support parameter is parsed into the mode character\n        to be used for "ban exception" modes. If no parameter is specified\n        then the character C{e} is assumed.\n        '
        self.assertEqual(self._parseFeature('EXCEPTS', 'Z'), 'Z')
        self.assertEqual(self._parseFeature('EXCEPTS'), 'e')

    def test_support_INVEX(self):
        if False:
            while True:
                i = 10
        '\n        The INVEX support parameter is parsed into the mode character to be\n        used for "invite exception" modes. If no parameter is specified then\n        the character C{I} is assumed.\n        '
        self.assertEqual(self._parseFeature('INVEX', 'Z'), 'Z')
        self.assertEqual(self._parseFeature('INVEX'), 'I')

class IRCClientWithoutLogin(irc.IRCClient):
    performLogin = 0

class CTCPTests(IRCTestCase):
    """
    Tests for L{twisted.words.protocols.irc.IRCClient} CTCP handling.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.file = StringIOWithoutClosing()
        self.transport = protocol.FileWrapper(self.file)
        self.client = IRCClientWithoutLogin()
        self.client.makeConnection(self.transport)
        self.addCleanup(self.transport.loseConnection)
        self.addCleanup(self.client.connectionLost, None)

    def test_ERRMSG(self):
        if False:
            for i in range(10):
                print('nop')
        'Testing CTCP query ERRMSG.\n\n        Not because this is this is an especially important case in the\n        field, but it does go through the entire dispatch/decode/encode\n        process.\n        '
        errQuery = ':nick!guy@over.there PRIVMSG #theChan :%(X)cERRMSG t%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
        errReply = 'NOTICE nick :%(X)cERRMSG t :No error has occurred.%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
        self.client.dataReceived(errQuery)
        reply = self.file.getvalue()
        self.assertEqualBufferValue(reply, errReply)

    def test_noNumbersVERSION(self):
        if False:
            return 10
        '\n        If attributes for version information on L{IRCClient} are set to\n        L{None}, the parts of the CTCP VERSION response they correspond to\n        are omitted.\n        '
        self.client.versionName = 'FrobozzIRC'
        self.client.ctcpQuery_VERSION('nick!guy@over.there', '#theChan', None)
        versionReply = 'NOTICE nick :%(X)cVERSION %(vname)s::%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF, 'vname': self.client.versionName}
        reply = self.file.getvalue()
        self.assertEqualBufferValue(reply, versionReply)

    def test_fullVERSION(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The response to a CTCP VERSION query includes the version number and\n        environment information, as specified by L{IRCClient.versionNum} and\n        L{IRCClient.versionEnv}.\n        '
        self.client.versionName = 'FrobozzIRC'
        self.client.versionNum = '1.2g'
        self.client.versionEnv = 'ZorkOS'
        self.client.ctcpQuery_VERSION('nick!guy@over.there', '#theChan', None)
        versionReply = 'NOTICE nick :%(X)cVERSION %(vname)s:%(vnum)s:%(venv)s%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF, 'vname': self.client.versionName, 'vnum': self.client.versionNum, 'venv': self.client.versionEnv}
        reply = self.file.getvalue()
        self.assertEqualBufferValue(reply, versionReply)

    def test_noDuplicateCTCPDispatch(self):
        if False:
            print('Hello World!')
        '\n        Duplicated CTCP messages are ignored and no reply is made.\n        '

        def testCTCP(user, channel, data):
            if False:
                return 10
            self.called += 1
        self.called = 0
        self.client.ctcpQuery_TESTTHIS = testCTCP
        self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}TESTTHIS{X}foo{X}TESTTHIS{X}'.format(X=irc.X_DELIM)])
        self.assertEqualBufferValue(self.file.getvalue(), '')
        self.assertEqual(self.called, 1)

    def test_noDefaultDispatch(self):
        if False:
            i = 10
            return i + 15
        '\n        The fallback handler is invoked for unrecognized CTCP messages.\n        '

        def unknownQuery(user, channel, tag, data):
            if False:
                i = 10
                return i + 15
            self.calledWith = (user, channel, tag, data)
            self.called += 1
        self.called = 0
        self.patch(self.client, 'ctcpUnknownQuery', unknownQuery)
        self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}NOTREAL{X}'.format(X=irc.X_DELIM)])
        self.assertEqualBufferValue(self.file.getvalue(), '')
        self.assertEqual(self.calledWith, ('foo!bar@baz.quux', '#chan', 'NOTREAL', None))
        self.assertEqual(self.called, 1)
        self.client.irc_PRIVMSG('foo!bar@baz.quux', ['#chan', '{X}NOTREAL{X}foo{X}NOTREAL{X}'.format(X=irc.X_DELIM)])
        self.assertEqual(self.called, 2)

class NoticingClient(IRCClientWithoutLogin):
    methods = {'created': ('when',), 'yourHost': ('info',), 'myInfo': ('servername', 'version', 'umodes', 'cmodes'), 'luserClient': ('info',), 'bounce': ('info',), 'isupport': ('options',), 'luserChannels': ('channels',), 'luserOp': ('ops',), 'luserMe': ('info',), 'receivedMOTD': ('motd',), 'privmsg': ('user', 'channel', 'message'), 'joined': ('channel',), 'left': ('channel',), 'noticed': ('user', 'channel', 'message'), 'modeChanged': ('user', 'channel', 'set', 'modes', 'args'), 'pong': ('user', 'secs'), 'signedOn': (), 'kickedFrom': ('channel', 'kicker', 'message'), 'nickChanged': ('nick',), 'userJoined': ('user', 'channel'), 'userLeft': ('user', 'channel'), 'userKicked': ('user', 'channel', 'kicker', 'message'), 'action': ('user', 'channel', 'data'), 'topicUpdated': ('user', 'channel', 'newTopic'), 'userRenamed': ('oldname', 'newname')}

    def __init__(self, *a, **kw):
        if False:
            print('Hello World!')
        self.calls = []

    def __getattribute__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if name.startswith('__') and name.endswith('__'):
            return super().__getattribute__(name)
        try:
            args = super().__getattribute__('methods')[name]
        except KeyError:
            return super().__getattribute__(name)
        else:
            return self.makeMethod(name, args)

    def makeMethod(self, fname, args):
        if False:
            return 10

        def method(*a, **kw):
            if False:
                for i in range(10):
                    print('nop')
            if len(a) > len(args):
                raise TypeError('TypeError: %s() takes %d arguments (%d given)' % (fname, len(args), len(a)))
            for (name, value) in zip(args, a):
                if name in kw:
                    raise TypeError("TypeError: %s() got multiple values for keyword argument '%s'" % (fname, name))
                else:
                    kw[name] = value
            if len(kw) != len(args):
                raise TypeError('TypeError: %s() takes %d arguments (%d given)' % (fname, len(args), len(a)))
            self.calls.append((fname, kw))
        return method

def pop(dict, key, default):
    if False:
        return 10
    try:
        value = dict[key]
    except KeyError:
        return default
    else:
        del dict[key]
        return value

class ClientImplementationTests(IRCTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport = StringTransport()
        self.client = NoticingClient()
        self.client.makeConnection(self.transport)
        self.addCleanup(self.transport.loseConnection)
        self.addCleanup(self.client.connectionLost, None)

    def _serverTestImpl(self, code, msg, func, **kw):
        if False:
            for i in range(10):
                print('nop')
        host = pop(kw, 'host', 'server.host')
        nick = pop(kw, 'nick', 'nickname')
        args = pop(kw, 'args', '')
        message = ':' + host + ' ' + code + ' ' + nick + ' ' + args + ' :' + msg + '\r\n'
        self.client.dataReceived(message)
        self.assertEqual(self.client.calls, [(func, kw)])

    def testYourHost(self):
        if False:
            print('Hello World!')
        msg = 'Your host is some.host[blah.blah/6667], running version server-version-3'
        self._serverTestImpl('002', msg, 'yourHost', info=msg)

    def testCreated(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'This server was cobbled together Fri Aug 13 18:00:25 UTC 2004'
        self._serverTestImpl('003', msg, 'created', when=msg)

    def testMyInfo(self):
        if False:
            i = 10
            return i + 15
        msg = 'server.host server-version abcDEF bcdEHI'
        self._serverTestImpl('004', msg, 'myInfo', servername='server.host', version='server-version', umodes='abcDEF', cmodes='bcdEHI')

    def testLuserClient(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'There are 9227 victims and 9542 hiding on 24 servers'
        self._serverTestImpl('251', msg, 'luserClient', info=msg)

    def _sendISUPPORT(self):
        if False:
            print('Hello World!')
        args = 'MODES=4 CHANLIMIT=#:20 NICKLEN=16 USERLEN=10 HOSTLEN=63 TOPICLEN=450 KICKLEN=450 CHANNELLEN=30 KEYLEN=23 CHANTYPES=# PREFIX=(ov)@+ CASEMAPPING=ascii CAPAB IRCD=dancer'
        msg = 'are available on this server'
        self._serverTestImpl('005', msg, 'isupport', args=args, options=['MODES=4', 'CHANLIMIT=#:20', 'NICKLEN=16', 'USERLEN=10', 'HOSTLEN=63', 'TOPICLEN=450', 'KICKLEN=450', 'CHANNELLEN=30', 'KEYLEN=23', 'CHANTYPES=#', 'PREFIX=(ov)@+', 'CASEMAPPING=ascii', 'CAPAB', 'IRCD=dancer'])

    def test_ISUPPORT(self):
        if False:
            return 10
        '\n        The client parses ISUPPORT messages sent by the server and calls\n        L{IRCClient.isupport}.\n        '
        self._sendISUPPORT()

    def testBounce(self):
        if False:
            return 10
        msg = 'Try server some.host, port 321'
        self._serverTestImpl('010', msg, 'bounce', info=msg)

    def testLuserChannels(self):
        if False:
            while True:
                i = 10
        args = '7116'
        msg = 'channels formed'
        self._serverTestImpl('254', msg, 'luserChannels', args=args, channels=int(args))

    def testLuserOp(self):
        if False:
            for i in range(10):
                print('nop')
        args = '34'
        msg = 'flagged staff members'
        self._serverTestImpl('252', msg, 'luserOp', args=args, ops=int(args))

    def testLuserMe(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'I have 1937 clients and 0 servers'
        self._serverTestImpl('255', msg, 'luserMe', info=msg)

    def test_receivedMOTD(self):
        if False:
            return 10
        '\n        Lines received in I{RPL_MOTDSTART} and I{RPL_MOTD} are delivered to\n        L{IRCClient.receivedMOTD} when I{RPL_ENDOFMOTD} is received.\n        '
        lines = [':host.name 375 nickname :- host.name Message of the Day -', ':host.name 372 nickname :- Welcome to host.name', ':host.name 376 nickname :End of /MOTD command.']
        for L in lines:
            self.assertEqual(self.client.calls, [])
            self.client.dataReceived(L + '\r\n')
        self.assertEqual(self.client.calls, [('receivedMOTD', {'motd': ['host.name Message of the Day -', 'Welcome to host.name']})])
        self.assertIdentical(self.client.motd, None)

    def test_withoutMOTDSTART(self):
        if False:
            i = 10
            return i + 15
        '\n        If L{IRCClient} receives I{RPL_MOTD} and I{RPL_ENDOFMOTD} without\n        receiving I{RPL_MOTDSTART}, L{IRCClient.receivedMOTD} is still\n        called with a list of MOTD lines.\n        '
        lines = [':host.name 372 nickname :- Welcome to host.name', ':host.name 376 nickname :End of /MOTD command.']
        for L in lines:
            self.client.dataReceived(L + '\r\n')
        self.assertEqual(self.client.calls, [('receivedMOTD', {'motd': ['Welcome to host.name']})])

    def _clientTestImpl(self, sender, group, type, msg, func, **kw):
        if False:
            i = 10
            return i + 15
        ident = pop(kw, 'ident', 'ident')
        host = pop(kw, 'host', 'host')
        wholeUser = sender + '!' + ident + '@' + host
        message = ':' + wholeUser + ' ' + type + ' ' + group + ' :' + msg + '\r\n'
        self.client.dataReceived(message)
        self.assertEqual(self.client.calls, [(func, kw)])
        self.client.calls = []

    def testPrivmsg(self):
        if False:
            while True:
                i = 10
        msg = 'Tooty toot toot.'
        self._clientTestImpl('sender', '#group', 'PRIVMSG', msg, 'privmsg', ident='ident', host='host', user='sender!ident@host', channel='#group', message=msg)
        self._clientTestImpl('sender', 'recipient', 'PRIVMSG', msg, 'privmsg', ident='ident', host='host', user='sender!ident@host', channel='recipient', message=msg)

    def test_getChannelModeParams(self):
        if False:
            return 10
        '\n        L{IRCClient.getChannelModeParams} uses ISUPPORT information, either\n        given by the server or defaults, to determine which channel modes\n        require arguments when being added or removed.\n        '
        (add, remove) = map(sorted, self.client.getChannelModeParams())
        self.assertEqual(add, ['b', 'h', 'k', 'l', 'o', 'v'])
        self.assertEqual(remove, ['b', 'h', 'o', 'v'])

        def removeFeature(name):
            if False:
                print('Hello World!')
            name = '-' + name
            msg = 'are available on this server'
            self._serverTestImpl('005', msg, 'isupport', args=name, options=[name])
            self.assertIdentical(self.client.supported.getFeature(name), None)
            self.client.calls = []
        removeFeature('CHANMODES')
        (add, remove) = map(sorted, self.client.getChannelModeParams())
        self.assertEqual(add, ['h', 'o', 'v'])
        self.assertEqual(remove, ['h', 'o', 'v'])
        removeFeature('PREFIX')
        (add, remove) = map(sorted, self.client.getChannelModeParams())
        self.assertEqual(add, [])
        self.assertEqual(remove, [])
        self._sendISUPPORT()
        self.assertNotIdentical(self.client.supported.getFeature('PREFIX'), None)

    def test_getUserModeParams(self):
        if False:
            while True:
                i = 10
        '\n        L{IRCClient.getUserModeParams} returns a list of user modes (modes that\n        the user sets on themself, outside of channel modes) that require\n        parameters when added and removed, respectively.\n        '
        (add, remove) = map(sorted, self.client.getUserModeParams())
        self.assertEqual(add, [])
        self.assertEqual(remove, [])

    def _sendModeChange(self, msg, args='', target=None):
        if False:
            return 10
        '\n        Build a MODE string and send it to the client.\n        '
        if target is None:
            target = '#chan'
        message = f':Wolf!~wolf@yok.utu.fi MODE {target} {msg} {args}\r\n'
        self.client.dataReceived(message)

    def _parseModeChange(self, results, target=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse the results, do some test and return the data to check.\n        '
        if target is None:
            target = '#chan'
        for (n, result) in enumerate(results):
            (method, data) = result
            self.assertEqual(method, 'modeChanged')
            self.assertEqual(data['user'], 'Wolf!~wolf@yok.utu.fi')
            self.assertEqual(data['channel'], target)
            results[n] = tuple((data[key] for key in ('set', 'modes', 'args')))
        return results

    def _checkModeChange(self, expected, target=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compare the expected result with the one returned by the client.\n        '
        result = self._parseModeChange(self.client.calls, target)
        self.assertEqual(result, expected)
        self.client.calls = []

    def test_modeMissingDirection(self):
        if False:
            while True:
                i = 10
        "\n        Mode strings that do not begin with a directional character, C{'+'} or\n        C{'-'}, have C{'+'} automatically prepended.\n        "
        self._sendModeChange('s')
        self._checkModeChange([(True, 's', (None,))])

    def test_noModeParameters(self):
        if False:
            i = 10
            return i + 15
        "\n        No parameters are passed to L{IRCClient.modeChanged} for modes that\n        don't take any parameters.\n        "
        self._sendModeChange('-s')
        self._checkModeChange([(False, 's', (None,))])
        self._sendModeChange('+n')
        self._checkModeChange([(True, 'n', (None,))])

    def test_oneModeParameter(self):
        if False:
            while True:
                i = 10
        '\n        Parameters are passed to L{IRCClient.modeChanged} for modes that take\n        parameters.\n        '
        self._sendModeChange('+o', 'a_user')
        self._checkModeChange([(True, 'o', ('a_user',))])
        self._sendModeChange('-o', 'a_user')
        self._checkModeChange([(False, 'o', ('a_user',))])

    def test_mixedModes(self):
        if False:
            i = 10
            return i + 15
        "\n        Mixing adding and removing modes that do and don't take parameters\n        invokes L{IRCClient.modeChanged} with mode characters and parameters\n        that match up.\n        "
        self._sendModeChange('+osv', 'a_user another_user')
        self._checkModeChange([(True, 'osv', ('a_user', None, 'another_user'))])
        self._sendModeChange('+v-os', 'a_user another_user')
        self._checkModeChange([(True, 'v', ('a_user',)), (False, 'os', ('another_user', None))])

    def test_tooManyModeParameters(self):
        if False:
            print('Hello World!')
        '\n        Passing an argument to modes that take no parameters results in\n        L{IRCClient.modeChanged} not being called and an error being logged.\n        '
        self._sendModeChange('+s', 'wrong')
        self._checkModeChange([])
        errors = self.flushLoggedErrors(irc.IRCBadModes)
        self.assertEqual(len(errors), 1)
        self.assertSubstring('Too many parameters', errors[0].getErrorMessage())

    def test_tooFewModeParameters(self):
        if False:
            return 10
        '\n        Passing no arguments to modes that do take parameters results in\n        L{IRCClient.modeChange} not being called and an error being logged.\n        '
        self._sendModeChange('+o')
        self._checkModeChange([])
        errors = self.flushLoggedErrors(irc.IRCBadModes)
        self.assertEqual(len(errors), 1)
        self.assertSubstring('Not enough parameters', errors[0].getErrorMessage())

    def test_userMode(self):
        if False:
            i = 10
            return i + 15
        '\n        A C{MODE} message whose target is our user (the nickname of our user,\n        to be precise), as opposed to a channel, will be parsed according to\n        the modes specified by L{IRCClient.getUserModeParams}.\n        '
        target = self.client.nickname
        self._sendModeChange('+o', target=target)
        self._checkModeChange([(True, 'o', (None,))], target=target)

        def getUserModeParams():
            if False:
                while True:
                    i = 10
            return ['Z', '']
        self.patch(self.client, 'getUserModeParams', getUserModeParams)
        self._sendModeChange('+Z', 'an_arg', target=target)
        self._checkModeChange([(True, 'Z', ('an_arg',))], target=target)

    def test_heartbeat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When the I{RPL_WELCOME} message is received a heartbeat is started that\n        will send a I{PING} message to the IRC server every\n        L{irc.IRCClient.heartbeatInterval} seconds. When the transport is\n        closed the heartbeat looping call is stopped too.\n        '

        def _createHeartbeat():
            if False:
                i = 10
                return i + 15
            heartbeat = self._originalCreateHeartbeat()
            heartbeat.clock = self.clock
            return heartbeat
        self.clock = task.Clock()
        self._originalCreateHeartbeat = self.client._createHeartbeat
        self.patch(self.client, '_createHeartbeat', _createHeartbeat)
        self.assertIdentical(self.client._heartbeat, None)
        self.client.irc_RPL_WELCOME('foo', [])
        self.assertNotIdentical(self.client._heartbeat, None)
        self.assertEqual(self.client.hostname, 'foo')
        self.assertEqualBufferValue(self.transport.value(), '')
        self.clock.advance(self.client.heartbeatInterval)
        self.assertEqualBufferValue(self.transport.value(), 'PING foo\r\n')
        self.transport.loseConnection()
        self.client.connectionLost(None)
        self.assertEqual(len(self.clock.getDelayedCalls()), 0)
        self.assertIdentical(self.client._heartbeat, None)

    def test_heartbeatDisabled(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If L{irc.IRCClient.heartbeatInterval} is set to L{None} then no\n        heartbeat is created.\n        '
        self.assertIdentical(self.client._heartbeat, None)
        self.client.heartbeatInterval = None
        self.client.irc_RPL_WELCOME('foo', [])
        self.assertIdentical(self.client._heartbeat, None)

class BasicServerFunctionalityTests(IRCTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.f = StringIOWithoutClosing()
        self.t = protocol.FileWrapper(self.f)
        self.p = irc.IRC()
        self.p.makeConnection(self.t)

    def check(self, s):
        if False:
            print('Hello World!')
        '\n        Make sure that the internal buffer equals a specified value.\n\n        @param s: the value to compare against buffer\n        @type s: L{bytes} or L{unicode}\n        '
        bufferValue = self.f.getvalue()
        if isinstance(s, str):
            bufferValue = bufferValue.decode('utf-8')
        self.assertEqual(bufferValue, s)

    def test_sendMessage(self):
        if False:
            return 10
        "\n        Passing a command and parameters to L{IRC.sendMessage} results in a\n        query string that consists of the command and parameters, separated by\n        a space, ending with '\r\n'.\n        "
        self.p.sendMessage('CMD', 'param1', 'param2')
        self.check('CMD param1 param2\r\n')

    def test_sendCommand(self):
        if False:
            i = 10
            return i + 15
        "\n        Passing a command and parameters to L{IRC.sendCommand} results in a\n        query string that consists of the command and parameters, separated by\n        a space, ending with '\r\n'.\n\n        The format is described in more detail in\n        U{RFC 1459 <https://tools.ietf.org/html/rfc1459.html#section-2.3>}.\n        "
        self.p.sendCommand('CMD', ('param1', 'param2'))
        self.check('CMD param1 param2\r\n')

    def test_sendUnicodeCommand(self):
        if False:
            print('Hello World!')
        '\n        Passing unicode parameters to L{IRC.sendCommand} encodes the parameters\n        in UTF-8.\n        '
        self.p.sendCommand('CMD', ('param¹', 'param²'))
        self.check(b'CMD param\xc2\xb9 param\xc2\xb2\r\n')

    def test_sendMessageNoCommand(self):
        if False:
            return 10
        '\n        Passing L{None} as the command to L{IRC.sendMessage} raises a\n        C{ValueError}.\n        '
        error = self.assertRaises(ValueError, self.p.sendMessage, None, 'param1', 'param2')
        self.assertEqual(str(error), 'IRC message requires a command.')

    def test_sendCommandNoCommand(self):
        if False:
            return 10
        '\n        Passing L{None} as the command to L{IRC.sendCommand} raises a\n        C{ValueError}.\n        '
        error = self.assertRaises(ValueError, self.p.sendCommand, None, ('param1', 'param2'))
        self.assertEqual(error.args[0], 'IRC message requires a command.')

    def test_sendMessageInvalidCommand(self):
        if False:
            print('Hello World!')
        '\n        Passing an invalid string command to L{IRC.sendMessage} raises a\n        C{ValueError}.\n        '
        error = self.assertRaises(ValueError, self.p.sendMessage, ' ', 'param1', 'param2')
        self.assertEqual(str(error), "Somebody screwed up, 'cuz this doesn't look like a command to me:  ")

    def test_sendCommandInvalidCommand(self):
        if False:
            print('Hello World!')
        '\n        Passing an invalid string command to L{IRC.sendCommand} raises a\n        C{ValueError}.\n        '
        error = self.assertRaises(ValueError, self.p.sendCommand, ' ', ('param1', 'param2'))
        self.assertEqual(error.args[0], 'Invalid command: " "')

    def test_sendCommandWithPrefix(self):
        if False:
            i = 10
            return i + 15
        '\n        Passing a command and parameters with a specified prefix to\n        L{IRC.sendCommand} results in a proper query string including the\n        specified line prefix.\n        '
        self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com')
        self.check(b':irc.example.com CMD param1 param2\r\n')

    def test_sendCommandWithTags(self):
        if False:
            while True:
                i = 10
        "\n        Passing a command and parameters with a specified prefix and tags\n        to L{IRC.sendCommand} results in a proper query string including the\n        specified line prefix and appropriate tags syntax.  The query string\n        should be output as follows:\n        @tags :prefix COMMAND param1 param2\r\n\n        The tags are a string of IRCv3 tags, preceded by '@'.  The rest\n        of the string is as described in test_sendMessage.  For more on\n        the message tag format, see U{the IRCv3 specification\n        <https://ircv3.net/specs/core/message-tags-3.2.html>}.\n        "
        sendTags = {'aaa': 'bbb', 'ccc': None, 'example.com/ddd': 'eee'}
        expectedTags = (b'aaa=bbb', b'ccc', b'example.com/ddd=eee')
        self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        outMsg = self.f.getvalue()
        (outTagStr, outLine) = outMsg.split(b' ', 1)
        outTags = outTagStr[1:].split(b';')
        self.assertEqual(outLine, b':irc.example.com CMD param1 param2\r\n')
        self.assertEqual(sorted(expectedTags), sorted(outTags))

    def test_sendCommandValidateEmptyTags(self):
        if False:
            return 10
        '\n        Passing empty tag names to L{IRC.sendCommand} raises a C{ValueError}.\n        '
        sendTags = {'aaa': 'bbb', 'ccc': None, '': ''}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'A tag name is required.')

    def test_sendCommandValidateNoneTags(self):
        if False:
            print('Hello World!')
        '\n        Passing None as a tag name to L{IRC.sendCommand} raises a\n        C{ValueError}.\n        '
        sendTags = {'aaa': 'bbb', 'ccc': None, None: 'beep'}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'A tag name is required.')

    def test_sendCommandValidateTagsWithSpaces(self):
        if False:
            i = 10
            return i + 15
        '\n        Passing a tag name containing spaces to L{IRC.sendCommand} raises a\n        C{ValueError}.\n        '
        sendTags = {'aaa bbb': 'ccc'}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'Tag contains invalid characters.')

    def test_sendCommandValidateTagsWithInvalidChars(self):
        if False:
            return 10
        '\n        Passing a tag name containing invalid characters to L{IRC.sendCommand}\n        raises a C{ValueError}.\n        '
        sendTags = {'aaa_b^@': 'ccc'}
        error = self.assertRaises(ValueError, self.p.sendCommand, 'CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        self.assertEqual(error.args[0], 'Tag contains invalid characters.')

    def test_sendCommandValidateTagValueEscaping(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tags with values containing invalid characters passed to\n        L{IRC.sendCommand} are escaped.\n        '
        sendTags = {'aaa': 'bbb', 'ccc': 'test\r\n \\;;'}
        expectedTags = (b'aaa=bbb', b'ccc=test\\r\\n\\s\\\\\\:\\:')
        self.p.sendCommand('CMD', ('param1', 'param2'), 'irc.example.com', sendTags)
        outMsg = self.f.getvalue()
        (outTagStr, outLine) = outMsg.split(b' ', 1)
        outTags = outTagStr[1:].split(b';')
        self.assertEqual(sorted(outTags), sorted(expectedTags))

    def testPrivmsg(self):
        if False:
            for i in range(10):
                print('nop')
        self.p.privmsg('this-is-sender', 'this-is-recip', 'this is message')
        self.check(':this-is-sender PRIVMSG this-is-recip :this is message\r\n')

    def testNotice(self):
        if False:
            while True:
                i = 10
        self.p.notice('this-is-sender', 'this-is-recip', 'this is notice')
        self.check(':this-is-sender NOTICE this-is-recip :this is notice\r\n')

    def testAction(self):
        if False:
            i = 10
            return i + 15
        self.p.action('this-is-sender', 'this-is-recip', 'this is action')
        self.check(':this-is-sender ACTION this-is-recip :this is action\r\n')

    def testJoin(self):
        if False:
            return 10
        self.p.join('this-person', '#this-channel')
        self.check(':this-person JOIN #this-channel\r\n')

    def testPart(self):
        if False:
            for i in range(10):
                print('nop')
        self.p.part('this-person', '#that-channel')
        self.check(':this-person PART #that-channel\r\n')

    def testWhois(self):
        if False:
            while True:
                i = 10
        '\n        Verify that a whois by the client receives the right protocol actions\n        from the server.\n        '
        timestamp = int(time.time() - 100)
        hostname = self.p.hostname
        req = 'requesting-nick'
        targ = 'target-nick'
        self.p.whois(req, targ, 'target', 'host.com', 'Target User', 'irc.host.com', 'A fake server', False, 12, timestamp, ['#fakeusers', '#fakemisc'])
        lines = [':%(hostname)s 311 %(req)s %(targ)s target host.com * :Target User', ':%(hostname)s 312 %(req)s %(targ)s irc.host.com :A fake server', ':%(hostname)s 317 %(req)s %(targ)s 12 %(timestamp)s :seconds idle, signon time', ':%(hostname)s 319 %(req)s %(targ)s :#fakeusers #fakemisc', ':%(hostname)s 318 %(req)s %(targ)s :End of WHOIS list.', '']
        expected = '\r\n'.join(lines) % dict(hostname=hostname, timestamp=timestamp, req=req, targ=targ)
        self.check(expected)

class DummyClient(irc.IRCClient):
    """
    A L{twisted.words.protocols.irc.IRCClient} that stores sent lines in a
    C{list} rather than transmitting them.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.lines = []

    def connectionMade(self):
        if False:
            print('Hello World!')
        irc.IRCClient.connectionMade(self)
        self.lines = []

    def _truncateLine(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Truncate an IRC line to the maximum allowed length.\n        '
        return line[:irc.MAX_COMMAND_LENGTH - len(self.delimiter)]

    def lineReceived(self, line):
        if False:
            i = 10
            return i + 15
        line = self._truncateLine(line)
        return irc.IRCClient.lineReceived(self, line)

    def sendLine(self, m):
        if False:
            while True:
                i = 10
        self.lines.append(self._truncateLine(m))

class ClientInviteTests(IRCTestCase):
    """
    Tests for L{IRCClient.invite}.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a L{DummyClient} to call C{invite} on in test methods.\n        '
        self.client = DummyClient()

    def test_channelCorrection(self):
        if False:
            i = 10
            return i + 15
        '\n        If the channel name passed to L{IRCClient.invite} does not begin with a\n        channel prefix character, one is prepended to it.\n        '
        self.client.invite('foo', 'bar')
        self.assertEqual(self.client.lines, ['INVITE foo #bar'])

    def test_invite(self):
        if False:
            while True:
                i = 10
        '\n        L{IRCClient.invite} sends an I{INVITE} message with the specified\n        username and a channel.\n        '
        self.client.invite('foo', '#bar')
        self.assertEqual(self.client.lines, ['INVITE foo #bar'])

class ClientMsgTests(IRCTestCase):
    """
    Tests for messages sent with L{twisted.words.protocols.irc.IRCClient}.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.client = DummyClient()
        self.client.connectionMade()

    def test_singleLine(self):
        if False:
            return 10
        '\n        A message containing no newlines is sent in a single command.\n        '
        self.client.msg('foo', 'bar')
        self.assertEqual(self.client.lines, ['PRIVMSG foo :bar'])

    def test_invalidMaxLength(self):
        if False:
            i = 10
            return i + 15
        '\n        Specifying a C{length} value to L{IRCClient.msg} that is too short to\n        contain the protocol command to send a message raises C{ValueError}.\n        '
        self.assertRaises(ValueError, self.client.msg, 'foo', 'bar', 0)
        self.assertRaises(ValueError, self.client.msg, 'foo', 'bar', 3)

    def test_multipleLine_msg(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Messages longer than the C{length} parameter to L{IRCClient.msg} will\n        be split and sent in multiple commands.\n        '
        maxLen_msg = len('PRIVMSG foo :') + 3 + 2
        self.client.msg('foo', 'barbazbo', maxLen_msg)
        self.assertEqual(self.client.lines, ['PRIVMSG foo :bar', 'PRIVMSG foo :baz', 'PRIVMSG foo :bo'])

    def test_multipleLine_notice(self):
        if False:
            i = 10
            return i + 15
        '\n        Messages longer than the C{length} parameter to L{IRCClient.notice}\n        will be split and sent in multiple commands.\n        '
        maxLen_notice = len('NOTICE foo :') + 3 + 2
        self.client.notice('foo', 'barbazbo', maxLen_notice)
        self.assertEqual(self.client.lines, ['NOTICE foo :bar', 'NOTICE foo :baz', 'NOTICE foo :bo'])

    def test_sufficientWidth_msg(self):
        if False:
            return 10
        '\n        Messages exactly equal in length to the C{length} parameter to\n        L{IRCClient.msg} are sent in a single command.\n        '
        msg = 'barbazbo'
        maxLen = len(f'PRIVMSG foo :{msg}') + 2
        self.client.msg('foo', msg, maxLen)
        self.assertEqual(self.client.lines, [f'PRIVMSG foo :{msg}'])
        self.client.lines = []
        self.client.msg('foo', msg, maxLen - 1)
        self.assertEqual(2, len(self.client.lines))
        self.client.lines = []
        self.client.msg('foo', msg, maxLen + 1)
        self.assertEqual(1, len(self.client.lines))

    def test_sufficientWidth_notice(self):
        if False:
            while True:
                i = 10
        '\n        Messages exactly equal in length to the C{length} parameter to\n        L{IRCClient.notice} are sent in a single command.\n        '
        msg = 'barbazbo'
        maxLen = len(f'NOTICE foo :{msg}') + 2
        self.client.notice('foo', msg, maxLen)
        self.assertEqual(self.client.lines, [f'NOTICE foo :{msg}'])
        self.client.lines = []
        self.client.notice('foo', msg, maxLen - 1)
        self.assertEqual(2, len(self.client.lines))
        self.client.lines = []
        self.client.notice('foo', msg, maxLen + 1)
        self.assertEqual(1, len(self.client.lines))

    def test_newlinesAtStart_msg(self):
        if False:
            print('Hello World!')
        '\n        An LF at the beginning of the message is ignored.\n        '
        self.client.lines = []
        self.client.msg('foo', '\nbar')
        self.assertEqual(self.client.lines, ['PRIVMSG foo :bar'])

    def test_newlinesAtStart_notice(self):
        if False:
            while True:
                i = 10
        '\n        An LF at the beginning of the notice is ignored.\n        '
        self.client.lines = []
        self.client.notice('foo', '\nbar')
        self.assertEqual(self.client.lines, ['NOTICE foo :bar'])

    def test_newlinesAtEnd_msg(self):
        if False:
            print('Hello World!')
        '\n        An LF at the end of the message is ignored.\n        '
        self.client.lines = []
        self.client.msg('foo', 'bar\n')
        self.assertEqual(self.client.lines, ['PRIVMSG foo :bar'])

    def test_newlinesAtEnd_notice(self):
        if False:
            return 10
        '\n        An LF at the end of the notice is ignored.\n        '
        self.client.lines = []
        self.client.notice('foo', 'bar\n')
        self.assertEqual(self.client.lines, ['NOTICE foo :bar'])

    def test_newlinesWithinMessage_msg(self):
        if False:
            while True:
                i = 10
        '\n        An LF within a message causes a new line.\n        '
        self.client.lines = []
        self.client.msg('foo', 'bar\nbaz')
        self.assertEqual(self.client.lines, ['PRIVMSG foo :bar', 'PRIVMSG foo :baz'])

    def test_newlinesWithinMessage_notice(self):
        if False:
            return 10
        '\n        An LF within a notice causes a new line.\n        '
        self.client.lines = []
        self.client.notice('foo', 'bar\nbaz')
        self.assertEqual(self.client.lines, ['NOTICE foo :bar', 'NOTICE foo :baz'])

    def test_consecutiveNewlines_msg(self):
        if False:
            while True:
                i = 10
        '\n        Consecutive LFs in messages do not cause a blank line.\n        '
        self.client.lines = []
        self.client.msg('foo', 'bar\n\nbaz')
        self.assertEqual(self.client.lines, ['PRIVMSG foo :bar', 'PRIVMSG foo :baz'])

    def test_consecutiveNewlines_notice(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Consecutive LFs in notices do not cause a blank line.\n        '
        self.client.lines = []
        self.client.notice('foo', 'bar\n\nbaz')
        self.assertEqual(self.client.lines, ['NOTICE foo :bar', 'NOTICE foo :baz'])

    def assertLongMessageSplitting_msg(self, message, expectedNumCommands, length=None):
        if False:
            while True:
                i = 10
        '\n        Assert that messages sent by L{IRCClient.msg} are split into an\n        expected number of commands and the original message is transmitted in\n        its entirety over those commands.\n        '
        responsePrefix = ':{}!{}@{} '.format(self.client.nickname, self.client.realname, self.client.hostname)
        self.client.msg('foo', message, length=length)
        privmsg = []
        self.patch(self.client, 'privmsg', lambda *a: privmsg.append(a))
        for line in self.client.lines:
            self.client.lineReceived(responsePrefix + line)
        self.assertEqual(len(privmsg), expectedNumCommands)
        receivedMessage = ''.join((message for (user, target, message) in privmsg))
        self.assertEqual(message, receivedMessage)

    def assertLongMessageSplitting_notice(self, message, expectedNumCommands, length=None):
        if False:
            return 10
        '\n        Assert that messages sent by l{IRCClient.notice} are split into an\n        expected number of commands and the original message is transmitted in\n        its entirety over those commands.\n        '
        responsePrefix = ':{}!{}@{} '.format(self.client.nickname, self.client.realname, self.client.hostname)
        self.client.notice('foo', message, length=length)
        notice = []
        self.patch(self.client, 'noticed', lambda *a: notice.append(a))
        for line in self.client.lines:
            self.client.lineReceived(responsePrefix + line)
        self.assertEqual(len(notice), expectedNumCommands)
        receivedMessage = ''.join((message for (user, target, message) in notice))
        self.assertEqual(message, receivedMessage)

    def test_splitLongMessagesWithDefault_msg(self):
        if False:
            while True:
                i = 10
        '\n        If a maximum message length is not provided to L{IRCClient.msg} a\n        best-guess effort is made to determine a safe maximum,  messages longer\n        than this are split into multiple commands with the intent of\n        delivering long messages without losing data due to message truncation\n        when the server relays them.\n        '
        message = 'o' * (irc.MAX_COMMAND_LENGTH - 2)
        self.assertLongMessageSplitting_msg(message, 2)

    def test_splitLongMessagesWithDefault_notice(self):
        if False:
            print('Hello World!')
        '\n        If a maximum message length is not provided to L{IRCClient.notice} a\n        best-guess effort is made to determine a safe maximum,  messages longer\n        than this are split into multiple commands with the intent of\n        delivering long messages without losing data due to message truncation\n        when the server relays them.\n        '
        message = 'o' * (irc.MAX_COMMAND_LENGTH - 2)
        self.assertLongMessageSplitting_notice(message, 2)

    def test_splitLongMessagesWithOverride_msg(self):
        if False:
            print('Hello World!')
        '\n        The maximum message length can be specified to L{IRCClient.msg},\n        messages longer than this are split into multiple commands with the\n        intent of delivering long messages without losing data due to message\n        truncation when the server relays them.\n        '
        message = 'o' * (irc.MAX_COMMAND_LENGTH - 2)
        self.assertLongMessageSplitting_msg(message, 3, length=irc.MAX_COMMAND_LENGTH // 2)

    def test_splitLongMessagesWithOverride_notice(self):
        if False:
            return 10
        '\n        The maximum message length can be specified to L{IRCClient.notice},\n        messages longer than this are split into multiple commands with the\n        intent of delivering long messages without losing data due to message\n        truncation when the server relays them.\n        '
        message = 'o' * (irc.MAX_COMMAND_LENGTH - 2)
        self.assertLongMessageSplitting_notice(message, 3, length=irc.MAX_COMMAND_LENGTH // 2)

    def test_newlinesBeforeLineBreaking(self):
        if False:
            i = 10
            return i + 15
        '\n        IRCClient breaks on newlines before it breaks long lines.\n        '
        longline = 'o' * (irc.MAX_COMMAND_LENGTH // 2)
        self.client.msg('foo', longline + '\n' + longline)
        self.assertEqual(self.client.lines, ['PRIVMSG foo :' + longline, 'PRIVMSG foo :' + longline])

    def test_lineBreakOnWordBoundaries(self):
        if False:
            i = 10
            return i + 15
        '\n        IRCClient prefers to break long lines at word boundaries.\n        '
        longline = 'o' * (irc.MAX_COMMAND_LENGTH // 2)
        self.client.msg('foo', longline + ' ' + longline)
        self.assertEqual(self.client.lines, ['PRIVMSG foo :' + longline, 'PRIVMSG foo :' + longline])

    def test_splitSanity(self):
        if False:
            print('Hello World!')
        "\n        L{twisted.words.protocols.irc.split} raises C{ValueError} if given a\n        length less than or equal to C{0} and returns C{[]} when splitting\n        C{''}.\n        "
        self.assertRaises(ValueError, irc.split, 'foo', -1)
        self.assertRaises(ValueError, irc.split, 'foo', 0)
        self.assertEqual([], irc.split('', 1))
        self.assertEqual([], irc.split(''))

    def test_splitDelimiters(self):
        if False:
            while True:
                i = 10
        '\n        L{twisted.words.protocols.irc.split} skips any delimiter (space or\n        newline) that it finds at the very beginning of the string segment it\n        is operating on.  Nothing should be added to the output list because of\n        it.\n        '
        r = irc.split('xx yyz', 2)
        self.assertEqual(['xx', 'yy', 'z'], r)
        r = irc.split('xx\nyyz', 2)
        self.assertEqual(['xx', 'yy', 'z'], r)

    def test_splitValidatesLength(self):
        if False:
            i = 10
            return i + 15
        '\n        L{twisted.words.protocols.irc.split} raises C{ValueError} if given a\n        length less than or equal to C{0}.\n        '
        self.assertRaises(ValueError, irc.split, 'foo', 0)
        self.assertRaises(ValueError, irc.split, 'foo', -1)

    def test_say(self):
        if False:
            return 10
        '\n        L{IRCClient.say} prepends the channel prefix C{"#"} if necessary and\n        then sends the message to the server for delivery to that channel.\n        '
        self.client.say('thechannel', 'the message')
        self.assertEqual(self.client.lines, ['PRIVMSG #thechannel :the message'])

class ClientTests(IRCTestCase):
    """
    Tests for the protocol-level behavior of IRCClient methods intended to
    be called by application code.
    """

    def setUp(self):
        if False:
            return 10
        '\n        Create and connect a new L{IRCClient} to a new L{StringTransport}.\n        '
        self.transport = StringTransport()
        self.protocol = IRCClient()
        self.protocol.performLogin = False
        self.protocol.makeConnection(self.transport)
        self.assertEqualBufferValue(self.transport.value(), '')
        self.addCleanup(self.transport.loseConnection)
        self.addCleanup(self.protocol.connectionLost, None)

    def getLastLine(self, transport):
        if False:
            while True:
                i = 10
        '\n        Return the last IRC message in the transport buffer.\n        '
        line = transport.value()
        if bytes != str and isinstance(line, bytes):
            line = line.decode('utf-8')
        return line.split('\r\n')[-2]

    def test_away(self):
        if False:
            print('Hello World!')
        '\n        L{IRCClient.away} sends an AWAY command with the specified message.\n        '
        message = "Sorry, I'm not here."
        self.protocol.away(message)
        expected = [f'AWAY :{message}', '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_back(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.back} sends an AWAY command with an empty message.\n        '
        self.protocol.back()
        expected = ['AWAY :', '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_whois(self):
        if False:
            print('Hello World!')
        '\n        L{IRCClient.whois} sends a WHOIS message.\n        '
        self.protocol.whois('alice')
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), ['WHOIS alice', ''])

    def test_whoisWithServer(self):
        if False:
            return 10
        '\n        L{IRCClient.whois} sends a WHOIS message with a server name if a\n        value is passed for the C{server} parameter.\n        '
        self.protocol.whois('alice', 'example.org')
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), ['WHOIS example.org alice', ''])

    def test_register(self):
        if False:
            print('Hello World!')
        '\n        L{IRCClient.register} sends NICK and USER commands with the\n        username, name, hostname, server name, and real name specified.\n        '
        username = 'testuser'
        hostname = 'testhost'
        servername = 'testserver'
        self.protocol.realname = 'testname'
        self.protocol.password = None
        self.protocol.register(username, hostname, servername)
        expected = [f'NICK {username}', 'USER %s %s %s :%s' % (username, hostname, servername, self.protocol.realname), '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_registerWithPassword(self):
        if False:
            print('Hello World!')
        '\n        If the C{password} attribute of L{IRCClient} is not L{None}, the\n        C{register} method also sends a PASS command with it as the\n        argument.\n        '
        username = 'testuser'
        hostname = 'testhost'
        servername = 'testserver'
        self.protocol.realname = 'testname'
        self.protocol.password = 'testpass'
        self.protocol.register(username, hostname, servername)
        expected = [f'PASS {self.protocol.password}', f'NICK {username}', 'USER %s %s %s :%s' % (username, hostname, servername, self.protocol.realname), '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_registerWithTakenNick(self):
        if False:
            while True:
                i = 10
        '\n        Verify that the client repeats the L{IRCClient.setNick} method with a\n        new value when presented with an C{ERR_NICKNAMEINUSE} while trying to\n        register.\n        '
        username = 'testuser'
        hostname = 'testhost'
        servername = 'testserver'
        self.protocol.realname = 'testname'
        self.protocol.password = 'testpass'
        self.protocol.register(username, hostname, servername)
        self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertNotEqual(lastLine, f'NICK {username}')
        self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, 'NICK {}'.format(username + '__'))

    def test_overrideAlterCollidedNick(self):
        if False:
            return 10
        '\n        L{IRCClient.alterCollidedNick} determines how a nickname is altered upon\n        collision while a user is trying to change to that nickname.\n        '
        nick = 'foo'
        self.protocol.alterCollidedNick = lambda nick: nick + '***'
        self.protocol.register(nick)
        self.protocol.irc_ERR_NICKNAMEINUSE('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, 'NICK {}'.format(nick + '***'))

    def test_nickChange(self):
        if False:
            while True:
                i = 10
        '\n        When a NICK command is sent after signon, C{IRCClient.nickname} is set\n        to the new nickname I{after} the server sends an acknowledgement.\n        '
        oldnick = 'foo'
        newnick = 'bar'
        self.protocol.register(oldnick)
        self.protocol.irc_RPL_WELCOME('prefix', ['param'])
        self.protocol.setNick(newnick)
        self.assertEqual(self.protocol.nickname, oldnick)
        self.protocol.irc_NICK(f'{oldnick}!quux@qux', [newnick])
        self.assertEqual(self.protocol.nickname, newnick)

    def test_erroneousNick(self):
        if False:
            while True:
                i = 10
        '\n        Trying to register an illegal nickname results in the default legal\n        nickname being set, and trying to change a nickname to an illegal\n        nickname results in the old nickname being kept.\n        '
        badnick = 'foo'
        self.assertEqual(self.protocol._registered, False)
        self.protocol.register(badnick)
        self.protocol.irc_ERR_ERRONEUSNICKNAME('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, f'NICK {self.protocol.erroneousNickFallback}')
        self.protocol.irc_RPL_WELCOME('prefix', ['param'])
        self.assertEqual(self.protocol._registered, True)
        self.protocol.setNick(self.protocol.erroneousNickFallback)
        self.assertEqual(self.protocol.nickname, self.protocol.erroneousNickFallback)
        oldnick = self.protocol.nickname
        self.protocol.setNick(badnick)
        self.protocol.irc_ERR_ERRONEUSNICKNAME('prefix', ['param'])
        lastLine = self.getLastLine(self.transport)
        self.assertEqual(lastLine, f'NICK {badnick}')
        self.assertEqual(self.protocol.nickname, oldnick)

    def test_describe(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.desrcibe} sends a CTCP ACTION message to the target\n        specified.\n        '
        target = 'foo'
        channel = '#bar'
        action = 'waves'
        self.protocol.describe(target, action)
        self.protocol.describe(channel, action)
        expected = [f'PRIVMSG {target} :\x01ACTION {action}\x01', f'PRIVMSG {channel} :\x01ACTION {action}\x01', '']
        self.assertEqualBufferValue(self.transport.value().split(b'\r\n'), expected)

    def test_noticedDoesntPrivmsg(self):
        if False:
            while True:
                i = 10
        "\n        The default implementation of L{IRCClient.noticed} doesn't invoke\n        C{privmsg()}\n        "

        def privmsg(user, channel, message):
            if False:
                i = 10
                return i + 15
            self.fail('privmsg() should not have been called')
        self.protocol.privmsg = privmsg
        self.protocol.irc_NOTICE('spam', ['#greasyspooncafe', "I don't want any spam!"])

    def test_ping(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{IRCClient.ping}\n        '
        self.protocol.ping('otheruser')
        self.assertTrue(self.transport.value().startswith(b'PRIVMSG otheruser :\x01PING'))
        self.transport.clear()
        self.protocol.ping('otheruser', 'are you there')
        self.assertEqual(self.transport.value(), b'PRIVMSG otheruser :\x01PING are you there\x01\r\n')
        self.transport.clear()
        self.protocol._pings = {}
        for pingNum in range(self.protocol._MAX_PINGRING + 3):
            self.protocol._pings['otheruser', str(pingNum)] = time.time() + pingNum
        self.assertEqual(len(self.protocol._pings), self.protocol._MAX_PINGRING + 3)
        self.protocol.ping('otheruser', 'I sent a lot of pings')
        self.assertEqual(len(self.protocol._pings), self.protocol._MAX_PINGRING)
        self.assertEqual(self.transport.value(), b'PRIVMSG otheruser :\x01PING I sent a lot of pings\x01\r\n')

class CollectorClient(irc.IRCClient):
    """
    A client that saves in a list the names of the methods that got called.
    """

    def __init__(self, methodsList):
        if False:
            while True:
                i = 10
        "\n        @param methodsList: list of methods' names that should be replaced.\n        @type methodsList: C{list}\n        "
        self.methods = []
        self.nickname = 'Wolf'
        for method in methodsList:

            def fake_method(method=method):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Collects C{method}s.\n                '

                def inner(*args):
                    if False:
                        while True:
                            i = 10
                    self.methods.append((method, args))
                return inner
            setattr(self, method, fake_method())

class DccTests(IRCTestCase):
    """
    Tests for C{dcc_*} methods.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        methods = ['dccDoSend', 'dccDoAcceptResume', 'dccDoResume', 'dccDoChat']
        self.user = 'Wolf!~wolf@yok.utu.fi'
        self.channel = '#twisted'
        self.client = CollectorClient(methods)

    def test_dccSend(self):
        if False:
            return 10
        '\n        L{irc.IRCClient.dcc_SEND} invokes L{irc.IRCClient.dccDoSend}.\n        '
        self.client.dcc_SEND(self.user, self.channel, 'foo.txt 127.0.0.1 1025')
        self.assertEqual(self.client.methods, [('dccDoSend', (self.user, '127.0.0.1', 1025, 'foo.txt', -1, ['foo.txt', '127.0.0.1', '1025']))])

    def test_dccSendNotImplemented(self):
        if False:
            return 10
        '\n        L{irc.IRCClient.dccDoSend} is raises C{NotImplementedError}\n        '
        client = irc.IRCClient()
        self.assertRaises(NotImplementedError, client.dccSend, 'username', None)

    def test_dccSendMalformedRequest(self):
        if False:
            while True:
                i = 10
        '\n        L{irc.IRCClient.dcc_SEND} raises L{irc.IRCBadMessage} when it is passed\n        a malformed query string.\n        '
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_SEND, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC SEND request: ['foo']")

    def test_dccSendIndecipherableAddress(self):
        if False:
            return 10
        "\n        L{irc.IRCClient.dcc_SEND} raises L{irc.IRCBadMessage} when it is passed\n        a query string that doesn't contain a valid address.\n        "
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_SEND, self.user, self.channel, 'foo.txt #23 sd@d')
        self.assertEqual(str(result), "Indecipherable address '#23'")

    def test_dccSendIndecipherablePort(self):
        if False:
            print('Hello World!')
        "\n        L{irc.IRCClient.dcc_SEND} raises L{irc.IRCBadMessage} when it is passed\n        a query string that doesn't contain a valid port number.\n        "
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_SEND, self.user, self.channel, 'foo.txt 127.0.0.1 sd@d')
        self.assertEqual(str(result), "Indecipherable port 'sd@d'")

    def test_dccAccept(self):
        if False:
            print('Hello World!')
        '\n        L{irc.IRCClient.dcc_ACCEPT} invokes L{irc.IRCClient.dccDoAcceptResume}.\n        '
        self.client.dcc_ACCEPT(self.user, self.channel, 'foo.txt 1025 2')
        self.assertEqual(self.client.methods, [('dccDoAcceptResume', (self.user, 'foo.txt', 1025, 2))])

    def test_dccAcceptMalformedRequest(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{irc.IRCClient.dcc_ACCEPT} raises L{irc.IRCBadMessage} when it is\n        passed a malformed query string.\n        '
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_ACCEPT, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC SEND ACCEPT request: ['foo']")

    def test_dccResume(self):
        if False:
            while True:
                i = 10
        '\n        L{irc.IRCClient.dcc_RESUME} invokes L{irc.IRCClient.dccDoResume}.\n        '
        self.client.dcc_RESUME(self.user, self.channel, 'foo.txt 1025 2')
        self.assertEqual(self.client.methods, [('dccDoResume', (self.user, 'foo.txt', 1025, 2))])

    def test_dccResumeMalformedRequest(self):
        if False:
            i = 10
            return i + 15
        '\n        L{irc.IRCClient.dcc_RESUME} raises L{irc.IRCBadMessage} when it is\n        passed a malformed query string.\n        '
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_RESUME, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC SEND RESUME request: ['foo']")

    def test_dccChat(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{irc.IRCClient.dcc_CHAT} invokes L{irc.IRCClient.dccDoChat}.\n        '
        self.client.dcc_CHAT(self.user, self.channel, 'foo.txt 127.0.0.1 1025')
        self.assertEqual(self.client.methods, [('dccDoChat', (self.user, self.channel, '127.0.0.1', 1025, ['foo.txt', '127.0.0.1', '1025']))])

    def test_dccChatMalformedRequest(self):
        if False:
            return 10
        '\n        L{irc.IRCClient.dcc_CHAT} raises L{irc.IRCBadMessage} when it is\n        passed a malformed query string.\n        '
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_CHAT, self.user, self.channel, 'foo')
        self.assertEqual(str(result), "malformed DCC CHAT request: ['foo']")

    def test_dccChatIndecipherablePort(self):
        if False:
            return 10
        "\n        L{irc.IRCClient.dcc_CHAT} raises L{irc.IRCBadMessage} when it is passed\n        a query string that doesn't contain a valid port number.\n        "
        result = self.assertRaises(irc.IRCBadMessage, self.client.dcc_CHAT, self.user, self.channel, 'foo.txt 127.0.0.1 sd@d')
        self.assertEqual(str(result), "Indecipherable port 'sd@d'")

class ServerToClientTests(IRCTestCase):
    """
    Tests for the C{irc_*} methods sent from the server to the client.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.user = 'Wolf!~wolf@yok.utu.fi'
        self.channel = '#twisted'
        methods = ['joined', 'userJoined', 'left', 'userLeft', 'userQuit', 'noticed', 'kickedFrom', 'userKicked', 'topicUpdated']
        self.client = CollectorClient(methods)

    def test_irc_JOIN(self):
        if False:
            return 10
        '\n        L{IRCClient.joined} is called when I join a channel;\n        L{IRCClient.userJoined} is called when someone else joins.\n        '
        self.client.irc_JOIN(self.user, [self.channel])
        self.client.irc_JOIN('Svadilfari!~svadi@yok.utu.fi', ['#python'])
        self.assertEqual(self.client.methods, [('joined', (self.channel,)), ('userJoined', ('Svadilfari', '#python'))])

    def test_irc_PART(self):
        if False:
            print('Hello World!')
        '\n        L{IRCClient.left} is called when I part the channel;\n        L{IRCClient.userLeft} is called when someone else parts.\n        '
        self.client.irc_PART(self.user, [self.channel])
        self.client.irc_PART('Svadilfari!~svadi@yok.utu.fi', ['#python'])
        self.assertEqual(self.client.methods, [('left', (self.channel,)), ('userLeft', ('Svadilfari', '#python'))])

    def test_irc_QUIT(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.userQuit} is called whenever someone quits\n        the channel (myself included).\n        '
        self.client.irc_QUIT('Svadilfari!~svadi@yok.utu.fi', ['Adios.'])
        self.client.irc_QUIT(self.user, ['Farewell.'])
        self.assertEqual(self.client.methods, [('userQuit', ('Svadilfari', 'Adios.')), ('userQuit', ('Wolf', 'Farewell.'))])

    def test_irc_NOTICE(self):
        if False:
            return 10
        '\n        L{IRCClient.noticed} is called when a notice is received.\n        '
        msg = '%(X)cextended%(X)cdata1%(X)cextended%(X)cdata2%(X)c%(EOL)s' % {'X': irc.X_DELIM, 'EOL': irc.CR + irc.LF}
        self.client.irc_NOTICE(self.user, [self.channel, msg])
        self.assertEqual(self.client.methods, [('noticed', (self.user, '#twisted', 'data1 data2'))])

    def test_irc_KICK(self):
        if False:
            return 10
        '\n        L{IRCClient.kickedFrom} is called when I get kicked from the channel;\n        L{IRCClient.userKicked} is called when someone else gets kicked.\n        '
        self.client.irc_KICK('Svadilfari!~svadi@yok.utu.fi', ['#python', 'WOLF', 'shoryuken!'])
        self.client.irc_KICK(self.user, [self.channel, 'Svadilfari', 'hadouken!'])
        self.assertEqual(self.client.methods, [('kickedFrom', ('#python', 'Svadilfari', 'shoryuken!')), ('userKicked', ('Svadilfari', self.channel, 'Wolf', 'hadouken!'))])

    def test_irc_TOPIC(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.topicUpdated} is called when someone sets the topic.\n        '
        self.client.irc_TOPIC(self.user, [self.channel, 'new topic is new'])
        self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, 'new topic is new'))])

    def test_irc_RPL_TOPIC(self):
        if False:
            print('Hello World!')
        '\n        L{IRCClient.topicUpdated} is called when the topic is initially\n        reported.\n        '
        self.client.irc_RPL_TOPIC(self.user, ['?', self.channel, 'new topic is new'])
        self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, 'new topic is new'))])

    def test_irc_RPL_NOTOPIC(self):
        if False:
            print('Hello World!')
        '\n        L{IRCClient.topicUpdated} is called when the topic is removed.\n        '
        self.client.irc_RPL_NOTOPIC(self.user, ['?', self.channel])
        self.assertEqual(self.client.methods, [('topicUpdated', ('Wolf', self.channel, ''))])

class CTCPQueryTests(IRCTestCase):
    """
    Tests for the C{ctcpQuery_*} methods.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.user = 'Wolf!~wolf@yok.utu.fi'
        self.channel = '#twisted'
        self.client = CollectorClient(['ctcpMakeReply'])

    def test_ctcpQuery_PING(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{IRCClient.ctcpQuery_PING} calls L{IRCClient.ctcpMakeReply} with the\n        correct args.\n        '
        self.client.ctcpQuery_PING(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('PING', 'data')]))])

    def test_ctcpQuery_FINGER(self):
        if False:
            print('Hello World!')
        '\n        L{IRCClient.ctcpQuery_FINGER} calls L{IRCClient.ctcpMakeReply} with the\n        correct args.\n        '
        self.client.fingerReply = 'reply'
        self.client.ctcpQuery_FINGER(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('FINGER', 'reply')]))])

    def test_ctcpQuery_SOURCE(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.ctcpQuery_SOURCE} calls L{IRCClient.ctcpMakeReply} with the\n        correct args.\n        '
        self.client.sourceURL = 'url'
        self.client.ctcpQuery_SOURCE(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('SOURCE', 'url'), ('SOURCE', None)]))])

    def test_ctcpQuery_USERINFO(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.ctcpQuery_USERINFO} calls L{IRCClient.ctcpMakeReply} with\n        the correct args.\n        '
        self.client.userinfo = 'info'
        self.client.ctcpQuery_USERINFO(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('USERINFO', 'info')]))])

    def test_ctcpQuery_CLIENTINFO(self):
        if False:
            while True:
                i = 10
        '\n        L{IRCClient.ctcpQuery_CLIENTINFO} calls L{IRCClient.ctcpMakeReply} with\n        the correct args.\n        '
        self.client.ctcpQuery_CLIENTINFO(self.user, self.channel, '')
        self.client.ctcpQuery_CLIENTINFO(self.user, self.channel, 'PING PONG')
        info = 'ACTION CLIENTINFO DCC ERRMSG FINGER PING SOURCE TIME USERINFO VERSION'
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('CLIENTINFO', info)])), ('ctcpMakeReply', ('Wolf', [('CLIENTINFO', None)]))])

    def test_ctcpQuery_TIME(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.ctcpQuery_TIME} calls L{IRCClient.ctcpMakeReply} with the\n        correct args.\n        '
        self.client.ctcpQuery_TIME(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods[0][1][0], 'Wolf')

    def test_ctcpQuery_DCC(self):
        if False:
            i = 10
            return i + 15
        '\n        L{IRCClient.ctcpQuery_DCC} calls L{IRCClient.ctcpMakeReply} with the\n        correct args.\n        '
        self.client.ctcpQuery_DCC(self.user, self.channel, 'data')
        self.assertEqual(self.client.methods, [('ctcpMakeReply', ('Wolf', [('ERRMSG', "DCC data :Unknown DCC type 'DATA'")]))])

class DccChatFactoryTests(IRCTestCase):
    """
    Tests for L{DccChatFactory}.
    """

    def test_buildProtocol(self):
        if False:
            return 10
        '\n        An instance of the L{irc.DccChat} protocol is returned, which has the\n        factory property set to the factory which created it.\n        '
        queryData = ('fromUser', None, None)
        factory = irc.DccChatFactory(None, queryData)
        protocol = factory.buildProtocol('127.0.0.1')
        self.assertIsInstance(protocol, irc.DccChat)
        self.assertEqual(protocol.factory, factory)

class DccDescribeTests(IRCTestCase):
    """
    Tests for L{dccDescribe}.
    """

    def test_address(self):
        if False:
            print('Hello World!')
        '\n        L{irc.dccDescribe} supports long IP addresses.\n        '
        result = irc.dccDescribe('CHAT arg 3232235522 6666')
        self.assertEqual(result, 'CHAT for host 192.168.0.2, port 6666')

class DccFileReceiveTests(IRCTestCase):
    """
    Tests for L{DccFileReceive}.
    """

    def makeConnectedDccFileReceive(self, filename, resumeOffset=0, overwrite=None):
        if False:
            i = 10
            return i + 15
        '\n        Factory helper that returns a L{DccFileReceive} instance\n        for a specific test case.\n\n        @param filename: Path to the local file where received data is stored.\n        @type filename: L{str}\n\n        @param resumeOffset: An integer representing the amount of bytes from\n            where the transfer of data should be resumed.\n        @type resumeOffset: L{int}\n\n        @param overwrite: A boolean specifying whether the file to write to\n            should be overwritten by calling L{DccFileReceive.set_overwrite}\n            or not.\n        @type overwrite: L{bool}\n\n        @return: An instance of L{DccFileReceive}.\n        @rtype: L{DccFileReceive}\n        '
        protocol = irc.DccFileReceive(filename, resumeOffset=resumeOffset)
        if overwrite:
            protocol.set_overwrite(True)
        transport = StringTransport()
        protocol.makeConnection(transport)
        return protocol

    def allDataReceivedForProtocol(self, protocol, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Arrange the protocol so that it received all data.\n\n        @param protocol: The protocol which will receive the data.\n        @type: L{DccFileReceive}\n\n        @param data: The received data.\n        @type data: L{bytest}\n        '
        protocol.dataReceived(data)
        protocol.connectionLost(None)

    def test_resumeFromResumeOffset(self):
        if False:
            print('Hello World!')
        '\n        If given a resumeOffset argument, L{DccFileReceive} will attempt to\n        resume from that number of bytes if the file exists.\n        '
        fp = FilePath(self.mktemp())
        fp.setContent(b'Twisted is awesome!')
        protocol = self.makeConnectedDccFileReceive(fp.path, resumeOffset=11)
        self.allDataReceivedForProtocol(protocol, b'amazing!')
        self.assertEqual(fp.getContent(), b'Twisted is amazing!')

    def test_resumeFromResumeOffsetInTheMiddleOfAlreadyWrittenData(self):
        if False:
            while True:
                i = 10
        '\n        When resuming from an offset somewhere in the middle of the file,\n        for example, if there are 50 bytes in a file, and L{DccFileReceive}\n        is given a resumeOffset of 25, and after that 15 more bytes are\n        written to the file, then the resultant file should have just 40\n        bytes of data.\n        '
        fp = FilePath(self.mktemp())
        fp.setContent(b'Twisted is amazing!')
        protocol = self.makeConnectedDccFileReceive(fp.path, resumeOffset=11)
        self.allDataReceivedForProtocol(protocol, b'cool!')
        self.assertEqual(fp.getContent(), b'Twisted is cool!')

    def test_setOverwrite(self):
        if False:
            print('Hello World!')
        '\n        When local file already exists it can be overwritten using the\n        L{DccFileReceive.set_overwrite} method.\n        '
        fp = FilePath(self.mktemp())
        fp.setContent(b'I love contributing to Twisted!')
        protocol = self.makeConnectedDccFileReceive(fp.path, overwrite=True)
        self.allDataReceivedForProtocol(protocol, b'Twisted rocks!')
        self.assertEqual(fp.getContent(), b'Twisted rocks!')

    def test_fileDoesNotExist(self):
        if False:
            print('Hello World!')
        '\n        If the file does not already exist, then L{DccFileReceive} will\n        create one and write the data to it.\n        '
        fp = FilePath(self.mktemp())
        protocol = self.makeConnectedDccFileReceive(fp.path)
        self.allDataReceivedForProtocol(protocol, b'I <3 Twisted')
        self.assertEqual(fp.getContent(), b'I <3 Twisted')

    def test_resumeWhenFileDoesNotExist(self):
        if False:
            while True:
                i = 10
        '\n        If given a resumeOffset to resume writing to a file that does not\n        exist, L{DccFileReceive} will raise L{OSError}.\n        '
        fp = FilePath(self.mktemp())
        error = self.assertRaises(OSError, self.makeConnectedDccFileReceive, fp.path, resumeOffset=1)
        self.assertEqual(errno.ENOENT, error.errno)

    def test_fileAlreadyExistsNoOverwrite(self):
        if False:
            while True:
                i = 10
        '\n        If the file already exists and overwrite action was not asked,\n        L{OSError} is raised.\n        '
        fp = FilePath(self.mktemp())
        fp.touch()
        self.assertRaises(OSError, self.makeConnectedDccFileReceive, fp.path)

    def test_failToOpenLocalFile(self):
        if False:
            print('Hello World!')
        '\n        L{IOError} is raised when failing to open the requested path.\n        '
        fp = FilePath(self.mktemp()).child('child-with-no-existing-parent')
        self.assertRaises(IOError, self.makeConnectedDccFileReceive, fp.path)