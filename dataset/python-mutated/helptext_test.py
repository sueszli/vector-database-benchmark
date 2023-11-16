"""Tests for the helptext module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import textwrap
from fire import formatting
from fire import helptext
from fire import test_components as tc
from fire import testutils
from fire import trace
import six

class HelpTest(testutils.BaseTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(HelpTest, self).setUp()
        os.environ['ANSI_COLORS_DISABLED'] = '1'

    def testHelpTextNoDefaults(self):
        if False:
            i = 10
            return i + 15
        component = tc.NoDefaults
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='NoDefaults'))
        self.assertIn('NAME\n    NoDefaults', help_screen)
        self.assertIn('SYNOPSIS\n    NoDefaults', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextNoDefaultsObject(self):
        if False:
            print('Hello World!')
        component = tc.NoDefaults()
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='NoDefaults'))
        self.assertIn('NAME\n    NoDefaults', help_screen)
        self.assertIn('SYNOPSIS\n    NoDefaults COMMAND', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:', help_screen)
        self.assertIn('double', help_screen)
        self.assertIn('triple', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunction(self):
        if False:
            while True:
                i = 10
        component = tc.NoDefaults().double
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='double'))
        self.assertIn('NAME\n    double', help_screen)
        self.assertIn('SYNOPSIS\n    double COUNT', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('POSITIONAL ARGUMENTS\n    COUNT', help_screen)
        self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)

    def testHelpTextFunctionWithDefaults(self):
        if False:
            return 10
        component = tc.WithDefaults().triple
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='triple'))
        self.assertIn('NAME\n    triple', help_screen)
        self.assertIn('SYNOPSIS\n    triple <flags>', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('FLAGS\n    -c, --count=COUNT\n        Default: 0', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunctionWithLongDefaults(self):
        if False:
            for i in range(10):
                print('nop')
        component = tc.WithDefaults().text
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='text'))
        self.assertIn('NAME\n    text', help_screen)
        self.assertIn('SYNOPSIS\n    text <flags>', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn("FLAGS\n    -s, --string=STRING\n        Default: '00010203040506070809101112131415161718192021222324252627282...", help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunctionWithKwargs(self):
        if False:
            i = 10
            return i + 15
        component = tc.fn_with_kwarg
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='text'))
        self.assertIn('NAME\n    text', help_screen)
        self.assertIn('SYNOPSIS\n    text ARG1 ARG2 <flags>', help_screen)
        self.assertIn('DESCRIPTION\n    Function with kwarg', help_screen)
        self.assertIn('FLAGS\n    --arg3\n        Description of arg3.\n    Additional undocumented flags may also be accepted.', help_screen)

    def testHelpTextFunctionWithKwargsAndDefaults(self):
        if False:
            return 10
        component = tc.fn_with_kwarg_and_defaults
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='text'))
        self.assertIn('NAME\n    text', help_screen)
        self.assertIn('SYNOPSIS\n    text ARG1 ARG2 <flags>', help_screen)
        self.assertIn('DESCRIPTION\n    Function with kwarg', help_screen)
        self.assertIn('FLAGS\n    -o, --opt=OPT\n        Default: True\n    The following flags are also accepted.\n    --arg3\n        Description of arg3.\n    Additional undocumented flags may also be accepted.', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithDefaultsAndTypes(self):
        if False:
            print('Hello World!')
        component = tc.py3.WithDefaultsAndTypes().double
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='double'))
        self.assertIn('NAME\n    double', help_screen)
        self.assertIn('SYNOPSIS\n    double <flags>', help_screen)
        self.assertIn('DESCRIPTION', help_screen)
        self.assertIn('FLAGS\n    -c, --count=COUNT\n        Type: float\n        Default: 0', help_screen)
        self.assertNotIn('NOTES', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithTypesAndDefaultNone(self):
        if False:
            return 10
        component = tc.py3.WithDefaultsAndTypes().get_int
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='get_int'))
        self.assertIn('NAME\n    get_int', help_screen)
        self.assertIn('SYNOPSIS\n    get_int <flags>', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('FLAGS\n    -v, --value=VALUE\n        Type: Optional[int]\n        Default: None', help_screen)
        self.assertNotIn('NOTES', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithTypes(self):
        if False:
            for i in range(10):
                print('nop')
        component = tc.py3.WithTypes().double
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='double'))
        self.assertIn('NAME\n    double', help_screen)
        self.assertIn('SYNOPSIS\n    double COUNT', help_screen)
        self.assertIn('DESCRIPTION', help_screen)
        self.assertIn('POSITIONAL ARGUMENTS\n    COUNT\n        Type: float', help_screen)
        self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)

    @testutils.skipIf(sys.version_info[0:2] < (3, 5), 'Python < 3.5 does not support type hints.')
    def testHelpTextFunctionWithLongTypes(self):
        if False:
            for i in range(10):
                print('nop')
        component = tc.py3.WithTypes().long_type
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, name='long_type'))
        self.assertIn('NAME\n    long_type', help_screen)
        self.assertIn('SYNOPSIS\n    long_type LONG_OBJ', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('NOTES\n    You can also use flags syntax for POSITIONAL ARGUMENTS', help_screen)

    def testHelpTextFunctionWithBuiltin(self):
        if False:
            i = 10
            return i + 15
        component = 'test'.upper
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'upper'))
        self.assertIn('NAME\n    upper', help_screen)
        self.assertIn('SYNOPSIS\n    upper', help_screen)
        self.assertIn('DESCRIPTION\n', help_screen)
        self.assertNotIn('NOTES', help_screen)

    def testHelpTextFunctionIntType(self):
        if False:
            i = 10
            return i + 15
        component = int
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'int'))
        self.assertIn('NAME\n    int', help_screen)
        self.assertIn('SYNOPSIS\n    int', help_screen)
        self.assertIn('DESCRIPTION\n', help_screen)

    def testHelpTextEmptyList(self):
        if False:
            return 10
        component = []
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'list'))
        self.assertIn('NAME\n    list', help_screen)
        self.assertIn('SYNOPSIS\n    list COMMAND', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:\n', help_screen)

    def testHelpTextShortList(self):
        if False:
            print('Hello World!')
        component = [10]
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'list'))
        self.assertIn('NAME\n    list', help_screen)
        self.assertIn('SYNOPSIS\n    list COMMAND', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:\n', help_screen)
        self.assertIn('     append\n', help_screen)

    def testHelpTextInt(self):
        if False:
            print('Hello World!')
        component = 7
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, '7'))
        self.assertIn('NAME\n    7', help_screen)
        self.assertIn('SYNOPSIS\n    7 COMMAND | VALUE', help_screen)
        self.assertNotIn('DESCRIPTION', help_screen)
        self.assertIn('COMMANDS\n    COMMAND is one of the following:\n', help_screen)
        self.assertIn('VALUES\n    VALUE is one of the following:\n', help_screen)

    def testHelpTextNoInit(self):
        if False:
            return 10
        component = tc.OldStyleEmpty
        help_screen = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'OldStyleEmpty'))
        self.assertIn('NAME\n    OldStyleEmpty', help_screen)
        self.assertIn('SYNOPSIS\n    OldStyleEmpty', help_screen)

    @testutils.skipIf(six.PY2, 'Python 2 does not support keyword-only arguments.')
    def testHelpTextKeywordOnlyArgumentsWithDefault(self):
        if False:
            print('Hello World!')
        component = tc.py3.KeywordOnly.with_default
        output = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'with_default'))
        self.assertIn('NAME\n    with_default', output)
        self.assertIn('FLAGS\n    -x, --x=X', output)

    @testutils.skipIf(six.PY2, 'Python 2 does not support keyword-only arguments.')
    def testHelpTextKeywordOnlyArgumentsWithoutDefault(self):
        if False:
            while True:
                i = 10
        component = tc.py3.KeywordOnly.double
        output = helptext.HelpText(component=component, trace=trace.FireTrace(component, 'double'))
        self.assertIn('NAME\n    double', output)
        self.assertIn('FLAGS\n    -c, --count=COUNT (required)', output)

    @testutils.skipIf(six.PY2, 'Python 2 does not support required name-only arguments.')
    def testHelpTextFunctionMixedDefaults(self):
        if False:
            while True:
                i = 10
        component = tc.py3.HelpTextComponent().identity
        t = trace.FireTrace(component, name='FunctionMixedDefaults')
        output = helptext.HelpText(component, trace=t)
        self.assertIn('NAME\n    FunctionMixedDefaults', output)
        self.assertIn('FunctionMixedDefaults <flags>', output)
        self.assertIn('--alpha=ALPHA (required)', output)
        self.assertIn("--beta=BETA\n        Default: '0'", output)

    def testHelpScreen(self):
        if False:
            i = 10
            return i + 15
        component = tc.ClassWithDocstring()
        t = trace.FireTrace(component, name='ClassWithDocstring')
        help_output = helptext.HelpText(component, t)
        expected_output = '\nNAME\n    ClassWithDocstring - Test class for testing help text output.\n\nSYNOPSIS\n    ClassWithDocstring COMMAND | VALUE\n\nDESCRIPTION\n    This is some detail description of this test class.\n\nCOMMANDS\n    COMMAND is one of the following:\n\n     print_msg\n       Prints a message.\n\nVALUES\n    VALUE is one of the following:\n\n     message\n       The default message to print.'
        self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())

    def testHelpScreenForFunctionDocstringWithLineBreak(self):
        if False:
            while True:
                i = 10
        component = tc.ClassWithMultilineDocstring.example_generator
        t = trace.FireTrace(component, name='example_generator')
        help_output = helptext.HelpText(component, t)
        expected_output = '\n    NAME\n        example_generator - Generators have a ``Yields`` section instead of a ``Returns`` section.\n\n    SYNOPSIS\n        example_generator N\n\n    DESCRIPTION\n        Generators have a ``Yields`` section instead of a ``Returns`` section.\n\n    POSITIONAL ARGUMENTS\n        N\n            The upper limit of the range to generate, from 0 to `n` - 1.\n\n    NOTES\n        You can also use flags syntax for POSITIONAL ARGUMENTS'
        self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())

    def testHelpScreenForFunctionFunctionWithDefaultArgs(self):
        if False:
            while True:
                i = 10
        component = tc.WithDefaults().double
        t = trace.FireTrace(component, name='double')
        help_output = helptext.HelpText(component, t)
        expected_output = '\n    NAME\n        double - Returns the input multiplied by 2.\n\n    SYNOPSIS\n        double <flags>\n\n    DESCRIPTION\n        Returns the input multiplied by 2.\n\n    FLAGS\n        -c, --count=COUNT\n            Default: 0\n            Input number that you want to double.'
        self.assertEqual(textwrap.dedent(expected_output).strip(), help_output.strip())

    def testHelpTextUnderlineFlag(self):
        if False:
            for i in range(10):
                print('nop')
        component = tc.WithDefaults().triple
        t = trace.FireTrace(component, name='triple')
        help_screen = helptext.HelpText(component, t)
        self.assertIn(formatting.Bold('NAME') + '\n    triple', help_screen)
        self.assertIn(formatting.Bold('SYNOPSIS') + '\n    triple <flags>', help_screen)
        self.assertIn(formatting.Bold('FLAGS') + '\n    -c, --' + formatting.Underline('count'), help_screen)

    def testHelpTextBoldCommandName(self):
        if False:
            return 10
        component = tc.ClassWithDocstring()
        t = trace.FireTrace(component, name='ClassWithDocstring')
        help_screen = helptext.HelpText(component, t)
        self.assertIn(formatting.Bold('NAME') + '\n    ClassWithDocstring', help_screen)
        self.assertIn(formatting.Bold('COMMANDS') + '\n', help_screen)
        self.assertIn(formatting.BoldUnderline('COMMAND') + ' is one of the following:\n', help_screen)
        self.assertIn(formatting.Bold('print_msg') + '\n', help_screen)

    def testHelpTextObjectWithGroupAndValues(self):
        if False:
            i = 10
            return i + 15
        component = tc.TypedProperties()
        t = trace.FireTrace(component, name='TypedProperties')
        help_screen = helptext.HelpText(component=component, trace=t, verbose=True)
        print(help_screen)
        self.assertIn('GROUPS', help_screen)
        self.assertIn('GROUP is one of the following:', help_screen)
        self.assertIn('charlie\n       Class with functions that have default arguments.', help_screen)
        self.assertIn('VALUES', help_screen)
        self.assertIn('VALUE is one of the following:', help_screen)
        self.assertIn('alpha', help_screen)

    def testHelpTextNameSectionCommandWithSeparator(self):
        if False:
            for i in range(10):
                print('nop')
        component = 9
        t = trace.FireTrace(component, name='int', separator='-')
        t.AddSeparator()
        help_screen = helptext.HelpText(component=component, trace=t, verbose=False)
        self.assertIn('int -', help_screen)
        self.assertNotIn('int - -', help_screen)

    def testHelpTextNameSectionCommandWithSeparatorVerbose(self):
        if False:
            print('Hello World!')
        component = tc.WithDefaults().double
        t = trace.FireTrace(component, name='double', separator='-')
        t.AddSeparator()
        help_screen = helptext.HelpText(component=component, trace=t, verbose=True)
        self.assertIn('double -', help_screen)
        self.assertIn('double - -', help_screen)

    def testHelpTextMultipleKeywoardArgumentsWithShortArgs(self):
        if False:
            for i in range(10):
                print('nop')
        component = tc.fn_with_multiple_defaults
        t = trace.FireTrace(component, name='shortargs')
        help_screen = helptext.HelpText(component, t)
        self.assertIn(formatting.Bold('NAME') + '\n    shortargs', help_screen)
        self.assertIn(formatting.Bold('SYNOPSIS') + '\n    shortargs <flags>', help_screen)
        self.assertIn(formatting.Bold('FLAGS') + '\n    -f, --first', help_screen)
        self.assertIn('\n    --last', help_screen)
        self.assertIn('\n    --late', help_screen)

class UsageTest(testutils.BaseTestCase):

    def testUsageOutput(self):
        if False:
            for i in range(10):
                print('nop')
        component = tc.NoDefaults()
        t = trace.FireTrace(component, name='NoDefaults')
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = '\n    Usage: NoDefaults <command>\n      available commands:    double | triple\n\n    For detailed information on this command, run:\n      NoDefaults --help'
        self.assertEqual(usage_output, textwrap.dedent(expected_output).lstrip('\n'))

    def testUsageOutputVerbose(self):
        if False:
            return 10
        component = tc.NoDefaults()
        t = trace.FireTrace(component, name='NoDefaults')
        usage_output = helptext.UsageText(component, trace=t, verbose=True)
        expected_output = '\n    Usage: NoDefaults <command>\n      available commands:    double | triple\n\n    For detailed information on this command, run:\n      NoDefaults --help'
        self.assertEqual(usage_output, textwrap.dedent(expected_output).lstrip('\n'))

    def testUsageOutputMethod(self):
        if False:
            while True:
                i = 10
        component = tc.NoDefaults().double
        t = trace.FireTrace(component, name='NoDefaults')
        t.AddAccessedProperty(component, 'double', ['double'], None, None)
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = '\n    Usage: NoDefaults double COUNT\n\n    For detailed information on this command, run:\n      NoDefaults double --help'
        self.assertEqual(usage_output, textwrap.dedent(expected_output).lstrip('\n'))

    def testUsageOutputFunctionWithHelp(self):
        if False:
            while True:
                i = 10
        component = tc.function_with_help
        t = trace.FireTrace(component, name='function_with_help')
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = '\n    Usage: function_with_help <flags>\n      optional flags:        --help\n\n    For detailed information on this command, run:\n      function_with_help -- --help'
        self.assertEqual(usage_output, textwrap.dedent(expected_output).lstrip('\n'))

    def testUsageOutputFunctionWithDocstring(self):
        if False:
            while True:
                i = 10
        component = tc.multiplier_with_docstring
        t = trace.FireTrace(component, name='multiplier_with_docstring')
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = '\n    Usage: multiplier_with_docstring NUM <flags>\n      optional flags:        --rate\n\n    For detailed information on this command, run:\n      multiplier_with_docstring --help'
        self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)

    @testutils.skipIf(six.PY2, 'Python 2 does not support required name-only arguments.')
    def testUsageOutputFunctionMixedDefaults(self):
        if False:
            return 10
        component = tc.py3.HelpTextComponent().identity
        t = trace.FireTrace(component, name='FunctionMixedDefaults')
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = '\n    Usage: FunctionMixedDefaults <flags>\n      optional flags:        --beta\n      required flags:        --alpha\n\n    For detailed information on this command, run:\n      FunctionMixedDefaults --help'
        expected_output = textwrap.dedent(expected_output).lstrip('\n')
        self.assertEqual(expected_output, usage_output)

    def testUsageOutputCallable(self):
        if False:
            while True:
                i = 10
        component = tc.CallableWithKeywordArgument()
        t = trace.FireTrace(component, name='CallableWithKeywordArgument', separator='@')
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = '\n    Usage: CallableWithKeywordArgument <command> | <flags>\n      available commands:    print_msg\n      flags are accepted\n\n    For detailed information on this command, run:\n      CallableWithKeywordArgument -- --help'
        self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)

    def testUsageOutputConstructorWithParameter(self):
        if False:
            while True:
                i = 10
        component = tc.InstanceVars
        t = trace.FireTrace(component, name='InstanceVars')
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = '\n    Usage: InstanceVars --arg1=ARG1 --arg2=ARG2\n\n    For detailed information on this command, run:\n      InstanceVars --help'
        self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)

    def testUsageOutputConstructorWithParameterVerbose(self):
        if False:
            print('Hello World!')
        component = tc.InstanceVars
        t = trace.FireTrace(component, name='InstanceVars')
        usage_output = helptext.UsageText(component, trace=t, verbose=True)
        expected_output = '\n    Usage: InstanceVars <command> | --arg1=ARG1 --arg2=ARG2\n      available commands:    run\n\n    For detailed information on this command, run:\n      InstanceVars --help'
        self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)

    def testUsageOutputEmptyDict(self):
        if False:
            for i in range(10):
                print('nop')
        component = {}
        t = trace.FireTrace(component, name='EmptyDict')
        usage_output = helptext.UsageText(component, trace=t, verbose=True)
        expected_output = '\n    Usage: EmptyDict\n\n    For detailed information on this command, run:\n      EmptyDict --help'
        self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)

    def testUsageOutputNone(self):
        if False:
            while True:
                i = 10
        component = None
        t = trace.FireTrace(component, name='None')
        usage_output = helptext.UsageText(component, trace=t, verbose=True)
        expected_output = '\n    Usage: None\n\n    For detailed information on this command, run:\n      None --help'
        self.assertEqual(textwrap.dedent(expected_output).lstrip('\n'), usage_output)

    def testInitRequiresFlagSyntaxSubclassNamedTuple(self):
        if False:
            print('Hello World!')
        component = tc.SubPoint
        t = trace.FireTrace(component, name='SubPoint')
        usage_output = helptext.UsageText(component, trace=t, verbose=False)
        expected_output = 'Usage: SubPoint --x=X --y=Y'
        self.assertIn(expected_output, usage_output)
if __name__ == '__main__':
    testutils.main()