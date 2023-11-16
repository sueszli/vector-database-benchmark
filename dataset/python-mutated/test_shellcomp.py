"""
Test cases for twisted.python._shellcomp
"""
import sys
from io import BytesIO
from typing import List, Optional
from twisted.python import _shellcomp, reflect, usage
from twisted.python.usage import CompleteFiles, CompleteList, Completer, Completions
from twisted.trial import unittest

class ZshScriptTestMeta(type):
    """
    Metaclass of ZshScriptTestMixin.
    """

    def __new__(cls, name, bases, attrs):
        if False:
            while True:
                i = 10

        def makeTest(cmdName, optionsFQPN):
            if False:
                return 10

            def runTest(self):
                if False:
                    for i in range(10):
                        print('nop')
                return test_genZshFunction(self, cmdName, optionsFQPN)
            return runTest
        if 'generateFor' in attrs:
            for (cmdName, optionsFQPN) in attrs['generateFor']:
                test = makeTest(cmdName, optionsFQPN)
                attrs['test_genZshFunction_' + cmdName] = test
        return type.__new__(cls, name, bases, attrs)

class ZshScriptTestMixin(metaclass=ZshScriptTestMeta):
    """
    Integration test helper to show that C{usage.Options} classes can have zsh
    completion functions generated for them without raising errors.

    In your subclasses set a class variable like so::

      #            | cmd name | Fully Qualified Python Name of Options class |
      #
      generateFor = [('conch',  'twisted.conch.scripts.conch.ClientOptions'),
                     ('twistd', 'twisted.scripts.twistd.ServerOptions'),
                     ]

    Each package that contains Twisted scripts should contain one TestCase
    subclass which also inherits from this mixin, and contains a C{generateFor}
    list appropriate for the scripts in that package.
    """

def test_genZshFunction(self, cmdName, optionsFQPN):
    if False:
        for i in range(10):
            print('nop')
    "\n    Generate completion functions for given twisted command - no errors\n    should be raised\n\n    @type cmdName: C{str}\n    @param cmdName: The name of the command-line utility e.g. 'twistd'\n\n    @type optionsFQPN: C{str}\n    @param optionsFQPN: The Fully Qualified Python Name of the C{Options}\n        class to be tested.\n    "
    outputFile = BytesIO()
    self.patch(usage.Options, '_shellCompFile', outputFile)
    try:
        o = reflect.namedAny(optionsFQPN)()
    except Exception as e:
        raise unittest.SkipTest("Couldn't import or instantiate Options class: %s" % (e,))
    try:
        o.parseOptions(['', '--_shell-completion', 'zsh:2'])
    except ImportError as e:
        raise unittest.SkipTest('ImportError calling parseOptions(): %s', (e,))
    except SystemExit:
        pass
    else:
        self.fail('SystemExit not raised')
    outputFile.seek(0)
    self.assertEqual(1, len(outputFile.read(1)))
    outputFile.seek(0)
    outputFile.truncate()
    if hasattr(o, 'subCommands'):
        for (cmd, short, parser, doc) in o.subCommands:
            try:
                o.parseOptions([cmd, '', '--_shell-completion', 'zsh:3'])
            except ImportError as e:
                raise unittest.SkipTest('ImportError calling parseOptions() on subcommand: %s', (e,))
            except SystemExit:
                pass
            else:
                self.fail('SystemExit not raised')
            outputFile.seek(0)
            self.assertEqual(1, len(outputFile.read(1)))
            outputFile.seek(0)
            outputFile.truncate()
    self.flushWarnings()

class ZshTests(unittest.TestCase):
    """
    Tests for zsh completion code
    """

    def test_accumulateMetadata(self):
        if False:
            while True:
                i = 10
        "\n        Are `compData' attributes you can place on Options classes\n        picked up correctly?\n        "
        opts = FighterAceExtendedOptions()
        ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
        descriptions = FighterAceOptions.compData.descriptions.copy()
        descriptions.update(FighterAceExtendedOptions.compData.descriptions)
        self.assertEqual(ag.descriptions, descriptions)
        self.assertEqual(ag.multiUse, set(FighterAceOptions.compData.multiUse))
        self.assertEqual(ag.mutuallyExclusive, FighterAceOptions.compData.mutuallyExclusive)
        optActions = FighterAceOptions.compData.optActions.copy()
        optActions.update(FighterAceExtendedOptions.compData.optActions)
        self.assertEqual(ag.optActions, optActions)
        self.assertEqual(ag.extraActions, FighterAceOptions.compData.extraActions)

    def test_mutuallyExclusiveCornerCase(self):
        if False:
            return 10
        '\n        Exercise a corner-case of ZshArgumentsGenerator.makeExcludesDict()\n        where the long option name already exists in the `excludes` dict being\n        built.\n        '

        class OddFighterAceOptions(FighterAceExtendedOptions):
            optFlags = [['anatra', None, 'Select the Anatra DS as your dogfighter aircraft']]
            compData = Completions(mutuallyExclusive=[['anatra', 'fokker', 'albatros', 'spad', 'bristol']])
        opts = OddFighterAceOptions()
        ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
        expected = {'albatros': {'anatra', 'b', 'bristol', 'f', 'fokker', 's', 'spad'}, 'anatra': {'a', 'albatros', 'b', 'bristol', 'f', 'fokker', 's', 'spad'}, 'bristol': {'a', 'albatros', 'anatra', 'f', 'fokker', 's', 'spad'}, 'fokker': {'a', 'albatros', 'anatra', 'b', 'bristol', 's', 'spad'}, 'spad': {'a', 'albatros', 'anatra', 'b', 'bristol', 'f', 'fokker'}}
        self.assertEqual(ag.excludes, expected)

    def test_accumulateAdditionalOptions(self):
        if False:
            while True:
                i = 10
        '\n        We pick up options that are only defined by having an\n        appropriately named method on your Options class,\n        e.g. def opt_foo(self, foo)\n        '
        opts = FighterAceExtendedOptions()
        ag = _shellcomp.ZshArgumentsGenerator(opts, 'ace', BytesIO())
        self.assertIn('nocrash', ag.flagNameToDefinition)
        self.assertIn('nocrash', ag.allOptionsNameToDefinition)
        self.assertIn('difficulty', ag.paramNameToDefinition)
        self.assertIn('difficulty', ag.allOptionsNameToDefinition)

    def test_verifyZshNames(self):
        if False:
            print('Hello World!')
        "\n        Using a parameter/flag name that doesn't exist\n        will raise an error\n        "

        class TmpOptions(FighterAceExtendedOptions):
            compData = Completions(optActions={'detaill': None})
        self.assertRaises(ValueError, _shellcomp.ZshArgumentsGenerator, TmpOptions(), 'ace', BytesIO())

        class TmpOptions2(FighterAceExtendedOptions):
            compData = Completions(mutuallyExclusive=[('foo', 'bar')])
        self.assertRaises(ValueError, _shellcomp.ZshArgumentsGenerator, TmpOptions2(), 'ace', BytesIO())

    def test_zshCode(self):
        if False:
            i = 10
            return i + 15
        '\n        Generate a completion function, and test the textual output\n        against a known correct output\n        '
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        self.patch(sys, 'argv', ['silly', '', '--_shell-completion', 'zsh:2'])
        opts = SimpleProgOptions()
        self.assertRaises(SystemExit, opts.parseOptions)
        self.assertEqual(testOutput1, outputFile.getvalue())

    def test_zshCodeWithSubs(self):
        if False:
            return 10
        '\n        Generate a completion function with subcommands,\n        and test the textual output against a known correct output\n        '
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        self.patch(sys, 'argv', ['silly2', '', '--_shell-completion', 'zsh:2'])
        opts = SimpleProgWithSubcommands()
        self.assertRaises(SystemExit, opts.parseOptions)
        self.assertEqual(testOutput2, outputFile.getvalue())

    def test_incompleteCommandLine(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Completion still happens even if a command-line is given\n        that would normally throw UsageError.\n        '
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        opts = FighterAceOptions()
        self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', 'server', '--unknown-option', '--unknown-option2', '--_shell-completion', 'zsh:5'])
        outputFile.seek(0)
        self.assertEqual(1, len(outputFile.read(1)))

    def test_incompleteCommandLine_case2(self):
        if False:
            print('Hello World!')
        '\n        Completion still happens even if a command-line is given\n        that would normally throw UsageError.\n\n        The existence of --unknown-option prior to the subcommand\n        will break subcommand detection... but we complete anyway\n        '
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        opts = FighterAceOptions()
        self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', '--unknown-option', 'server', '--list-server', '--_shell-completion', 'zsh:5'])
        outputFile.seek(0)
        self.assertEqual(1, len(outputFile.read(1)))
        outputFile.seek(0)
        outputFile.truncate()

    def test_incompleteCommandLine_case3(self):
        if False:
            i = 10
            return i + 15
        '\n        Completion still happens even if a command-line is given\n        that would normally throw UsageError.\n\n        Break subcommand detection in a different way by providing\n        an invalid subcommand name.\n        '
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        opts = FighterAceOptions()
        self.assertRaises(SystemExit, opts.parseOptions, ['--fokker', 'unknown-subcommand', '--list-server', '--_shell-completion', 'zsh:4'])
        outputFile.seek(0)
        self.assertEqual(1, len(outputFile.read(1)))

    def test_skipSubcommandList(self):
        if False:
            return 10
        "\n        Ensure the optimization which skips building the subcommand list\n        under certain conditions isn't broken.\n        "
        outputFile = BytesIO()
        self.patch(usage.Options, '_shellCompFile', outputFile)
        opts = FighterAceOptions()
        self.assertRaises(SystemExit, opts.parseOptions, ['--alba', '--_shell-completion', 'zsh:2'])
        outputFile.seek(0)
        self.assertEqual(1, len(outputFile.read(1)))

    def test_poorlyDescribedOptMethod(self):
        if False:
            return 10
        '\n        Test corner case fetching an option description from a method docstring\n        '
        opts = FighterAceOptions()
        argGen = _shellcomp.ZshArgumentsGenerator(opts, 'ace', None)
        descr = argGen.getDescription('silly')
        self.assertEqual(descr, 'silly')

    def test_brokenActions(self):
        if False:
            i = 10
            return i + 15
        '\n        A C{Completer} with repeat=True may only be used as the\n        last item in the extraActions list.\n        '

        class BrokenActions(usage.Options):
            compData = usage.Completions(extraActions=[usage.Completer(repeat=True), usage.Completer()])
        outputFile = BytesIO()
        opts = BrokenActions()
        self.patch(opts, '_shellCompFile', outputFile)
        self.assertRaises(ValueError, opts.parseOptions, ['', '--_shell-completion', 'zsh:2'])

    def test_optMethodsDontOverride(self):
        if False:
            while True:
                i = 10
        '\n        opt_* methods on Options classes should not override the\n        data provided in optFlags or optParameters.\n        '

        class Options(usage.Options):
            optFlags = [['flag', 'f', 'A flag']]
            optParameters = [['param', 'p', None, 'A param']]

            def opt_flag(self):
                if False:
                    i = 10
                    return i + 15
                'junk description'

            def opt_param(self, param):
                if False:
                    while True:
                        i = 10
                'junk description'
        opts = Options()
        argGen = _shellcomp.ZshArgumentsGenerator(opts, 'ace', None)
        self.assertEqual(argGen.getDescription('flag'), 'A flag')
        self.assertEqual(argGen.getDescription('param'), 'A param')

class EscapeTests(unittest.TestCase):

    def test_escape(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify _shellcomp.escape() function\n        '
        esc = _shellcomp.escape
        test = '$'
        self.assertEqual(esc(test), "'$'")
        test = 'A--\'$"\\`--B'
        self.assertEqual(esc(test), '"A--\'\\$\\"\\\\\\`--B"')

class CompleterNotImplementedTests(unittest.TestCase):
    """
    Test that using an unknown shell constant with SubcommandAction
    raises NotImplementedError

    The other Completer() subclasses are tested in test_usage.py
    """

    def test_unknownShell(self):
        if False:
            return 10
        '\n        Using an unknown shellType should raise NotImplementedError\n        '
        action = _shellcomp.SubcommandAction()
        self.assertRaises(NotImplementedError, action._shellCode, None, 'bad_shell_type')

class FighterAceServerOptions(usage.Options):
    """
    Options for FighterAce 'server' subcommand
    """
    optFlags = [['list-server', None, 'List this server with the online FighterAce network']]
    optParameters = [['packets-per-second', None, 'Number of update packets to send per second', '20']]

class FighterAceOptions(usage.Options):
    """
    Command-line options for an imaginary `Fighter Ace` game
    """
    optFlags: List[List[Optional[str]]] = [['fokker', 'f', 'Select the Fokker Dr.I as your dogfighter aircraft'], ['albatros', 'a', 'Select the Albatros D-III as your dogfighter aircraft'], ['spad', 's', 'Select the SPAD S.VII as your dogfighter aircraft'], ['bristol', 'b', 'Select the Bristol Scout as your dogfighter aircraft'], ['physics', 'p', 'Enable secret Twisted physics engine'], ['jam', 'j', 'Enable a small chance that your machine guns will jam!'], ['verbose', 'v', 'Verbose logging (may be specified more than once)']]
    optParameters: List[List[Optional[str]]] = [['pilot-name', None, "What's your name, Ace?", 'Manfred von Richthofen'], ['detail', 'd', 'Select the level of rendering detail (1-5)', '3']]
    subCommands = [['server', None, FighterAceServerOptions, 'Start FighterAce game-server.']]
    compData = Completions(descriptions={'physics': 'Twisted-Physics', 'detail': 'Rendering detail level'}, multiUse=['verbose'], mutuallyExclusive=[['fokker', 'albatros', 'spad', 'bristol']], optActions={'detail': CompleteList(['12345'])}, extraActions=[CompleteFiles(descr='saved game file to load')])

    def opt_silly(self):
        if False:
            print('Hello World!')
        ' '

class FighterAceExtendedOptions(FighterAceOptions):
    """
    Extend the options and zsh metadata provided by FighterAceOptions.
    _shellcomp must accumulate options and metadata from all classes in the
    hiearchy so this is important to test.
    """
    optFlags = [['no-stalls', None, 'Turn off the ability to stall your aircraft']]
    optParameters = [['reality-level', None, 'Select the level of physics reality (1-5)', '5']]
    compData = Completions(descriptions={'no-stalls': "Can't stall your plane"}, optActions={'reality-level': Completer(descr='Physics reality level')})

    def opt_nocrash(self):
        if False:
            return 10
        "\n        Select that you can't crash your plane\n        "

    def opt_difficulty(self, difficulty):
        if False:
            i = 10
            return i + 15
        '\n        How tough are you? (1-10)\n        '

def _accuracyAction():
    if False:
        for i in range(10):
            print('nop')
    return CompleteList(['1', '2', '3'], descr="Accuracy'`?")

class SimpleProgOptions(usage.Options):
    """
    Command-line options for a `Silly` imaginary program
    """
    optFlags = [['color', 'c', 'Turn on color output'], ['gray', 'g', 'Turn on gray-scale output'], ['verbose', 'v', 'Verbose logging (may be specified more than once)']]
    optParameters = [['optimization', None, '5', 'Select the level of optimization (1-5)'], ['accuracy', 'a', '3', 'Select the level of accuracy (1-3)']]
    compData = Completions(descriptions={'color': 'Color on', 'optimization': 'Optimization level'}, multiUse=['verbose'], mutuallyExclusive=[['color', 'gray']], optActions={'optimization': CompleteList(['1', '2', '3', '4', '5'], descr='Optimization?'), 'accuracy': _accuracyAction}, extraActions=[CompleteFiles(descr='output file')])

    def opt_X(self):
        if False:
            i = 10
            return i + 15
        '\n        usage.Options does not recognize single-letter opt_ methods\n        '

class SimpleProgSub1(usage.Options):
    optFlags = [['sub-opt', 's', 'Sub Opt One']]

class SimpleProgSub2(usage.Options):
    optFlags = [['sub-opt', 's', 'Sub Opt Two']]

class SimpleProgWithSubcommands(SimpleProgOptions):
    optFlags = [['some-option'], ['other-option', 'o']]
    optParameters = [['some-param'], ['other-param', 'p'], ['another-param', 'P', 'Yet Another Param']]
    subCommands = [['sub1', None, SimpleProgSub1, 'Sub Command 1'], ['sub2', None, SimpleProgSub2, 'Sub Command 2']]
testOutput1 = b'#compdef silly\n\n_arguments -s -A "-*" \\\n\':output file (*):_files -g "*"\' \\\n"(--accuracy)-a[Select the level of accuracy (1-3)]:Accuracy\'\\`?:(1 2 3)" \\\n"(-a)--accuracy=[Select the level of accuracy (1-3)]:Accuracy\'\\`?:(1 2 3)" \\\n\'(--color --gray -g)-c[Color on]\' \\\n\'(--gray -c -g)--color[Color on]\' \\\n\'(--color --gray -c)-g[Turn on gray-scale output]\' \\\n\'(--color -c -g)--gray[Turn on gray-scale output]\' \\\n\'--help[Display this help and exit.]\' \\\n\'--optimization=[Optimization level]:Optimization?:(1 2 3 4 5)\' \\\n\'*-v[Verbose logging (may be specified more than once)]\' \\\n\'*--verbose[Verbose logging (may be specified more than once)]\' \\\n\'--version[Display Twisted version and exit.]\' \\\n&& return 0\n'
testOutput2 = b'#compdef silly2\n\n_arguments -s -A "-*" \\\n\'*::subcmd:->subcmd\' \\\n\':output file (*):_files -g "*"\' \\\n"(--accuracy)-a[Select the level of accuracy (1-3)]:Accuracy\'\\`?:(1 2 3)" \\\n"(-a)--accuracy=[Select the level of accuracy (1-3)]:Accuracy\'\\`?:(1 2 3)" \\\n\'(--another-param)-P[another-param]:another-param:_files\' \\\n\'(-P)--another-param=[another-param]:another-param:_files\' \\\n\'(--color --gray -g)-c[Color on]\' \\\n\'(--gray -c -g)--color[Color on]\' \\\n\'(--color --gray -c)-g[Turn on gray-scale output]\' \\\n\'(--color -c -g)--gray[Turn on gray-scale output]\' \\\n\'--help[Display this help and exit.]\' \\\n\'--optimization=[Optimization level]:Optimization?:(1 2 3 4 5)\' \\\n\'(--other-option)-o[other-option]\' \\\n\'(-o)--other-option[other-option]\' \\\n\'(--other-param)-p[other-param]:other-param:_files\' \\\n\'(-p)--other-param=[other-param]:other-param:_files\' \\\n\'--some-option[some-option]\' \\\n\'--some-param=[some-param]:some-param:_files\' \\\n\'*-v[Verbose logging (may be specified more than once)]\' \\\n\'*--verbose[Verbose logging (may be specified more than once)]\' \\\n\'--version[Display Twisted version and exit.]\' \\\n&& return 0\nlocal _zsh_subcmds_array\n_zsh_subcmds_array=(\n"sub1:Sub Command 1"\n"sub2:Sub Command 2"\n)\n\n_describe "sub-command" _zsh_subcmds_array\n'