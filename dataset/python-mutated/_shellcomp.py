"""
No public APIs are provided by this module. Internal use only.

This module implements dynamic tab-completion for any command that uses
twisted.python.usage. Currently, only zsh is supported. Bash support may
be added in the future.

Maintainer: Eric P. Mangold - twisted AT teratorn DOT org

In order for zsh completion to take place the shell must be able to find an
appropriate "stub" file ("completion function") that invokes this code and
displays the results to the user.

The stub used for Twisted commands is in the file C{twisted-completion.zsh},
which is also included in the official Zsh distribution at
C{Completion/Unix/Command/_twisted}. Use this file as a basis for completion
functions for your own commands. You should only need to change the first line
to something like C{#compdef mycommand}.

The main public documentation exists in the L{twisted.python.usage.Options}
docstring, the L{twisted.python.usage.Completions} docstring, and the
Options howto.
"""
import getopt
import inspect
import itertools
from types import MethodType
from typing import Dict, List, Set
from twisted.python import reflect, usage, util
from twisted.python.compat import ioType

def shellComplete(config, cmdName, words, shellCompFile):
    if False:
        return 10
    '\n    Perform shell completion.\n\n    A completion function (shell script) is generated for the requested\n    shell and written to C{shellCompFile}, typically C{stdout}. The result\n    is then eval\'d by the shell to produce the desired completions.\n\n    @type config: L{twisted.python.usage.Options}\n    @param config: The L{twisted.python.usage.Options} instance to generate\n        completions for.\n\n    @type cmdName: C{str}\n    @param cmdName: The name of the command we\'re generating completions for.\n        In the case of zsh, this is used to print an appropriate\n        "#compdef $CMD" line at the top of the output. This is\n        not necessary for the functionality of the system, but it\n        helps in debugging, since the output we produce is properly\n        formed and may be saved in a file and used as a stand-alone\n        completion function.\n\n    @type words: C{list} of C{str}\n    @param words: The raw command-line words passed to use by the shell\n        stub function. argv[0] has already been stripped off.\n\n    @type shellCompFile: C{file}\n    @param shellCompFile: The file to write completion data to.\n    '
    if shellCompFile and ioType(shellCompFile) == str:
        shellCompFile = shellCompFile.buffer
    (shellName, position) = words[-1].split(':')
    position = int(position)
    position -= 2
    cWord = words[position]
    while position >= 1:
        if words[position - 1].startswith('-'):
            position -= 1
        else:
            break
    words = words[:position]
    subCommands = getattr(config, 'subCommands', None)
    if subCommands:
        args = None
        try:
            (opts, args) = getopt.getopt(words, config.shortOpt, config.longOpt)
        except getopt.error:
            pass
        if args:
            for (cmd, short, parser, doc) in config.subCommands:
                if args[0] == cmd or args[0] == short:
                    subOptions = parser()
                    subOptions.parent = config
                    gen: ZshBuilder = ZshSubcommandBuilder(subOptions, config, cmdName, shellCompFile)
                    gen.write()
                    return
        genSubs = True
        if cWord.startswith('-'):
            genSubs = False
        gen = ZshBuilder(config, cmdName, shellCompFile)
        gen.write(genSubs=genSubs)
    else:
        gen = ZshBuilder(config, cmdName, shellCompFile)
        gen.write()

class SubcommandAction(usage.Completer):

    def _shellCode(self, optName, shellType):
        if False:
            i = 10
            return i + 15
        if shellType == usage._ZSH:
            return '*::subcmd:->subcmd'
        raise NotImplementedError(f'Unknown shellType {shellType!r}')

class ZshBuilder:
    """
    Constructs zsh code that will complete options for a given usage.Options
    instance, possibly including a list of subcommand names.

    Completions for options to subcommands won't be generated because this
    class will never be used if the user is completing options for a specific
    subcommand. (See L{ZshSubcommandBuilder} below)

    @type options: L{twisted.python.usage.Options}
    @ivar options: The L{twisted.python.usage.Options} instance defined for this
        command.

    @type cmdName: C{str}
    @ivar cmdName: The name of the command we're generating completions for.

    @type file: C{file}
    @ivar file: The C{file} to write the completion function to.  The C{file}
        must have L{bytes} I/O semantics.
    """

    def __init__(self, options, cmdName, file):
        if False:
            return 10
        self.options = options
        self.cmdName = cmdName
        self.file = file

    def write(self, genSubs=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate the completion function and write it to the output file\n        @return: L{None}\n\n        @type genSubs: C{bool}\n        @param genSubs: Flag indicating whether or not completions for the list\n            of subcommand should be generated. Only has an effect\n            if the C{subCommands} attribute has been defined on the\n            L{twisted.python.usage.Options} instance.\n        '
        if genSubs and getattr(self.options, 'subCommands', None) is not None:
            gen = ZshArgumentsGenerator(self.options, self.cmdName, self.file)
            gen.extraActions.insert(0, SubcommandAction())
            gen.write()
            self.file.write(b'local _zsh_subcmds_array\n_zsh_subcmds_array=(\n')
            for (cmd, short, parser, desc) in self.options.subCommands:
                self.file.write(b'"' + cmd.encode('utf-8') + b':' + desc.encode('utf-8') + b'"\n')
            self.file.write(b')\n\n')
            self.file.write(b'_describe "sub-command" _zsh_subcmds_array\n')
        else:
            gen = ZshArgumentsGenerator(self.options, self.cmdName, self.file)
            gen.write()

class ZshSubcommandBuilder(ZshBuilder):
    """
    Constructs zsh code that will complete options for a given usage.Options
    instance, and also for a single sub-command. This will only be used in
    the case where the user is completing options for a specific subcommand.

    @type subOptions: L{twisted.python.usage.Options}
    @ivar subOptions: The L{twisted.python.usage.Options} instance defined for
        the sub command.
    """

    def __init__(self, subOptions, *args):
        if False:
            for i in range(10):
                print('nop')
        self.subOptions = subOptions
        ZshBuilder.__init__(self, *args)

    def write(self):
        if False:
            while True:
                i = 10
        '\n        Generate the completion function and write it to the output file\n        @return: L{None}\n        '
        gen = ZshArgumentsGenerator(self.options, self.cmdName, self.file)
        gen.extraActions.insert(0, SubcommandAction())
        gen.write()
        gen = ZshArgumentsGenerator(self.subOptions, self.cmdName, self.file)
        gen.write()

class ZshArgumentsGenerator:
    """
    Generate a call to the zsh _arguments completion function
    based on data in a usage.Options instance

    The first three instance variables are populated based on constructor
    arguments. The remaining non-constructor variables are populated by this
    class with data gathered from the C{Options} instance passed in, and its
    base classes.

    @type options: L{twisted.python.usage.Options}
    @ivar options: The L{twisted.python.usage.Options} instance to generate for

    @type cmdName: C{str}
    @ivar cmdName: The name of the command we're generating completions for.

    @type file: C{file}
    @ivar file: The C{file} to write the completion function to.  The C{file}
        must have L{bytes} I/O semantics.

    @type descriptions: C{dict}
    @ivar descriptions: A dict mapping long option names to alternate
        descriptions. When this variable is defined, the descriptions
        contained here will override those descriptions provided in the
        optFlags and optParameters variables.

    @type multiUse: C{list}
    @ivar multiUse: An iterable containing those long option names which may
        appear on the command line more than once. By default, options will
        only be completed one time.

    @type mutuallyExclusive: C{list} of C{tuple}
    @ivar mutuallyExclusive: A sequence of sequences, with each sub-sequence
        containing those long option names that are mutually exclusive. That is,
        those options that cannot appear on the command line together.

    @type optActions: C{dict}
    @ivar optActions: A dict mapping long option names to shell "actions".
        These actions define what may be completed as the argument to the
        given option, and should be given as instances of
        L{twisted.python.usage.Completer}.

        Callables may instead be given for the values in this dict. The
        callable should accept no arguments, and return a C{Completer}
        instance used as the action.

    @type extraActions: C{list} of C{twisted.python.usage.Completer}
    @ivar extraActions: Extra arguments are those arguments typically
        appearing at the end of the command-line, which are not associated
        with any particular named option. That is, the arguments that are
        given to the parseArgs() method of your usage.Options subclass.
    """

    def __init__(self, options, cmdName, file):
        if False:
            return 10
        self.options = options
        self.cmdName = cmdName
        self.file = file
        self.descriptions = {}
        self.multiUse = set()
        self.mutuallyExclusive = []
        self.optActions = {}
        self.extraActions = []
        for cls in reversed(inspect.getmro(options.__class__)):
            data = getattr(cls, 'compData', None)
            if data:
                self.descriptions.update(data.descriptions)
                self.optActions.update(data.optActions)
                self.multiUse.update(data.multiUse)
                self.mutuallyExclusive.extend(data.mutuallyExclusive)
                if data.extraActions:
                    self.extraActions = data.extraActions
        aCL = reflect.accumulateClassList
        optFlags: List[List[object]] = []
        optParams: List[List[object]] = []
        aCL(options.__class__, 'optFlags', optFlags)
        aCL(options.__class__, 'optParameters', optParams)
        for (i, optList) in enumerate(optFlags):
            if len(optList) != 3:
                optFlags[i] = util.padTo(3, optList)
        for (i, optList) in enumerate(optParams):
            if len(optList) != 5:
                optParams[i] = util.padTo(5, optList)
        self.optFlags = optFlags
        self.optParams = optParams
        paramNameToDefinition = {}
        for optList in optParams:
            paramNameToDefinition[optList[0]] = optList[1:]
        self.paramNameToDefinition = paramNameToDefinition
        flagNameToDefinition = {}
        for optList in optFlags:
            flagNameToDefinition[optList[0]] = optList[1:]
        self.flagNameToDefinition = flagNameToDefinition
        allOptionsNameToDefinition = {}
        allOptionsNameToDefinition.update(paramNameToDefinition)
        allOptionsNameToDefinition.update(flagNameToDefinition)
        self.allOptionsNameToDefinition = allOptionsNameToDefinition
        self.addAdditionalOptions()
        self.verifyZshNames()
        self.excludes = self.makeExcludesDict()

    def write(self):
        if False:
            return 10
        '\n        Write the zsh completion code to the file given to __init__\n        @return: L{None}\n        '
        self.writeHeader()
        self.writeExtras()
        self.writeOptions()
        self.writeFooter()

    def writeHeader(self):
        if False:
            print('Hello World!')
        '\n        This is the start of the code that calls _arguments\n        @return: L{None}\n        '
        self.file.write(b'#compdef ' + self.cmdName.encode('utf-8') + b'\n\n_arguments -s -A "-*" \\\n')

    def writeOptions(self):
        if False:
            while True:
                i = 10
        '\n        Write out zsh code for each option in this command\n        @return: L{None}\n        '
        optNames = list(self.allOptionsNameToDefinition.keys())
        optNames.sort()
        for longname in optNames:
            self.writeOpt(longname)

    def writeExtras(self):
        if False:
            print('Hello World!')
        '\n        Write out completion information for extra arguments appearing on the\n        command-line. These are extra positional arguments not associated\n        with a named option. That is, the stuff that gets passed to\n        Options.parseArgs().\n\n        @return: L{None}\n\n        @raise ValueError: If C{Completer} with C{repeat=True} is found and\n            is not the last item in the C{extraActions} list.\n        '
        for (i, action) in enumerate(self.extraActions):
            if action._repeat and i != len(self.extraActions) - 1:
                raise ValueError('Completer with repeat=True must be last item in Options.extraActions')
            self.file.write(escape(action._shellCode('', usage._ZSH)).encode('utf-8'))
            self.file.write(b' \\\n')

    def writeFooter(self):
        if False:
            print('Hello World!')
        '\n        Write the last bit of code that finishes the call to _arguments\n        @return: L{None}\n        '
        self.file.write(b'&& return 0\n')

    def verifyZshNames(self):
        if False:
            return 10
        '\n        Ensure that none of the option names given in the metadata are typoed\n        @return: L{None}\n        @raise ValueError: If unknown option names have been found.\n        '

        def err(name):
            if False:
                i = 10
                return i + 15
            raise ValueError('Unknown option name "%s" found while\nexamining Completions instances on %s' % (name, self.options))
        for name in itertools.chain(self.descriptions, self.optActions, self.multiUse):
            if name not in self.allOptionsNameToDefinition:
                err(name)
        for seq in self.mutuallyExclusive:
            for name in seq:
                if name not in self.allOptionsNameToDefinition:
                    err(name)

    def excludeStr(self, longname, buildShort=False):
        if False:
            while True:
                i = 10
        '\n        Generate an "exclusion string" for the given option\n\n        @type longname: C{str}\n        @param longname: The long option name (e.g. "verbose" instead of "v")\n\n        @type buildShort: C{bool}\n        @param buildShort: May be True to indicate we\'re building an excludes\n            string for the short option that corresponds to the given long opt.\n\n        @return: The generated C{str}\n        '
        if longname in self.excludes:
            exclusions = self.excludes[longname].copy()
        else:
            exclusions = set()
        if longname not in self.multiUse:
            if buildShort is False:
                short = self.getShortOption(longname)
                if short is not None:
                    exclusions.add(short)
            else:
                exclusions.add(longname)
        if not exclusions:
            return ''
        strings = []
        for optName in exclusions:
            if len(optName) == 1:
                strings.append('-' + optName)
            else:
                strings.append('--' + optName)
        strings.sort()
        return '(%s)' % ' '.join(strings)

    def makeExcludesDict(self) -> Dict[str, Set[str]]:
        if False:
            print('Hello World!')
        "\n        @return: A C{dict} that maps each option name appearing in\n            self.mutuallyExclusive to a set of those option names that is it\n            mutually exclusive with (can't appear on the cmd line with).\n        "
        longToShort = {}
        for optList in itertools.chain(self.optParams, self.optFlags):
            if optList[1] != None:
                longToShort[optList[0]] = optList[1]
        excludes: Dict[str, Set[str]] = {}
        for lst in self.mutuallyExclusive:
            for (i, longname) in enumerate(lst):
                tmp = set(lst[:i] + lst[i + 1:])
                for name in tmp.copy():
                    if name in longToShort:
                        tmp.add(longToShort[name])
                if longname in excludes:
                    excludes[longname] = excludes[longname].union(tmp)
                else:
                    excludes[longname] = tmp
        return excludes

    def writeOpt(self, longname):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write out the zsh code for the given argument. This is just part of the\n        one big call to _arguments\n\n        @type longname: C{str}\n        @param longname: The long option name (e.g. "verbose" instead of "v")\n\n        @return: L{None}\n        '
        if longname in self.flagNameToDefinition:
            longField = '--%s' % longname
        else:
            longField = '--%s=' % longname
        short = self.getShortOption(longname)
        if short != None:
            shortField = '-' + short
        else:
            shortField = ''
        descr = self.getDescription(longname)
        descriptionField = descr.replace('[', '\\[')
        descriptionField = descriptionField.replace(']', '\\]')
        descriptionField = '[%s]' % descriptionField
        actionField = self.getAction(longname)
        if longname in self.multiUse:
            multiField = '*'
        else:
            multiField = ''
        longExclusionsField = self.excludeStr(longname)
        if short:
            shortExclusionsField = self.excludeStr(longname, buildShort=True)
            self.file.write(escape('%s%s%s%s%s' % (shortExclusionsField, multiField, shortField, descriptionField, actionField)).encode('utf-8'))
            self.file.write(b' \\\n')
        self.file.write(escape('%s%s%s%s%s' % (longExclusionsField, multiField, longField, descriptionField, actionField)).encode('utf-8'))
        self.file.write(b' \\\n')

    def getAction(self, longname):
        if False:
            return 10
        '\n        Return a zsh "action" string for the given argument\n        @return: C{str}\n        '
        if longname in self.optActions:
            if callable(self.optActions[longname]):
                action = self.optActions[longname]()
            else:
                action = self.optActions[longname]
            return action._shellCode(longname, usage._ZSH)
        if longname in self.paramNameToDefinition:
            return f':{longname}:_files'
        return ''

    def getDescription(self, longname):
        if False:
            return 10
        '\n        Return the description to be used for this argument\n        @return: C{str}\n        '
        if longname in self.descriptions:
            return self.descriptions[longname]
        try:
            descr = self.flagNameToDefinition[longname][1]
        except KeyError:
            try:
                descr = self.paramNameToDefinition[longname][2]
            except KeyError:
                descr = None
        if descr is not None:
            return descr
        longMangled = longname.replace('-', '_')
        obj = getattr(self.options, 'opt_%s' % longMangled, None)
        if obj is not None:
            descr = descrFromDoc(obj)
            if descr is not None:
                return descr
        return longname

    def getShortOption(self, longname):
        if False:
            print('Hello World!')
        '\n        Return the short option letter or None\n        @return: C{str} or L{None}\n        '
        optList = self.allOptionsNameToDefinition[longname]
        return optList[0] or None

    def addAdditionalOptions(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Add additional options to the optFlags and optParams lists.\n        These will be defined by 'opt_foo' methods of the Options subclass\n        @return: L{None}\n        "
        methodsDict: Dict[str, MethodType] = {}
        reflect.accumulateMethods(self.options, methodsDict, 'opt_')
        methodToShort = {}
        for name in methodsDict.copy():
            if len(name) == 1:
                methodToShort[methodsDict[name]] = name
                del methodsDict[name]
        for (methodName, methodObj) in methodsDict.items():
            longname = methodName.replace('_', '-')
            if longname in self.allOptionsNameToDefinition:
                continue
            descr = self.getDescription(longname)
            short = None
            if methodObj in methodToShort:
                short = methodToShort[methodObj]
            reqArgs = methodObj.__func__.__code__.co_argcount
            if reqArgs == 2:
                self.optParams.append([longname, short, None, descr])
                self.paramNameToDefinition[longname] = [short, None, descr]
                self.allOptionsNameToDefinition[longname] = [short, None, descr]
            else:
                self.optFlags.append([longname, short, descr])
                self.flagNameToDefinition[longname] = [short, descr]
                self.allOptionsNameToDefinition[longname] = [short, None, descr]

def descrFromDoc(obj):
    if False:
        return 10
    '\n    Generate an appropriate description from docstring of the given object\n    '
    if obj.__doc__ is None or obj.__doc__.isspace():
        return None
    lines = [x.strip() for x in obj.__doc__.split('\n') if x and (not x.isspace())]
    return ' '.join(lines)

def escape(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Shell escape the given string\n\n    Implementation borrowed from now-deprecated commands.mkarg() in the stdlib\n    '
    if "'" not in x:
        return "'" + x + "'"
    s = '"'
    for c in x:
        if c in '\\$"`':
            s = s + '\\'
        s = s + c
    s = s + '"'
    return s