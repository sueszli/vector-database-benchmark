__revision__ = 'src/engine/SCons/Script/SConsOptions.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import optparse
import re
import sys
import textwrap
no_hyphen_re = re.compile('(\\s+|(?<=[\\w\\!\\"\\\'\\&\\.\\,\\?])-{2,}(?=\\w))')
try:
    from gettext import gettext
except ImportError:

    def gettext(message):
        if False:
            while True:
                i = 10
        return message
_ = gettext
import SCons.Node.FS
import SCons.Platform.virtualenv
import SCons.Warnings
OptionValueError = optparse.OptionValueError
SUPPRESS_HELP = optparse.SUPPRESS_HELP
diskcheck_all = SCons.Node.FS.diskcheck_types()

def diskcheck_convert(value):
    if False:
        while True:
            i = 10
    if value is None:
        return []
    if not SCons.Util.is_List(value):
        value = value.split(',')
    result = []
    for v in value:
        v = v.lower()
        if v == 'all':
            result = diskcheck_all
        elif v == 'none':
            result = []
        elif v in diskcheck_all:
            result.append(v)
        else:
            raise ValueError(v)
    return result

class SConsValues(optparse.Values):
    """
    Holder class for uniform access to SCons options, regardless
    of whether or not they can be set on the command line or in the
    SConscript files (using the SetOption() function).

    A SCons option value can originate three different ways:

        1)  set on the command line;
        2)  set in an SConscript file;
        3)  the default setting (from the the op.add_option()
            calls in the Parser() function, below).

    The command line always overrides a value set in a SConscript file,
    which in turn always overrides default settings.  Because we want
    to support user-specified options in the SConscript file itself,
    though, we may not know about all of the options when the command
    line is first parsed, so we can't make all the necessary precedence
    decisions at the time the option is configured.

    The solution implemented in this class is to keep these different sets
    of settings separate (command line, SConscript file, and default)
    and to override the __getattr__() method to check them in turn.
    This should allow the rest of the code to just fetch values as
    attributes of an instance of this class, without having to worry
    about where they came from.

    Note that not all command line options are settable from SConscript
    files, and the ones that are must be explicitly added to the
    "settable" list in this class, and optionally validated and coerced
    in the set_option() method.
    """

    def __init__(self, defaults):
        if False:
            return 10
        self.__dict__['__defaults__'] = defaults
        self.__dict__['__SConscript_settings__'] = {}

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        '\n        Fetches an options value, checking first for explicit settings\n        from the command line (which are direct attributes), then the\n        SConscript file settings, then the default values.\n        '
        try:
            return self.__dict__[attr]
        except KeyError:
            try:
                return self.__dict__['__SConscript_settings__'][attr]
            except KeyError:
                try:
                    return getattr(self.__dict__['__defaults__'], attr)
                except KeyError:
                    raise AttributeError(attr)
    settable = ['clean', 'diskcheck', 'duplicate', 'help', 'implicit_cache', 'max_drift', 'md5_chunksize', 'no_exec', 'num_jobs', 'random', 'stack_size', 'warn', 'silent']

    def set_option(self, name, value):
        if False:
            print('Hello World!')
        '\n        Sets an option from an SConscript file.\n        '
        if name not in self.settable:
            raise SCons.Errors.UserError('This option is not settable from a SConscript file: %s' % name)
        if name == 'num_jobs':
            try:
                value = int(value)
                if value < 1:
                    raise ValueError
            except ValueError:
                raise SCons.Errors.UserError('A positive integer is required: %s' % repr(value))
        elif name == 'max_drift':
            try:
                value = int(value)
            except ValueError:
                raise SCons.Errors.UserError('An integer is required: %s' % repr(value))
        elif name == 'duplicate':
            try:
                value = str(value)
            except ValueError:
                raise SCons.Errors.UserError('A string is required: %s' % repr(value))
            if value not in SCons.Node.FS.Valid_Duplicates:
                raise SCons.Errors.UserError('Not a valid duplication style: %s' % value)
            SCons.Node.FS.set_duplicate(value)
        elif name == 'diskcheck':
            try:
                value = diskcheck_convert(value)
            except ValueError as v:
                raise SCons.Errors.UserError('Not a valid diskcheck value: %s' % v)
            if 'diskcheck' not in self.__dict__:
                SCons.Node.FS.set_diskcheck(value)
        elif name == 'stack_size':
            try:
                value = int(value)
            except ValueError:
                raise SCons.Errors.UserError('An integer is required: %s' % repr(value))
        elif name == 'md5_chunksize':
            try:
                value = int(value)
            except ValueError:
                raise SCons.Errors.UserError('An integer is required: %s' % repr(value))
        elif name == 'warn':
            if SCons.Util.is_String(value):
                value = [value]
            value = self.__SConscript_settings__.get(name, []) + value
            SCons.Warnings.process_warn_strings(value)
        self.__SConscript_settings__[name] = value

class SConsOption(optparse.Option):

    def convert_value(self, opt, value):
        if False:
            return 10
        if value is not None:
            if self.nargs in (1, '?'):
                return self.check_value(opt, value)
            else:
                return tuple([self.check_value(opt, v) for v in value])

    def process(self, opt, value, values, parser):
        if False:
            i = 10
            return i + 15
        value = self.convert_value(opt, value)
        return self.take_action(self.action, self.dest, opt, value, values, parser)

    def _check_nargs_optional(self):
        if False:
            i = 10
            return i + 15
        if self.nargs == '?' and self._short_opts:
            fmt = "option %s: nargs='?' is incompatible with short options"
            raise SCons.Errors.UserError(fmt % self._short_opts[0])
    CHECK_METHODS = optparse.Option.CHECK_METHODS + [_check_nargs_optional]
    CONST_ACTIONS = optparse.Option.CONST_ACTIONS + optparse.Option.TYPED_ACTIONS

class SConsOptionGroup(optparse.OptionGroup):
    """
    A subclass for SCons-specific option groups.

    The only difference between this and the base class is that we print
    the group's help text flush left, underneath their own title but
    lined up with the normal "SCons Options".
    """

    def format_help(self, formatter):
        if False:
            while True:
                i = 10
        '\n        Format an option group\'s help text, outdenting the title so it\'s\n        flush with the "SCons Options" title we print at the top.\n        '
        formatter.dedent()
        result = formatter.format_heading(self.title)
        formatter.indent()
        result = result + optparse.OptionContainer.format_help(self, formatter)
        return result

class SConsOptionParser(optparse.OptionParser):
    preserve_unknown_options = False

    def error(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self.print_usage(sys.stderr)
        sys.stderr.write('SCons Error: %s\n' % msg)
        sys.exit(2)

    def _process_long_opt(self, rargs, values):
        if False:
            while True:
                i = 10
        '\n        SCons-specific processing of long options.\n\n        This is copied directly from the normal\n        optparse._process_long_opt() method, except that, if configured\n        to do so, we catch the exception thrown when an unknown option\n        is encountered and just stick it back on the "leftover" arguments\n        for later (re-)processing.\n        '
        arg = rargs.pop(0)
        if '=' in arg:
            (opt, next_arg) = arg.split('=', 1)
            rargs.insert(0, next_arg)
            had_explicit_value = True
        else:
            opt = arg
            had_explicit_value = False
        try:
            opt = self._match_long_opt(opt)
        except optparse.BadOptionError:
            if self.preserve_unknown_options:
                self.largs.append(arg)
                if had_explicit_value:
                    rargs.pop(0)
                return
            raise
        option = self._long_opt[opt]
        if option.takes_value():
            nargs = option.nargs
            if nargs == '?':
                if had_explicit_value:
                    value = rargs.pop(0)
                else:
                    value = option.const
            elif len(rargs) < nargs:
                if nargs == 1:
                    if not option.choices:
                        self.error(_('%s option requires an argument') % opt)
                    else:
                        msg = _('%s option requires an argument ' % opt)
                        msg += _('(choose from %s)' % ', '.join(option.choices))
                        self.error(msg)
                else:
                    self.error(_('%s option requires %d arguments') % (opt, nargs))
            elif nargs == 1:
                value = rargs.pop(0)
            else:
                value = tuple(rargs[0:nargs])
                del rargs[0:nargs]
        elif had_explicit_value:
            self.error(_('%s option does not take a value') % opt)
        else:
            value = None
        option.process(opt, value, values, self)

    def reparse_local_options(self):
        if False:
            i = 10
            return i + 15
        '\n        Re-parse the leftover command-line options stored\n        in self.largs, so that any value overridden on the\n        command line is immediately available if the user turns\n        around and does a GetOption() right away.\n\n        We mimic the processing of the single args\n        in the original OptionParser._process_args(), but here we\n        allow exact matches for long-opts only (no partial\n        argument names!).\n\n        Else, this would lead to problems in add_local_option()\n        below. When called from there, we try to reparse the\n        command-line arguments that\n          1. haven\'t been processed so far (self.largs), but\n          2. are possibly not added to the list of options yet.\n\n        So, when we only have a value for "--myargument" yet,\n        a command-line argument of "--myarg=test" would set it.\n        Responsible for this behaviour is the method\n        _match_long_opt(), which allows for partial matches of\n        the option name, as long as the common prefix appears to\n        be unique.\n        This would lead to further confusion, because we might want\n        to add another option "--myarg" later on (see issue #2929).\n\n        '
        rargs = []
        largs_restore = []
        skip = False
        for l in self.largs:
            if skip:
                largs_restore.append(l)
            elif len(l) > 2 and l[0:2] == '--':
                lopt = (l,)
                if '=' in l:
                    lopt = l.split('=', 1)
                if lopt[0] in self._long_opt:
                    rargs.append('='.join(lopt))
                else:
                    largs_restore.append('='.join(lopt))
            elif l == '--' or l == '-':
                largs_restore.append(l)
                skip = True
            else:
                rargs.append(l)
        self.parse_args(rargs, self.values)
        self.largs = self.largs + largs_restore

    def add_local_option(self, *args, **kw):
        if False:
            print('Hello World!')
        '\n        Adds a local option to the parser.\n\n        This is initiated by a SetOption() call to add a user-defined\n        command-line option.  We add the option to a separate option\n        group for the local options, creating the group if necessary.\n        '
        try:
            group = self.local_option_group
        except AttributeError:
            group = SConsOptionGroup(self, 'Local Options')
            group = self.add_option_group(group)
            self.local_option_group = group
        result = group.add_option(*args, **kw)
        if result:
            setattr(self.values.__defaults__, result.dest, result.default)
            self.reparse_local_options()
        return result

class SConsIndentedHelpFormatter(optparse.IndentedHelpFormatter):

    def format_usage(self, usage):
        if False:
            return 10
        return 'usage: %s\n' % usage

    def format_heading(self, heading):
        if False:
            print('Hello World!')
        '\n        This translates any heading of "options" or "Options" into\n        "SCons Options."  Unfortunately, we have to do this here,\n        because those titles are hard-coded in the optparse calls.\n        '
        if heading == 'Options':
            heading = 'SCons Options'
        return optparse.IndentedHelpFormatter.format_heading(self, heading)

    def format_option(self, option):
        if False:
            while True:
                i = 10
        "\n        A copy of the normal optparse.IndentedHelpFormatter.format_option()\n        method.  This has been snarfed so we can modify text wrapping to\n        out liking:\n\n        --  add our own regular expression that doesn't break on hyphens\n            (so things like --no-print-directory don't get broken);\n\n        --  wrap the list of options themselves when it's too long\n            (the wrapper.fill(opts) call below);\n\n        --  set the subsequent_indent when wrapping the help_text.\n        "
        result = []
        opts = self.option_strings[option]
        opt_width = self.help_position - self.current_indent - 2
        if len(opts) > opt_width:
            wrapper = textwrap.TextWrapper(width=self.width, initial_indent='  ', subsequent_indent='  ')
            wrapper.wordsep_re = no_hyphen_re
            opts = wrapper.fill(opts) + '\n'
            indent_first = self.help_position
        else:
            opts = '%*s%-*s  ' % (self.current_indent, '', opt_width, opts)
            indent_first = 0
        result.append(opts)
        if option.help:
            help_text = self.expand_default(option)
            wrapper = textwrap.TextWrapper(width=self.help_width, subsequent_indent='  ')
            wrapper.wordsep_re = no_hyphen_re
            help_lines = wrapper.wrap(help_text)
            result.append('%*s%s\n' % (indent_first, '', help_lines[0]))
            for line in help_lines[1:]:
                result.append('%*s%s\n' % (self.help_position, '', line))
        elif opts[-1] != '\n':
            result.append('\n')
        return ''.join(result)

def Parser(version):
    if False:
        i = 10
        return i + 15
    '\n    Returns an options parser object initialized with the standard\n    SCons options.\n    '
    formatter = SConsIndentedHelpFormatter(max_help_position=30)
    op = SConsOptionParser(option_class=SConsOption, add_help_option=False, formatter=formatter, usage='usage: scons [OPTION] [TARGET] ...')
    op.preserve_unknown_options = True
    op.version = version

    def opt_ignore(option, opt, value, parser):
        if False:
            print('Hello World!')
        sys.stderr.write('Warning:  ignoring %s option\n' % opt)
    op.add_option('-b', '-d', '-e', '-m', '-S', '-t', '-w', '--environment-overrides', '--no-keep-going', '--no-print-directory', '--print-directory', '--stop', '--touch', action='callback', callback=opt_ignore, help='Ignored for compatibility.')
    op.add_option('-c', '--clean', '--remove', dest='clean', default=False, action='store_true', help='Remove specified targets and dependencies.')
    op.add_option('-C', '--directory', nargs=1, type='string', dest='directory', default=[], action='append', help='Change to DIR before doing anything.', metavar='DIR')
    op.add_option('--cache-debug', nargs=1, dest='cache_debug', default=None, action='store', help='Print CacheDir debug info to FILE.', metavar='FILE')
    op.add_option('--cache-disable', '--no-cache', dest='cache_disable', default=False, action='store_true', help='Do not retrieve built targets from CacheDir.')
    op.add_option('--cache-force', '--cache-populate', dest='cache_force', default=False, action='store_true', help='Copy already-built targets into the CacheDir.')
    op.add_option('--cache-readonly', dest='cache_readonly', default=False, action='store_true', help='Do not update CacheDir with built targets.')
    op.add_option('--cache-show', dest='cache_show', default=False, action='store_true', help='Print build actions for files from CacheDir.')

    def opt_invalid(group, value, options):
        if False:
            print('Hello World!')
        'report an invalid option from a group'
        errmsg = "`%s' is not a valid %s option type, try:\n" % (value, group)
        return errmsg + '    %s' % ', '.join(options)

    def opt_invalid_rm(group, value, msg):
        if False:
            print('Hello World!')
        'report an invalid option from a group: recognized but removed'
        errmsg = "`%s' is not a valid %s option type " % (value, group)
        return errmsg + msg
    config_options = ['auto', 'force', 'cache']
    opt_config_help = 'Controls Configure subsystem: %s.' % ', '.join(config_options)
    op.add_option('--config', nargs=1, choices=config_options, dest='config', default='auto', help=opt_config_help, metavar='MODE')
    op.add_option('-D', dest='climb_up', default=None, action='store_const', const=2, help='Search up directory tree for SConstruct,       build all Default() targets.')
    deprecated_debug_options = {}
    removed_debug_options = {'dtree': '; please use --tree=derived instead', 'nomemoizer': '; there is no replacement', 'stree': '; please use --tree=all,status instead', 'tree': '; please use --tree=all instead'}
    debug_options = ['count', 'duplicate', 'explain', 'findlibs', 'includes', 'memoizer', 'memory', 'objects', 'pdb', 'prepare', 'presub', 'stacktrace', 'time', 'action-timestamps']

    def opt_debug(option, opt, value__, parser, debug_options=debug_options, deprecated_debug_options=deprecated_debug_options, removed_debug_options=removed_debug_options):
        if False:
            print('Hello World!')
        for value in value__.split(','):
            if value in debug_options:
                parser.values.debug.append(value)
            elif value in deprecated_debug_options:
                parser.values.debug.append(value)
                try:
                    parser.values.delayed_warnings
                except AttributeError:
                    parser.values.delayed_warnings = []
                msg = deprecated_debug_options[value]
                w = 'The --debug=%s option is deprecated%s.' % (value, msg)
                t = (SCons.Warnings.DeprecatedDebugOptionsWarning, w)
                parser.values.delayed_warnings.append(t)
            elif value in removed_debug_options:
                msg = removed_debug_options[value]
                raise OptionValueError(opt_invalid_rm('debug', value, msg))
            else:
                raise OptionValueError(opt_invalid('debug', value, debug_options))
    opt_debug_help = 'Print various types of debugging information: %s.' % ', '.join(debug_options)
    op.add_option('--debug', nargs=1, type='string', dest='debug', default=[], action='callback', callback=opt_debug, help=opt_debug_help, metavar='TYPE')

    def opt_diskcheck(option, opt, value, parser):
        if False:
            while True:
                i = 10
        try:
            diskcheck_value = diskcheck_convert(value)
        except ValueError as e:
            raise OptionValueError("`%s' is not a valid diskcheck type" % e)
        setattr(parser.values, option.dest, diskcheck_value)
    op.add_option('--diskcheck', nargs=1, type='string', dest='diskcheck', default=None, action='callback', callback=opt_diskcheck, help='Enable specific on-disk checks.', metavar='TYPE')

    def opt_duplicate(option, opt, value, parser):
        if False:
            print('Hello World!')
        if value not in SCons.Node.FS.Valid_Duplicates:
            raise OptionValueError(opt_invalid('duplication', value, SCons.Node.FS.Valid_Duplicates))
        setattr(parser.values, option.dest, value)
        SCons.Node.FS.set_duplicate(value)
    opt_duplicate_help = 'Set the preferred duplication methods. Must be one of ' + ', '.join(SCons.Node.FS.Valid_Duplicates)
    op.add_option('--duplicate', nargs=1, type='string', dest='duplicate', default='hard-soft-copy', action='callback', callback=opt_duplicate, help=opt_duplicate_help)
    if not SCons.Platform.virtualenv.virtualenv_enabled_by_default:
        op.add_option('--enable-virtualenv', dest='enable_virtualenv', action='store_true', help='Import certain virtualenv variables to SCons')
    op.add_option('-f', '--file', '--makefile', '--sconstruct', nargs=1, type='string', dest='file', default=[], action='append', help='Read FILE as the top-level SConstruct file.')
    op.add_option('-h', '--help', dest='help', default=False, action='store_true', help='Print defined help message, or this one.')
    op.add_option('-H', '--help-options', action='help', help='Print this message and exit.')
    op.add_option('-i', '--ignore-errors', dest='ignore_errors', default=False, action='store_true', help='Ignore errors from build actions.')
    op.add_option('-I', '--include-dir', nargs=1, dest='include_dir', default=[], action='append', help='Search DIR for imported Python modules.', metavar='DIR')
    op.add_option('--ignore-virtualenv', dest='ignore_virtualenv', action='store_true', help='Do not import virtualenv variables to SCons')
    op.add_option('--implicit-cache', dest='implicit_cache', default=False, action='store_true', help='Cache implicit dependencies')

    def opt_implicit_deps(option, opt, value, parser):
        if False:
            while True:
                i = 10
        setattr(parser.values, 'implicit_cache', True)
        setattr(parser.values, option.dest, True)
    op.add_option('--implicit-deps-changed', dest='implicit_deps_changed', default=False, action='callback', callback=opt_implicit_deps, help='Ignore cached implicit dependencies.')
    op.add_option('--implicit-deps-unchanged', dest='implicit_deps_unchanged', default=False, action='callback', callback=opt_implicit_deps, help='Ignore changes in implicit dependencies.')
    op.add_option('--interact', '--interactive', dest='interactive', default=False, action='store_true', help='Run in interactive mode.')
    op.add_option('-j', '--jobs', nargs=1, type='int', dest='num_jobs', default=1, action='store', help='Allow N jobs at once.', metavar='N')
    op.add_option('-k', '--keep-going', dest='keep_going', default=False, action='store_true', help="Keep going when a target can't be made.")
    op.add_option('--max-drift', nargs=1, type='int', dest='max_drift', default=SCons.Node.FS.default_max_drift, action='store', help='Set maximum system clock drift to N seconds.', metavar='N')
    op.add_option('--md5-chunksize', nargs=1, type='int', dest='md5_chunksize', default=SCons.Node.FS.File.md5_chunksize, action='store', help='Set chunk-size for MD5 signature computation to N kilobytes.', metavar='N')
    op.add_option('-n', '--no-exec', '--just-print', '--dry-run', '--recon', dest='no_exec', default=False, action='store_true', help="Don't build; just print commands.")
    op.add_option('--no-site-dir', dest='no_site_dir', default=False, action='store_true', help="Don't search or use the usual site_scons dir.")
    op.add_option('--profile', nargs=1, dest='profile_file', default=None, action='store', help='Profile SCons and put results in FILE.', metavar='FILE')
    op.add_option('-q', '--question', dest='question', default=False, action='store_true', help="Don't build; exit status says if up to date.")
    op.add_option('-Q', dest='no_progress', default=False, action='store_true', help='Suppress "Reading/Building" progress messages.')
    op.add_option('--random', dest='random', default=False, action='store_true', help='Build dependencies in random order.')
    op.add_option('-s', '--silent', '--quiet', dest='silent', default=False, action='store_true', help="Don't print commands.")
    op.add_option('--site-dir', nargs=1, dest='site_dir', default=None, action='store', help='Use DIR instead of the usual site_scons dir.', metavar='DIR')
    op.add_option('--stack-size', nargs=1, type='int', dest='stack_size', action='store', help='Set the stack size of the threads used to run jobs to N kilobytes.', metavar='N')
    op.add_option('--taskmastertrace', nargs=1, dest='taskmastertrace_file', default=None, action='store', help='Trace Node evaluation to FILE.', metavar='FILE')
    tree_options = ['all', 'derived', 'prune', 'status']

    def opt_tree(option, opt, value, parser, tree_options=tree_options):
        if False:
            for i in range(10):
                print('nop')
        from . import Main
        tp = Main.TreePrinter()
        for o in value.split(','):
            if o == 'all':
                tp.derived = False
            elif o == 'derived':
                tp.derived = True
            elif o == 'prune':
                tp.prune = True
            elif o == 'status':
                tp.status = True
            else:
                raise OptionValueError(opt_invalid('--tree', o, tree_options))
        parser.values.tree_printers.append(tp)
    opt_tree_help = 'Print a dependency tree in various formats: %s.' % ', '.join(tree_options)
    op.add_option('--tree', nargs=1, type='string', dest='tree_printers', default=[], action='callback', callback=opt_tree, help=opt_tree_help, metavar='OPTIONS')
    op.add_option('-u', '--up', '--search-up', dest='climb_up', default=0, action='store_const', const=1, help='Search up directory tree for SConstruct,       build targets at or below current directory.')
    op.add_option('-U', dest='climb_up', default=0, action='store_const', const=3, help='Search up directory tree for SConstruct,       build Default() targets from local SConscript.')

    def opt_version(option, opt, value, parser):
        if False:
            while True:
                i = 10
        sys.stdout.write(parser.version + '\n')
        sys.exit(0)
    op.add_option('-v', '--version', action='callback', callback=opt_version, help='Print the SCons version number and exit.')

    def opt_warn(option, opt, value, parser, tree_options=tree_options):
        if False:
            for i in range(10):
                print('nop')
        if SCons.Util.is_String(value):
            value = value.split(',')
        parser.values.warn.extend(value)
    op.add_option('--warn', '--warning', nargs=1, type='string', dest='warn', default=[], action='callback', callback=opt_warn, help='Enable or disable warnings.', metavar='WARNING-SPEC')
    op.add_option('-Y', '--repository', '--srcdir', nargs=1, dest='repository', default=[], action='append', help='Search REPOSITORY for source and target files.')

    def opt_not_yet(option, opt, value, parser):
        if False:
            while True:
                i = 10
        msg = 'Warning:  the %s option is not yet implemented\n' % opt
        sys.stderr.write(msg)
    op.add_option('-l', '--load-average', '--max-load', nargs=1, type='float', dest='load_average', default=0, action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    op.add_option('--list-actions', dest='list_actions', action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    op.add_option('--list-derived', dest='list_derived', action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    op.add_option('--list-where', dest='list_where', action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    op.add_option('-o', '--old-file', '--assume-old', nargs=1, type='string', dest='old_file', default=[], action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    op.add_option('--override', nargs=1, type='string', action='callback', callback=opt_not_yet, dest='override', help=SUPPRESS_HELP)
    op.add_option('-p', action='callback', callback=opt_not_yet, dest='p', help=SUPPRESS_HELP)
    op.add_option('-r', '-R', '--no-builtin-rules', '--no-builtin-variables', action='callback', callback=opt_not_yet, dest='no_builtin_rules', help=SUPPRESS_HELP)
    op.add_option('--write-filenames', nargs=1, type='string', dest='write_filenames', action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    op.add_option('-W', '--new-file', '--assume-new', '--what-if', nargs=1, type='string', dest='new_file', action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    op.add_option('--warn-undefined-variables', dest='warn_undefined_variables', action='callback', callback=opt_not_yet, help=SUPPRESS_HELP)
    return op