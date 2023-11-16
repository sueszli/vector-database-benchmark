"""
The :class:`~traitlets.config.application.Application` object for the command
line :command:`ipython` program.
"""
import logging
import os
import sys
import warnings
from traitlets.config.loader import Config
from traitlets.config.application import boolean_flag, catch_config_error
from IPython.core import release
from IPython.core import usage
from IPython.core.completer import IPCompleter
from IPython.core.crashhandler import CrashHandler
from IPython.core.formatters import PlainTextFormatter
from IPython.core.history import HistoryManager
from IPython.core.application import ProfileDir, BaseIPythonApplication, base_flags, base_aliases
from IPython.core.magic import MagicsManager
from IPython.core.magics import ScriptMagics, LoggingMagics
from IPython.core.shellapp import InteractiveShellApp, shell_flags, shell_aliases
from IPython.extensions.storemagic import StoreMagics
from .interactiveshell import TerminalInteractiveShell
from IPython.paths import get_ipython_dir
from traitlets import Bool, List, default, observe, Type
_examples = '\nipython --matplotlib       # enable matplotlib integration\nipython --matplotlib=qt    # enable matplotlib integration with qt4 backend\n\nipython --log-level=DEBUG  # set logging to DEBUG\nipython --profile=foo      # start with profile foo\n\nipython profile create foo # create profile foo w/ default config files\nipython help profile       # show the help for the profile subcmd\n\nipython locate             # print the path to the IPython directory\nipython locate profile foo # print the path to the directory for profile `foo`\n'

class IPAppCrashHandler(CrashHandler):
    """sys.excepthook for IPython itself, leaves a detailed report on disk."""

    def __init__(self, app):
        if False:
            for i in range(10):
                print('nop')
        contact_name = release.author
        contact_email = release.author_email
        bug_tracker = 'https://github.com/ipython/ipython/issues'
        super(IPAppCrashHandler, self).__init__(app, contact_name, contact_email, bug_tracker)

    def make_report(self, traceback):
        if False:
            print('Hello World!')
        'Return a string containing a crash report.'
        sec_sep = self.section_sep
        report = [super(IPAppCrashHandler, self).make_report(traceback)]
        rpt_add = report.append
        try:
            rpt_add(sec_sep + 'History of session input:')
            for line in self.app.shell.user_ns['_ih']:
                rpt_add(line)
            rpt_add('\n*** Last line of input (may not be in above history):\n')
            rpt_add(self.app.shell._last_input_line + '\n')
        except:
            pass
        return ''.join(report)
flags = dict(base_flags)
flags.update(shell_flags)
frontend_flags = {}
addflag = lambda *args: frontend_flags.update(boolean_flag(*args))
addflag('autoedit-syntax', 'TerminalInteractiveShell.autoedit_syntax', 'Turn on auto editing of files with syntax errors.', 'Turn off auto editing of files with syntax errors.')
addflag('simple-prompt', 'TerminalInteractiveShell.simple_prompt', 'Force simple minimal prompt using `raw_input`', 'Use a rich interactive prompt with prompt_toolkit')
addflag('banner', 'TerminalIPythonApp.display_banner', 'Display a banner upon starting IPython.', "Don't display a banner upon starting IPython.")
addflag('confirm-exit', 'TerminalInteractiveShell.confirm_exit', "Set to confirm when you try to exit IPython with an EOF (Control-D\n    in Unix, Control-Z/Enter in Windows). By typing 'exit' or 'quit',\n    you can force a direct exit without any confirmation.", "Don't prompt the user when exiting.")
addflag('term-title', 'TerminalInteractiveShell.term_title', 'Enable auto setting the terminal title.', 'Disable auto setting the terminal title.')
classic_config = Config()
classic_config.InteractiveShell.cache_size = 0
classic_config.PlainTextFormatter.pprint = False
classic_config.TerminalInteractiveShell.prompts_class = 'IPython.terminal.prompts.ClassicPrompts'
classic_config.InteractiveShell.separate_in = ''
classic_config.InteractiveShell.separate_out = ''
classic_config.InteractiveShell.separate_out2 = ''
classic_config.InteractiveShell.colors = 'NoColor'
classic_config.InteractiveShell.xmode = 'Plain'
frontend_flags['classic'] = (classic_config, 'Gives IPython a similar feel to the classic Python prompt.')
frontend_flags['quick'] = ({'TerminalIPythonApp': {'quick': True}}, 'Enable quick startup with no config files.')
frontend_flags['i'] = ({'TerminalIPythonApp': {'force_interact': True}}, 'If running code from the command line, become interactive afterwards.\n    It is often useful to follow this with `--` to treat remaining flags as\n    script arguments.\n    ')
flags.update(frontend_flags)
aliases = dict(base_aliases)
aliases.update(shell_aliases)

class LocateIPythonApp(BaseIPythonApplication):
    description = 'print the path to the IPython dir'
    subcommands = dict(profile=('IPython.core.profileapp.ProfileLocate', 'print the path to an IPython profile directory'))

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        if self.subapp is not None:
            return self.subapp.start()
        else:
            print(self.ipython_dir)

class TerminalIPythonApp(BaseIPythonApplication, InteractiveShellApp):
    name = u'ipython'
    description = usage.cl_usage
    crash_handler_class = IPAppCrashHandler
    examples = _examples
    flags = flags
    aliases = aliases
    classes = List()
    interactive_shell_class = Type(klass=object, default_value=TerminalInteractiveShell, help='Class to use to instantiate the TerminalInteractiveShell object. Useful for custom Frontends').tag(config=True)

    @default('classes')
    def _classes_default(self):
        if False:
            for i in range(10):
                print('nop')
        'This has to be in a method, for TerminalIPythonApp to be available.'
        return [InteractiveShellApp, self.__class__, TerminalInteractiveShell, HistoryManager, MagicsManager, ProfileDir, PlainTextFormatter, IPCompleter, ScriptMagics, LoggingMagics, StoreMagics]
    subcommands = dict(profile=('IPython.core.profileapp.ProfileApp', 'Create and manage IPython profiles.'), kernel=('ipykernel.kernelapp.IPKernelApp', 'Start a kernel without an attached frontend.'), locate=('IPython.terminal.ipapp.LocateIPythonApp', LocateIPythonApp.description), history=('IPython.core.historyapp.HistoryApp', 'Manage the IPython history database.'))
    auto_create = Bool(True)
    quick = Bool(False, help='Start IPython quickly by skipping the loading of config files.').tag(config=True)

    @observe('quick')
    def _quick_changed(self, change):
        if False:
            return 10
        if change['new']:
            self.load_config_file = lambda *a, **kw: None
    display_banner = Bool(True, help='Whether to display a banner upon starting IPython.').tag(config=True)
    force_interact = Bool(False, help="If a command or file is given via the command-line,\n        e.g. 'ipython foo.py', start an interactive shell after executing the\n        file or command.").tag(config=True)

    @observe('force_interact')
    def _force_interact_changed(self, change):
        if False:
            i = 10
            return i + 15
        if change['new']:
            self.interact = True

    @observe('file_to_run', 'code_to_run', 'module_to_run')
    def _file_to_run_changed(self, change):
        if False:
            print('Hello World!')
        new = change['new']
        if new:
            self.something_to_run = True
        if new and (not self.force_interact):
            self.interact = False
    something_to_run = Bool(False)

    @catch_config_error
    def initialize(self, argv=None):
        if False:
            i = 10
            return i + 15
        'Do actions after construct, but before starting the app.'
        super(TerminalIPythonApp, self).initialize(argv)
        if self.subapp is not None:
            return
        if self.extra_args and (not self.something_to_run):
            self.file_to_run = self.extra_args[0]
        self.init_path()
        self.init_shell()
        self.init_banner()
        self.init_gui_pylab()
        self.init_extensions()
        self.init_code()

    def init_shell(self):
        if False:
            i = 10
            return i + 15
        'initialize the InteractiveShell instance'
        self.shell = self.interactive_shell_class.instance(parent=self, profile_dir=self.profile_dir, ipython_dir=self.ipython_dir, user_ns=self.user_ns)
        self.shell.configurables.append(self)

    def init_banner(self):
        if False:
            i = 10
            return i + 15
        'optionally display the banner'
        if self.display_banner and self.interact:
            self.shell.show_banner()
        if self.log_level <= logging.INFO:
            print()

    def _pylab_changed(self, name, old, new):
        if False:
            while True:
                i = 10
        "Replace --pylab='inline' with --pylab='auto'"
        if new == 'inline':
            warnings.warn("'inline' not available as pylab backend, using 'auto' instead.")
            self.pylab = 'auto'

    def start(self):
        if False:
            while True:
                i = 10
        if self.subapp is not None:
            return self.subapp.start()
        if self.interact:
            self.log.debug("Starting IPython's mainloop...")
            self.shell.mainloop()
        else:
            self.log.debug('IPython not interactive...')
            self.shell.restore_term_title()
            if not self.shell.last_execution_succeeded:
                sys.exit(1)

def load_default_config(ipython_dir=None):
    if False:
        i = 10
        return i + 15
    'Load the default config file from the default ipython_dir.\n\n    This is useful for embedded shells.\n    '
    if ipython_dir is None:
        ipython_dir = get_ipython_dir()
    profile_dir = os.path.join(ipython_dir, 'profile_default')
    app = TerminalIPythonApp()
    app.config_file_paths.append(profile_dir)
    app.load_config_file()
    return app.config
launch_new_instance = TerminalIPythonApp.launch_instance