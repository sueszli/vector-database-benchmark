"""
Prefiltering components.

Prefilters transform user input before it is exec'd by Python.  These
transforms are used to implement additional syntax such as !ls and %magic.
"""
from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import ESC_MAGIC, ESC_QUOTE, ESC_QUOTE2, ESC_PAREN
from .macro import Macro
from .splitinput import LineInfo
from traitlets import List, Integer, Unicode, Bool, Instance, CRegExp

class PrefilterError(Exception):
    pass
re_fun_name = re.compile('[^\\W\\d]([\\w.]*) *$')
re_exclude_auto = re.compile('^[,&^\\|\\*/\\+-]|^is |^not |^in |^and |^or ')

def is_shadowed(identifier, ip):
    if False:
        print('Hello World!')
    "Is the given identifier defined in one of the namespaces which shadow\n    the alias and magic namespaces?  Note that an identifier is different\n    than ifun, because it can not contain a '.' character."
    return identifier in ip.user_ns or identifier in ip.user_global_ns or identifier in ip.ns_table['builtin'] or iskeyword(identifier)

class PrefilterManager(Configurable):
    """Main prefilter component.

    The IPython prefilter is run on all user input before it is run.  The
    prefilter consumes lines of input and produces transformed lines of
    input.

    The implementation consists of two phases:

    1. Transformers
    2. Checkers and handlers

    Over time, we plan on deprecating the checkers and handlers and doing
    everything in the transformers.

    The transformers are instances of :class:`PrefilterTransformer` and have
    a single method :meth:`transform` that takes a line and returns a
    transformed line.  The transformation can be accomplished using any
    tool, but our current ones use regular expressions for speed.

    After all the transformers have been run, the line is fed to the checkers,
    which are instances of :class:`PrefilterChecker`.  The line is passed to
    the :meth:`check` method, which either returns `None` or a
    :class:`PrefilterHandler` instance.  If `None` is returned, the other
    checkers are tried.  If an :class:`PrefilterHandler` instance is returned,
    the line is passed to the :meth:`handle` method of the returned
    handler and no further checkers are tried.

    Both transformers and checkers have a `priority` attribute, that determines
    the order in which they are called.  Smaller priorities are tried first.

    Both transformers and checkers also have `enabled` attribute, which is
    a boolean that determines if the instance is used.

    Users or developers can change the priority or enabled attribute of
    transformers or checkers, but they must call the :meth:`sort_checkers`
    or :meth:`sort_transformers` method after changing the priority.
    """
    multi_line_specials = Bool(True).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)

    def __init__(self, shell=None, **kwargs):
        if False:
            while True:
                i = 10
        super(PrefilterManager, self).__init__(shell=shell, **kwargs)
        self.shell = shell
        self._transformers = []
        self.init_handlers()
        self.init_checkers()

    def sort_transformers(self):
        if False:
            for i in range(10):
                print('nop')
        'Sort the transformers by priority.\n\n        This must be called after the priority of a transformer is changed.\n        The :meth:`register_transformer` method calls this automatically.\n        '
        self._transformers.sort(key=lambda x: x.priority)

    @property
    def transformers(self):
        if False:
            while True:
                i = 10
        'Return a list of checkers, sorted by priority.'
        return self._transformers

    def register_transformer(self, transformer):
        if False:
            return 10
        'Register a transformer instance.'
        if transformer not in self._transformers:
            self._transformers.append(transformer)
            self.sort_transformers()

    def unregister_transformer(self, transformer):
        if False:
            for i in range(10):
                print('nop')
        'Unregister a transformer instance.'
        if transformer in self._transformers:
            self._transformers.remove(transformer)

    def init_checkers(self):
        if False:
            for i in range(10):
                print('nop')
        'Create the default checkers.'
        self._checkers = []
        for checker in _default_checkers:
            checker(shell=self.shell, prefilter_manager=self, parent=self)

    def sort_checkers(self):
        if False:
            while True:
                i = 10
        'Sort the checkers by priority.\n\n        This must be called after the priority of a checker is changed.\n        The :meth:`register_checker` method calls this automatically.\n        '
        self._checkers.sort(key=lambda x: x.priority)

    @property
    def checkers(self):
        if False:
            return 10
        'Return a list of checkers, sorted by priority.'
        return self._checkers

    def register_checker(self, checker):
        if False:
            while True:
                i = 10
        'Register a checker instance.'
        if checker not in self._checkers:
            self._checkers.append(checker)
            self.sort_checkers()

    def unregister_checker(self, checker):
        if False:
            return 10
        'Unregister a checker instance.'
        if checker in self._checkers:
            self._checkers.remove(checker)

    def init_handlers(self):
        if False:
            return 10
        'Create the default handlers.'
        self._handlers = {}
        self._esc_handlers = {}
        for handler in _default_handlers:
            handler(shell=self.shell, prefilter_manager=self, parent=self)

    @property
    def handlers(self):
        if False:
            print('Hello World!')
        'Return a dict of all the handlers.'
        return self._handlers

    def register_handler(self, name, handler, esc_strings):
        if False:
            print('Hello World!')
        'Register a handler instance by name with esc_strings.'
        self._handlers[name] = handler
        for esc_str in esc_strings:
            self._esc_handlers[esc_str] = handler

    def unregister_handler(self, name, handler, esc_strings):
        if False:
            while True:
                i = 10
        'Unregister a handler instance by name with esc_strings.'
        try:
            del self._handlers[name]
        except KeyError:
            pass
        for esc_str in esc_strings:
            h = self._esc_handlers.get(esc_str)
            if h is handler:
                del self._esc_handlers[esc_str]

    def get_handler_by_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Get a handler by its name.'
        return self._handlers.get(name)

    def get_handler_by_esc(self, esc_str):
        if False:
            while True:
                i = 10
        'Get a handler by its escape string.'
        return self._esc_handlers.get(esc_str)

    def prefilter_line_info(self, line_info):
        if False:
            for i in range(10):
                print('nop')
        'Prefilter a line that has been converted to a LineInfo object.\n\n        This implements the checker/handler part of the prefilter pipe.\n        '
        handler = self.find_handler(line_info)
        return handler.handle(line_info)

    def find_handler(self, line_info):
        if False:
            i = 10
            return i + 15
        'Find a handler for the line_info by trying checkers.'
        for checker in self.checkers:
            if checker.enabled:
                handler = checker.check(line_info)
                if handler:
                    return handler
        return self.get_handler_by_name('normal')

    def transform_line(self, line, continue_prompt):
        if False:
            i = 10
            return i + 15
        'Calls the enabled transformers in order of increasing priority.'
        for transformer in self.transformers:
            if transformer.enabled:
                line = transformer.transform(line, continue_prompt)
        return line

    def prefilter_line(self, line, continue_prompt=False):
        if False:
            return 10
        'Prefilter a single input line as text.\n\n        This method prefilters a single line of text by calling the\n        transformers and then the checkers/handlers.\n        '
        self.shell._last_input_line = line
        if not line:
            return ''
        if not continue_prompt or (continue_prompt and self.multi_line_specials):
            line = self.transform_line(line, continue_prompt)
        line_info = LineInfo(line, continue_prompt)
        stripped = line.strip()
        normal_handler = self.get_handler_by_name('normal')
        if not stripped:
            return normal_handler.handle(line_info)
        if continue_prompt and (not self.multi_line_specials):
            return normal_handler.handle(line_info)
        prefiltered = self.prefilter_line_info(line_info)
        return prefiltered

    def prefilter_lines(self, lines, continue_prompt=False):
        if False:
            i = 10
            return i + 15
        'Prefilter multiple input lines of text.\n\n        This is the main entry point for prefiltering multiple lines of\n        input.  This simply calls :meth:`prefilter_line` for each line of\n        input.\n\n        This covers cases where there are multiple lines in the user entry,\n        which is the case when the user goes back to a multiline history\n        entry and presses enter.\n        '
        llines = lines.rstrip('\n').split('\n')
        if len(llines) > 1:
            out = '\n'.join([self.prefilter_line(line, lnum > 0) for (lnum, line) in enumerate(llines)])
        else:
            out = self.prefilter_line(llines[0], continue_prompt)
        return out

class PrefilterTransformer(Configurable):
    """Transform a line of user input."""
    priority = Integer(100).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)
    enabled = Bool(True).tag(config=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        if False:
            return 10
        super(PrefilterTransformer, self).__init__(shell=shell, prefilter_manager=prefilter_manager, **kwargs)
        self.prefilter_manager.register_transformer(self)

    def transform(self, line, continue_prompt):
        if False:
            return 10
        'Transform a line, returning the new one.'
        return None

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s(priority=%r, enabled=%r)>' % (self.__class__.__name__, self.priority, self.enabled)

class PrefilterChecker(Configurable):
    """Inspect an input line and return a handler for that line."""
    priority = Integer(100).tag(config=True)
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)
    enabled = Bool(True).tag(config=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        if False:
            i = 10
            return i + 15
        super(PrefilterChecker, self).__init__(shell=shell, prefilter_manager=prefilter_manager, **kwargs)
        self.prefilter_manager.register_checker(self)

    def check(self, line_info):
        if False:
            for i in range(10):
                print('nop')
        'Inspect line_info and return a handler instance or None.'
        return None

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<%s(priority=%r, enabled=%r)>' % (self.__class__.__name__, self.priority, self.enabled)

class EmacsChecker(PrefilterChecker):
    priority = Integer(100).tag(config=True)
    enabled = Bool(False).tag(config=True)

    def check(self, line_info):
        if False:
            return 10
        'Emacs ipython-mode tags certain input lines.'
        if line_info.line.endswith('# PYTHON-MODE'):
            return self.prefilter_manager.get_handler_by_name('emacs')
        else:
            return None

class MacroChecker(PrefilterChecker):
    priority = Integer(250).tag(config=True)

    def check(self, line_info):
        if False:
            for i in range(10):
                print('nop')
        obj = self.shell.user_ns.get(line_info.ifun)
        if isinstance(obj, Macro):
            return self.prefilter_manager.get_handler_by_name('macro')
        else:
            return None

class IPyAutocallChecker(PrefilterChecker):
    priority = Integer(300).tag(config=True)

    def check(self, line_info):
        if False:
            print('Hello World!')
        'Instances of IPyAutocall in user_ns get autocalled immediately'
        obj = self.shell.user_ns.get(line_info.ifun, None)
        if isinstance(obj, IPyAutocall):
            obj.set_ip(self.shell)
            return self.prefilter_manager.get_handler_by_name('auto')
        else:
            return None

class AssignmentChecker(PrefilterChecker):
    priority = Integer(600).tag(config=True)

    def check(self, line_info):
        if False:
            return 10
        "Check to see if user is assigning to a var for the first time, in\n        which case we want to avoid any sort of automagic / autocall games.\n\n        This allows users to assign to either alias or magic names true python\n        variables (the magic/alias systems always take second seat to true\n        python code).  E.g. ls='hi', or ls,that=1,2"
        if line_info.the_rest:
            if line_info.the_rest[0] in '=,':
                return self.prefilter_manager.get_handler_by_name('normal')
        else:
            return None

class AutoMagicChecker(PrefilterChecker):
    priority = Integer(700).tag(config=True)

    def check(self, line_info):
        if False:
            i = 10
            return i + 15
        "If the ifun is magic, and automagic is on, run it.  Note: normal,\n        non-auto magic would already have been triggered via '%' in\n        check_esc_chars. This just checks for automagic.  Also, before\n        triggering the magic handler, make sure that there is nothing in the\n        user namespace which could shadow it."
        if not self.shell.automagic or not self.shell.find_magic(line_info.ifun):
            return None
        if line_info.continue_prompt and (not self.prefilter_manager.multi_line_specials):
            return None
        head = line_info.ifun.split('.', 1)[0]
        if is_shadowed(head, self.shell):
            return None
        return self.prefilter_manager.get_handler_by_name('magic')

class PythonOpsChecker(PrefilterChecker):
    priority = Integer(900).tag(config=True)

    def check(self, line_info):
        if False:
            while True:
                i = 10
        "If the 'rest' of the line begins with a function call or pretty much\n        any python operator, we should simply execute the line (regardless of\n        whether or not there's a possible autocall expansion).  This avoids\n        spurious (and very confusing) geattr() accesses."
        if line_info.the_rest and line_info.the_rest[0] in '!=()<>,+*/%^&|':
            return self.prefilter_manager.get_handler_by_name('normal')
        else:
            return None

class AutocallChecker(PrefilterChecker):
    priority = Integer(1000).tag(config=True)
    function_name_regexp = CRegExp(re_fun_name, help='RegExp to identify potential function names.').tag(config=True)
    exclude_regexp = CRegExp(re_exclude_auto, help='RegExp to exclude strings with this start from autocalling.').tag(config=True)

    def check(self, line_info):
        if False:
            for i in range(10):
                print('nop')
        'Check if the initial word/function is callable and autocall is on.'
        if not self.shell.autocall:
            return None
        oinfo = line_info.ofind(self.shell)
        if not oinfo.found:
            return None
        ignored_funs = ['b', 'f', 'r', 'u', 'br', 'rb', 'fr', 'rf']
        ifun = line_info.ifun
        line = line_info.line
        if ifun.lower() in ignored_funs and (line.startswith(ifun + "'") or line.startswith(ifun + '"')):
            return None
        if callable(oinfo.obj) and (not self.exclude_regexp.match(line_info.the_rest)) and self.function_name_regexp.match(line_info.ifun):
            return self.prefilter_manager.get_handler_by_name('auto')
        else:
            return None

class PrefilterHandler(Configurable):
    handler_name = Unicode('normal')
    esc_strings = List([])
    shell = Instance('IPython.core.interactiveshell.InteractiveShellABC', allow_none=True)
    prefilter_manager = Instance('IPython.core.prefilter.PrefilterManager', allow_none=True)

    def __init__(self, shell=None, prefilter_manager=None, **kwargs):
        if False:
            return 10
        super(PrefilterHandler, self).__init__(shell=shell, prefilter_manager=prefilter_manager, **kwargs)
        self.prefilter_manager.register_handler(self.handler_name, self, self.esc_strings)

    def handle(self, line_info):
        if False:
            while True:
                i = 10
        'Handle normal input lines. Use as a template for handlers.'
        line = line_info.line
        continue_prompt = line_info.continue_prompt
        if continue_prompt and self.shell.autoindent and line.isspace() and (0 < abs(len(line) - self.shell.indent_current_nsp) <= 2):
            line = ''
        return line

    def __str__(self):
        if False:
            return 10
        return '<%s(name=%s)>' % (self.__class__.__name__, self.handler_name)

class MacroHandler(PrefilterHandler):
    handler_name = Unicode('macro')

    def handle(self, line_info):
        if False:
            for i in range(10):
                print('nop')
        obj = self.shell.user_ns.get(line_info.ifun)
        pre_space = line_info.pre_whitespace
        line_sep = '\n' + pre_space
        return pre_space + line_sep.join(obj.value.splitlines())

class MagicHandler(PrefilterHandler):
    handler_name = Unicode('magic')
    esc_strings = List([ESC_MAGIC])

    def handle(self, line_info):
        if False:
            while True:
                i = 10
        'Execute magic functions.'
        ifun = line_info.ifun
        the_rest = line_info.the_rest
        t_arg_s = ifun + ' ' + the_rest
        (t_magic_name, _, t_magic_arg_s) = t_arg_s.partition(' ')
        t_magic_name = t_magic_name.lstrip(ESC_MAGIC)
        cmd = '%sget_ipython().run_line_magic(%r, %r)' % (line_info.pre_whitespace, t_magic_name, t_magic_arg_s)
        return cmd

class AutoHandler(PrefilterHandler):
    handler_name = Unicode('auto')
    esc_strings = List([ESC_PAREN, ESC_QUOTE, ESC_QUOTE2])

    def handle(self, line_info):
        if False:
            while True:
                i = 10
        'Handle lines which can be auto-executed, quoting if requested.'
        line = line_info.line
        ifun = line_info.ifun
        the_rest = line_info.the_rest
        esc = line_info.esc
        continue_prompt = line_info.continue_prompt
        obj = line_info.ofind(self.shell).obj
        if continue_prompt:
            return line
        force_auto = isinstance(obj, IPyAutocall)
        try:
            auto_rewrite = obj.rewrite
        except Exception:
            auto_rewrite = True
        if esc == ESC_QUOTE:
            newcmd = '%s("%s")' % (ifun, '", "'.join(the_rest.split()))
        elif esc == ESC_QUOTE2:
            newcmd = '%s("%s")' % (ifun, the_rest)
        elif esc == ESC_PAREN:
            newcmd = '%s(%s)' % (ifun, ','.join(the_rest.split()))
        else:
            if force_auto:
                do_rewrite = not the_rest.startswith('(')
            elif not the_rest:
                do_rewrite = self.shell.autocall >= 2
            elif the_rest.startswith('[') and hasattr(obj, '__getitem__'):
                do_rewrite = False
            else:
                do_rewrite = True
            if do_rewrite:
                if the_rest.endswith(';'):
                    newcmd = '%s(%s);' % (ifun.rstrip(), the_rest[:-1])
                else:
                    newcmd = '%s(%s)' % (ifun.rstrip(), the_rest)
            else:
                normal_handler = self.prefilter_manager.get_handler_by_name('normal')
                return normal_handler.handle(line_info)
        if auto_rewrite:
            self.shell.auto_rewrite_input(newcmd)
        return newcmd

class EmacsHandler(PrefilterHandler):
    handler_name = Unicode('emacs')
    esc_strings = List([])

    def handle(self, line_info):
        if False:
            i = 10
            return i + 15
        'Handle input lines marked by python-mode.'
        return line_info.line
_default_checkers = [EmacsChecker, MacroChecker, IPyAutocallChecker, AssignmentChecker, AutoMagicChecker, PythonOpsChecker, AutocallChecker]
_default_handlers = [PrefilterHandler, MacroHandler, MagicHandler, AutoHandler, EmacsHandler]