"""
Sphinx directive to support embedded IPython code.

IPython provides an extension for `Sphinx <http://www.sphinx-doc.org/>`_ to
highlight and run code.

This directive allows pasting of entire interactive IPython sessions, prompts
and all, and their code will actually get re-executed at doc build time, with
all prompts renumbered sequentially. It also allows you to input code as a pure
python input by giving the argument python to the directive. The output looks
like an interactive ipython section.

Here is an example of how the IPython directive can
**run** python code, at build time.

.. ipython::

   In [1]: 1+1

   In [1]: import datetime
      ...: datetime.date.fromisoformat('2022-02-22')

It supports IPython construct that plain
Python does not understand (like magics):

.. ipython::

   In [0]: import time

   In [0]: %pdoc time.sleep

This will also support top-level async when using IPython 7.0+

.. ipython::

   In [2]: import asyncio
      ...: print('before')
      ...: await asyncio.sleep(1)
      ...: print('after')


The namespace will persist across multiple code chucks, Let's define a variable:

.. ipython::

   In [0]: who = "World"

And now say hello:

.. ipython::

   In [0]: print('Hello,', who)

If the current section raises an exception, you can add the ``:okexcept:`` flag
to the current block, otherwise the build will fail.

.. ipython::
   :okexcept:

   In [1]: 1/0

IPython Sphinx directive module
===============================

To enable this directive, simply list it in your Sphinx ``conf.py`` file
(making sure the directory where you placed it is visible to sphinx, as is
needed for all Sphinx directives). For example, to enable syntax highlighting
and the IPython directive::

    extensions = ['IPython.sphinxext.ipython_console_highlighting',
                  'IPython.sphinxext.ipython_directive']

The IPython directive outputs code-blocks with the language 'ipython'. So
if you do not have the syntax highlighting extension enabled as well, then
all rendered code-blocks will be uncolored. By default this directive assumes
that your prompts are unchanged IPython ones, but this can be customized.
The configurable options that can be placed in conf.py are:

ipython_savefig_dir:
    The directory in which to save the figures. This is relative to the
    Sphinx source directory. The default is `html_static_path`.
ipython_rgxin:
    The compiled regular expression to denote the start of IPython input
    lines. The default is ``re.compile('In \\[(\\d+)\\]:\\s?(.*)\\s*')``. You
    shouldn't need to change this.
ipython_warning_is_error: [default to True]
    Fail the build if something unexpected happen, for example if a block raise
    an exception but does not have the `:okexcept:` flag. The exact behavior of
    what is considered strict, may change between the sphinx directive version.
ipython_rgxout:
    The compiled regular expression to denote the start of IPython output
    lines. The default is ``re.compile('Out\\[(\\d+)\\]:\\s?(.*)\\s*')``. You
    shouldn't need to change this.
ipython_promptin:
    The string to represent the IPython input prompt in the generated ReST.
    The default is ``'In [%d]:'``. This expects that the line numbers are used
    in the prompt.
ipython_promptout:
    The string to represent the IPython prompt in the generated ReST. The
    default is ``'Out [%d]:'``. This expects that the line numbers are used
    in the prompt.
ipython_mplbackend:
    The string which specifies if the embedded Sphinx shell should import
    Matplotlib and set the backend. The value specifies a backend that is
    passed to `matplotlib.use()` before any lines in `ipython_execlines` are
    executed. If not specified in conf.py, then the default value of 'agg' is
    used. To use the IPython directive without matplotlib as a dependency, set
    the value to `None`. It may end up that matplotlib is still imported
    if the user specifies so in `ipython_execlines` or makes use of the
    @savefig pseudo decorator.
ipython_execlines:
    A list of strings to be exec'd in the embedded Sphinx shell. Typical
    usage is to make certain packages always available. Set this to an empty
    list if you wish to have no imports always available. If specified in
    ``conf.py`` as `None`, then it has the effect of making no imports available.
    If omitted from conf.py altogether, then the default value of
    ['import numpy as np', 'import matplotlib.pyplot as plt'] is used.
ipython_holdcount
    When the @suppress pseudo-decorator is used, the execution count can be
    incremented or not. The default behavior is to hold the execution count,
    corresponding to a value of `True`. Set this to `False` to increment
    the execution count after each suppressed command.

As an example, to use the IPython directive when `matplotlib` is not available,
one sets the backend to `None`::

    ipython_mplbackend = None

An example usage of the directive is:

.. code-block:: rst

    .. ipython::

        In [1]: x = 1

        In [2]: y = x**2

        In [3]: print(y)

See http://matplotlib.org/sampledoc/ipython_directive.html for additional
documentation.

Pseudo-Decorators
=================

Note: Only one decorator is supported per input. If more than one decorator
is specified, then only the last one is used.

In addition to the Pseudo-Decorators/options described at the above link,
several enhancements have been made. The directive will emit a message to the
console at build-time if code-execution resulted in an exception or warning.
You can suppress these on a per-block basis by specifying the :okexcept:
or :okwarning: options:

.. code-block:: rst

    .. ipython::
        :okexcept:
        :okwarning:

        In [1]: 1/0
        In [2]: # raise warning.

To Do
=====

- Turn the ad-hoc test() function into a real test suite.
- Break up ipython-specific functionality from matplotlib stuff into better
  separated code.

"""
import atexit
import errno
import os
import pathlib
import re
import sys
import tempfile
import ast
import warnings
import shutil
from io import StringIO
from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive
from sphinx.util import logging
from traitlets.config import Config
from IPython import InteractiveShell
from IPython.core.profiledir import ProfileDir
use_matplotlib = False
try:
    import matplotlib
    use_matplotlib = True
except Exception:
    pass
(COMMENT, INPUT, OUTPUT) = range(3)
PSEUDO_DECORATORS = ['suppress', 'verbatim', 'savefig', 'doctest']

def block_parser(part, rgxin, rgxout, fmtin, fmtout):
    if False:
        print('Hello World!')
    '\n    part is a string of ipython text, comprised of at most one\n    input, one output, comments, and blank lines.  The block parser\n    parses the text into a list of::\n\n      blocks = [ (TOKEN0, data0), (TOKEN1, data1), ...]\n\n    where TOKEN is one of [COMMENT | INPUT | OUTPUT ] and\n    data is, depending on the type of token::\n\n      COMMENT : the comment string\n\n      INPUT: the (DECORATOR, INPUT_LINE, REST) where\n         DECORATOR: the input decorator (or None)\n         INPUT_LINE: the input as string (possibly multi-line)\n         REST : any stdout generated by the input line (not OUTPUT)\n\n      OUTPUT: the output string, possibly multi-line\n\n    '
    block = []
    lines = part.split('\n')
    N = len(lines)
    i = 0
    decorator = None
    while 1:
        if i == N:
            break
        line = lines[i]
        i += 1
        line_stripped = line.strip()
        if line_stripped.startswith('#'):
            block.append((COMMENT, line))
            continue
        if any((line_stripped.startswith('@' + pseudo_decorator) for pseudo_decorator in PSEUDO_DECORATORS)):
            if decorator:
                raise RuntimeError('Applying multiple pseudo-decorators on one line is not supported')
            else:
                decorator = line_stripped
                continue
        matchin = rgxin.match(line)
        if matchin:
            (lineno, inputline) = (int(matchin.group(1)), matchin.group(2))
            continuation = '   %s:' % ''.join(['.'] * (len(str(lineno)) + 2))
            Nc = len(continuation)
            rest = []
            while i < N:
                nextline = lines[i]
                matchout = rgxout.match(nextline)
                if matchout or nextline.startswith('#'):
                    break
                elif nextline.startswith(continuation):
                    nextline = nextline[Nc:]
                    if nextline and nextline[0] == ' ':
                        nextline = nextline[1:]
                    inputline += '\n' + nextline
                else:
                    rest.append(nextline)
                i += 1
            block.append((INPUT, (decorator, inputline, '\n'.join(rest))))
            continue
        matchout = rgxout.match(line)
        if matchout:
            (lineno, output) = (int(matchout.group(1)), matchout.group(2))
            if i < N - 1:
                output = '\n'.join([output] + lines[i:])
            block.append((OUTPUT, output))
            break
    return block

class EmbeddedSphinxShell(object):
    """An embedded IPython instance to run inside Sphinx"""

    def __init__(self, exec_lines=None):
        if False:
            print('Hello World!')
        self.cout = StringIO()
        if exec_lines is None:
            exec_lines = []
        config = Config()
        config.HistoryManager.hist_file = ':memory:'
        config.InteractiveShell.autocall = False
        config.InteractiveShell.autoindent = False
        config.InteractiveShell.colors = 'NoColor'
        tmp_profile_dir = tempfile.mkdtemp(prefix='profile_')
        profname = 'auto_profile_sphinx_build'
        pdir = os.path.join(tmp_profile_dir, profname)
        profile = ProfileDir.create_profile_dir(pdir)
        IP = InteractiveShell.instance(config=config, profile_dir=profile)
        atexit.register(self.cleanup)
        self.IP = IP
        self.user_ns = self.IP.user_ns
        self.user_global_ns = self.IP.user_global_ns
        self.input = ''
        self.output = ''
        self.tmp_profile_dir = tmp_profile_dir
        self.is_verbatim = False
        self.is_doctest = False
        self.is_suppress = False
        self.directive = None
        self._pyplot_imported = False
        for line in exec_lines:
            self.process_input_line(line, store_history=False)

    def cleanup(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.tmp_profile_dir, ignore_errors=True)

    def clear_cout(self):
        if False:
            for i in range(10):
                print('nop')
        self.cout.seek(0)
        self.cout.truncate(0)

    def process_input_line(self, line, store_history):
        if False:
            i = 10
            return i + 15
        return self.process_input_lines([line], store_history=store_history)

    def process_input_lines(self, lines, store_history=True):
        if False:
            while True:
                i = 10
        'process the input, capturing stdout'
        stdout = sys.stdout
        source_raw = '\n'.join(lines)
        try:
            sys.stdout = self.cout
            self.IP.run_cell(source_raw, store_history=store_history)
        finally:
            sys.stdout = stdout

    def process_image(self, decorator):
        if False:
            return 10
        '\n        # build out an image directive like\n        # .. image:: somefile.png\n        #    :width 4in\n        #\n        # from an input like\n        # savefig somefile.png width=4in\n        '
        savefig_dir = self.savefig_dir
        source_dir = self.source_dir
        saveargs = decorator.split(' ')
        filename = saveargs[1]
        path = pathlib.Path(savefig_dir, filename)
        outfile = '/' + path.relative_to(source_dir).as_posix()
        imagerows = ['.. image:: %s' % outfile]
        for kwarg in saveargs[2:]:
            (arg, val) = kwarg.split('=')
            arg = arg.strip()
            val = val.strip()
            imagerows.append('   :%s: %s' % (arg, val))
        image_file = os.path.basename(outfile)
        image_directive = '\n'.join(imagerows)
        return (image_file, image_directive)

    def process_input(self, data, input_prompt, lineno):
        if False:
            return 10
        '\n        Process data block for INPUT token.\n\n        '
        (decorator, input, rest) = data
        image_file = None
        image_directive = None
        is_verbatim = decorator == '@verbatim' or self.is_verbatim
        is_doctest = decorator is not None and decorator.startswith('@doctest') or self.is_doctest
        is_suppress = decorator == '@suppress' or self.is_suppress
        is_okexcept = decorator == '@okexcept' or self.is_okexcept
        is_okwarning = decorator == '@okwarning' or self.is_okwarning
        is_savefig = decorator is not None and decorator.startswith('@savefig')
        input_lines = input.split('\n')
        if len(input_lines) > 1:
            if input_lines[-1] != '':
                input_lines.append('')
        continuation = '   %s:' % ''.join(['.'] * (len(str(lineno)) + 2))
        if is_savefig:
            (image_file, image_directive) = self.process_image(decorator)
        ret = []
        is_semicolon = False
        if is_suppress and self.hold_count:
            store_history = False
        else:
            store_history = True
        with warnings.catch_warnings(record=True) as ws:
            if input_lines[0].endswith(';'):
                is_semicolon = True
            if is_verbatim:
                self.process_input_lines([''])
                self.IP.execution_count += 1
            else:
                self.process_input_lines(input_lines, store_history=store_history)
        if not is_suppress:
            for (i, line) in enumerate(input_lines):
                if i == 0:
                    formatted_line = '%s %s' % (input_prompt, line)
                else:
                    formatted_line = '%s %s' % (continuation, line)
                ret.append(formatted_line)
        if not is_suppress and len(rest.strip()) and is_verbatim:
            ret.append(rest)
        self.cout.seek(0)
        processed_output = self.cout.read()
        if not is_suppress and (not is_semicolon):
            ret.append(processed_output)
        elif is_semicolon:
            ret.append('')
        filename = 'Unknown'
        lineno = 0
        if self.directive.state:
            filename = self.directive.state.document.current_source
            lineno = self.directive.state.document.current_line
        logger = logging.getLogger(__name__)
        if not is_okexcept and ('Traceback' in processed_output or 'SyntaxError' in processed_output):
            s = '\n>>>' + '-' * 73 + '\n'
            s += 'Exception in %s at block ending on line %s\n' % (filename, lineno)
            s += 'Specify :okexcept: as an option in the ipython:: block to suppress this message\n'
            s += processed_output + '\n'
            s += '<<<' + '-' * 73
            logger.warning(s)
            if self.warning_is_error:
                raise RuntimeError('Unexpected exception in `{}` line {}'.format(filename, lineno))
        if not is_okwarning:
            for w in ws:
                s = '\n>>>' + '-' * 73 + '\n'
                s += 'Warning in %s at block ending on line %s\n' % (filename, lineno)
                s += 'Specify :okwarning: as an option in the ipython:: block to suppress this message\n'
                s += '-' * 76 + '\n'
                s += warnings.formatwarning(w.message, w.category, w.filename, w.lineno, w.line)
                s += '<<<' + '-' * 73
                logger.warning(s)
                if self.warning_is_error:
                    raise RuntimeError('Unexpected warning in `{}` line {}'.format(filename, lineno))
        self.clear_cout()
        return (ret, input_lines, processed_output, is_doctest, decorator, image_file, image_directive)

    def process_output(self, data, output_prompt, input_lines, output, is_doctest, decorator, image_file):
        if False:
            return 10
        '\n        Process data block for OUTPUT token.\n\n        '
        TAB = ' ' * 4
        if is_doctest and output is not None:
            found = output
            found = found.strip()
            submitted = data.strip()
            if self.directive is None:
                source = 'Unavailable'
                content = 'Unavailable'
            else:
                source = self.directive.state.document.current_source
                content = self.directive.content
                content = '\n'.join([TAB + line for line in content])
            ind = found.find(output_prompt)
            if ind < 0:
                e = 'output does not contain output prompt\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nInput line(s):\n{TAB}{2}\n\nOutput line(s):\n{TAB}{3}\n\n'
                e = e.format(source, content, '\n'.join(input_lines), repr(found), TAB=TAB)
                raise RuntimeError(e)
            found = found[len(output_prompt):].strip()
            if decorator.strip() == '@doctest':
                if found != submitted:
                    e = 'doctest failure\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nOn input line(s):\n{TAB}{2}\n\nwe found output:\n{TAB}{3}\n\ninstead of the expected:\n{TAB}{4}\n\n'
                    e = e.format(source, content, '\n'.join(input_lines), repr(found), repr(submitted), TAB=TAB)
                    raise RuntimeError(e)
            else:
                self.custom_doctest(decorator, input_lines, found, submitted)
        out_data = []
        is_verbatim = decorator == '@verbatim' or self.is_verbatim
        if is_verbatim and data.strip():
            out_data.append('{0} {1}\n'.format(output_prompt, data))
        return out_data

    def process_comment(self, data):
        if False:
            return 10
        'Process data fPblock for COMMENT token.'
        if not self.is_suppress:
            return [data]

    def save_image(self, image_file):
        if False:
            return 10
        '\n        Saves the image file to disk.\n        '
        self.ensure_pyplot()
        command = 'plt.gcf().savefig("%s")' % image_file
        self.process_input_line('bookmark ipy_thisdir', store_history=False)
        self.process_input_line('cd -b ipy_savedir', store_history=False)
        self.process_input_line(command, store_history=False)
        self.process_input_line('cd -b ipy_thisdir', store_history=False)
        self.process_input_line('bookmark -d ipy_thisdir', store_history=False)
        self.clear_cout()

    def process_block(self, block):
        if False:
            return 10
        '\n        process block from the block_parser and return a list of processed lines\n        '
        ret = []
        output = None
        input_lines = None
        lineno = self.IP.execution_count
        input_prompt = self.promptin % lineno
        output_prompt = self.promptout % lineno
        image_file = None
        image_directive = None
        found_input = False
        for (token, data) in block:
            if token == COMMENT:
                out_data = self.process_comment(data)
            elif token == INPUT:
                found_input = True
                (out_data, input_lines, output, is_doctest, decorator, image_file, image_directive) = self.process_input(data, input_prompt, lineno)
            elif token == OUTPUT:
                if not found_input:
                    TAB = ' ' * 4
                    linenumber = 0
                    source = 'Unavailable'
                    content = 'Unavailable'
                    if self.directive:
                        linenumber = self.directive.state.document.current_line
                        source = self.directive.state.document.current_source
                        content = self.directive.content
                        content = '\n'.join([TAB + line for line in content])
                    e = '\n\nInvalid block: Block contains an output prompt without an input prompt.\n\nDocument source: {0}\n\nContent begins at line {1}: \n\n{2}\n\nProblematic block within content: \n\n{TAB}{3}\n\n'
                    e = e.format(source, linenumber, content, block, TAB=TAB)
                    sys.stdout.write(e)
                    raise RuntimeError('An invalid block was detected.')
                out_data = self.process_output(data, output_prompt, input_lines, output, is_doctest, decorator, image_file)
                if out_data:
                    assert ret[-1] == ''
                    del ret[-1]
            if out_data:
                ret.extend(out_data)
        if image_file is not None:
            self.save_image(image_file)
        return (ret, image_directive)

    def ensure_pyplot(self):
        if False:
            print('Hello World!')
        '\n        Ensures that pyplot has been imported into the embedded IPython shell.\n\n        Also, makes sure to set the backend appropriately if not set already.\n\n        '
        if not self._pyplot_imported:
            if 'matplotlib.backends' not in sys.modules:
                import matplotlib
                matplotlib.use('agg')
            self.process_input_line('import matplotlib.pyplot as plt', store_history=False)
            self._pyplot_imported = True

    def process_pure_python(self, content):
        if False:
            i = 10
            return i + 15
        '\n        content is a list of strings. it is unedited directive content\n\n        This runs it line by line in the InteractiveShell, prepends\n        prompts as needed capturing stderr and stdout, then returns\n        the content as a list as if it were ipython code\n        '
        output = []
        savefig = False
        multiline = False
        multiline_start = None
        fmtin = self.promptin
        ct = 0
        for (lineno, line) in enumerate(content):
            line_stripped = line.strip()
            if not len(line):
                output.append(line)
                continue
            if any((line_stripped.startswith('@' + pseudo_decorator) for pseudo_decorator in PSEUDO_DECORATORS)):
                output.extend([line])
                if 'savefig' in line:
                    savefig = True
                continue
            if line_stripped.startswith('#'):
                output.extend([line])
                continue
            continuation = u'   %s:' % ''.join(['.'] * (len(str(ct)) + 2))
            if not multiline:
                modified = u'%s %s' % (fmtin % ct, line_stripped)
                output.append(modified)
                ct += 1
                try:
                    ast.parse(line_stripped)
                    output.append(u'')
                except Exception:
                    multiline = True
                    multiline_start = lineno
            else:
                modified = u'%s %s' % (continuation, line)
                output.append(modified)
                if len(content) > lineno + 1:
                    nextline = content[lineno + 1]
                    if len(nextline) - len(nextline.lstrip()) > 3:
                        continue
                try:
                    mod = ast.parse('\n'.join(content[multiline_start:lineno + 1]))
                    if isinstance(mod.body[0], ast.FunctionDef):
                        for element in mod.body[0].body:
                            if isinstance(element, ast.Return):
                                multiline = False
                    else:
                        output.append(u'')
                        multiline = False
                except Exception:
                    pass
            if savefig:
                self.ensure_pyplot()
                self.process_input_line('plt.clf()', store_history=False)
                self.clear_cout()
                savefig = False
        return output

    def custom_doctest(self, decorator, input_lines, found, submitted):
        if False:
            return 10
        '\n        Perform a specialized doctest.\n\n        '
        from .custom_doctests import doctests
        args = decorator.split()
        doctest_type = args[1]
        if doctest_type in doctests:
            doctests[doctest_type](self, args, input_lines, found, submitted)
        else:
            e = 'Invalid option to @doctest: {0}'.format(doctest_type)
            raise Exception(e)

class IPythonDirective(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 4
    final_argumuent_whitespace = True
    option_spec = {'python': directives.unchanged, 'suppress': directives.flag, 'verbatim': directives.flag, 'doctest': directives.flag, 'okexcept': directives.flag, 'okwarning': directives.flag}
    shell = None
    seen_docs = set()

    def get_config_options(self):
        if False:
            return 10
        config = self.state.document.settings.env.config
        savefig_dir = config.ipython_savefig_dir
        source_dir = self.state.document.settings.env.srcdir
        savefig_dir = os.path.join(source_dir, savefig_dir)
        rgxin = config.ipython_rgxin
        rgxout = config.ipython_rgxout
        warning_is_error = config.ipython_warning_is_error
        promptin = config.ipython_promptin
        promptout = config.ipython_promptout
        mplbackend = config.ipython_mplbackend
        exec_lines = config.ipython_execlines
        hold_count = config.ipython_holdcount
        return (savefig_dir, source_dir, rgxin, rgxout, promptin, promptout, mplbackend, exec_lines, hold_count, warning_is_error)

    def setup(self):
        if False:
            print('Hello World!')
        (savefig_dir, source_dir, rgxin, rgxout, promptin, promptout, mplbackend, exec_lines, hold_count, warning_is_error) = self.get_config_options()
        try:
            os.makedirs(savefig_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        if self.shell is None:
            if mplbackend and 'matplotlib.backends' not in sys.modules and use_matplotlib:
                import matplotlib
                matplotlib.use(mplbackend)
            self.shell = EmbeddedSphinxShell(exec_lines)
            self.shell.directive = self
        if not self.state.document.current_source in self.seen_docs:
            self.shell.IP.history_manager.reset()
            self.shell.IP.execution_count = 1
            self.seen_docs.add(self.state.document.current_source)
        self.shell.rgxin = rgxin
        self.shell.rgxout = rgxout
        self.shell.promptin = promptin
        self.shell.promptout = promptout
        self.shell.savefig_dir = savefig_dir
        self.shell.source_dir = source_dir
        self.shell.hold_count = hold_count
        self.shell.warning_is_error = warning_is_error
        self.shell.process_input_line('bookmark ipy_savedir "%s"' % savefig_dir, store_history=False)
        self.shell.clear_cout()
        return (rgxin, rgxout, promptin, promptout)

    def teardown(self):
        if False:
            for i in range(10):
                print('nop')
        self.shell.process_input_line('bookmark -d ipy_savedir', store_history=False)
        self.shell.clear_cout()

    def run(self):
        if False:
            print('Hello World!')
        debug = False
        (rgxin, rgxout, promptin, promptout) = self.setup()
        options = self.options
        self.shell.is_suppress = 'suppress' in options
        self.shell.is_doctest = 'doctest' in options
        self.shell.is_verbatim = 'verbatim' in options
        self.shell.is_okexcept = 'okexcept' in options
        self.shell.is_okwarning = 'okwarning' in options
        if 'python' in self.arguments:
            content = self.content
            self.content = self.shell.process_pure_python(content)
        parts = '\n'.join(self.content).split('\n\n')
        lines = ['.. code-block:: ipython', '']
        figures = []
        logger = logging.getLogger(__name__)
        for part in parts:
            block = block_parser(part, rgxin, rgxout, promptin, promptout)
            if len(block):
                (rows, figure) = self.shell.process_block(block)
                for row in rows:
                    lines.extend(['   {0}'.format(line) for line in row.split('\n')])
                if figure is not None:
                    figures.append(figure)
            else:
                message = 'Code input with no code at {}, line {}'.format(self.state.document.current_source, self.state.document.current_line)
                if self.shell.warning_is_error:
                    raise RuntimeError(message)
                else:
                    logger.warning(message)
        for figure in figures:
            lines.append('')
            lines.extend(figure.split('\n'))
            lines.append('')
        if len(lines) > 2:
            if debug:
                print('\n'.join(lines))
            else:
                self.state_machine.insert_input(lines, self.state_machine.input_lines.source(0))
        self.teardown()
        return []

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    setup.app = app
    app.add_directive('ipython', IPythonDirective)
    app.add_config_value('ipython_savefig_dir', 'savefig', 'env')
    app.add_config_value('ipython_warning_is_error', True, 'env')
    app.add_config_value('ipython_rgxin', re.compile('In \\[(\\d+)\\]:\\s?(.*)\\s*'), 'env')
    app.add_config_value('ipython_rgxout', re.compile('Out\\[(\\d+)\\]:\\s?(.*)\\s*'), 'env')
    app.add_config_value('ipython_promptin', 'In [%d]:', 'env')
    app.add_config_value('ipython_promptout', 'Out[%d]:', 'env')
    app.add_config_value('ipython_mplbackend', 'agg', 'env')
    execlines = ['import numpy as np']
    if use_matplotlib:
        execlines.append('import matplotlib.pyplot as plt')
    app.add_config_value('ipython_execlines', execlines, 'env')
    app.add_config_value('ipython_holdcount', True, 'env')
    metadata = {'parallel_read_safe': True, 'parallel_write_safe': True}
    return metadata

def test():
    if False:
        print('Hello World!')
    examples = ["\nIn [9]: pwd\nOut[9]: '/home/jdhunter/py4science/book'\n\nIn [10]: cd bookdata/\n/home/jdhunter/py4science/book/bookdata\n\nIn [2]: from pylab import *\n\nIn [2]: ion()\n\nIn [3]: im = imread('stinkbug.png')\n\n@savefig mystinkbug.png width=4in\nIn [4]: imshow(im)\nOut[4]: <matplotlib.image.AxesImage object at 0x39ea850>\n\n", "\n\nIn [1]: x = 'hello world'\n\n# string methods can be\n# used to alter the string\n@doctest\nIn [2]: x.upper()\nOut[2]: 'HELLO WORLD'\n\n@verbatim\nIn [3]: x.st<TAB>\nx.startswith  x.strip\n", "\n\nIn [130]: url = 'http://ichart.finance.yahoo.com/table.csv?s=CROX\\\n   .....: &d=9&e=22&f=2009&g=d&a=1&br=8&c=2006&ignore=.csv'\n\nIn [131]: print url.split('&')\n['http://ichart.finance.yahoo.com/table.csv?s=CROX', 'd=9', 'e=22', 'f=2009', 'g=d', 'a=1', 'b=8', 'c=2006', 'ignore=.csv']\n\nIn [60]: import urllib\n\n", '\\\n\nIn [133]: import numpy.random\n\n@suppress\nIn [134]: numpy.random.seed(2358)\n\n@doctest\nIn [135]: numpy.random.rand(10,2)\nOut[135]:\narray([[ 0.64524308,  0.59943846],\n       [ 0.47102322,  0.8715456 ],\n       [ 0.29370834,  0.74776844],\n       [ 0.99539577,  0.1313423 ],\n       [ 0.16250302,  0.21103583],\n       [ 0.81626524,  0.1312433 ],\n       [ 0.67338089,  0.72302393],\n       [ 0.7566368 ,  0.07033696],\n       [ 0.22591016,  0.77731835],\n       [ 0.0072729 ,  0.34273127]])\n\n', '\nIn [106]: print x\njdh\n\nIn [109]: for i in range(10):\n   .....:     print i\n   .....:\n   .....:\n0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n', "\n\nIn [144]: from pylab import *\n\nIn [145]: ion()\n\n# use a semicolon to suppress the output\n@savefig test_hist.png width=4in\nIn [151]: hist(np.random.randn(10000), 100);\n\n\n@savefig test_plot.png width=4in\nIn [151]: plot(np.random.randn(10000), 'o');\n   ", '\n# use a semicolon to suppress the output\nIn [151]: plt.clf()\n\n@savefig plot_simple.png width=4in\nIn [151]: plot([1,2,3])\n\n@savefig hist_simple.png width=4in\nIn [151]: hist(np.random.randn(10000), 100);\n\n', "\n# update the current fig\nIn [151]: ylabel('number')\n\nIn [152]: title('normal distribution')\n\n\n@savefig hist_with_text.png\nIn [153]: grid(True)\n\n@doctest float\nIn [154]: 0.1 + 0.2\nOut[154]: 0.3\n\n@doctest float\nIn [155]: np.arange(16).reshape(4,4)\nOut[155]:\narray([[ 0,  1,  2,  3],\n       [ 4,  5,  6,  7],\n       [ 8,  9, 10, 11],\n       [12, 13, 14, 15]])\n\nIn [1]: x = np.arange(16, dtype=float).reshape(4,4)\n\nIn [2]: x[0,0] = np.inf\n\nIn [3]: x[0,1] = np.nan\n\n@doctest float\nIn [4]: x\nOut[4]:\narray([[ inf,  nan,   2.,   3.],\n       [  4.,   5.,   6.,   7.],\n       [  8.,   9.,  10.,  11.],\n       [ 12.,  13.,  14.,  15.]])\n\n\n        "]
    examples = examples[1:]
    options = {}
    for example in examples:
        content = example.split('\n')
        IPythonDirective('debug', arguments=None, options=options, content=content, lineno=0, content_offset=None, block_text=None, state=None, state_machine=None)
if __name__ == '__main__':
    if not os.path.isdir('_static'):
        os.mkdir('_static')
    test()
    print('All OK? Check figures in _static/')