"""The base class for xonsh shell"""
import io
import os
import sys
import time
from xonsh.ansi_colors import ansi_partial_color_format
from xonsh.built_ins import XSH
from xonsh.codecache import code_cache_check, code_cache_name, get_cache_filename, run_compiled_code, should_use_cache, update_cache
from xonsh.completer import Completer
from xonsh.events import events
from xonsh.lazyimps import pyghooks, pygments
from xonsh.platform import HAS_PYGMENTS, ON_WINDOWS
from xonsh.prompt.base import PromptFormatter, multiline_prompt
from xonsh.shell import transform_command
from xonsh.tools import DefaultNotGiven, XonshError, check_for_partial_string, format_std_prepost, get_line_continuation, print_exception
if ON_WINDOWS:
    import ctypes
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleTitleW.argtypes = [ctypes.c_wchar_p]

class _TeeStdBuf(io.RawIOBase):
    """A dispatcher for bytes to two buffers, as std stream buffer and an
    in memory buffer.
    """

    def __init__(self, stdbuf, membuf, encoding=None, errors=None, prestd=b'', poststd=b''):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        stdbuf : BytesIO-like or StringIO-like\n            The std stream buffer.\n        membuf : BytesIO-like\n            The in memory stream buffer.\n        encoding : str or None, optional\n            The encoding of the stream. Only used if stdbuf is a text stream,\n            rather than a binary one. Defaults to $XONSH_ENCODING if None.\n        errors : str or None, optional\n            The error form for the encoding of the stream. Only used if stdbuf\n            is a text stream, rather than a binary one. Deafults to\n            $XONSH_ENCODING_ERRORS if None.\n        prestd : bytes, optional\n            The prefix to prepend to the standard buffer.\n        poststd : bytes, optional\n            The postfix to append to the standard buffer.\n        '
        self.stdbuf = stdbuf
        self.membuf = membuf
        env = XSH.env
        self.encoding = env.get('XONSH_ENCODING') if encoding is None else encoding
        self.errors = env.get('XONSH_ENCODING_ERRORS') if errors is None else errors
        self.prestd = prestd
        self.poststd = poststd
        self._std_is_binary = not hasattr(stdbuf, 'encoding') or hasattr(stdbuf, '_redirect_to')

    def fileno(self):
        if False:
            i = 10
            return i + 15
        'Returns the file descriptor of the std buffer.'
        return self.stdbuf.fileno()

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            i = 10
            return i + 15
        'Sets the location in both the stdbuf and the membuf.'
        self.stdbuf.seek(offset, whence)
        self.membuf.seek(offset, whence)

    def truncate(self, size=None):
        if False:
            i = 10
            return i + 15
        'Truncate both buffers.'
        self.stdbuf.truncate(size)
        self.membuf.truncate(size)

    def readinto(self, b):
        if False:
            i = 10
            return i + 15
        'Read bytes into buffer from both streams.'
        if self._std_is_binary:
            self.stdbuf.readinto(b)
        return self.membuf.readinto(b)

    def write(self, b):
        if False:
            return 10
        'Write bytes into both buffers.'
        std_b = b
        if self.prestd:
            std_b = self.prestd + b
        if self.poststd:
            std_b += self.poststd
        if self._std_is_binary:
            self.stdbuf.write(std_b)
        else:
            self.stdbuf.write(std_b.decode(encoding=self.encoding, errors=self.errors))
        return self.membuf.write(b)

class _TeeStd(io.TextIOBase):
    """Tees a std stream into an in-memory container and the original stream."""

    def __init__(self, name, mem, prestd='', poststd=''):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parameters\n        ----------\n        name : str\n            The name of the buffer in the sys module, e.g. 'stdout'.\n        mem : io.TextIOBase-like\n            The in-memory text-based representation.\n        prestd : str, optional\n            The prefix to prepend to the standard stream.\n        poststd : str, optional\n            The postfix to append to the standard stream.\n        "
        self._name = name
        self.std = std = getattr(sys, name)
        self.mem = mem
        self.prestd = prestd
        self.poststd = poststd
        preb = prestd.encode(encoding=mem.encoding, errors=mem.errors)
        postb = poststd.encode(encoding=mem.encoding, errors=mem.errors)
        if hasattr(std, 'buffer'):
            buffer = _TeeStdBuf(std.buffer, mem.buffer, prestd=preb, poststd=postb)
        else:
            buffer = _TeeStdBuf(std, mem.buffer, encoding=mem.encoding, errors=mem.errors, prestd=preb, poststd=postb)
        self.buffer = buffer
        setattr(sys, name, self)

    @property
    def encoding(self):
        if False:
            for i in range(10):
                print('nop')
        'The encoding of the in-memory buffer.'
        return self.mem.encoding

    @property
    def errors(self):
        if False:
            print('Hello World!')
        'The errors of the in-memory buffer.'
        return self.mem.errors

    @property
    def newlines(self):
        if False:
            for i in range(10):
                print('nop')
        'The newlines of the in-memory buffer.'
        return self.mem.newlines

    def _replace_std(self):
        if False:
            while True:
                i = 10
        std = self.std
        if std is None:
            return
        setattr(sys, self._name, std)
        self.std = self._name = None

    def __del__(self):
        if False:
            i = 10
            return i + 15
        self._replace_std()

    def close(self):
        if False:
            print('Hello World!')
        'Restores the original std stream.'
        self._replace_std()

    def write(self, s):
        if False:
            while True:
                i = 10
        'Writes data to the original std stream and the in-memory object.'
        self.mem.write(s)
        if self.std is None:
            return
        std_s = s
        if self.prestd:
            std_s = self.prestd + std_s
        if self.poststd:
            std_s += self.poststd
        self.std.write(std_s)

    def flush(self):
        if False:
            while True:
                i = 10
        'Flushes both the original stdout and the buffer.'
        getattr(getattr(self, 'std', lambda : None), 'flush', lambda : None)()
        getattr(getattr(self, 'mem', lambda : None), 'flush', lambda : None)()

    def fileno(self):
        if False:
            return 10
        'Tunnel fileno() calls to the std stream.'
        return self.std.fileno()

    def seek(self, offset, whence=io.SEEK_SET):
        if False:
            while True:
                i = 10
        'Seek to a location in both streams.'
        self.std.seek(offset, whence)
        self.mem.seek(offset, whence)

    def truncate(self, size=None):
        if False:
            i = 10
            return i + 15
        'Seek to a location in both streams.'
        self.std.truncate(size)
        self.mem.truncate(size)

    def detach(self):
        if False:
            i = 10
            return i + 15
        'This operation is not supported.'
        raise io.UnsupportedOperation

    def read(self, size=None):
        if False:
            while True:
                i = 10
        'Read from the in-memory stream and seek to a new location in the\n        std stream.\n        '
        s = self.mem.read(size)
        loc = self.std.tell()
        self.std.seek(loc + len(s))
        return s

    def readline(self, size=-1):
        if False:
            while True:
                i = 10
        'Read a line from the in-memory stream and seek to a new location\n        in the std stream.\n        '
        s = self.mem.readline(size)
        loc = self.std.tell()
        self.std.seek(loc + len(s))
        return s

    def isatty(self) -> bool:
        if False:
            return 10
        'delegate the method to the underlying io-wrapper'
        if self.std:
            return self.std.isatty()
        return super().isatty()

class Tee:
    """Class that merges tee'd stdout and stderr into a single stream.

    This represents what a user would actually see on the command line.
    This class has the same interface as io.TextIOWrapper, except that
    the buffer is optional.
    """

    def __init__(self, buffer=None, encoding=None, errors=None, newline=None, line_buffering=False, write_through=False):
        if False:
            i = 10
            return i + 15
        self.buffer = io.BytesIO() if buffer is None else buffer
        self.memory = io.TextIOWrapper(self.buffer, encoding=encoding, errors=errors, newline=newline, line_buffering=line_buffering, write_through=write_through)
        self.stdout = _TeeStd('stdout', self.memory)
        env = XSH.env
        prestderr = format_std_prepost(env.get('XONSH_STDERR_PREFIX'))
        poststderr = format_std_prepost(env.get('XONSH_STDERR_POSTFIX'))
        self.stderr = _TeeStd('stderr', self.memory, prestd=prestderr, poststd=poststderr)

    @property
    def line_buffering(self):
        if False:
            for i in range(10):
                print('nop')
        return self.memory.line_buffering

    def __del__(self):
        if False:
            while True:
                i = 10
        del self.stdout, self.stderr
        self.stdout = self.stderr = None

    def close(self):
        if False:
            while True:
                i = 10
        'Closes the buffer as well as the stdout and stderr tees.'
        self.stdout.close()
        self.stderr.close()
        self.memory.close()

    def getvalue(self):
        if False:
            i = 10
            return i + 15
        'Gets the current contents of the in-memory buffer.'
        m = self.memory
        loc = m.tell()
        m.seek(0)
        s = m.read()
        m.seek(loc)
        return s

class BaseShell:
    """The xonsh shell."""

    def __init__(self, execer, ctx, **kwargs):
        if False:
            return 10
        '\n\n        Notes\n        -----\n        classes inheriting multiple base classes should call them explicitly\n        as done for ``ReadlineShell``\n        '
        self.execer = execer
        self.ctx = ctx
        self.completer = Completer() if kwargs.get('completer', True) else None
        self.buffer = []
        self.need_more_lines = False
        self.src_starts_with_space = False
        self.mlprompt = None
        self._styler = DefaultNotGiven
        self.prompt_formatter = PromptFormatter()
        self.accumulated_inputs = ''
        self.precwd = None

    @property
    def styler(self):
        if False:
            for i in range(10):
                print('nop')
        if self._styler is DefaultNotGiven:
            if HAS_PYGMENTS:
                from xonsh.pyghooks import XonshStyle
                env = XSH.env
                self._styler = XonshStyle(env.get('XONSH_COLOR_STYLE'))
            else:
                self._styler = None
        return self._styler

    @styler.setter
    def styler(self, value):
        if False:
            i = 10
            return i + 15
        self._styler = value

    @styler.deleter
    def styler(self):
        if False:
            while True:
                i = 10
        self._styler = DefaultNotGiven

    def emptyline(self):
        if False:
            for i in range(10):
                print('nop')
        'Called when an empty line has been entered.'
        self.need_more_lines = False
        self.default('')

    def singleline(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Reads a single line of input from the shell.'
        msg = '{0} has not implemented singleline().'
        raise RuntimeError(msg.format(self.__class__.__name__))

    def precmd(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Called just before execution of line.'
        try:
            self.precwd = os.getcwd()
        except FileNotFoundError:
            self.precwd = os.path.expanduser('~')
        return line if self.need_more_lines else line.lstrip()

    def default(self, line, raw_line=None):
        if False:
            return 10
        'Implements code execution.'
        line = line if line.endswith('\n') else line + '\n'
        if not self.need_more_lines:
            if not raw_line:
                self.src_starts_with_space = False
            else:
                self.src_starts_with_space = raw_line[0].isspace()
        (src, code) = self.push(line)
        if code is None:
            return
        events.on_precommand.fire(cmd=src)
        env = XSH.env
        hist = XSH.history
        ts1 = None
        enc = env.get('XONSH_ENCODING')
        err = env.get('XONSH_ENCODING_ERRORS')
        tee = Tee(encoding=enc, errors=err)
        ts0 = time.time()
        try:
            exc_info = run_compiled_code(code, self.ctx, None, 'single')
            if exc_info != (None, None, None):
                raise exc_info[1]
            ts1 = time.time()
            if hist is not None and hist.last_cmd_rtn is None:
                hist.last_cmd_rtn = 0
        except XonshError as e:
            print(e.args[0], file=sys.stderr)
            if hist is not None and hist.last_cmd_rtn is None:
                hist.last_cmd_rtn = 1
        except (SystemExit, KeyboardInterrupt) as err:
            raise err
        except BaseException:
            print_exception(exc_info=exc_info)
            if hist is not None and hist.last_cmd_rtn is None:
                hist.last_cmd_rtn = 1
        finally:
            ts1 = ts1 or time.time()
            tee_out = tee.getvalue()
            self._append_history(inp=src, ts=[ts0, ts1], spc=self.src_starts_with_space, tee_out=tee_out, cwd=self.precwd)
            self.accumulated_inputs += src
            if tee_out and env.get('XONSH_APPEND_NEWLINE') and (not tee_out.endswith(os.linesep)):
                print(os.linesep, end='')
            tee.close()
            self._fix_cwd()
        if XSH.exit:
            return True

    def _append_history(self, tee_out=None, **info):
        if False:
            return 10
        'Append information about the command to the history.\n\n        This also handles on_postcommand because this is the place where all the\n        information is available.\n        '
        hist = XSH.history
        info['rtn'] = hist.last_cmd_rtn if hist is not None else None
        XSH.env['LAST_RETURN_CODE'] = info['rtn'] or 0
        tee_out = tee_out or None
        last_out = hist.last_cmd_out if hist is not None else None
        if last_out is None and tee_out is None:
            pass
        elif last_out is None and tee_out is not None:
            info['out'] = tee_out
        elif last_out is not None and tee_out is None:
            info['out'] = last_out
        else:
            info['out'] = tee_out + '\n' + last_out
        events.on_postcommand.fire(cmd=info['inp'], rtn=info['rtn'], out=info.get('out', None), ts=info['ts'])
        if hist is not None:
            hist.append(info)
            hist.last_cmd_rtn = hist.last_cmd_out = None

    def _fix_cwd(self):
        if False:
            while True:
                i = 10
        'Check if the cwd changed out from under us.'
        env = XSH.env
        try:
            cwd = os.getcwd()
        except OSError:
            cwd = None
        if cwd is None:
            pwd = env.get('PWD', None)
            if pwd is None:
                env['PWD'] = '<invalid directory>'
            elif os.path.isdir(pwd):
                pass
            else:
                msg = '{UNDERLINE_INTENSE_WHITE}{BACKGROUND_INTENSE_BLACK}'
                msg += 'xonsh: working directory does not exist: ' + pwd
                msg += '{RESET}'
                self.print_color(msg, file=sys.stderr)
        elif 'PWD' not in env:
            env['PWD'] = cwd
        elif os.path.realpath(cwd) != os.path.realpath(env['PWD']):
            old = env['PWD']
            env['PWD'] = cwd
            env['OLDPWD'] = old
            events.on_chdir.fire(olddir=old, newdir=cwd)

    def push(self, line):
        if False:
            i = 10
            return i + 15
        'Pushes a line onto the buffer and compiles the code in a way that\n        enables multiline input.\n        '
        self.buffer.append(line)
        if self.need_more_lines:
            return (None, None)
        src = ''.join(self.buffer)
        src = transform_command(src)
        return self.compile(src)

    def compile(self, src):
        if False:
            for i in range(10):
                print('nop')
        'Compiles source code and returns the (possibly modified) source and\n        a valid code object.\n        '
        _cache = should_use_cache(self.execer, 'single')
        if _cache:
            codefname = code_cache_name(src)
            cachefname = get_cache_filename(codefname, code=True)
            (usecache, code) = code_cache_check(cachefname)
            if usecache:
                self.reset_buffer()
                return (src, code)
        lincont = get_line_continuation()
        if src.endswith(lincont + '\n'):
            self.need_more_lines = True
            return (src, None)
        try:
            code = self.execer.compile(src, mode='single', glbs=self.ctx, locs=None, filename='<stdin>', compile_empty_tree=False)
            if _cache:
                update_cache(code, cachefname)
            self.reset_buffer()
        except SyntaxError:
            partial_string_info = check_for_partial_string(src)
            in_partial_string = partial_string_info[0] is not None and partial_string_info[1] is None
            if (src == '\n' or src.endswith('\n\n')) and (not in_partial_string):
                self.reset_buffer()
                print_exception()
                return (src, None)
            self.need_more_lines = True
            code = None
        except Exception:
            self.reset_buffer()
            print_exception()
            code = None
        return (src, code)

    def reset_buffer(self):
        if False:
            while True:
                i = 10
        'Resets the line buffer.'
        self.buffer.clear()
        self.need_more_lines = False
        self.mlprompt = None

    def settitle(self):
        if False:
            for i in range(10):
                print('nop')
        'Sets terminal title.'
        env = XSH.env
        term = env.get('TERM', None)
        if term is None and (not ON_WINDOWS) or term in ['dumb', 'eterm-color', 'linux']:
            return
        t = env.get('TITLE')
        if t is None:
            return
        t = self.prompt_formatter(t)
        if ON_WINDOWS and 'ANSICON' not in env:
            kernel32.SetConsoleTitleW(t)
        else:
            with open(1, 'wb', closefd=False) as f:
                f.write(f'\x1b]0;{t}\x07'.encode())
                f.flush()

    @property
    def prompt(self):
        if False:
            return 10
        'Obtains the current prompt string.'
        XSH.env['PROMPT_FIELDS'].reset()
        if self.need_more_lines:
            if self.mlprompt is None:
                try:
                    self.mlprompt = multiline_prompt()
                except Exception:
                    print_exception()
                    self.mlprompt = '<multiline prompt error> '
            return self.mlprompt
        env = XSH.env
        p = env.get('PROMPT')
        try:
            p = self.prompt_formatter(p)
        except Exception:
            print_exception()
        self.settitle()
        return p

    def format_color(self, string, hide=False, force_string=False, **kwargs):
        if False:
            i = 10
            return i + 15
        "Formats the colors in a string. ``BaseShell``'s default implementation\n        of this method uses colors based on ANSI color codes.\n        "
        style = XSH.env.get('XONSH_COLOR_STYLE')
        return ansi_partial_color_format(string, hide=hide, style=style)

    def print_color(self, string, hide=False, **kwargs):
        if False:
            while True:
                i = 10
        "Prints a string in color. This base implementation's colors are based\n        on ANSI color codes if a string was given as input. If a list of token\n        pairs is given, it will color based on pygments, if available. If\n        pygments is not available, it will print a colorless string.\n        "
        if isinstance(string, str):
            s = self.format_color(string, hide=hide)
        elif HAS_PYGMENTS:
            env = XSH.env
            self.styler.style_name = env.get('XONSH_COLOR_STYLE')
            style_proxy = pyghooks.xonsh_style_proxy(self.styler)
            formatter = pyghooks.XonshTerminal256Formatter(style=style_proxy)
            s = pygments.format(string, formatter).rstrip()
        else:
            s = ''.join([x for (_, x) in string])
        print(s, **kwargs)

    def color_style_names(self):
        if False:
            while True:
                i = 10
        'Returns an iterable of all available style names.'
        return ()

    def color_style(self):
        if False:
            print('Hello World!')
        'Returns the current color map.'
        return {}

    def restore_tty_sanity(self):
        if False:
            i = 10
            return i + 15
        'An interface for resetting the TTY stdin mode. This is highly\n        dependent on the shell backend. Also it is mostly optional since\n        it only affects ^Z backgrounding behaviour.\n        '
        pass