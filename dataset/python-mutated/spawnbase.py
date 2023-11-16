from io import StringIO, BytesIO
import codecs
import os
import sys
import re
import errno
from .exceptions import ExceptionPexpect, EOF, TIMEOUT
from .expect import Expecter, searcher_string, searcher_re
PY3 = sys.version_info[0] >= 3
text_type = str if PY3 else unicode

class _NullCoder(object):
    """Pass bytes through unchanged."""

    @staticmethod
    def encode(b, final=False):
        if False:
            for i in range(10):
                print('nop')
        return b

    @staticmethod
    def decode(b, final=False):
        if False:
            return 10
        return b

class SpawnBase(object):
    """A base class providing the backwards-compatible spawn API for Pexpect.

    This should not be instantiated directly: use :class:`pexpect.spawn` or
    :class:`pexpect.fdpexpect.fdspawn`.
    """
    encoding = None
    pid = None
    flag_eof = False

    def __init__(self, timeout=30, maxread=2000, searchwindowsize=None, logfile=None, encoding=None, codec_errors='strict'):
        if False:
            i = 10
            return i + 15
        self.stdin = sys.stdin
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.searcher = None
        self.ignorecase = False
        self.before = None
        self.after = None
        self.match = None
        self.match_index = None
        self.terminated = True
        self.exitstatus = None
        self.signalstatus = None
        self.status = None
        self.child_fd = -1
        self.timeout = timeout
        self.delimiter = EOF
        self.logfile = logfile
        self.logfile_read = None
        self.logfile_send = None
        self.maxread = maxread
        self.searchwindowsize = searchwindowsize
        self.delaybeforesend = 0.05
        self.delayafterclose = 0.1
        self.delayafterterminate = 0.1
        self.delayafterread = 0.0001
        self.softspace = False
        self.name = '<' + repr(self) + '>'
        self.closed = True
        self.encoding = encoding
        self.codec_errors = codec_errors
        if encoding is None:
            self._encoder = self._decoder = _NullCoder()
            self.string_type = bytes
            self.buffer_type = BytesIO
            self.crlf = b'\r\n'
            if PY3:
                self.allowed_string_types = (bytes, str)
                self.linesep = os.linesep.encode('ascii')

                def write_to_stdout(b):
                    if False:
                        return 10
                    try:
                        return sys.stdout.buffer.write(b)
                    except AttributeError:
                        return sys.stdout.write(b.decode('ascii', 'replace'))
                self.write_to_stdout = write_to_stdout
            else:
                self.allowed_string_types = (basestring,)
                self.linesep = os.linesep
                self.write_to_stdout = sys.stdout.write
        else:
            self._encoder = codecs.getincrementalencoder(encoding)(codec_errors)
            self._decoder = codecs.getincrementaldecoder(encoding)(codec_errors)
            self.string_type = text_type
            self.buffer_type = StringIO
            self.crlf = u'\r\n'
            self.allowed_string_types = (text_type,)
            if PY3:
                self.linesep = os.linesep
            else:
                self.linesep = os.linesep.decode('ascii')
            self.write_to_stdout = sys.stdout.write
        self.async_pw_transport = None
        self._buffer = self.buffer_type()
        self._before = self.buffer_type()

    def _log(self, s, direction):
        if False:
            print('Hello World!')
        if self.logfile is not None:
            self.logfile.write(s)
            self.logfile.flush()
        second_log = self.logfile_send if direction == 'send' else self.logfile_read
        if second_log is not None:
            second_log.write(s)
            second_log.flush()

    def _coerce_expect_string(self, s):
        if False:
            print('Hello World!')
        if self.encoding is None and (not isinstance(s, bytes)):
            return s.encode('ascii')
        return s

    def _coerce_send_string(self, s):
        if False:
            print('Hello World!')
        if self.encoding is None and (not isinstance(s, bytes)):
            return s.encode('utf-8')
        return s

    def _get_buffer(self):
        if False:
            return 10
        return self._buffer.getvalue()

    def _set_buffer(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._buffer = self.buffer_type()
        self._buffer.write(value)
    buffer = property(_get_buffer, _set_buffer)

    def read_nonblocking(self, size=1, timeout=None):
        if False:
            print('Hello World!')
        'This reads data from the file descriptor.\n\n        This is a simple implementation suitable for a regular file. Subclasses using ptys or pipes should override it.\n\n        The timeout parameter is ignored.\n        '
        try:
            s = os.read(self.child_fd, size)
        except OSError as err:
            if err.args[0] == errno.EIO:
                self.flag_eof = True
                raise EOF('End Of File (EOF). Exception style platform.')
            raise
        if s == b'':
            self.flag_eof = True
            raise EOF('End Of File (EOF). Empty string style platform.')
        s = self._decoder.decode(s, final=False)
        self._log(s, 'read')
        return s

    def _pattern_type_err(self, pattern):
        if False:
            while True:
                i = 10
        raise TypeError('got {badtype} ({badobj!r}) as pattern, must be one of: {goodtypes}, pexpect.EOF, pexpect.TIMEOUT'.format(badtype=type(pattern), badobj=pattern, goodtypes=', '.join([str(ast) for ast in self.allowed_string_types])))

    def compile_pattern_list(self, patterns):
        if False:
            i = 10
            return i + 15
        'This compiles a pattern-string or a list of pattern-strings.\n        Patterns must be a StringType, EOF, TIMEOUT, SRE_Pattern, or a list of\n        those. Patterns may also be None which results in an empty list (you\n        might do this if waiting for an EOF or TIMEOUT condition without\n        expecting any pattern).\n\n        This is used by expect() when calling expect_list(). Thus expect() is\n        nothing more than::\n\n             cpl = self.compile_pattern_list(pl)\n             return self.expect_list(cpl, timeout)\n\n        If you are using expect() within a loop it may be more\n        efficient to compile the patterns first and then call expect_list().\n        This avoid calls in a loop to compile_pattern_list()::\n\n             cpl = self.compile_pattern_list(my_pattern)\n             while some_condition:\n                ...\n                i = self.expect_list(cpl, timeout)\n                ...\n        '
        if patterns is None:
            return []
        if not isinstance(patterns, list):
            patterns = [patterns]
        compile_flags = re.DOTALL
        if self.ignorecase:
            compile_flags = compile_flags | re.IGNORECASE
        compiled_pattern_list = []
        for (idx, p) in enumerate(patterns):
            if isinstance(p, self.allowed_string_types):
                p = self._coerce_expect_string(p)
                compiled_pattern_list.append(re.compile(p, compile_flags))
            elif p is EOF:
                compiled_pattern_list.append(EOF)
            elif p is TIMEOUT:
                compiled_pattern_list.append(TIMEOUT)
            elif isinstance(p, type(re.compile(''))):
                compiled_pattern_list.append(p)
            else:
                self._pattern_type_err(p)
        return compiled_pattern_list

    def expect(self, pattern, timeout=-1, searchwindowsize=-1, async_=False, **kw):
        if False:
            for i in range(10):
                print('nop')
        'This seeks through the stream until a pattern is matched. The\n        pattern is overloaded and may take several types. The pattern can be a\n        StringType, EOF, a compiled re, or a list of any of those types.\n        Strings will be compiled to re types. This returns the index into the\n        pattern list. If the pattern was not a list this returns index 0 on a\n        successful match. This may raise exceptions for EOF or TIMEOUT. To\n        avoid the EOF or TIMEOUT exceptions add EOF or TIMEOUT to the pattern\n        list. That will cause expect to match an EOF or TIMEOUT condition\n        instead of raising an exception.\n\n        If you pass a list of patterns and more than one matches, the first\n        match in the stream is chosen. If more than one pattern matches at that\n        point, the leftmost in the pattern list is chosen. For example::\n\n            # the input is \'foobar\'\n            index = p.expect([\'bar\', \'foo\', \'foobar\'])\n            # returns 1(\'foo\') even though \'foobar\' is a "better" match\n\n        Please note, however, that buffering can affect this behavior, since\n        input arrives in unpredictable chunks. For example::\n\n            # the input is \'foobar\'\n            index = p.expect([\'foobar\', \'foo\'])\n            # returns 0(\'foobar\') if all input is available at once,\n            # but returns 1(\'foo\') if parts of the final \'bar\' arrive late\n\n        When a match is found for the given pattern, the class instance\n        attribute *match* becomes an re.MatchObject result.  Should an EOF\n        or TIMEOUT pattern match, then the match attribute will be an instance\n        of that exception class.  The pairing before and after class\n        instance attributes are views of the data preceding and following\n        the matching pattern.  On general exception, class attribute\n        *before* is all data received up to the exception, while *match* and\n        *after* attributes are value None.\n\n        When the keyword argument timeout is -1 (default), then TIMEOUT will\n        raise after the default value specified by the class timeout\n        attribute. When None, TIMEOUT will not be raised and may block\n        indefinitely until match.\n\n        When the keyword argument searchwindowsize is -1 (default), then the\n        value specified by the class maxread attribute is used.\n\n        A list entry may be EOF or TIMEOUT instead of a string. This will\n        catch these exceptions and return the index of the list entry instead\n        of raising the exception. The attribute \'after\' will be set to the\n        exception type. The attribute \'match\' will be None. This allows you to\n        write code like this::\n\n                index = p.expect([\'good\', \'bad\', pexpect.EOF, pexpect.TIMEOUT])\n                if index == 0:\n                    do_something()\n                elif index == 1:\n                    do_something_else()\n                elif index == 2:\n                    do_some_other_thing()\n                elif index == 3:\n                    do_something_completely_different()\n\n        instead of code like this::\n\n                try:\n                    index = p.expect([\'good\', \'bad\'])\n                    if index == 0:\n                        do_something()\n                    elif index == 1:\n                        do_something_else()\n                except EOF:\n                    do_some_other_thing()\n                except TIMEOUT:\n                    do_something_completely_different()\n\n        These two forms are equivalent. It all depends on what you want. You\n        can also just expect the EOF if you are waiting for all output of a\n        child to finish. For example::\n\n                p = pexpect.spawn(\'/bin/ls\')\n                p.expect(pexpect.EOF)\n                print p.before\n\n        If you are trying to optimize for speed then see expect_list().\n\n        On Python 3.4, or Python 3.3 with asyncio installed, passing\n        ``async_=True``  will make this return an :mod:`asyncio` coroutine,\n        which you can yield from to get the same result that this method would\n        normally give directly. So, inside a coroutine, you can replace this code::\n\n            index = p.expect(patterns)\n\n        With this non-blocking form::\n\n            index = yield from p.expect(patterns, async_=True)\n        '
        if 'async' in kw:
            async_ = kw.pop('async')
        if kw:
            raise TypeError('Unknown keyword arguments: {}'.format(kw))
        compiled_pattern_list = self.compile_pattern_list(pattern)
        return self.expect_list(compiled_pattern_list, timeout, searchwindowsize, async_)

    def expect_list(self, pattern_list, timeout=-1, searchwindowsize=-1, async_=False, **kw):
        if False:
            i = 10
            return i + 15
        'This takes a list of compiled regular expressions and returns the\n        index into the pattern_list that matched the child output. The list may\n        also contain EOF or TIMEOUT(which are not compiled regular\n        expressions). This method is similar to the expect() method except that\n        expect_list() does not recompile the pattern list on every call. This\n        may help if you are trying to optimize for speed, otherwise just use\n        the expect() method.  This is called by expect().\n\n\n        Like :meth:`expect`, passing ``async_=True`` will make this return an\n        asyncio coroutine.\n        '
        if timeout == -1:
            timeout = self.timeout
        if 'async' in kw:
            async_ = kw.pop('async')
        if kw:
            raise TypeError('Unknown keyword arguments: {}'.format(kw))
        exp = Expecter(self, searcher_re(pattern_list), searchwindowsize)
        if async_:
            from ._async import expect_async
            return expect_async(exp, timeout)
        else:
            return exp.expect_loop(timeout)

    def expect_exact(self, pattern_list, timeout=-1, searchwindowsize=-1, async_=False, **kw):
        if False:
            print('Hello World!')
        "This is similar to expect(), but uses plain string matching instead\n        of compiled regular expressions in 'pattern_list'. The 'pattern_list'\n        may be a string; a list or other sequence of strings; or TIMEOUT and\n        EOF.\n\n        This call might be faster than expect() for two reasons: string\n        searching is faster than RE matching and it is possible to limit the\n        search to just the end of the input buffer.\n\n        This method is also useful when you don't want to have to worry about\n        escaping regular expression characters that you want to match.\n\n        Like :meth:`expect`, passing ``async_=True`` will make this return an\n        asyncio coroutine.\n        "
        if timeout == -1:
            timeout = self.timeout
        if 'async' in kw:
            async_ = kw.pop('async')
        if kw:
            raise TypeError('Unknown keyword arguments: {}'.format(kw))
        if isinstance(pattern_list, self.allowed_string_types) or pattern_list in (TIMEOUT, EOF):
            pattern_list = [pattern_list]

        def prepare_pattern(pattern):
            if False:
                while True:
                    i = 10
            if pattern in (TIMEOUT, EOF):
                return pattern
            if isinstance(pattern, self.allowed_string_types):
                return self._coerce_expect_string(pattern)
            self._pattern_type_err(pattern)
        try:
            pattern_list = iter(pattern_list)
        except TypeError:
            self._pattern_type_err(pattern_list)
        pattern_list = [prepare_pattern(p) for p in pattern_list]
        exp = Expecter(self, searcher_string(pattern_list), searchwindowsize)
        if async_:
            from ._async import expect_async
            return expect_async(exp, timeout)
        else:
            return exp.expect_loop(timeout)

    def expect_loop(self, searcher, timeout=-1, searchwindowsize=-1):
        if False:
            i = 10
            return i + 15
        "This is the common loop used inside expect. The 'searcher' should be\n        an instance of searcher_re or searcher_string, which describes how and\n        what to search for in the input.\n\n        See expect() for other arguments, return value and exceptions. "
        exp = Expecter(self, searcher, searchwindowsize)
        return exp.expect_loop(timeout)

    def read(self, size=-1):
        if False:
            return 10
        'This reads at most "size" bytes from the file (less if the read hits\n        EOF before obtaining size bytes). If the size argument is negative or\n        omitted, read all data until EOF is reached. The bytes are returned as\n        a string object. An empty string is returned when EOF is encountered\n        immediately. '
        if size == 0:
            return self.string_type()
        if size < 0:
            self.expect(self.delimiter)
            return self.before
        cre = re.compile(self._coerce_expect_string('.{%d}' % size), re.DOTALL)
        index = self.expect([cre, self.delimiter])
        if index == 0:
            return self.after
        return self.before

    def readline(self, size=-1):
        if False:
            print('Hello World!')
        'This reads and returns one entire line. The newline at the end of\n        line is returned as part of the string, unless the file ends without a\n        newline. An empty string is returned if EOF is encountered immediately.\n        This looks for a newline as a CR/LF pair (\\r\\n) even on UNIX because\n        this is what the pseudotty device returns. So contrary to what you may\n        expect you will receive newlines as \\r\\n.\n\n        If the size argument is 0 then an empty string is returned. In all\n        other cases the size argument is ignored, which is not standard\n        behavior for a file-like object. '
        if size == 0:
            return self.string_type()
        index = self.expect([self.crlf, self.delimiter])
        if index == 0:
            return self.before + self.crlf
        else:
            return self.before

    def __iter__(self):
        if False:
            return 10
        'This is to support iterators over a file-like object.\n        '
        return iter(self.readline, self.string_type())

    def readlines(self, sizehint=-1):
        if False:
            for i in range(10):
                print('nop')
        "This reads until EOF using readline() and returns a list containing\n        the lines thus read. The optional 'sizehint' argument is ignored.\n        Remember, because this reads until EOF that means the child\n        process should have closed its stdout. If you run this method on\n        a child that is still running with its stdout open then this\n        method will block until it timesout."
        lines = []
        while True:
            line = self.readline()
            if not line:
                break
            lines.append(line)
        return lines

    def fileno(self):
        if False:
            return 10
        'Expose file descriptor for a file-like interface\n        '
        return self.child_fd

    def flush(self):
        if False:
            while True:
                i = 10
        'This does nothing. It is here to support the interface for a\n        File-like object. '
        pass

    def isatty(self):
        if False:
            print('Hello World!')
        'Overridden in subclass using tty'
        return False

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, etype, evalue, tb):
        if False:
            return 10
        self.close()