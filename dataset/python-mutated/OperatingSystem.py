import fnmatch
import glob
import os
import pathlib
import re
import shutil
import tempfile
import time
from datetime import datetime
from robot.version import get_version
from robot.api import logger
from robot.api.deco import keyword
from robot.utils import abspath, ConnectionCache, console_decode, del_env_var, get_env_var, get_env_vars, get_time, is_truthy, is_string, normpath, parse_time, plural_or_not, safe_str, secs_to_timestr, seq2str, set_env_var, timestr_to_secs, CONSOLE_ENCODING, WINDOWS
__version__ = get_version()
PROCESSES = ConnectionCache('No active processes.')

class OperatingSystem:
    """A library providing keywords for operating system related tasks.

    ``OperatingSystem`` is Robot Framework's standard library that
    enables various operating system related tasks to be performed in
    the system where Robot Framework is running. It can, among other
    things, execute commands (e.g. `Run`), create and remove files and
    directories (e.g. `Create File`, `Remove Directory`), check
    whether files or directories exists or contain something
    (e.g. `File Should Exist`, `Directory Should Be Empty`) and
    manipulate environment variables (e.g. `Set Environment Variable`).

    == Table of contents ==

    %TOC%

    = Path separators =

    Because Robot Framework uses the backslash (``\\``) as an escape character
    in its data, using a literal backslash requires duplicating it like
    in ``c:\\\\path\\\\file.txt``. That can be inconvenient especially with
    longer Windows paths, and thus all keywords expecting paths as arguments
    convert forward slashes to backslashes automatically on Windows. This also
    means that paths like ``${CURDIR}/path/file.txt`` are operating system
    independent.

    Notice that the automatic path separator conversion does not work if
    the path is only a part of an argument like with the `Run` keyword.
    In these cases the built-in variable ``${/}`` that contains ``\\`` or ``/``,
    depending on the operating system, can be used instead.

    = Pattern matching =

    Many keywords accept arguments as either _glob_ or _regular expression_ patterns.

    == Glob patterns ==

    Some keywords, for example `List Directory`, support so called
    [http://en.wikipedia.org/wiki/Glob_(programming)|glob patterns] where:

    | ``*``        | matches any string, even an empty string                |
    | ``?``        | matches any single character                            |
    | ``[chars]``  | matches one character in the bracket                    |
    | ``[!chars]`` | matches one character not in the bracket                |
    | ``[a-z]``    | matches one character from the range in the bracket     |
    | ``[!a-z]``   | matches one character not from the range in the bracket |

    Unless otherwise noted, matching is case-insensitive on case-insensitive
    operating systems such as Windows.

    == Regular expressions ==

    Some keywords, for example `Grep File`, support
    [http://en.wikipedia.org/wiki/Regular_expression|regular expressions]
    that are more powerful but also more complicated that glob patterns.
    The regular expression support is implemented using Python's
    [http://docs.python.org/library/re.html|re module] and its documentation
    should be consulted for more information about the syntax.

    Because the backslash character (``\\``) is an escape character in
    Robot Framework data, possible backslash characters in regular
    expressions need to be escaped with another backslash like ``\\\\d\\\\w+``.
    Strings that may contain special characters but should be handled
    as literal strings, can be escaped with the `Regexp Escape` keyword
    from the BuiltIn library.

    = Tilde expansion =

    Paths beginning with ``~`` or ``~username`` are expanded to the current or
    specified user's home directory, respectively. The resulting path is
    operating system dependent, but typically e.g. ``~/robot`` is expanded to
    ``C:\\Users\\<user>\\robot`` on Windows and ``/home/<user>/robot`` on Unixes.

    = ``pathlib.Path`` support =

    Starting from Robot Framework 6.0, arguments representing paths can be given
    as [https://docs.python.org/3/library/pathlib.html pathlib.Path] instances
    in addition to strings.

    All keywords returning paths return them as strings. This may change in
    the future so that the return value type matches the argument type.

    = Boolean arguments =

    Some keywords accept arguments that are handled as Boolean values true or
    false. If such an argument is given as a string, it is considered false if
    it is an empty string or equal to ``FALSE``, ``NONE``, ``NO``, ``OFF`` or
    ``0``, case-insensitively. Other strings are considered true regardless
    their value, and other argument types are tested using the same
    [http://docs.python.org/library/stdtypes.html#truth|rules as in Python].

    True examples:
    | `Remove Directory` | ${path} | recursive=True    | # Strings are generally true.    |
    | `Remove Directory` | ${path} | recursive=yes     | # Same as the above.             |
    | `Remove Directory` | ${path} | recursive=${TRUE} | # Python ``True`` is true.       |
    | `Remove Directory` | ${path} | recursive=${42}   | # Numbers other than 0 are true. |

    False examples:
    | `Remove Directory` | ${path} | recursive=False    | # String ``false`` is false.   |
    | `Remove Directory` | ${path} | recursive=no       | # Also string ``no`` is false. |
    | `Remove Directory` | ${path} | recursive=${EMPTY} | # Empty string is false.       |
    | `Remove Directory` | ${path} | recursive=${FALSE} | # Python ``False`` is false.   |

    = Example =

    | ***** Settings *****
    | Library         OperatingSystem
    |
    | ***** Variables *****
    | ${PATH}         ${CURDIR}/example.txt
    |
    | ***** Test Cases *****
    | Example
    |     `Create File`          ${PATH}    Some text
    |     `File Should Exist`    ${PATH}
    |     `Copy File`            ${PATH}    ~/file.txt
    """
    ROBOT_LIBRARY_SCOPE = 'GLOBAL'
    ROBOT_LIBRARY_VERSION = __version__

    def run(self, command):
        if False:
            print('Hello World!')
        'Runs the given command in the system and returns the output.\n\n        The execution status of the command *is not checked* by this\n        keyword, and it must be done separately based on the returned\n        output. If the execution return code is needed, either `Run\n        And Return RC` or `Run And Return RC And Output` can be used.\n\n        The standard error stream is automatically redirected to the standard\n        output stream by adding ``2>&1`` after the executed command. This\n        automatic redirection is done only when the executed command does not\n        contain additional output redirections. You can thus freely forward\n        the standard error somewhere else, for example, like\n        ``my_command 2>stderr.txt``.\n\n        The returned output contains everything written into the standard\n        output or error streams by the command (unless either of them\n        is redirected explicitly). Many commands add an extra newline\n        (``\\n``) after the output to make it easier to read in the\n        console. To ease processing the returned output, this possible\n        trailing newline is stripped by this keyword.\n\n        Examples:\n        | ${output} =        | Run       | ls -lhF /tmp |\n        | Log                | ${output} |\n        | ${result} =        | Run       | ${CURDIR}${/}tester.py arg1 arg2 |\n        | Should Not Contain | ${result} | FAIL |\n        | ${stdout} =        | Run       | /opt/script.sh 2>/tmp/stderr.txt |\n        | Should Be Equal    | ${stdout} | TEST PASSED |\n        | File Should Be Empty | /tmp/stderr.txt |\n\n        *TIP:* `Run Process` keyword provided by the\n        [http://robotframework.org/robotframework/latest/libraries/Process.html|\n        Process library] supports better process configuration and is generally\n        recommended as a replacement for this keyword.\n        '
        return self._run(command)[1]

    def run_and_return_rc(self, command):
        if False:
            return 10
        'Runs the given command in the system and returns the return code.\n\n        The return code (RC) is returned as a positive integer in\n        range from 0 to 255 as returned by the executed command. On\n        some operating systems (notable Windows) original return codes\n        can be something else, but this keyword always maps them to\n        the 0-255 range. Since the RC is an integer, it must be\n        checked e.g. with the keyword `Should Be Equal As Integers`\n        instead of `Should Be Equal` (both are built-in keywords).\n\n        Examples:\n        | ${rc} = | Run and Return RC | ${CURDIR}${/}script.py arg |\n        | Should Be Equal As Integers | ${rc} | 0 |\n        | ${rc} = | Run and Return RC | /path/to/example.rb arg1 arg2 |\n        | Should Be True | 0 < ${rc} < 42 |\n\n        See `Run` and `Run And Return RC And Output` if you need to get the\n        output of the executed command.\n\n        *TIP:* `Run Process` keyword provided by the\n        [http://robotframework.org/robotframework/latest/libraries/Process.html|\n        Process library] supports better process configuration and is generally\n        recommended as a replacement for this keyword.\n        '
        return self._run(command)[0]

    def run_and_return_rc_and_output(self, command):
        if False:
            print('Hello World!')
        'Runs the given command in the system and returns the RC and output.\n\n        The return code (RC) is returned similarly as with `Run And Return RC`\n        and the output similarly as with `Run`.\n\n        Examples:\n        | ${rc} | ${output} =  | Run and Return RC and Output | ${CURDIR}${/}mytool |\n        | Should Be Equal As Integers | ${rc}    | 0    |\n        | Should Not Contain   | ${output}       | FAIL |\n        | ${rc} | ${stdout} =  | Run and Return RC and Output | /opt/script.sh 2>/tmp/stderr.txt |\n        | Should Be True       | ${rc} > 42      |\n        | Should Be Equal      | ${stdout}       | TEST PASSED |\n        | File Should Be Empty | /tmp/stderr.txt |\n\n        *TIP:* `Run Process` keyword provided by the\n        [http://robotframework.org/robotframework/latest/libraries/Process.html|\n        Process library] supports better process configuration and is generally\n        recommended as a replacement for this keyword.\n        '
        return self._run(command)

    def _run(self, command):
        if False:
            while True:
                i = 10
        process = _Process(command)
        self._info("Running command '%s'." % process)
        stdout = process.read()
        rc = process.close()
        return (rc, stdout)

    def get_file(self, path, encoding='UTF-8', encoding_errors='strict'):
        if False:
            return 10
        'Returns the contents of a specified file.\n\n        This keyword reads the specified file and returns the contents.\n        Line breaks in content are converted to platform independent form.\n        See also `Get Binary File`.\n\n        ``encoding`` defines the encoding of the file. The default value is\n        ``UTF-8``, which means that UTF-8 and ASCII encoded files are read\n        correctly. In addition to the encodings supported by the underlying\n        Python implementation, the following special encoding values can be\n        used:\n\n        - ``SYSTEM``: Use the default system encoding.\n        - ``CONSOLE``: Use the console encoding. Outside Windows this is same\n          as the system encoding.\n\n        ``encoding_errors`` argument controls what to do if decoding some bytes\n        fails. All values accepted by ``decode`` method in Python are valid, but\n        in practice the following values are most useful:\n\n        - ``strict``: Fail if characters cannot be decoded (default).\n        - ``ignore``: Ignore characters that cannot be decoded.\n        - ``replace``: Replace characters that cannot be decoded with\n          a replacement character.\n        '
        path = self._absnorm(path)
        self._link("Getting file '%s'.", path)
        encoding = self._map_encoding(encoding)
        with open(path, encoding=encoding, errors=encoding_errors, newline='') as f:
            return f.read().replace('\r\n', '\n')

    def _map_encoding(self, encoding):
        if False:
            print('Hello World!')
        return {'SYSTEM': None, 'CONSOLE': CONSOLE_ENCODING}.get(encoding.upper(), encoding)

    def get_binary_file(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Returns the contents of a specified file.\n\n        This keyword reads the specified file and returns the contents as is.\n        See also `Get File`.\n        '
        path = self._absnorm(path)
        self._link("Getting file '%s'.", path)
        with open(path, 'rb') as f:
            return f.read()

    def grep_file(self, path, pattern, encoding='UTF-8', encoding_errors='strict', regexp=False):
        if False:
            print('Hello World!')
        'Returns the lines of the specified file that match the ``pattern``.\n\n        This keyword reads a file from the file system using the defined\n        ``path``, ``encoding`` and ``encoding_errors`` similarly as `Get File`.\n        A difference is that only the lines that match the given ``pattern`` are\n        returned. Lines are returned as a single string concatenated back together\n        with newlines and the number of matched lines is automatically logged.\n        Possible trailing newline is never returned.\n\n        A line matches if it contains the ``pattern`` anywhere in it i.e. it does\n        not need to match the pattern fully. There are two supported pattern types:\n\n        - By default the pattern is considered a _glob_ pattern where, for example,\n          ``*`` and ``?`` can be used as wildcards.\n        - If the ``regexp`` argument is given a true value, the pattern is\n          considered to be a _regular expression_. These patterns are more\n          powerful but also more complicated than glob patterns. They often use\n          the backslash character and it needs to be escaped in Robot Framework\n          date like `\\\\`.\n\n        For more information about glob and regular expression syntax, see\n        the `Pattern matching` section. With this keyword matching is always\n        case-sensitive.\n\n        Examples:\n        | ${errors} = | Grep File | /var/log/myapp.log | ERROR |\n        | ${ret} = | Grep File | ${CURDIR}/file.txt | [Ww]ildc??d ex*ple |\n        | ${ret} = | Grep File | ${CURDIR}/file.txt | [Ww]ildc\\\\w+d ex.*ple | regexp=True |\n\n        Special encoding values ``SYSTEM`` and ``CONSOLE`` that `Get File` supports\n        are supported by this keyword only with Robot Framework 4.0 and newer.\n\n        Support for regular expressions is new in Robot Framework 5.0.\n        '
        path = self._absnorm(path)
        if not regexp:
            pattern = fnmatch.translate(f'{pattern}*')
        reobj = re.compile(pattern)
        encoding = self._map_encoding(encoding)
        lines = []
        total_lines = 0
        self._link("Reading file '%s'.", path)
        with open(path, encoding=encoding, errors=encoding_errors) as file:
            for line in file:
                total_lines += 1
                line = line.rstrip('\r\n')
                if reobj.search(line):
                    lines.append(line)
            self._info('%d out of %d lines matched' % (len(lines), total_lines))
            return '\n'.join(lines)

    def log_file(self, path, encoding='UTF-8', encoding_errors='strict'):
        if False:
            for i in range(10):
                print('nop')
        'Wrapper for `Get File` that also logs the returned file.\n\n        The file is logged with the INFO level. If you want something else,\n        just use `Get File` and the built-in keyword `Log` with the desired\n        level.\n\n        See `Get File` for more information about ``encoding`` and\n        ``encoding_errors`` arguments.\n        '
        content = self.get_file(path, encoding, encoding_errors)
        self._info(content)
        return content

    def should_exist(self, path, msg=None):
        if False:
            return 10
        'Fails unless the given path (file or directory) exists.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        if not self._glob(path):
            self._fail(msg, "Path '%s' does not exist." % path)
        self._link("Path '%s' exists.", path)

    def should_not_exist(self, path, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails if the given path (file or directory) exists.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        matches = self._glob(path)
        if matches:
            self._fail(msg, self._get_matches_error('Path', path, matches))
        self._link("Path '%s' does not exist.", path)

    def _glob(self, path):
        if False:
            return 10
        return glob.glob(path) if not os.path.exists(path) else [path]

    def _get_matches_error(self, what, path, matches):
        if False:
            print('Hello World!')
        if not self._is_glob_path(path):
            return "%s '%s' exists." % (what, path)
        return "%s '%s' matches %s." % (what, path, seq2str(sorted(matches)))

    def _is_glob_path(self, path):
        if False:
            while True:
                i = 10
        return '*' in path or '?' in path or ('[' in path and ']' in path)

    def file_should_exist(self, path, msg=None):
        if False:
            while True:
                i = 10
        'Fails unless the given ``path`` points to an existing file.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        matches = [p for p in self._glob(path) if os.path.isfile(p)]
        if not matches:
            self._fail(msg, "File '%s' does not exist." % path)
        self._link("File '%s' exists.", path)

    def file_should_not_exist(self, path, msg=None):
        if False:
            i = 10
            return i + 15
        'Fails if the given path points to an existing file.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        matches = [p for p in self._glob(path) if os.path.isfile(p)]
        if matches:
            self._fail(msg, self._get_matches_error('File', path, matches))
        self._link("File '%s' does not exist.", path)

    def directory_should_exist(self, path, msg=None):
        if False:
            i = 10
            return i + 15
        'Fails unless the given path points to an existing directory.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        matches = [p for p in self._glob(path) if os.path.isdir(p)]
        if not matches:
            self._fail(msg, "Directory '%s' does not exist." % path)
        self._link("Directory '%s' exists.", path)

    def directory_should_not_exist(self, path, msg=None):
        if False:
            i = 10
            return i + 15
        'Fails if the given path points to an existing file.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        matches = [p for p in self._glob(path) if os.path.isdir(p)]
        if matches:
            self._fail(msg, self._get_matches_error('Directory', path, matches))
        self._link("Directory '%s' does not exist.", path)

    def wait_until_removed(self, path, timeout='1 minute'):
        if False:
            i = 10
            return i + 15
        'Waits until the given file or directory is removed.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n        If the path is a pattern, the keyword waits until all matching\n        items are removed.\n\n        The optional ``timeout`` can be used to control the maximum time of\n        waiting. The timeout is given as a timeout string, e.g. in a format\n        ``15 seconds``, ``1min 10s`` or just ``10``. The time string format is\n        described in an appendix of Robot Framework User Guide.\n\n        If the timeout is negative, the keyword is never timed-out. The keyword\n        returns immediately, if the path does not exist in the first place.\n        '
        path = self._absnorm(path)
        timeout = timestr_to_secs(timeout)
        maxtime = time.time() + timeout
        while self._glob(path):
            if timeout >= 0 and time.time() > maxtime:
                self._fail("'%s' was not removed in %s." % (path, secs_to_timestr(timeout)))
            time.sleep(0.1)
        self._link("'%s' was removed.", path)

    def wait_until_created(self, path, timeout='1 minute'):
        if False:
            i = 10
            return i + 15
        'Waits until the given file or directory is created.\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n        If the path is a pattern, the keyword returns when an item matching\n        it is created.\n\n        The optional ``timeout`` can be used to control the maximum time of\n        waiting. The timeout is given as a timeout string, e.g. in a format\n        ``15 seconds``, ``1min 10s`` or just ``10``. The time string format is\n        described in an appendix of Robot Framework User Guide.\n\n        If the timeout is negative, the keyword is never timed-out. The keyword\n        returns immediately, if the path already exists.\n        '
        path = self._absnorm(path)
        timeout = timestr_to_secs(timeout)
        maxtime = time.time() + timeout
        while not self._glob(path):
            if timeout >= 0 and time.time() > maxtime:
                self._fail("'%s' was not created in %s." % (path, secs_to_timestr(timeout)))
            time.sleep(0.1)
        self._link("'%s' was created.", path)

    def directory_should_be_empty(self, path, msg=None):
        if False:
            i = 10
            return i + 15
        'Fails unless the specified directory is empty.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        items = self._list_dir(path)
        if items:
            self._fail(msg, "Directory '%s' is not empty. Contents: %s." % (path, seq2str(items, lastsep=', ')))
        self._link("Directory '%s' is empty.", path)

    def directory_should_not_be_empty(self, path, msg=None):
        if False:
            while True:
                i = 10
        'Fails if the specified directory is empty.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        items = self._list_dir(path)
        if not items:
            self._fail(msg, "Directory '%s' is empty." % path)
        self._link("Directory '%%s' contains %d item%s." % (len(items), plural_or_not(items)), path)

    def file_should_be_empty(self, path, msg=None):
        if False:
            for i in range(10):
                print('nop')
        'Fails unless the specified file is empty.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        if not os.path.isfile(path):
            self._error("File '%s' does not exist." % path)
        size = os.stat(path).st_size
        if size > 0:
            self._fail(msg, "File '%s' is not empty. Size: %d bytes." % (path, size))
        self._link("File '%s' is empty.", path)

    def file_should_not_be_empty(self, path, msg=None):
        if False:
            return 10
        'Fails if the specified file is empty.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        path = self._absnorm(path)
        if not os.path.isfile(path):
            self._error("File '%s' does not exist." % path)
        size = os.stat(path).st_size
        if size == 0:
            self._fail(msg, "File '%s' is empty." % path)
        self._link("File '%%s' contains %d bytes." % size, path)

    def create_file(self, path, content='', encoding='UTF-8'):
        if False:
            i = 10
            return i + 15
        'Creates a file with the given content and encoding.\n\n        If the directory where the file is created does not exist, it is\n        automatically created along with possible missing intermediate\n        directories. Possible existing file is overwritten.\n\n        On Windows newline characters (``\\n``) in content are automatically\n        converted to Windows native newline sequence (``\\r\\n``).\n\n        See `Get File` for more information about possible ``encoding`` values,\n        including special values ``SYSTEM`` and ``CONSOLE``.\n\n        Examples:\n        | Create File | ${dir}/example.txt | Hello, world!       |         |\n        | Create File | ${path}            | Hyv\\xe4 esimerkki  | Latin-1 |\n        | Create File | /tmp/foo.txt       | 3\\nlines\\nhere\\n | SYSTEM  |\n\n        Use `Append To File` if you want to append to an existing file\n        and `Create Binary File` if you need to write bytes without encoding.\n        `File Should Not Exist` can be used to avoid overwriting existing\n        files.\n        '
        path = self._write_to_file(path, content, encoding)
        self._link("Created file '%s'.", path)

    def _write_to_file(self, path, content, encoding=None, mode='w'):
        if False:
            i = 10
            return i + 15
        path = self._absnorm(path)
        parent = os.path.dirname(path)
        if not os.path.exists(parent):
            os.makedirs(parent)
        if encoding:
            encoding = self._map_encoding(encoding)
        with open(path, mode, encoding=encoding) as f:
            f.write(content)
        return path

    def create_binary_file(self, path, content):
        if False:
            while True:
                i = 10
        'Creates a binary file with the given content.\n\n        If content is given as a Unicode string, it is first converted to bytes\n        character by character. All characters with ordinal below 256 can be\n        used and are converted to bytes with same values. Using characters\n        with higher ordinal is an error.\n\n        Byte strings, and possible other types, are written to the file as is.\n\n        If the directory for the file does not exist, it is created, along\n        with missing intermediate directories.\n\n        Examples:\n        | Create Binary File | ${dir}/example.png | ${image content} |\n        | Create Binary File | ${path}            | \\x01\\x00\\xe4\\x00 |\n\n        Use `Create File` if you want to create a text file using a certain\n        encoding. `File Should Not Exist` can be used to avoid overwriting\n        existing files.\n        '
        if is_string(content):
            content = bytes((ord(c) for c in content))
        path = self._write_to_file(path, content, mode='wb')
        self._link("Created binary file '%s'.", path)

    def append_to_file(self, path, content, encoding='UTF-8'):
        if False:
            for i in range(10):
                print('nop')
        'Appends the given content to the specified file.\n\n        If the file exists, the given text is written to its end. If the file\n        does not exist, it is created.\n\n        Other than not overwriting possible existing files, this keyword works\n        exactly like `Create File`. See its documentation for more details\n        about the usage.\n        '
        path = self._write_to_file(path, content, encoding, mode='a')
        self._link("Appended to file '%s'.", path)

    def remove_file(self, path):
        if False:
            print('Hello World!')
        'Removes a file with the given path.\n\n        Passes if the file does not exist, but fails if the path does\n        not point to a regular file (e.g. it points to a directory).\n\n        The path can be given as an exact path or as a glob pattern.\n        See the `Glob patterns` section for details about the supported syntax.\n        If the path is a pattern, all files matching it are removed.\n        '
        path = self._absnorm(path)
        matches = self._glob(path)
        if not matches:
            self._link("File '%s' does not exist.", path)
        for match in matches:
            if not os.path.isfile(match):
                self._error("Path '%s' is not a file." % match)
            os.remove(match)
            self._link("Removed file '%s'.", match)

    def remove_files(self, *paths):
        if False:
            while True:
                i = 10
        'Uses `Remove File` to remove multiple files one-by-one.\n\n        Example:\n        | Remove Files | ${TEMPDIR}${/}foo.txt | ${TEMPDIR}${/}bar.txt | ${TEMPDIR}${/}zap.txt |\n        '
        for path in paths:
            self.remove_file(path)

    def empty_directory(self, path):
        if False:
            return 10
        'Deletes all the content from the given directory.\n\n        Deletes both files and sub-directories, but the specified directory\n        itself if not removed. Use `Remove Directory` if you want to remove\n        the whole directory.\n        '
        path = self._absnorm(path)
        for item in self._list_dir(path, absolute=True):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
        self._link("Emptied directory '%s'.", path)

    def create_directory(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Creates the specified directory.\n\n        Also possible intermediate directories are created. Passes if the\n        directory already exists, but fails if the path exists and is not\n        a directory.\n        '
        path = self._absnorm(path)
        if os.path.isdir(path):
            self._link("Directory '%s' already exists.", path)
        elif os.path.exists(path):
            self._error("Path '%s' is not a directory." % path)
        else:
            os.makedirs(path)
            self._link("Created directory '%s'.", path)

    def remove_directory(self, path, recursive=False):
        if False:
            return 10
        'Removes the directory pointed to by the given ``path``.\n\n        If the second argument ``recursive`` is given a true value (see\n        `Boolean arguments`), the directory is removed recursively. Otherwise\n        removing fails if the directory is not empty.\n\n        If the directory pointed to by the ``path`` does not exist, the keyword\n        passes, but it fails, if the ``path`` points to a file.\n        '
        path = self._absnorm(path)
        if not os.path.exists(path):
            self._link("Directory '%s' does not exist.", path)
        elif not os.path.isdir(path):
            self._error("Path '%s' is not a directory." % path)
        else:
            if is_truthy(recursive):
                shutil.rmtree(path)
            else:
                self.directory_should_be_empty(path, "Directory '%s' is not empty." % path)
                os.rmdir(path)
            self._link("Removed directory '%s'.", path)

    def copy_file(self, source, destination):
        if False:
            print('Hello World!')
        'Copies the source file into the destination.\n\n        Source must be a path to an existing file or a glob pattern (see\n        `Glob patterns`) that matches exactly one file. How the\n        destination is interpreted is explained below.\n\n        1) If the destination is an existing file, the source file is copied\n        over it.\n\n        2) If the destination is an existing directory, the source file is\n        copied into it. A possible file with the same name as the source is\n        overwritten.\n\n        3) If the destination does not exist and it ends with a path\n        separator (``/`` or ``\\``), it is considered a directory. That\n        directory is created and a source file copied into it.\n        Possible missing intermediate directories are also created.\n\n        4) If the destination does not exist and it does not end with a path\n        separator, it is considered a file. If the path to the file does not\n        exist, it is created.\n\n        The resulting destination path is returned.\n\n        See also `Copy Files`, `Move File`, and `Move Files`.\n        '
        (source, destination) = self._prepare_copy_and_move_file(source, destination)
        if not self._are_source_and_destination_same_file(source, destination):
            (source, destination) = self._atomic_copy(source, destination)
            self._link("Copied file from '%s' to '%s'.", source, destination)
        return destination

    def _prepare_copy_and_move_file(self, source, destination):
        if False:
            i = 10
            return i + 15
        source = self._normalize_copy_and_move_source(source)
        destination = self._normalize_copy_and_move_destination(destination)
        if os.path.isdir(destination):
            destination = os.path.join(destination, os.path.basename(source))
        return (source, destination)

    def _normalize_copy_and_move_source(self, source):
        if False:
            print('Hello World!')
        source = self._absnorm(source)
        sources = self._glob(source)
        if len(sources) > 1:
            self._error("Multiple matches with source pattern '%s'." % source)
        if sources:
            source = sources[0]
        if not os.path.exists(source):
            self._error("Source file '%s' does not exist." % source)
        if not os.path.isfile(source):
            self._error("Source file '%s' is not a regular file." % source)
        return source

    def _normalize_copy_and_move_destination(self, destination):
        if False:
            i = 10
            return i + 15
        if isinstance(destination, pathlib.Path):
            destination = str(destination)
        is_dir = os.path.isdir(destination) or destination.endswith(('/', '\\'))
        destination = self._absnorm(destination)
        directory = destination if is_dir else os.path.dirname(destination)
        self._ensure_destination_directory_exists(directory)
        return destination

    def _ensure_destination_directory_exists(self, path):
        if False:
            print('Hello World!')
        if not os.path.exists(path):
            os.makedirs(path)
        elif not os.path.isdir(path):
            self._error("Destination '%s' exists and is not a directory." % path)

    def _are_source_and_destination_same_file(self, source, destination):
        if False:
            while True:
                i = 10
        if self._force_normalize(source) == self._force_normalize(destination):
            self._link("Source '%s' and destination '%s' point to the same file.", source, destination)
            return True
        return False

    def _force_normalize(self, path):
        if False:
            for i in range(10):
                print('nop')
        return os.path.realpath(normpath(path, case_normalize=True))

    def _atomic_copy(self, source, destination):
        if False:
            for i in range(10):
                print('nop')
        'Copy file atomically (or at least try to).\n\n        This method tries to ensure that a file copy operation will not fail\n        if the destination file is removed during copy operation. The problem\n        is that copying a file is typically not an atomic operation.\n\n        Luckily moving files is atomic in almost every platform, assuming files\n        are on the same filesystem, and we can use that as a workaround:\n        - First move the source to a temporary directory that is ensured to\n          be on the same filesystem as the destination.\n        - Move the temporary file over the real destination.\n\n        See also https://github.com/robotframework/robotframework/issues/1502\n        '
        temp_directory = tempfile.mkdtemp(dir=os.path.dirname(destination))
        temp_file = os.path.join(temp_directory, os.path.basename(source))
        try:
            shutil.copy(source, temp_file)
            if os.path.exists(destination):
                os.remove(destination)
            shutil.move(temp_file, destination)
        finally:
            shutil.rmtree(temp_directory)
        return (source, destination)

    def move_file(self, source, destination):
        if False:
            i = 10
            return i + 15
        'Moves the source file into the destination.\n\n        Arguments have exactly same semantics as with `Copy File` keyword.\n        Destination file path is returned.\n\n        If the source and destination are on the same filesystem, rename\n        operation is used. Otherwise file is copied to the destination\n        filesystem and then removed from the original filesystem.\n\n        See also `Move Files`, `Copy File`, and `Copy Files`.\n        '
        (source, destination) = self._prepare_copy_and_move_file(source, destination)
        if not self._are_source_and_destination_same_file(destination, source):
            shutil.move(source, destination)
            self._link("Moved file from '%s' to '%s'.", source, destination)
        return destination

    def copy_files(self, *sources_and_destination):
        if False:
            for i in range(10):
                print('nop')
        'Copies specified files to the target directory.\n\n        Source files can be given as exact paths and as glob patterns (see\n        `Glob patterns`). At least one source must be given, but it is\n        not an error if it is a pattern that does not match anything.\n\n        Last argument must be the destination directory. If the destination\n        does not exist, it will be created.\n\n        Examples:\n        | Copy Files | ${dir}/file1.txt  | ${dir}/file2.txt | ${dir2} |\n        | Copy Files | ${dir}/file-*.txt | ${dir2}          |         |\n\n        See also `Copy File`, `Move File`, and `Move Files`.\n        '
        (sources, destination) = self._prepare_copy_and_move_files(sources_and_destination)
        for source in sources:
            self.copy_file(source, destination)

    def _prepare_copy_and_move_files(self, items):
        if False:
            i = 10
            return i + 15
        if len(items) < 2:
            self._error('Must contain destination and at least one source.')
        sources = self._glob_files(items[:-1])
        destination = self._absnorm(items[-1])
        self._ensure_destination_directory_exists(destination)
        return (sources, destination)

    def _glob_files(self, patterns):
        if False:
            while True:
                i = 10
        files = []
        for pattern in patterns:
            files.extend(self._glob(self._absnorm(pattern)))
        return files

    def move_files(self, *sources_and_destination):
        if False:
            for i in range(10):
                print('nop')
        'Moves specified files to the target directory.\n\n        Arguments have exactly same semantics as with `Copy Files` keyword.\n\n        See also `Move File`, `Copy File`, and `Copy Files`.\n        '
        (sources, destination) = self._prepare_copy_and_move_files(sources_and_destination)
        for source in sources:
            self.move_file(source, destination)

    def copy_directory(self, source, destination):
        if False:
            return 10
        'Copies the source directory into the destination.\n\n        If the destination exists, the source is copied under it. Otherwise\n        the destination directory and the possible missing intermediate\n        directories are created.\n        '
        (source, destination) = self._prepare_copy_and_move_directory(source, destination)
        shutil.copytree(source, destination)
        self._link("Copied directory from '%s' to '%s'.", source, destination)

    def _prepare_copy_and_move_directory(self, source, destination):
        if False:
            i = 10
            return i + 15
        source = self._absnorm(source)
        destination = self._absnorm(destination)
        if not os.path.exists(source):
            self._error("Source '%s' does not exist." % source)
        if not os.path.isdir(source):
            self._error("Source '%s' is not a directory." % source)
        if os.path.exists(destination) and (not os.path.isdir(destination)):
            self._error("Destination '%s' is not a directory." % destination)
        if os.path.exists(destination):
            base = os.path.basename(source)
            destination = os.path.join(destination, base)
        else:
            parent = os.path.dirname(destination)
            if not os.path.exists(parent):
                os.makedirs(parent)
        return (source, destination)

    def move_directory(self, source, destination):
        if False:
            while True:
                i = 10
        'Moves the source directory into a destination.\n\n        Uses `Copy Directory` keyword internally, and ``source`` and\n        ``destination`` arguments have exactly same semantics as with\n        that keyword.\n        '
        (source, destination) = self._prepare_copy_and_move_directory(source, destination)
        shutil.move(source, destination)
        self._link("Moved directory from '%s' to '%s'.", source, destination)

    @keyword(types=None)
    def get_environment_variable(self, name, default=None):
        if False:
            i = 10
            return i + 15
        'Returns the value of an environment variable with the given name.\n\n        If no environment variable is found, returns possible default value.\n        If no default value is given, the keyword fails.\n\n        Returned variables are automatically decoded to Unicode using\n        the system encoding.\n\n        Note that you can also access environment variables directly using\n        the variable syntax ``%{ENV_VAR_NAME}``.\n        '
        value = get_env_var(name, default)
        if value is None:
            self._error("Environment variable '%s' does not exist." % name)
        return value

    def set_environment_variable(self, name, value):
        if False:
            while True:
                i = 10
        'Sets an environment variable to a specified value.\n\n        Values are converted to strings automatically. Set variables are\n        automatically encoded using the system encoding.\n        '
        set_env_var(name, value)
        self._info("Environment variable '%s' set to value '%s'." % (name, value))

    def append_to_environment_variable(self, name, *values, **config):
        if False:
            print('Hello World!')
        'Appends given ``values`` to environment variable ``name``.\n\n        If the environment variable already exists, values are added after it,\n        and otherwise a new environment variable is created.\n\n        Values are, by default, joined together using the operating system\n        path separator (``;`` on Windows, ``:`` elsewhere). This can be changed\n        by giving a separator after the values like ``separator=value``. No\n        other configuration parameters are accepted.\n\n        Examples (assuming ``NAME`` and ``NAME2`` do not exist initially):\n        | Append To Environment Variable | NAME     | first  |       |\n        | Should Be Equal                | %{NAME}  | first  |       |\n        | Append To Environment Variable | NAME     | second | third |\n        | Should Be Equal                | %{NAME}  | first${:}second${:}third |\n        | Append To Environment Variable | NAME2    | first  | separator=-     |\n        | Should Be Equal                | %{NAME2} | first  |                 |\n        | Append To Environment Variable | NAME2    | second | separator=-     |\n        | Should Be Equal                | %{NAME2} | first-second             |\n        '
        sentinel = object()
        initial = self.get_environment_variable(name, sentinel)
        if initial is not sentinel:
            values = (initial,) + values
        separator = config.pop('separator', os.pathsep)
        if config:
            config = ['='.join(i) for i in sorted(config.items())]
            self._error('Configuration %s not accepted.' % seq2str(config, lastsep=' or '))
        self.set_environment_variable(name, separator.join(values))

    def remove_environment_variable(self, *names):
        if False:
            return 10
        'Deletes the specified environment variable.\n\n        Does nothing if the environment variable is not set.\n\n        It is possible to remove multiple variables by passing them to this\n        keyword as separate arguments.\n        '
        for name in names:
            value = del_env_var(name)
            if value:
                self._info("Environment variable '%s' deleted." % name)
            else:
                self._info("Environment variable '%s' does not exist." % name)

    def environment_variable_should_be_set(self, name, msg=None):
        if False:
            return 10
        'Fails if the specified environment variable is not set.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        value = get_env_var(name)
        if not value:
            self._fail(msg, "Environment variable '%s' is not set." % name)
        self._info("Environment variable '%s' is set to '%s'." % (name, value))

    def environment_variable_should_not_be_set(self, name, msg=None):
        if False:
            while True:
                i = 10
        'Fails if the specified environment variable is set.\n\n        The default error message can be overridden with the ``msg`` argument.\n        '
        value = get_env_var(name)
        if value:
            self._fail(msg, "Environment variable '%s' is set to '%s'." % (name, value))
        self._info("Environment variable '%s' is not set." % name)

    def get_environment_variables(self):
        if False:
            while True:
                i = 10
        'Returns currently available environment variables as a dictionary.\n\n        Both keys and values are decoded to Unicode using the system encoding.\n        Altering the returned dictionary has no effect on the actual environment\n        variables.\n        '
        return get_env_vars()

    def log_environment_variables(self, level='INFO'):
        if False:
            print('Hello World!')
        'Logs all environment variables using the given log level.\n\n        Environment variables are also returned the same way as with\n        `Get Environment Variables` keyword.\n        '
        variables = get_env_vars()
        for name in sorted(variables, key=lambda item: item.lower()):
            self._log('%s = %s' % (name, variables[name]), level)
        return variables

    def join_path(self, base, *parts):
        if False:
            return 10
        "Joins the given path part(s) to the given base path.\n\n        The path separator (``/`` or ``\\``) is inserted when needed and\n        the possible absolute paths handled as expected. The resulted\n        path is also normalized.\n\n        Examples:\n        | ${path} = | Join Path | my        | path  |\n        | ${p2} =   | Join Path | my/       | path/ |\n        | ${p3} =   | Join Path | my        | path  | my | file.txt |\n        | ${p4} =   | Join Path | my        | /path |\n        | ${p5} =   | Join Path | /my/path/ | ..    | path2 |\n        =>\n        - ${path} = 'my/path'\n        - ${p2} = 'my/path'\n        - ${p3} = 'my/path/my/file.txt'\n        - ${p4} = '/path'\n        - ${p5} = '/my/path2'\n        "
        parts = [str(p) if isinstance(p, pathlib.Path) else p.replace('/', os.sep) for p in (base,) + parts]
        return self.normalize_path(os.path.join(*parts))

    def join_paths(self, base, *paths):
        if False:
            for i in range(10):
                print('nop')
        "Joins given paths with base and returns resulted paths.\n\n        See `Join Path` for more information.\n\n        Examples:\n        | @{p1} = | Join Paths | base     | example       | other |          |\n        | @{p2} = | Join Paths | /my/base | /example      | other |          |\n        | @{p3} = | Join Paths | my/base  | example/path/ | other | one/more |\n        =>\n        - @{p1} = ['base/example', 'base/other']\n        - @{p2} = ['/example', '/my/base/other']\n        - @{p3} = ['my/base/example/path', 'my/base/other', 'my/base/one/more']\n        "
        return [self.join_path(base, path) for path in paths]

    def normalize_path(self, path, case_normalize=False):
        if False:
            i = 10
            return i + 15
        "Normalizes the given path.\n\n        - Collapses redundant separators and up-level references.\n        - Converts ``/`` to ``\\`` on Windows.\n        - Replaces initial ``~`` or ``~user`` by that user's home directory.\n        - If ``case_normalize`` is given a true value (see `Boolean arguments`)\n          on Windows, converts the path to all lowercase.\n        - Converts ``pathlib.Path`` instances to ``str``.\n\n        Examples:\n        | ${path1} = | Normalize Path | abc/           |\n        | ${path2} = | Normalize Path | abc/../def     |\n        | ${path3} = | Normalize Path | abc/./def//ghi |\n        | ${path4} = | Normalize Path | ~robot/stuff   |\n        =>\n        - ${path1} = 'abc'\n        - ${path2} = 'def'\n        - ${path3} = 'abc/def/ghi'\n        - ${path4} = '/home/robot/stuff'\n\n        On Windows result would use ``\\`` instead of ``/`` and home directory\n        would be different.\n        "
        if isinstance(path, pathlib.Path):
            path = str(path)
        else:
            path = path.replace('/', os.sep)
        path = os.path.normpath(os.path.expanduser(path))
        if case_normalize:
            path = os.path.normcase(path)
        return path or '.'

    def split_path(self, path):
        if False:
            while True:
                i = 10
        "Splits the given path from the last path separator (``/`` or ``\\``).\n\n        The given path is first normalized (e.g. a possible trailing\n        path separator is removed, special directories ``..`` and ``.``\n        removed). The parts that are split are returned as separate\n        components.\n\n        Examples:\n        | ${path1} | ${dir} =  | Split Path | abc/def         |\n        | ${path2} | ${file} = | Split Path | abc/def/ghi.txt |\n        | ${path3} | ${d2}  =  | Split Path | abc/../def/ghi/ |\n        =>\n        - ${path1} = 'abc' & ${dir} = 'def'\n        - ${path2} = 'abc/def' & ${file} = 'ghi.txt'\n        - ${path3} = 'def' & ${d2} = 'ghi'\n        "
        return os.path.split(self.normalize_path(path))

    def split_extension(self, path):
        if False:
            print('Hello World!')
        "Splits the extension from the given path.\n\n        The given path is first normalized (e.g. possible trailing\n        path separators removed, special directories ``..`` and ``.``\n        removed). The base path and extension are returned as separate\n        components so that the dot used as an extension separator is\n        removed. If the path contains no extension, an empty string is\n        returned for it. Possible leading and trailing dots in the file\n        name are never considered to be extension separators.\n\n        Examples:\n        | ${path} | ${ext} = | Split Extension | file.extension    |\n        | ${p2}   | ${e2} =  | Split Extension | path/file.ext     |\n        | ${p3}   | ${e3} =  | Split Extension | path/file         |\n        | ${p4}   | ${e4} =  | Split Extension | p1/../p2/file.ext |\n        | ${p5}   | ${e5} =  | Split Extension | path/.file.ext    |\n        | ${p6}   | ${e6} =  | Split Extension | path/.file        |\n        =>\n        - ${path} = 'file' & ${ext} = 'extension'\n        - ${p2} = 'path/file' & ${e2} = 'ext'\n        - ${p3} = 'path/file' & ${e3} = ''\n        - ${p4} = 'p2/file' & ${e4} = 'ext'\n        - ${p5} = 'path/.file' & ${e5} = 'ext'\n        - ${p6} = 'path/.file' & ${e6} = ''\n        "
        path = self.normalize_path(path)
        basename = os.path.basename(path)
        if basename.startswith('.' * basename.count('.')):
            return (path, '')
        if path.endswith('.'):
            path2 = path.rstrip('.')
            trailing_dots = '.' * (len(path) - len(path2))
            path = path2
        else:
            trailing_dots = ''
        (basepath, extension) = os.path.splitext(path)
        if extension.startswith('.'):
            extension = extension[1:]
        if extension:
            extension += trailing_dots
        else:
            basepath += trailing_dots
        return (basepath, extension)

    def get_modified_time(self, path, format='timestamp'):
        if False:
            while True:
                i = 10
        "Returns the last modification time of a file or directory.\n\n        How time is returned is determined based on the given ``format``\n        string as follows. Note that all checks are case-insensitive.\n        Returned time is also automatically logged.\n\n        1) If ``format`` contains the word ``epoch``, the time is returned\n           in seconds after the UNIX epoch. The return value is always\n           an integer.\n\n        2) If ``format`` contains any of the words ``year``, ``month``,\n           ``day``, ``hour``, ``min`` or ``sec``, only the selected parts are\n           returned. The order of the returned parts is always the one\n           in the previous sentence and the order of the words in\n           ``format`` is not significant. The parts are returned as\n           zero-padded strings (e.g. May -> ``05``).\n\n        3) Otherwise, and by default, the time is returned as a\n           timestamp string in the format ``2006-02-24 15:08:31``.\n\n        Examples (when the modified time of ``${CURDIR}`` is\n        2006-03-29 15:06:21):\n        | ${time} = | Get Modified Time | ${CURDIR} |\n        | ${secs} = | Get Modified Time | ${CURDIR} | epoch |\n        | ${year} = | Get Modified Time | ${CURDIR} | return year |\n        | ${y} | ${d} = | Get Modified Time | ${CURDIR} | year,day |\n        | @{time} = | Get Modified Time | ${CURDIR} | year,month,day,hour,min,sec |\n        =>\n        - ${time} = '2006-03-29 15:06:21'\n        - ${secs} = 1143637581\n        - ${year} = '2006'\n        - ${y} = '2006' & ${d} = '29'\n        - @{time} = ['2006', '03', '29', '15', '06', '21']\n        "
        path = self._absnorm(path)
        if not os.path.exists(path):
            self._error("Path '%s' does not exist." % path)
        mtime = get_time(format, os.stat(path).st_mtime)
        self._link("Last modified time of '%%s' is %s." % mtime, path)
        return mtime

    def set_modified_time(self, path, mtime):
        if False:
            i = 10
            return i + 15
        'Sets the file modification and access times.\n\n        Changes the modification and access times of the given file to\n        the value determined by ``mtime``. The time can be given in\n        different formats described below. Note that all checks\n        involving strings are case-insensitive. Modified time can only\n        be set to regular files.\n\n        1) If ``mtime`` is a number, or a string that can be converted\n           to a number, it is interpreted as seconds since the UNIX\n           epoch (1970-01-01 00:00:00 UTC). This documentation was\n           originally written about 1177654467 seconds after the epoch.\n\n        2) If ``mtime`` is a timestamp, that time will be used. Valid\n           timestamp formats are ``YYYY-MM-DD hh:mm:ss`` and\n           ``YYYYMMDD hhmmss``.\n\n        3) If ``mtime`` is equal to ``NOW``, the current local time is used.\n\n        4) If ``mtime`` is equal to ``UTC``, the current time in\n           [http://en.wikipedia.org/wiki/Coordinated_Universal_Time|UTC]\n           is used.\n\n        5) If ``mtime`` is in the format like ``NOW - 1 day`` or ``UTC + 1\n           hour 30 min``, the current local/UTC time plus/minus the time\n           specified with the time string is used. The time string format\n           is described in an appendix of Robot Framework User Guide.\n\n        Examples:\n        | Set Modified Time | /path/file | 1177654467         | # Time given as epoch seconds |\n        | Set Modified Time | /path/file | 2007-04-27 9:14:27 | # Time given as a timestamp   |\n        | Set Modified Time | /path/file | NOW                | # The local time of execution |\n        | Set Modified Time | /path/file | NOW - 1 day        | # 1 day subtracted from the local time |\n        | Set Modified Time | /path/file | UTC + 1h 2min 3s   | # 1h 2min 3s added to the UTC time |\n        '
        mtime = parse_time(mtime)
        path = self._absnorm(path)
        if not os.path.exists(path):
            self._error("File '%s' does not exist." % path)
        if not os.path.isfile(path):
            self._error("Path '%s' is not a regular file." % path)
        os.utime(path, (mtime, mtime))
        time.sleep(0.1)
        tstamp = datetime.fromtimestamp(mtime).isoformat(' ', timespec='seconds')
        self._link("Set modified time of '%%s' to %s." % tstamp, path)

    def get_file_size(self, path):
        if False:
            while True:
                i = 10
        'Returns and logs file size as an integer in bytes.'
        path = self._absnorm(path)
        if not os.path.isfile(path):
            self._error("File '%s' does not exist." % path)
        size = os.stat(path).st_size
        plural = plural_or_not(size)
        self._link("Size of file '%%s' is %d byte%s." % (size, plural), path)
        return size

    def list_directory(self, path, pattern=None, absolute=False):
        if False:
            while True:
                i = 10
        "Returns and logs items in a directory, optionally filtered with ``pattern``.\n\n        File and directory names are returned in case-sensitive alphabetical\n        order, e.g. ``['A Name', 'Second', 'a lower case name', 'one more']``.\n        Implicit directories ``.`` and ``..`` are not returned. The returned\n        items are automatically logged.\n\n        File and directory names are returned relative to the given path\n        (e.g. ``'file.txt'``) by default. If you want them be returned in\n        absolute format (e.g. ``'/home/robot/file.txt'``), give the ``absolute``\n        argument a true value (see `Boolean arguments`).\n\n        If ``pattern`` is given, only items matching it are returned. The pattern\n        is considered to be a _glob pattern_ and the full syntax is explained in\n        the `Glob patterns` section. With this keyword matching is always\n        case-sensitive.\n\n        Examples (using also other `List Directory` variants):\n        | @{items} = | List Directory           | ${TEMPDIR} |\n        | @{files} = | List Files In Directory  | /tmp | *.txt | absolute |\n        | ${count} = | Count Files In Directory | ${CURDIR} | ??? |\n        "
        items = self._list_dir(path, pattern, absolute)
        self._info('%d item%s:\n%s' % (len(items), plural_or_not(items), '\n'.join(items)))
        return items

    def list_files_in_directory(self, path, pattern=None, absolute=False):
        if False:
            return 10
        'Wrapper for `List Directory` that returns only files.'
        files = self._list_files_in_dir(path, pattern, absolute)
        self._info('%d file%s:\n%s' % (len(files), plural_or_not(files), '\n'.join(files)))
        return files

    def list_directories_in_directory(self, path, pattern=None, absolute=False):
        if False:
            while True:
                i = 10
        'Wrapper for `List Directory` that returns only directories.'
        dirs = self._list_dirs_in_dir(path, pattern, absolute)
        self._info('%d director%s:\n%s' % (len(dirs), 'y' if len(dirs) == 1 else 'ies', '\n'.join(dirs)))
        return dirs

    def count_items_in_directory(self, path, pattern=None):
        if False:
            print('Hello World!')
        'Returns and logs the number of all items in the given directory.\n\n        The argument ``pattern`` has the same semantics as with `List Directory`\n        keyword. The count is returned as an integer, so it must be checked e.g.\n        with the built-in keyword `Should Be Equal As Integers`.\n        '
        count = len(self._list_dir(path, pattern))
        self._info('%s item%s.' % (count, plural_or_not(count)))
        return count

    def count_files_in_directory(self, path, pattern=None):
        if False:
            for i in range(10):
                print('nop')
        'Wrapper for `Count Items In Directory` returning only file count.'
        count = len(self._list_files_in_dir(path, pattern))
        self._info('%s file%s.' % (count, plural_or_not(count)))
        return count

    def count_directories_in_directory(self, path, pattern=None):
        if False:
            for i in range(10):
                print('nop')
        'Wrapper for `Count Items In Directory` returning only directory count.'
        count = len(self._list_dirs_in_dir(path, pattern))
        self._info('%s director%s.' % (count, 'y' if count == 1 else 'ies'))
        return count

    def _list_dir(self, path, pattern=None, absolute=False):
        if False:
            i = 10
            return i + 15
        path = self._absnorm(path)
        self._link("Listing contents of directory '%s'.", path)
        if not os.path.isdir(path):
            self._error("Directory '%s' does not exist." % path)
        items = sorted((safe_str(item) for item in os.listdir(path)))
        if pattern:
            items = [i for i in items if fnmatch.fnmatchcase(i, pattern)]
        if is_truthy(absolute):
            path = os.path.normpath(path)
            items = [os.path.join(path, item) for item in items]
        return items

    def _list_files_in_dir(self, path, pattern=None, absolute=False):
        if False:
            for i in range(10):
                print('nop')
        return [item for item in self._list_dir(path, pattern, absolute) if os.path.isfile(os.path.join(path, item))]

    def _list_dirs_in_dir(self, path, pattern=None, absolute=False):
        if False:
            for i in range(10):
                print('nop')
        return [item for item in self._list_dir(path, pattern, absolute) if os.path.isdir(os.path.join(path, item))]

    def touch(self, path):
        if False:
            while True:
                i = 10
        'Emulates the UNIX touch command.\n\n        Creates a file, if it does not exist. Otherwise changes its access and\n        modification times to the current time.\n\n        Fails if used with the directories or the parent directory of the given\n        file does not exist.\n        '
        path = self._absnorm(path)
        if os.path.isdir(path):
            self._error("Cannot touch '%s' because it is a directory." % path)
        if not os.path.exists(os.path.dirname(path)):
            self._error("Cannot touch '%s' because its parent directory does not exist." % path)
        if os.path.exists(path):
            mtime = round(time.time())
            os.utime(path, (mtime, mtime))
            self._link("Touched existing file '%s'.", path)
        else:
            open(path, 'w').close()
            self._link("Touched new file '%s'.", path)

    def _absnorm(self, path):
        if False:
            while True:
                i = 10
        return abspath(self.normalize_path(path))

    def _fail(self, *messages):
        if False:
            i = 10
            return i + 15
        raise AssertionError(next((msg for msg in messages if msg)))

    def _error(self, msg):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError(msg)

    def _info(self, msg):
        if False:
            for i in range(10):
                print('nop')
        self._log(msg, 'INFO')

    def _link(self, msg, *paths):
        if False:
            print('Hello World!')
        paths = tuple(('<a href="file://%s">%s</a>' % (p, p) for p in paths))
        self._log(msg % paths, 'HTML')

    def _warn(self, msg):
        if False:
            return 10
        self._log(msg, 'WARN')

    def _log(self, msg, level):
        if False:
            while True:
                i = 10
        logger.write(msg, level)

class _Process:

    def __init__(self, command):
        if False:
            i = 10
            return i + 15
        self._command = self._process_command(command)
        self._process = os.popen(self._command)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._command

    def read(self):
        if False:
            for i in range(10):
                print('nop')
        return self._process_output(self._process.read())

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            rc = self._process.close()
        except IOError:
            return 255
        if rc is None:
            return 0
        if WINDOWS:
            return rc % 256
        return rc >> 8

    def _process_command(self, command):
        if False:
            i = 10
            return i + 15
        if '>' not in command:
            if command.endswith('&'):
                command = command[:-1] + ' 2>&1 &'
            else:
                command += ' 2>&1'
        return command

    def _process_output(self, output):
        if False:
            for i in range(10):
                print('nop')
        if '\r\n' in output:
            output = output.replace('\r\n', '\n')
        if output.endswith('\n'):
            output = output[:-1]
        return console_decode(output)