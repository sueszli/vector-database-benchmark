"""text_file

provides the TextFile class, which gives an interface to text files
that (optionally) takes care of stripping comments, ignoring blank
lines, and joining lines with backslashes."""
import sys, io

class TextFile:
    """Provides a file-like object that takes care of all the things you
       commonly want to do when processing a text file that has some
       line-by-line syntax: strip comments (as long as "#" is your
       comment character), skip blank lines, join adjacent lines by
       escaping the newline (ie. backslash at end of line), strip
       leading and/or trailing whitespace.  All of these are optional
       and independently controllable.

       Provides a 'warn()' method so you can generate warning messages that
       report physical line number, even if the logical line in question
       spans multiple physical lines.  Also provides 'unreadline()' for
       implementing line-at-a-time lookahead.

       Constructor is called as:

           TextFile (filename=None, file=None, **options)

       It bombs (RuntimeError) if both 'filename' and 'file' are None;
       'filename' should be a string, and 'file' a file object (or
       something that provides 'readline()' and 'close()' methods).  It is
       recommended that you supply at least 'filename', so that TextFile
       can include it in warning messages.  If 'file' is not supplied,
       TextFile creates its own using 'io.open()'.

       The options are all boolean, and affect the value returned by
       'readline()':
         strip_comments [default: true]
           strip from "#" to end-of-line, as well as any whitespace
           leading up to the "#" -- unless it is escaped by a backslash
         lstrip_ws [default: false]
           strip leading whitespace from each line before returning it
         rstrip_ws [default: true]
           strip trailing whitespace (including line terminator!) from
           each line before returning it
         skip_blanks [default: true}
           skip lines that are empty *after* stripping comments and
           whitespace.  (If both lstrip_ws and rstrip_ws are false,
           then some lines may consist of solely whitespace: these will
           *not* be skipped, even if 'skip_blanks' is true.)
         join_lines [default: false]
           if a backslash is the last non-newline character on a line
           after stripping comments and whitespace, join the following line
           to it to form one "logical line"; if N consecutive lines end
           with a backslash, then N+1 physical lines will be joined to
           form one logical line.
         collapse_join [default: false]
           strip leading whitespace from lines that are joined to their
           predecessor; only matters if (join_lines and not lstrip_ws)
         errors [default: 'strict']
           error handler used to decode the file content

       Note that since 'rstrip_ws' can strip the trailing newline, the
       semantics of 'readline()' must differ from those of the builtin file
       object's 'readline()' method!  In particular, 'readline()' returns
       None for end-of-file: an empty string might just be a blank line (or
       an all-whitespace line), if 'rstrip_ws' is true but 'skip_blanks' is
       not."""
    default_options = {'strip_comments': 1, 'skip_blanks': 1, 'lstrip_ws': 0, 'rstrip_ws': 1, 'join_lines': 0, 'collapse_join': 0, 'errors': 'strict'}

    def __init__(self, filename=None, file=None, **options):
        if False:
            for i in range(10):
                print('nop')
        "Construct a new TextFile object.  At least one of 'filename'\n           (a string) and 'file' (a file-like object) must be supplied.\n           They keyword argument options are described above and affect\n           the values returned by 'readline()'."
        if filename is None and file is None:
            raise RuntimeError("you must supply either or both of 'filename' and 'file'")
        for opt in self.default_options.keys():
            if opt in options:
                setattr(self, opt, options[opt])
            else:
                setattr(self, opt, self.default_options[opt])
        for opt in options.keys():
            if opt not in self.default_options:
                raise KeyError("invalid TextFile option '%s'" % opt)
        if file is None:
            self.open(filename)
        else:
            self.filename = filename
            self.file = file
            self.current_line = 0
        self.linebuf = []

    def open(self, filename):
        if False:
            for i in range(10):
                print('nop')
        "Open a new file named 'filename'.  This overrides both the\n           'filename' and 'file' arguments to the constructor."
        self.filename = filename
        self.file = io.open(self.filename, 'r', errors=self.errors)
        self.current_line = 0

    def close(self):
        if False:
            i = 10
            return i + 15
        'Close the current file and forget everything we know about it\n           (filename, current line number).'
        file = self.file
        self.file = None
        self.filename = None
        self.current_line = None
        file.close()

    def gen_error(self, msg, line=None):
        if False:
            while True:
                i = 10
        outmsg = []
        if line is None:
            line = self.current_line
        outmsg.append(self.filename + ', ')
        if isinstance(line, (list, tuple)):
            outmsg.append('lines %d-%d: ' % tuple(line))
        else:
            outmsg.append('line %d: ' % line)
        outmsg.append(str(msg))
        return ''.join(outmsg)

    def error(self, msg, line=None):
        if False:
            while True:
                i = 10
        raise ValueError('error: ' + self.gen_error(msg, line))

    def warn(self, msg, line=None):
        if False:
            for i in range(10):
                print('nop')
        'Print (to stderr) a warning message tied to the current logical\n           line in the current file.  If the current logical line in the\n           file spans multiple physical lines, the warning refers to the\n           whole range, eg. "lines 3-5".  If \'line\' supplied, it overrides\n           the current line number; it may be a list or tuple to indicate a\n           range of physical lines, or an integer for a single physical\n           line.'
        sys.stderr.write('warning: ' + self.gen_error(msg, line) + '\n')

    def readline(self):
        if False:
            for i in range(10):
                print('nop')
        'Read and return a single logical line from the current file (or\n           from an internal buffer if lines have previously been "unread"\n           with \'unreadline()\').  If the \'join_lines\' option is true, this\n           may involve reading multiple physical lines concatenated into a\n           single string.  Updates the current line number, so calling\n           \'warn()\' after \'readline()\' emits a warning about the physical\n           line(s) just read.  Returns None on end-of-file, since the empty\n           string can occur if \'rstrip_ws\' is true but \'strip_blanks\' is\n           not.'
        if self.linebuf:
            line = self.linebuf[-1]
            del self.linebuf[-1]
            return line
        buildup_line = ''
        while True:
            line = self.file.readline()
            if line == '':
                line = None
            if self.strip_comments and line:
                pos = line.find('#')
                if pos == -1:
                    pass
                elif pos == 0 or line[pos - 1] != '\\':
                    eol = line[-1] == '\n' and '\n' or ''
                    line = line[0:pos] + eol
                    if line.strip() == '':
                        continue
                else:
                    line = line.replace('\\#', '#')
            if self.join_lines and buildup_line:
                if line is None:
                    self.warn('continuation line immediately precedes end-of-file')
                    return buildup_line
                if self.collapse_join:
                    line = line.lstrip()
                line = buildup_line + line
                if isinstance(self.current_line, list):
                    self.current_line[1] = self.current_line[1] + 1
                else:
                    self.current_line = [self.current_line, self.current_line + 1]
            else:
                if line is None:
                    return None
                if isinstance(self.current_line, list):
                    self.current_line = self.current_line[1] + 1
                else:
                    self.current_line = self.current_line + 1
            if self.lstrip_ws and self.rstrip_ws:
                line = line.strip()
            elif self.lstrip_ws:
                line = line.lstrip()
            elif self.rstrip_ws:
                line = line.rstrip()
            if (line == '' or line == '\n') and self.skip_blanks:
                continue
            if self.join_lines:
                if line[-1] == '\\':
                    buildup_line = line[:-1]
                    continue
                if line[-2:] == '\\\n':
                    buildup_line = line[0:-2] + '\n'
                    continue
            return line

    def readlines(self):
        if False:
            for i in range(10):
                print('nop')
        'Read and return the list of all logical lines remaining in the\n           current file.'
        lines = []
        while True:
            line = self.readline()
            if line is None:
                return lines
            lines.append(line)

    def unreadline(self, line):
        if False:
            while True:
                i = 10
        "Push 'line' (a string) onto an internal buffer that will be\n           checked by future 'readline()' calls.  Handy for implementing\n           a parser with line-at-a-time lookahead."
        self.linebuf.append(line)