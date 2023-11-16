"""Building Blocks of TensorFlow Debugger Command-Line Interface."""
import copy
import os
import re
import traceback
import numpy as np
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.platform import gfile
HELP_INDENT = '  '
EXPLICIT_USER_EXIT = 'explicit_user_exit'
REGEX_MATCH_LINES_KEY = 'regex_match_lines'
INIT_SCROLL_POS_KEY = 'init_scroll_pos'
MAIN_MENU_KEY = 'mm:'

class CommandLineExit(Exception):

    def __init__(self, exit_token=None):
        if False:
            while True:
                i = 10
        Exception.__init__(self)
        self._exit_token = exit_token

    @property
    def exit_token(self):
        if False:
            while True:
                i = 10
        return self._exit_token

class RichLine:
    """Rich single-line text.

  Attributes:
    text: A plain string, the raw text represented by this object.  Should not
      contain newlines.
    font_attr_segs: A list of (start, end, font attribute) triples, representing
      richness information applied to substrings of text.
  """

    def __init__(self, text='', font_attr=None):
        if False:
            for i in range(10):
                print('nop')
        'Construct a RichLine with no rich attributes or a single attribute.\n\n    Args:\n      text: Raw text string\n      font_attr: If specified, a single font attribute to be applied to the\n        entire text.  Extending this object via concatenation allows creation\n        of text with varying attributes.\n    '
        self.text = text
        if font_attr:
            self.font_attr_segs = [(0, len(text), font_attr)]
        else:
            self.font_attr_segs = []

    def __add__(self, other):
        if False:
            print('Hello World!')
        'Concatenate two chunks of maybe rich text to make a longer rich line.\n\n    Does not modify self.\n\n    Args:\n      other: Another piece of text to concatenate with this one.\n        If it is a plain str, it will be appended to this string with no\n        attributes.  If it is a RichLine, it will be appended to this string\n        with its attributes preserved.\n\n    Returns:\n      A new RichLine comprising both chunks of text, with appropriate\n        attributes applied to the corresponding substrings.\n    '
        ret = RichLine()
        if isinstance(other, str):
            ret.text = self.text + other
            ret.font_attr_segs = self.font_attr_segs[:]
            return ret
        elif isinstance(other, RichLine):
            ret.text = self.text + other.text
            ret.font_attr_segs = self.font_attr_segs[:]
            old_len = len(self.text)
            for (start, end, font_attr) in other.font_attr_segs:
                ret.font_attr_segs.append((old_len + start, old_len + end, font_attr))
            return ret
        else:
            raise TypeError('%r cannot be concatenated with a RichLine' % other)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.text)

def rich_text_lines_from_rich_line_list(rich_text_list, annotations=None):
    if False:
        while True:
            i = 10
    'Convert a list of RichLine objects or strings to a RichTextLines object.\n\n  Args:\n    rich_text_list: a list of RichLine objects or strings\n    annotations: annotations for the resultant RichTextLines object.\n\n  Returns:\n    A corresponding RichTextLines object.\n  '
    lines = []
    font_attr_segs = {}
    for (i, rl) in enumerate(rich_text_list):
        if isinstance(rl, RichLine):
            lines.append(rl.text)
            if rl.font_attr_segs:
                font_attr_segs[i] = rl.font_attr_segs
        else:
            lines.append(rl)
    return RichTextLines(lines, font_attr_segs, annotations=annotations)

def get_tensorflow_version_lines(include_dependency_versions=False):
    if False:
        while True:
            i = 10
    "Generate RichTextLines with TensorFlow version info.\n\n  Args:\n    include_dependency_versions: Include the version of TensorFlow's key\n      dependencies, such as numpy.\n\n  Returns:\n    A formatted, multi-line `RichTextLines` object.\n  "
    lines = ['TensorFlow version: %s' % pywrap_tf_session.__version__]
    lines.append('')
    if include_dependency_versions:
        lines.append('Dependency version(s):')
        lines.append('  numpy: %s' % np.__version__)
        lines.append('')
    return RichTextLines(lines)

class RichTextLines:
    """Rich multi-line text.

  Line-by-line text output, with font attributes (e.g., color) and annotations
  (e.g., indices in a multi-dimensional tensor). Used as the text output of CLI
  commands. Can be rendered on terminal environments such as curses.

  This is not to be confused with Rich Text Format (RTF). This class is for text
  lines only.
  """

    def __init__(self, lines, font_attr_segs=None, annotations=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor of RichTextLines.\n\n    Args:\n      lines: A list of str or a single str, representing text output to\n        screen. The latter case is for convenience when the text output is\n        single-line.\n      font_attr_segs: A map from 0-based row index to a list of 3-tuples.\n        It lists segments in each row that have special font attributes, such\n        as colors, that are not the default attribute. For example:\n        {1: [(0, 3, "red"), (4, 7, "green")], 2: [(10, 20, "yellow")]}\n\n        In each tuple, the 1st element is the start index of the segment. The\n        2nd element is the end index, in an "open interval" fashion. The 3rd\n        element is an object or a list of objects that represents the font\n        attribute. Colors are represented as strings as in the examples above.\n      annotations: A map from 0-based row index to any object for annotating\n        the row. A typical use example is annotating rows of the output as\n        indices in a multi-dimensional tensor. For example, consider the\n        following text representation of a 3x2x2 tensor:\n          [[[0, 0], [0, 0]],\n           [[0, 0], [0, 0]],\n           [[0, 0], [0, 0]]]\n        The annotation can indicate the indices of the first element shown in\n        each row, i.e.,\n          {0: [0, 0, 0], 1: [1, 0, 0], 2: [2, 0, 0]}\n        This information can make display of tensors on screen clearer and can\n        help the user navigate (scroll) to the desired location in a large\n        tensor.\n\n    Raises:\n      ValueError: If lines is of invalid type.\n    '
        if isinstance(lines, list):
            self._lines = lines
        elif isinstance(lines, str):
            self._lines = [lines]
        else:
            raise ValueError('Unexpected type in lines: %s' % type(lines))
        self._font_attr_segs = font_attr_segs
        if not self._font_attr_segs:
            self._font_attr_segs = {}
        self._annotations = annotations
        if not self._annotations:
            self._annotations = {}

    @property
    def lines(self):
        if False:
            print('Hello World!')
        return self._lines

    @property
    def font_attr_segs(self):
        if False:
            i = 10
            return i + 15
        return self._font_attr_segs

    @property
    def annotations(self):
        if False:
            for i in range(10):
                print('nop')
        return self._annotations

    def num_lines(self):
        if False:
            while True:
                i = 10
        return len(self._lines)

    def slice(self, begin, end):
        if False:
            for i in range(10):
                print('nop')
        'Slice a RichTextLines object.\n\n    The object itself is not changed. A sliced instance is returned.\n\n    Args:\n      begin: (int) Beginning line index (inclusive). Must be >= 0.\n      end: (int) Ending line index (exclusive). Must be >= 0.\n\n    Returns:\n      (RichTextLines) Sliced output instance of RichTextLines.\n\n    Raises:\n      ValueError: If begin or end is negative.\n    '
        if begin < 0 or end < 0:
            raise ValueError('Encountered negative index.')
        lines = self.lines[begin:end]
        font_attr_segs = {}
        for key in self.font_attr_segs:
            if key >= begin and key < end:
                font_attr_segs[key - begin] = self.font_attr_segs[key]
        annotations = {}
        for key in self.annotations:
            if not isinstance(key, int):
                annotations[key] = self.annotations[key]
            elif key >= begin and key < end:
                annotations[key - begin] = self.annotations[key]
        return RichTextLines(lines, font_attr_segs=font_attr_segs, annotations=annotations)

    def extend(self, other):
        if False:
            print('Hello World!')
        'Extend this instance of RichTextLines with another instance.\n\n    The extension takes effect on the text lines, the font attribute segments,\n    as well as the annotations. The line indices in the font attribute\n    segments and the annotations are adjusted to account for the existing\n    lines. If there are duplicate, non-line-index fields in the annotations,\n    the value from the input argument "other" will override that in this\n    instance.\n\n    Args:\n      other: (RichTextLines) The other RichTextLines instance to be appended at\n        the end of this instance.\n    '
        orig_num_lines = self.num_lines()
        self._lines.extend(other.lines)
        for line_index in other.font_attr_segs:
            self._font_attr_segs[orig_num_lines + line_index] = other.font_attr_segs[line_index]
        for key in other.annotations:
            if isinstance(key, int):
                self._annotations[orig_num_lines + key] = other.annotations[key]
            else:
                self._annotations[key] = other.annotations[key]

    def _extend_before(self, other):
        if False:
            print('Hello World!')
        'Add another RichTextLines object to the front.\n\n    Args:\n      other: (RichTextLines) The other object to add to the front to this\n        object.\n    '
        other_num_lines = other.num_lines()
        self._lines = other.lines + self._lines
        new_font_attr_segs = {}
        for line_index in self.font_attr_segs:
            new_font_attr_segs[other_num_lines + line_index] = self.font_attr_segs[line_index]
        new_font_attr_segs.update(other.font_attr_segs)
        self._font_attr_segs = new_font_attr_segs
        new_annotations = {}
        for key in self._annotations:
            if isinstance(key, int):
                new_annotations[other_num_lines + key] = self.annotations[key]
            else:
                new_annotations[key] = other.annotations[key]
        new_annotations.update(other.annotations)
        self._annotations = new_annotations

    def append(self, line, font_attr_segs=None):
        if False:
            i = 10
            return i + 15
        'Append a single line of text.\n\n    Args:\n      line: (str) The text to be added to the end.\n      font_attr_segs: (list of tuples) Font attribute segments of the appended\n        line.\n    '
        self._lines.append(line)
        if font_attr_segs:
            self._font_attr_segs[len(self._lines) - 1] = font_attr_segs

    def append_rich_line(self, rich_line):
        if False:
            return 10
        self.append(rich_line.text, rich_line.font_attr_segs)

    def prepend(self, line, font_attr_segs=None):
        if False:
            while True:
                i = 10
        'Prepend (i.e., add to the front) a single line of text.\n\n    Args:\n      line: (str) The text to be added to the front.\n      font_attr_segs: (list of tuples) Font attribute segments of the appended\n        line.\n    '
        other = RichTextLines(line)
        if font_attr_segs:
            other.font_attr_segs[0] = font_attr_segs
        self._extend_before(other)

    def write_to_file(self, file_path):
        if False:
            for i in range(10):
                print('nop')
        'Write the object itself to file, in a plain format.\n\n    The font_attr_segs and annotations are ignored.\n\n    Args:\n      file_path: (str) path of the file to write to.\n    '
        with gfile.Open(file_path, 'w') as f:
            for line in self._lines:
                f.write(line + '\n')

def regex_find(orig_screen_output, regex, font_attr):
    if False:
        while True:
            i = 10
    'Perform regex match in rich text lines.\n\n  Produces a new RichTextLines object with font_attr_segs containing highlighted\n  regex matches.\n\n  Example use cases include:\n  1) search for specific items in a large list of items, and\n  2) search for specific numerical values in a large tensor.\n\n  Args:\n    orig_screen_output: The original RichTextLines, in which the regex find\n      is to be performed.\n    regex: The regex used for matching.\n    font_attr: Font attribute used for highlighting the found result.\n\n  Returns:\n    A modified copy of orig_screen_output.\n\n  Raises:\n    ValueError: If input str regex is not a valid regular expression.\n  '
    new_screen_output = RichTextLines(orig_screen_output.lines, font_attr_segs=copy.deepcopy(orig_screen_output.font_attr_segs), annotations=orig_screen_output.annotations)
    try:
        re_prog = re.compile(regex)
    except re.error:
        raise ValueError('Invalid regular expression: "%s"' % regex)
    regex_match_lines = []
    for (i, line) in enumerate(new_screen_output.lines):
        find_it = re_prog.finditer(line)
        match_segs = []
        for match in find_it:
            match_segs.append((match.start(), match.end(), font_attr))
        if match_segs:
            if i not in new_screen_output.font_attr_segs:
                new_screen_output.font_attr_segs[i] = match_segs
            else:
                new_screen_output.font_attr_segs[i].extend(match_segs)
                new_screen_output.font_attr_segs[i] = sorted(new_screen_output.font_attr_segs[i], key=lambda x: x[0])
            regex_match_lines.append(i)
    new_screen_output.annotations[REGEX_MATCH_LINES_KEY] = regex_match_lines
    return new_screen_output

def wrap_rich_text_lines(inp, cols):
    if False:
        for i in range(10):
            print('nop')
    "Wrap RichTextLines according to maximum number of columns.\n\n  Produces a new RichTextLines object with the text lines, font_attr_segs and\n  annotations properly wrapped. This ought to be used sparingly, as in most\n  cases, command handlers producing RichTextLines outputs should know the\n  screen/panel width via the screen_info kwarg and should produce properly\n  length-limited lines in the output accordingly.\n\n  Args:\n    inp: Input RichTextLines object.\n    cols: Number of columns, as an int.\n\n  Returns:\n    1) A new instance of RichTextLines, with line lengths limited to cols.\n    2) A list of new (wrapped) line index. For example, if the original input\n      consists of three lines and only the second line is wrapped, and it's\n      wrapped into two lines, this return value will be: [0, 1, 3].\n  Raises:\n    ValueError: If inputs have invalid types.\n  "
    new_line_indices = []
    if not isinstance(inp, RichTextLines):
        raise ValueError('Invalid type of input screen_output')
    if not isinstance(cols, int):
        raise ValueError('Invalid type of input cols')
    out = RichTextLines([])
    row_counter = 0
    for (i, line) in enumerate(inp.lines):
        new_line_indices.append(out.num_lines())
        if i in inp.annotations:
            out.annotations[row_counter] = inp.annotations[i]
        if len(line) <= cols:
            out.lines.append(line)
            if i in inp.font_attr_segs:
                out.font_attr_segs[row_counter] = inp.font_attr_segs[i]
            row_counter += 1
        else:
            wlines = []
            osegs = []
            if i in inp.font_attr_segs:
                osegs = inp.font_attr_segs[i]
            idx = 0
            while idx < len(line):
                if idx + cols > len(line):
                    rlim = len(line)
                else:
                    rlim = idx + cols
                wlines.append(line[idx:rlim])
                for seg in osegs:
                    if seg[0] < rlim and seg[1] >= idx:
                        if seg[0] >= idx:
                            lb = seg[0] - idx
                        else:
                            lb = 0
                        if seg[1] < rlim:
                            rb = seg[1] - idx
                        else:
                            rb = rlim - idx
                        if rb > lb:
                            wseg = (lb, rb, seg[2])
                            if row_counter not in out.font_attr_segs:
                                out.font_attr_segs[row_counter] = [wseg]
                            else:
                                out.font_attr_segs[row_counter].append(wseg)
                idx += cols
                row_counter += 1
            out.lines.extend(wlines)
    for key in inp.annotations:
        if not isinstance(key, int):
            out.annotations[key] = inp.annotations[key]
    return (out, new_line_indices)

class CommandHandlerRegistry:
    """Registry of command handlers for CLI.

  Handler methods (callables) for user commands can be registered with this
  class, which then is able to dispatch commands to the correct handlers and
  retrieve the RichTextLines output.

  For example, suppose you have the following handler defined:
    def echo(argv, screen_info=None):
      return RichTextLines(["arguments = %s" % " ".join(argv),
                            "screen_info = " + repr(screen_info)])

  you can register the handler with the command prefix "echo" and alias "e":
    registry = CommandHandlerRegistry()
    registry.register_command_handler("echo", echo,
        "Echo arguments, along with screen info", prefix_aliases=["e"])

  then to invoke this command handler with some arguments and screen_info, do:
    registry.dispatch_command("echo", ["foo", "bar"], screen_info={"cols": 80})

  or with the prefix alias:
    registry.dispatch_command("e", ["foo", "bar"], screen_info={"cols": 80})

  The call will return a RichTextLines object which can be rendered by a CLI.
  """
    HELP_COMMAND = 'help'
    HELP_COMMAND_ALIASES = ['h']
    VERSION_COMMAND = 'version'
    VERSION_COMMAND_ALIASES = ['ver']

    def __init__(self):
        if False:
            while True:
                i = 10
        self._handlers = {}
        self._alias_to_prefix = {}
        self._prefix_to_aliases = {}
        self._prefix_to_help = {}
        self._help_intro = None
        self.register_command_handler(self.HELP_COMMAND, self._help_handler, 'Print this help message.', prefix_aliases=self.HELP_COMMAND_ALIASES)
        self.register_command_handler(self.VERSION_COMMAND, self._version_handler, 'Print the versions of TensorFlow and its key dependencies.', prefix_aliases=self.VERSION_COMMAND_ALIASES)

    def register_command_handler(self, prefix, handler, help_info, prefix_aliases=None):
        if False:
            print('Hello World!')
        'Register a callable as a command handler.\n\n    Args:\n      prefix: Command prefix, i.e., the first word in a command, e.g.,\n        "print" as in "print tensor_1".\n      handler: A callable of the following signature:\n          foo_handler(argv, screen_info=None),\n        where argv is the argument vector (excluding the command prefix) and\n          screen_info is a dictionary containing information about the screen,\n          such as number of columns, e.g., {"cols": 100}.\n        The callable should return:\n          1) a RichTextLines object representing the screen output.\n\n        The callable can also raise an exception of the type CommandLineExit,\n        which if caught by the command-line interface, will lead to its exit.\n        The exception can optionally carry an exit token of arbitrary type.\n      help_info: A help string.\n      prefix_aliases: Aliases for the command prefix, as a list of str. E.g.,\n        shorthands for the command prefix: ["p", "pr"]\n\n    Raises:\n      ValueError: If\n        1) the prefix is empty, or\n        2) handler is not callable, or\n        3) a handler is already registered for the prefix, or\n        4) elements in prefix_aliases clash with existing aliases.\n        5) help_info is not a str.\n    '
        if not prefix:
            raise ValueError('Empty command prefix')
        if prefix in self._handlers:
            raise ValueError('A handler is already registered for command prefix "%s"' % prefix)
        if not callable(handler):
            raise ValueError('handler is not callable')
        if not isinstance(help_info, str):
            raise ValueError('help_info is not a str')
        if prefix_aliases:
            for alias in prefix_aliases:
                if self._resolve_prefix(alias):
                    raise ValueError('The prefix alias "%s" clashes with existing prefixes or aliases.' % alias)
                self._alias_to_prefix[alias] = prefix
            self._prefix_to_aliases[prefix] = prefix_aliases
        self._handlers[prefix] = handler
        self._prefix_to_help[prefix] = help_info

    def dispatch_command(self, prefix, argv, screen_info=None):
        if False:
            while True:
                i = 10
        'Handles a command by dispatching it to a registered command handler.\n\n    Args:\n      prefix: Command prefix, as a str, e.g., "print".\n      argv: Command argument vector, excluding the command prefix, represented\n        as a list of str, e.g.,\n        ["tensor_1"]\n      screen_info: A dictionary containing screen info, e.g., {"cols": 100}.\n\n    Returns:\n      An instance of RichTextLines or None. If any exception is caught during\n      the invocation of the command handler, the RichTextLines will wrap the\n      error type and message.\n\n    Raises:\n      ValueError: If\n        1) prefix is empty, or\n        2) no command handler is registered for the command prefix, or\n        3) the handler is found for the prefix, but it fails to return a\n          RichTextLines or raise any exception.\n      CommandLineExit:\n        If the command handler raises this type of exception, this method will\n        simply pass it along.\n    '
        if not prefix:
            raise ValueError('Prefix is empty')
        resolved_prefix = self._resolve_prefix(prefix)
        if not resolved_prefix:
            raise ValueError('No handler is registered for command prefix "%s"' % prefix)
        handler = self._handlers[resolved_prefix]
        try:
            output = handler(argv, screen_info=screen_info)
        except CommandLineExit as e:
            raise e
        except SystemExit as e:
            lines = ['Syntax error for command: %s' % prefix, 'For help, do "help %s"' % prefix]
            output = RichTextLines(lines)
        except BaseException as e:
            lines = ['Error occurred during handling of command: %s %s:' % (resolved_prefix, ' '.join(argv)), '%s: %s' % (type(e), str(e))]
            lines.append('')
            lines.extend(traceback.format_exc().split('\n'))
            output = RichTextLines(lines)
        if not isinstance(output, RichTextLines) and output is not None:
            raise ValueError('Return value from command handler %s is not None or a RichTextLines instance' % str(handler))
        return output

    def is_registered(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        'Test if a command prefix or its alias is has a registered handler.\n\n    Args:\n      prefix: A prefix or its alias, as a str.\n\n    Returns:\n      True iff a handler is registered for prefix.\n    '
        return self._resolve_prefix(prefix) is not None

    def get_help(self, cmd_prefix=None):
        if False:
            for i in range(10):
                print('nop')
        'Compile help information into a RichTextLines object.\n\n    Args:\n      cmd_prefix: Optional command prefix. As the prefix itself or one of its\n        aliases.\n\n    Returns:\n      A RichTextLines object containing the help information. If cmd_prefix\n      is None, the return value will be the full command-line help. Otherwise,\n      it will be the help information for the specified command.\n    '
        if not cmd_prefix:
            help_info = RichTextLines([])
            if self._help_intro:
                help_info.extend(self._help_intro)
            sorted_prefixes = sorted(self._handlers)
            for cmd_prefix in sorted_prefixes:
                lines = self._get_help_for_command_prefix(cmd_prefix)
                lines.append('')
                lines.append('')
                help_info.extend(RichTextLines(lines))
            return help_info
        else:
            return RichTextLines(self._get_help_for_command_prefix(cmd_prefix))

    def set_help_intro(self, help_intro):
        if False:
            i = 10
            return i + 15
        'Set an introductory message to help output.\n\n    Args:\n      help_intro: (RichTextLines) Rich text lines appended to the\n        beginning of the output of the command "help", as introductory\n        information.\n    '
        self._help_intro = help_intro

    def _help_handler(self, args, screen_info=None):
        if False:
            print('Hello World!')
        'Command handler for "help".\n\n    "help" is a common command that merits built-in support from this class.\n\n    Args:\n      args: Command line arguments to "help" (not including "help" itself).\n      screen_info: (dict) Information regarding the screen, e.g., the screen\n        width in characters: {"cols": 80}\n\n    Returns:\n      (RichTextLines) Screen text output.\n    '
        _ = screen_info
        if not args:
            return self.get_help()
        elif len(args) == 1:
            return self.get_help(args[0])
        else:
            return RichTextLines(['ERROR: help takes only 0 or 1 input argument.'])

    def _version_handler(self, args, screen_info=None):
        if False:
            for i in range(10):
                print('nop')
        del args
        del screen_info
        return get_tensorflow_version_lines(include_dependency_versions=True)

    def _resolve_prefix(self, token):
        if False:
            for i in range(10):
                print('nop')
        'Resolve command prefix from the prefix itself or its alias.\n\n    Args:\n      token: a str to be resolved.\n\n    Returns:\n      If resolvable, the resolved command prefix.\n      If not resolvable, None.\n    '
        if token in self._handlers:
            return token
        elif token in self._alias_to_prefix:
            return self._alias_to_prefix[token]
        else:
            return None

    def _get_help_for_command_prefix(self, cmd_prefix):
        if False:
            while True:
                i = 10
        'Compile the help information for a given command prefix.\n\n    Args:\n      cmd_prefix: Command prefix, as the prefix itself or one of its aliases.\n\n    Returns:\n      A list of str as the help information for cmd_prefix. If the cmd_prefix\n        does not exist, the returned list of str will indicate that.\n    '
        lines = []
        resolved_prefix = self._resolve_prefix(cmd_prefix)
        if not resolved_prefix:
            lines.append('Invalid command prefix: "%s"' % cmd_prefix)
            return lines
        lines.append(resolved_prefix)
        if resolved_prefix in self._prefix_to_aliases:
            lines.append(HELP_INDENT + 'Aliases: ' + ', '.join(self._prefix_to_aliases[resolved_prefix]))
        lines.append('')
        help_lines = self._prefix_to_help[resolved_prefix].split('\n')
        for line in help_lines:
            lines.append(HELP_INDENT + line)
        return lines

class TabCompletionRegistry:
    """Registry for tab completion responses."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._comp_dict = {}

    def register_tab_comp_context(self, context_words, comp_items):
        if False:
            while True:
                i = 10
        'Register a tab-completion context.\n\n    Register that, for each word in context_words, the potential tab-completions\n    are the words in comp_items.\n\n    A context word is a pre-existing, completed word in the command line that\n    determines how tab-completion works for another, incomplete word in the same\n    command line.\n    Completion items consist of potential candidates for the incomplete word.\n\n    To give a general example, a context word can be "drink", and the completion\n    items can be ["coffee", "tea", "water"]\n\n    Note: A context word can be empty, in which case the context is for the\n     top-level commands.\n\n    Args:\n      context_words: A list of context words belonging to the context being\n        registered. It is a list of str, instead of a single string, to support\n        synonym words triggering the same tab-completion context, e.g.,\n        both "drink" and the short-hand "dr" can trigger the same context.\n      comp_items: A list of completion items, as a list of str.\n\n    Raises:\n      TypeError: if the input arguments are not all of the correct types.\n    '
        if not isinstance(context_words, list):
            raise TypeError('Incorrect type in context_list: Expected list, got %s' % type(context_words))
        if not isinstance(comp_items, list):
            raise TypeError('Incorrect type in comp_items: Expected list, got %s' % type(comp_items))
        sorted_comp_items = sorted(comp_items)
        for context_word in context_words:
            self._comp_dict[context_word] = sorted_comp_items

    def deregister_context(self, context_words):
        if False:
            for i in range(10):
                print('nop')
        'Deregister a list of context words.\n\n    Args:\n      context_words: A list of context words to deregister, as a list of str.\n\n    Raises:\n      KeyError: if there are word(s) in context_words that do not correspond\n        to any registered contexts.\n    '
        for context_word in context_words:
            if context_word not in self._comp_dict:
                raise KeyError('Cannot deregister unregistered context word "%s"' % context_word)
        for context_word in context_words:
            del self._comp_dict[context_word]

    def extend_comp_items(self, context_word, new_comp_items):
        if False:
            while True:
                i = 10
        'Add a list of completion items to a completion context.\n\n    Args:\n      context_word: A single completion word as a string. The extension will\n        also apply to all other context words of the same context.\n      new_comp_items: (list of str) New completion items to add.\n\n    Raises:\n      KeyError: if the context word has not been registered.\n    '
        if context_word not in self._comp_dict:
            raise KeyError('Context word "%s" has not been registered' % context_word)
        self._comp_dict[context_word].extend(new_comp_items)
        self._comp_dict[context_word] = sorted(self._comp_dict[context_word])

    def remove_comp_items(self, context_word, comp_items):
        if False:
            while True:
                i = 10
        'Remove a list of completion items from a completion context.\n\n    Args:\n      context_word: A single completion word as a string. The removal will\n        also apply to all other context words of the same context.\n      comp_items: Completion items to remove.\n\n    Raises:\n      KeyError: if the context word has not been registered.\n    '
        if context_word not in self._comp_dict:
            raise KeyError('Context word "%s" has not been registered' % context_word)
        for item in comp_items:
            self._comp_dict[context_word].remove(item)

    def get_completions(self, context_word, prefix):
        if False:
            i = 10
            return i + 15
        'Get the tab completions given a context word and a prefix.\n\n    Args:\n      context_word: The context word.\n      prefix: The prefix of the incomplete word.\n\n    Returns:\n      (1) None if no registered context matches the context_word.\n          A list of str for the matching completion items. Can be an empty list\n          of a matching context exists, but no completion item matches the\n          prefix.\n      (2) Common prefix of all the words in the first return value. If the\n          first return value is None, this return value will be None, too. If\n          the first return value is not None, i.e., a list, this return value\n          will be a str, which can be an empty str if there is no common\n          prefix among the items of the list.\n    '
        if context_word not in self._comp_dict:
            return (None, None)
        comp_items = self._comp_dict[context_word]
        comp_items = sorted([item for item in comp_items if item.startswith(prefix)])
        return (comp_items, self._common_prefix(comp_items))

    def _common_prefix(self, m):
        if False:
            while True:
                i = 10
        'Given a list of str, returns the longest common prefix.\n\n    Args:\n      m: (list of str) A list of strings.\n\n    Returns:\n      (str) The longest common prefix.\n    '
        if not m:
            return ''
        s1 = min(m)
        s2 = max(m)
        for (i, c) in enumerate(s1):
            if c != s2[i]:
                return s1[:i]
        return s1

class CommandHistory:
    """Keeps command history and supports lookup."""
    _HISTORY_FILE_NAME = '.tfdbg_history'

    def __init__(self, limit=100, history_file_path=None):
        if False:
            for i in range(10):
                print('nop')
        'CommandHistory constructor.\n\n    Args:\n      limit: Maximum number of the most recent commands that this instance\n        keeps track of, as an int.\n      history_file_path: (str) Manually specified path to history file. Used in\n        testing.\n    '
        self._commands = []
        self._limit = limit
        self._history_file_path = history_file_path or self._get_default_history_file_path()
        self._load_history_from_file()

    def _load_history_from_file(self):
        if False:
            return 10
        if os.path.isfile(self._history_file_path):
            try:
                with open(self._history_file_path, 'rt') as history_file:
                    commands = history_file.readlines()
                self._commands = [command.strip() for command in commands if command.strip()]
                if len(self._commands) > self._limit:
                    self._commands = self._commands[-self._limit:]
                    with open(self._history_file_path, 'wt') as history_file:
                        for command in self._commands:
                            history_file.write(command + '\n')
            except IOError:
                print('WARNING: writing history file failed.')

    def _add_command_to_history_file(self, command):
        if False:
            while True:
                i = 10
        try:
            with open(self._history_file_path, 'at') as history_file:
                history_file.write(command + '\n')
        except IOError:
            pass

    @classmethod
    def _get_default_history_file_path(cls):
        if False:
            return 10
        return os.path.join(os.path.expanduser('~'), cls._HISTORY_FILE_NAME)

    def add_command(self, command):
        if False:
            i = 10
            return i + 15
        'Add a command to the command history.\n\n    Args:\n      command: The history command, as a str.\n\n    Raises:\n      TypeError: if command is not a str.\n    '
        if self._commands and command == self._commands[-1]:
            return
        if not isinstance(command, str):
            raise TypeError('Attempt to enter non-str entry to command history')
        self._commands.append(command)
        if len(self._commands) > self._limit:
            self._commands = self._commands[-self._limit:]
        self._add_command_to_history_file(command)

    def most_recent_n(self, n):
        if False:
            for i in range(10):
                print('nop')
        'Look up the n most recent commands.\n\n    Args:\n      n: Number of most recent commands to look up.\n\n    Returns:\n      A list of n most recent commands, or all available most recent commands,\n      if n exceeds size of the command history, in chronological order.\n    '
        return self._commands[-n:]

    def lookup_prefix(self, prefix, n):
        if False:
            return 10
        'Look up the n most recent commands that starts with prefix.\n\n    Args:\n      prefix: The prefix to lookup.\n      n: Number of most recent commands to look up.\n\n    Returns:\n      A list of n most recent commands that have the specified prefix, or all\n      available most recent commands that have the prefix, if n exceeds the\n      number of history commands with the prefix.\n    '
        commands = [cmd for cmd in self._commands if cmd.startswith(prefix)]
        return commands[-n:]

class MenuItem:
    """A class for an item in a text-based menu."""

    def __init__(self, caption, content, enabled=True):
        if False:
            return 10
        'Menu constructor.\n\n    TODO(cais): Nested menu is currently not supported. Support it.\n\n    Args:\n      caption: (str) caption of the menu item.\n      content: Content of the menu item. For a menu item that triggers\n        a command, for example, content is the command string.\n      enabled: (bool) whether this menu item is enabled.\n    '
        self._caption = caption
        self._content = content
        self._enabled = enabled

    @property
    def caption(self):
        if False:
            while True:
                i = 10
        return self._caption

    @property
    def type(self):
        if False:
            print('Hello World!')
        return self._node_type

    @property
    def content(self):
        if False:
            while True:
                i = 10
        return self._content

    def is_enabled(self):
        if False:
            print('Hello World!')
        return self._enabled

    def disable(self):
        if False:
            while True:
                i = 10
        self._enabled = False

    def enable(self):
        if False:
            return 10
        self._enabled = True

class Menu:
    """A class for text-based menu."""

    def __init__(self, name=None):
        if False:
            while True:
                i = 10
        'Menu constructor.\n\n    Args:\n      name: (str or None) name of this menu.\n    '
        self._name = name
        self._items = []

    def append(self, item):
        if False:
            return 10
        'Append an item to the Menu.\n\n    Args:\n      item: (MenuItem) the item to be appended.\n    '
        self._items.append(item)

    def insert(self, index, item):
        if False:
            while True:
                i = 10
        self._items.insert(index, item)

    def num_items(self):
        if False:
            print('Hello World!')
        return len(self._items)

    def captions(self):
        if False:
            while True:
                i = 10
        return [item.caption for item in self._items]

    def caption_to_item(self, caption):
        if False:
            i = 10
            return i + 15
        'Get a MenuItem from the caption.\n\n    Args:\n      caption: (str) The caption to look up.\n\n    Returns:\n      (MenuItem) The first-match menu item with the caption, if any.\n\n    Raises:\n      LookupError: If a menu item with the caption does not exist.\n    '
        captions = self.captions()
        if caption not in captions:
            raise LookupError('There is no menu item with the caption "%s"' % caption)
        return self._items[captions.index(caption)]

    def format_as_single_line(self, prefix=None, divider=' | ', enabled_item_attrs=None, disabled_item_attrs=None):
        if False:
            print('Hello World!')
        'Format the menu as a single-line RichTextLines object.\n\n    Args:\n      prefix: (str) String added to the beginning of the line.\n      divider: (str) The dividing string between the menu items.\n      enabled_item_attrs: (list or str) Attributes applied to each enabled\n        menu item, e.g., ["bold", "underline"].\n      disabled_item_attrs: (list or str) Attributes applied to each\n        disabled menu item, e.g., ["red"].\n\n    Returns:\n      (RichTextLines) A single-line output representing the menu, with\n        font_attr_segs marking the individual menu items.\n    '
        if enabled_item_attrs is not None and (not isinstance(enabled_item_attrs, list)):
            enabled_item_attrs = [enabled_item_attrs]
        if disabled_item_attrs is not None and (not isinstance(disabled_item_attrs, list)):
            disabled_item_attrs = [disabled_item_attrs]
        menu_line = prefix if prefix is not None else ''
        attr_segs = []
        for item in self._items:
            menu_line += item.caption
            item_name_begin = len(menu_line) - len(item.caption)
            if item.is_enabled():
                final_attrs = [item]
                if enabled_item_attrs:
                    final_attrs.extend(enabled_item_attrs)
                attr_segs.append((item_name_begin, len(menu_line), final_attrs))
            elif disabled_item_attrs:
                attr_segs.append((item_name_begin, len(menu_line), disabled_item_attrs))
            menu_line += divider
        return RichTextLines(menu_line, font_attr_segs={0: attr_segs})