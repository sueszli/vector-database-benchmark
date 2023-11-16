"""
Simple utility for splitting user input. This is used by both inputsplitter and
prefilter.

Authors:

* Brian Granger
* Fernando Perez
"""
import re
import sys
from IPython.utils import py3compat
from IPython.utils.encoding import get_stream_enc
from IPython.core.oinspect import OInfo
line_split = re.compile("\n             ^(\\s*)               # any leading space\n             ([,;/%]|!!?|\\?\\??)?  # escape character or characters\n             \\s*(%{0,2}[\\w\\.\\*]*)     # function/method, possibly with leading %\n                                  # to correctly treat things like '?%magic'\n             (.*?$|$)             # rest of line\n             ", re.VERBOSE)

def split_user_input(line, pattern=None):
    if False:
        print('Hello World!')
    'Split user input into initial whitespace, escape character, function part\n    and the rest.\n    '
    encoding = get_stream_enc(sys.stdin, 'utf-8')
    line = py3compat.cast_unicode(line, encoding)
    if pattern is None:
        pattern = line_split
    match = pattern.match(line)
    if not match:
        try:
            (ifun, the_rest) = line.split(None, 1)
        except ValueError:
            (ifun, the_rest) = (line, u'')
        pre = re.match('^(\\s*)(.*)', line).groups()[0]
        esc = ''
    else:
        (pre, esc, ifun, the_rest) = match.groups()
    return (pre, esc or '', ifun.strip(), the_rest.lstrip())

class LineInfo(object):
    """A single line of input and associated info.

    Includes the following as properties:

    line
      The original, raw line

    continue_prompt
      Is this line a continuation in a sequence of multiline input?

    pre
      Any leading whitespace.

    esc
      The escape character(s) in pre or the empty string if there isn't one.
      Note that '!!' and '??' are possible values for esc. Otherwise it will
      always be a single character.

    ifun
      The 'function part', which is basically the maximal initial sequence
      of valid python identifiers and the '.' character. This is what is
      checked for alias and magic transformations, used for auto-calling,
      etc. In contrast to Python identifiers, it may start with "%" and contain
      "*".

    the_rest
      Everything else on the line.
    """

    def __init__(self, line, continue_prompt=False):
        if False:
            while True:
                i = 10
        self.line = line
        self.continue_prompt = continue_prompt
        (self.pre, self.esc, self.ifun, self.the_rest) = split_user_input(line)
        self.pre_char = self.pre.strip()
        if self.pre_char:
            self.pre_whitespace = ''
        else:
            self.pre_whitespace = self.pre

    def ofind(self, ip) -> OInfo:
        if False:
            while True:
                i = 10
        "Do a full, attribute-walking lookup of the ifun in the various\n        namespaces for the given IPython InteractiveShell instance.\n\n        Return a dict with keys: {found, obj, ospace, ismagic}\n\n        Note: can cause state changes because of calling getattr, but should\n        only be run if autocall is on and if the line hasn't matched any\n        other, less dangerous handlers.\n\n        Does cache the results of the call, so can be called multiple times\n        without worrying about *further* damaging state.\n        "
        return ip._ofind(self.ifun)

    def __str__(self):
        if False:
            while True:
                i = 10
        return 'LineInfo [%s|%s|%s|%s]' % (self.pre, self.esc, self.ifun, self.the_rest)