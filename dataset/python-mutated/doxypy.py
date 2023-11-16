__applicationName__ = 'doxypy'
__blurb__ = '\ndoxypy is an input filter for Doxygen. It preprocesses python\nfiles so that docstrings of classes and functions are reformatted\ninto Doxygen-conform documentation blocks.\n'
__doc__ = __blurb__ + '\nIn order to make Doxygen preprocess files through doxypy, simply\nadd the following lines to your Doxyfile:\n    FILTER_SOURCE_FILES = YES\n    INPUT_FILTER = "python /path/to/doxypy.py"\n'
__version__ = '0.4.2'
__date__ = '5th December 2008'
__website__ = 'http://code.foosel.org/doxypy'
__author__ = ("Philippe 'demod' Neumann (doxypy at demod dot org)", "Gina 'foosel' Haeussge (gina at foosel dot net)")
__licenseName__ = 'GPL v2'
__license__ = 'This program is free software: you can redistribute it and/or modify\nit under the terms of the GNU General Public License as published by\nthe Free Software Foundation, either version 2 of the License, or\n(at your option) any later version.\n\nThis program is distributed in the hope that it will be useful,\nbut WITHOUT ANY WARRANTY; without even the implied warranty of\nMERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\nGNU General Public License for more details.\n\nYou should have received a copy of the GNU General Public License\nalong with this program.  If not, see <http://www.gnu.org/licenses/>.\n'
import sys
import re
from argparse import ArgumentParser

class FSM(object):
    """Implements a finite state machine.

    Transitions are given as 4-tuples, consisting of an origin state, a target
    state, a condition for the transition (given as a reference to a function
    which gets called with a given piece of input) and a pointer to a function
    to be called upon the execution of the given transition.
    """
    '\n    @var transitions holds the transitions\n    @var current_state holds the current state\n    @var current_input holds the current input\n    @var current_transition hold the currently active transition\n    '

    def __init__(self, start_state=None, transitions=[]):
        if False:
            i = 10
            return i + 15
        self.transitions = transitions
        self.current_state = start_state
        self.current_input = None
        self.current_transition = None

    def setStartState(self, state):
        if False:
            while True:
                i = 10
        self.current_state = state

    def addTransition(self, from_state, to_state, condition, callback):
        if False:
            i = 10
            return i + 15
        self.transitions.append([from_state, to_state, condition, callback])

    def makeTransition(self, input):
        if False:
            i = 10
            return i + 15
        ' Makes a transition based on the given input.\n\n        @param    input    input to parse by the FSM\n        '
        for transition in self.transitions:
            [from_state, to_state, condition, callback] = transition
            if from_state == self.current_state:
                match = condition(input)
                if match:
                    self.current_state = to_state
                    self.current_input = input
                    self.current_transition = transition
                    if args.debug:
                        print("# FSM: executing (%s -> %s) for line '%s'" % (from_state, to_state, input), file=sys.stderr)
                    callback(match)
                    return

class Doxypy(object):

    def __init__(self):
        if False:
            return 10
        string_prefixes = '[uU]?[rR]?'
        self.start_single_comment_re = re.compile("^\\s*%s(''')" % string_prefixes)
        self.end_single_comment_re = re.compile("(''')\\s*$")
        self.start_double_comment_re = re.compile('^\\s*%s(""")' % string_prefixes)
        self.end_double_comment_re = re.compile('(""")\\s*$')
        self.single_comment_re = re.compile("^\\s*%s(''').*(''')\\s*$" % string_prefixes)
        self.double_comment_re = re.compile('^\\s*%s(""").*(""")\\s*$' % string_prefixes)
        self.defclass_re = re.compile('^(\\s*)(def .+:|class .+:)')
        self.empty_re = re.compile('^\\s*$')
        self.hashline_re = re.compile('^\\s*#.*$')
        self.importline_re = re.compile('^\\s*(import |from .+ import)')
        self.multiline_defclass_start_re = re.compile('^(\\s*)(def|class)(\\s.*)?$')
        self.multiline_defclass_end_re = re.compile(':\\s*$')
        transitions = [['FILEHEAD', 'FILEHEAD', self.single_comment_re.search, self.appendCommentLine], ['FILEHEAD', 'FILEHEAD', self.double_comment_re.search, self.appendCommentLine], ['FILEHEAD', 'FILEHEAD_COMMENT_SINGLE', self.start_single_comment_re.search, self.appendCommentLine], ['FILEHEAD_COMMENT_SINGLE', 'FILEHEAD', self.end_single_comment_re.search, self.appendCommentLine], ['FILEHEAD_COMMENT_SINGLE', 'FILEHEAD_COMMENT_SINGLE', self.catchall, self.appendCommentLine], ['FILEHEAD', 'FILEHEAD_COMMENT_DOUBLE', self.start_double_comment_re.search, self.appendCommentLine], ['FILEHEAD_COMMENT_DOUBLE', 'FILEHEAD', self.end_double_comment_re.search, self.appendCommentLine], ['FILEHEAD_COMMENT_DOUBLE', 'FILEHEAD_COMMENT_DOUBLE', self.catchall, self.appendCommentLine], ['FILEHEAD', 'FILEHEAD', self.empty_re.search, self.appendFileheadLine], ['FILEHEAD', 'FILEHEAD', self.hashline_re.search, self.appendFileheadLine], ['FILEHEAD', 'FILEHEAD', self.importline_re.search, self.appendFileheadLine], ['FILEHEAD', 'DEFCLASS', self.defclass_re.search, self.resetCommentSearch], ['FILEHEAD', 'DEFCLASS_MULTI', self.multiline_defclass_start_re.search, self.resetCommentSearch], ['FILEHEAD', 'DEFCLASS_BODY', self.catchall, self.appendFileheadLine], ['DEFCLASS', 'DEFCLASS_BODY', self.single_comment_re.search, self.appendCommentLine], ['DEFCLASS', 'DEFCLASS_BODY', self.double_comment_re.search, self.appendCommentLine], ['DEFCLASS', 'COMMENT_SINGLE', self.start_single_comment_re.search, self.appendCommentLine], ['COMMENT_SINGLE', 'DEFCLASS_BODY', self.end_single_comment_re.search, self.appendCommentLine], ['COMMENT_SINGLE', 'COMMENT_SINGLE', self.catchall, self.appendCommentLine], ['DEFCLASS', 'COMMENT_DOUBLE', self.start_double_comment_re.search, self.appendCommentLine], ['COMMENT_DOUBLE', 'DEFCLASS_BODY', self.end_double_comment_re.search, self.appendCommentLine], ['COMMENT_DOUBLE', 'COMMENT_DOUBLE', self.catchall, self.appendCommentLine], ['DEFCLASS', 'DEFCLASS', self.empty_re.search, self.appendDefclassLine], ['DEFCLASS', 'DEFCLASS', self.defclass_re.search, self.resetCommentSearch], ['DEFCLASS', 'DEFCLASS_MULTI', self.multiline_defclass_start_re.search, self.resetCommentSearch], ['DEFCLASS', 'DEFCLASS_BODY', self.catchall, self.stopCommentSearch], ['DEFCLASS_BODY', 'DEFCLASS', self.defclass_re.search, self.startCommentSearch], ['DEFCLASS_BODY', 'DEFCLASS_MULTI', self.multiline_defclass_start_re.search, self.startCommentSearch], ['DEFCLASS_BODY', 'DEFCLASS_BODY', self.catchall, self.appendNormalLine], ['DEFCLASS_MULTI', 'DEFCLASS', self.multiline_defclass_end_re.search, self.appendDefclassLine], ['DEFCLASS_MULTI', 'DEFCLASS_MULTI', self.catchall, self.appendDefclassLine]]
        self.fsm = FSM('FILEHEAD', transitions)
        self.outstream = sys.stdout
        self.output = []
        self.comment = []
        self.filehead = []
        self.defclass = []
        self.indent = ''

    def __closeComment(self):
        if False:
            return 10
        'Appends any open comment block and triggering block to the output.'
        if args.autobrief:
            if len(self.comment) == 1 or (len(self.comment) > 2 and self.comment[1].strip() == ''):
                self.comment[0] = self.__docstringSummaryToBrief(self.comment[0])
        if self.comment:
            block = self.makeCommentBlock()
            self.output.extend(block)
        if self.defclass:
            self.output.extend(self.defclass)

    def __docstringSummaryToBrief(self, line):
        if False:
            while True:
                i = 10
        'Adds \\brief to the docstrings summary line.\n\n        A \\brief is prepended, provided no other doxygen command is at the\n        start of the line.\n        '
        stripped = line.strip()
        if stripped and (not stripped[0] in ('@', '\\')):
            return '\\brief ' + line
        else:
            return line

    def __flushBuffer(self):
        if False:
            for i in range(10):
                print('nop')
        'Flushes the current outputbuffer to the outstream.'
        if self.output:
            try:
                if args.debug:
                    print('# OUTPUT: ', self.output, file=sys.stderr)
                print('\n'.join(self.output), file=self.outstream)
                self.outstream.flush()
            except IOError:
                pass
        self.output = []

    def catchall(self, input):
        if False:
            print('Hello World!')
        'The catchall-condition, always returns true.'
        return True

    def resetCommentSearch(self, match):
        if False:
            i = 10
            return i + 15
        'Restarts a new comment search for a different triggering line.\n\n        Closes the current commentblock and starts a new comment search.\n        '
        if args.debug:
            print('# CALLBACK: resetCommentSearch', file=sys.stderr)
        self.__closeComment()
        self.startCommentSearch(match)

    def startCommentSearch(self, match):
        if False:
            while True:
                i = 10
        'Starts a new comment search.\n\n        Saves the triggering line, resets the current comment and saves\n        the current indentation.\n        '
        if args.debug:
            print('# CALLBACK: startCommentSearch', file=sys.stderr)
        self.defclass = [self.fsm.current_input]
        self.comment = []
        self.indent = match.group(1)

    def stopCommentSearch(self, match):
        if False:
            print('Hello World!')
        'Stops a comment search.\n\n        Closes the current commentblock, resets    the triggering line and\n        appends the current line to the output.\n        '
        if args.debug:
            print('# CALLBACK: stopCommentSearch', file=sys.stderr)
        self.__closeComment()
        self.defclass = []
        self.output.append(self.fsm.current_input)

    def appendFileheadLine(self, match):
        if False:
            while True:
                i = 10
        'Appends a line in the FILEHEAD state.\n\n        Closes the open comment    block, resets it and appends the current line.\n        '
        if args.debug:
            print('# CALLBACK: appendFileheadLine', file=sys.stderr)
        self.__closeComment()
        self.comment = []
        self.output.append(self.fsm.current_input)

    def appendCommentLine(self, match):
        if False:
            print('Hello World!')
        'Appends a comment line.\n\n        The comment delimiter is removed from multiline start and ends as\n        well as singleline comments.\n        '
        if args.debug:
            print('# CALLBACK: appendCommentLine', file=sys.stderr)
        (from_state, to_state, condition, callback) = self.fsm.current_transition
        if from_state == 'DEFCLASS' and to_state == 'DEFCLASS_BODY' or (from_state == 'FILEHEAD' and to_state == 'FILEHEAD'):
            activeCommentDelim = match.group(1)
            line = self.fsm.current_input
            self.comment.append(line[line.find(activeCommentDelim) + len(activeCommentDelim):line.rfind(activeCommentDelim)])
            if to_state == 'DEFCLASS_BODY':
                self.__closeComment()
                self.defclass = []
        elif from_state == 'DEFCLASS' or from_state == 'FILEHEAD':
            activeCommentDelim = match.group(1)
            line = self.fsm.current_input
            self.comment.append(line[line.find(activeCommentDelim) + len(activeCommentDelim):])
        elif to_state == 'DEFCLASS_BODY' or to_state == 'FILEHEAD':
            activeCommentDelim = match.group(1)
            line = self.fsm.current_input
            self.comment.append(line[0:line.rfind(activeCommentDelim)])
            if to_state == 'DEFCLASS_BODY':
                self.__closeComment()
                self.defclass = []
        else:
            self.comment.append(self.fsm.current_input)

    def appendNormalLine(self, match):
        if False:
            return 10
        'Appends a line to the output.'
        if args.debug:
            print('# CALLBACK: appendNormalLine', file=sys.stderr)
        self.output.append(self.fsm.current_input)

    def appendDefclassLine(self, match):
        if False:
            i = 10
            return i + 15
        'Appends a line to the triggering block.'
        if args.debug:
            print('# CALLBACK: appendDefclassLine', file=sys.stderr)
        self.defclass.append(self.fsm.current_input)

    def makeCommentBlock(self):
        if False:
            print('Hello World!')
        'Indents the current comment block with respect to the current\n        indentation level.\n\n        @returns a list of indented comment lines\n        '
        doxyStart = '##'
        commentLines = self.comment
        commentLines = ['%s# %s' % (self.indent, x) for x in commentLines]
        l = [self.indent + doxyStart]
        l.extend(commentLines)
        return l

    def parse(self, input):
        if False:
            print('Hello World!')
        'Parses a python file given as input string and returns the doxygen-\n        compatible representation.\n\n        @param    input    the python code to parse\n        @returns the modified python code\n        '
        lines = input.split('\n')
        for line in lines:
            self.fsm.makeTransition(line)
        if self.fsm.current_state == 'DEFCLASS':
            self.__closeComment()
        return '\n'.join(self.output)

    def parseFile(self, filename):
        if False:
            return 10
        'Parses a python file given as input string and returns the doxygen-\n        compatible representation.\n\n        @param    input    the python code to parse\n        @returns the modified python code\n        '
        f = open(filename, 'r')
        for line in f:
            self.parseLine(line.rstrip('\r\n'))
        if self.fsm.current_state == 'DEFCLASS':
            self.__closeComment()
            self.__flushBuffer()
        f.close()

    def parseLine(self, line):
        if False:
            print('Hello World!')
        'Parse one line of python and flush the resulting output to the\n        outstream.\n\n        @param    line    the python code line to parse\n        '
        self.fsm.makeTransition(line)
        self.__flushBuffer()

def argParse():
    if False:
        while True:
            i = 10
    'Parses commandline args.'
    parser = ArgumentParser(prog=__applicationName__)
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument('--autobrief', action='store_true', help='use the docstring summary line as \\brief description')
    parser.add_argument('--debug', action='store_true', help='enable debug output on stderr')
    parser.add_argument('filename', metavar='FILENAME')
    return parser.parse_args()

def main():
    if False:
        while True:
            i = 10
    'Starts the parser on the file given by the filename as the first\n    argument on the commandline.\n    '
    global args
    args = argParse()
    fsm = Doxypy()
    fsm.parseFile(args.filename)
if __name__ == '__main__':
    main()