import sys, os
from calibre.ebooks.rtf2xml import copy, check_brackets
from calibre.ptempfile import better_mktemp
from polyglot.builtins import iteritems
from . import open_for_read, open_for_write

class AddBrackets:
    """
    Add brackets for old RTF.
    Logic:
    When control words without their own brackets are encountered
    and in the list of allowed words, this will add brackets
    to facilitate the treatment of the file
    """

    def __init__(self, in_file, bug_handler, copy=None, run_level=1):
        if False:
            i = 10
            return i + 15
        "\n        Required:\n            'file'--file to parse\n        Optional:\n            'copy'-- whether to make a copy of result for debugging\n            'temp_dir' --where to output temporary results (default is\n            directory from which the script is run.)\n        Returns:\n            nothing\n            "
        self.__file = in_file
        self.__bug_handler = bug_handler
        self.__copy = copy
        self.__write_to = better_mktemp()
        self.__run_level = run_level
        self.__state_dict = {'before_body': self.__before_body_func, 'in_body': self.__in_body_func, 'after_control_word': self.__after_control_word_func, 'in_ignore': self.__ignore_func}
        self.__accept = ['cw<ci<bold______', 'cw<ci<annotation', 'cw<ci<blue______', 'cw<ci<caps______', 'cw<ci<char-style', 'cw<ci<dbl-strike', 'cw<ci<emboss____', 'cw<ci<engrave___', 'cw<ci<font-color', 'cw<ci<font-down_', 'cw<ci<font-size_', 'cw<ci<font-style', 'cw<ci<font-up___', 'cw<ci<footnot-mk', 'cw<ci<green_____', 'cw<ci<hidden____', 'cw<ci<italics___', 'cw<ci<outline___', 'cw<ci<red_______', 'cw<ci<shadow____', 'cw<ci<small-caps', 'cw<ci<strike-thr', 'cw<ci<subscript_', 'cw<ci<superscrip', 'cw<ci<underlined']

    def __initiate_values(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Init temp values\n        '
        self.__state = 'before_body'
        self.__inline = {}
        self.__temp_group = []
        self.__open_bracket = False
        self.__found_brackets = False

    def __before_body_func(self, line):
        if False:
            return 10
        '\n        If we are before the body, not interest in changing anything\n        '
        if self.__token_info == 'mi<mk<body-open_':
            self.__state = 'in_body'
        self.__write_obj.write(line)

    def __in_body_func(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Select what action to take in body:\n            1-At the end of the file close the braket if a bracket was opened\n            This happens if there is achange\n            2-If an open bracket is found the code inside is ignore\n            (written without modifications)\n            3-If an accepted control word is found put the line\n            in a buffer then change state to after cw\n            4-Else simply write the line\n        '
        if line == 'cb<nu<clos-brack<0001\n' and self.__open_bracket:
            self.__write_obj.write('cb<nu<clos-brack<0003\n')
            self.__write_obj.write(line)
        elif self.__token_info == 'ob<nu<open-brack':
            self.__found_brackets = True
            self.__state = 'in_ignore'
            self.__ignore_count = self.__ob_count
            self.__write_obj.write(line)
        elif self.__token_info in self.__accept:
            self.__temp_group.append(line)
            self.__state = 'after_control_word'
        else:
            self.__write_obj.write(line)

    def __after_control_word_func(self, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        After a cw either add next allowed cw to temporary list or\n        change group and write it.\n        If the token leading to an exit is an open bracket go to\n        ignore otherwise goto in body\n        '
        if self.__token_info in self.__accept:
            self.__temp_group.append(line)
        else:
            self.__change_permanent_group()
            self.__write_group()
            self.__write_obj.write(line)
            if self.__token_info == 'ob<nu<open-brack':
                self.__state = 'in_ignore'
                self.__ignore_count = self.__ob_count
            else:
                self.__state = 'in_body'

    def __write_group(self):
        if False:
            while True:
                i = 10
        '\n        Write a temporary group after accepted control words end\n        But this is mostly useless in my opinion as there is no list of rejected cw\n        This may be a way to implement future old rtf processing for cw\n        Utility: open a group to just put brackets but why be so complicated?\n        Scheme: open brackets, write cw then go to body and back with cw after\n        '
        if self.__open_bracket:
            self.__write_obj.write('cb<nu<clos-brack<0003\n')
            self.__open_bracket = False
        inline_string = ''.join([f'{k}<nu<{v}\n' for (k, v) in iteritems(self.__inline) if v != 'false'])
        if inline_string:
            self.__write_obj.write('ob<nu<open-brack<0003\n%s' % inline_string)
            self.__open_bracket = True
        self.__temp_group = []

    def __change_permanent_group(self):
        if False:
            return 10
        '\n        Use temp group to change permanent group\n        If the control word is not accepted remove it\n        What is the interest as it is build to accept only accepted cw\n        in __after_control_word_func?\n        '
        self.__inline = {line[:16]: line[20:-1] for line in self.__temp_group if line[:16] in self.__accept}

    def __ignore_func(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Just copy data inside of RTF brackets already here.\n        '
        self.__write_obj.write(line)
        if self.__token_info == 'cb<nu<clos-brack' and self.__cb_count == self.__ignore_count:
            self.__state = 'in_body'

    def __check_brackets(self, in_file):
        if False:
            i = 10
            return i + 15
        '\n        Return True if brackets match\n        '
        check_brack_obj = check_brackets.CheckBrackets(file=in_file)
        return check_brack_obj.check_brackets()[0]

    def add_brackets(self):
        if False:
            i = 10
            return i + 15
        '\n        '
        self.__initiate_values()
        with open_for_read(self.__file) as read_obj:
            with open_for_write(self.__write_to) as self.__write_obj:
                for line in read_obj:
                    self.__token_info = line[:16]
                    if self.__token_info == 'ob<nu<open-brack':
                        self.__ob_count = line[-5:-1]
                    if self.__token_info == 'cb<nu<clos-brack':
                        self.__cb_count = line[-5:-1]
                    action = self.__state_dict.get(self.__state)
                    if action is None:
                        sys.stderr.write('No matching state in module add_brackets.py\n%s\n' % self.__state)
                    action(line)
        if self.__check_brackets(self.__write_to):
            copy_obj = copy.Copy(bug_handler=self.__bug_handler)
            if self.__copy:
                copy_obj.copy_file(self.__write_to, 'add_brackets.data')
            copy_obj.rename(self.__write_to, self.__file)
        elif self.__run_level > 0:
            sys.stderr.write('Sorry, but this files has a mix of old and new RTF.\nSome characteristics cannot be converted.\n')
        os.remove(self.__write_to)