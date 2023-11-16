from . import open_for_read

class CheckBrackets:
    """Check that brackets match up"""

    def __init__(self, bug_handler=None, file=None):
        if False:
            while True:
                i = 10
        self.__file = file
        self.__bug_handler = bug_handler
        self.__bracket_count = 0
        self.__ob_count = 0
        self.__cb_count = 0
        self.__open_bracket_num = []

    def open_brack(self, line):
        if False:
            for i in range(10):
                print('nop')
        num = line[-5:-1]
        self.__open_bracket_num.append(num)
        self.__bracket_count += 1

    def close_brack(self, line):
        if False:
            i = 10
            return i + 15
        num = line[-5:-1]
        try:
            last_num = self.__open_bracket_num.pop()
        except:
            return False
        if num != last_num:
            return False
        self.__bracket_count -= 1
        return True

    def check_brackets(self):
        if False:
            for i in range(10):
                print('nop')
        line_count = 0
        with open_for_read(self.__file) as read_obj:
            for line in read_obj:
                line_count += 1
                self.__token_info = line[:16]
                if self.__token_info == 'ob<nu<open-brack':
                    self.open_brack(line)
                if self.__token_info == 'cb<nu<clos-brack':
                    if not self.close_brack(line):
                        return (False, "closed bracket doesn't match, line %s" % line_count)
        if self.__bracket_count != 0:
            msg = "At end of file open and closed brackets don't match\ntotal number of brackets is %s" % self.__bracket_count
            return (False, msg)
        return (True, 'Brackets match!')