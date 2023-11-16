import re
_is_valid_var = re.compile('[_a-zA-Z]\\w*$')
_rm = re.compile('\\$[()]')
_remove = re.compile('\\$\\([^$]*(\\$[^)][^$]*)*\\$\\)')
_dollar_exps_str = '\\$[\\$\\(\\)]|\\$[_a-zA-Z][\\.\\w]*|\\${[^}]*}'
_dollar_exps = re.compile('(%s)' % _dollar_exps_str)
_separate_args = re.compile('(%s|\\s+|[^\\s$]+|\\$)' % _dollar_exps_str)
_space_sep = re.compile('[\\t ]+(?![^{]*})')

class ValueTypes:
    """
    Enum to store what type of value the variable holds.
    """
    UNKNOWN = 0
    STRING = 1
    CALLABLE = 2
    VARIABLE = 3

class EnvironmentValue:
    """
    Hold a single value. We're going to cache parsed version of the file
    We're going to keep track of variables which feed into this values evaluation
    """

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value
        self.var_type = ValueTypes.UNKNOWN
        if callable(self.value):
            self.var_type = ValueTypes.CALLABLE
        else:
            self.parse_value()

    def parse_value(self):
        if False:
            while True:
                i = 10
        '\n        Scan the string and break into component values\n        '
        try:
            if '$' not in self.value:
                self._parsed = self.value
                self.var_type = ValueTypes.STRING
            else:
                result = _dollar_exps.sub(sub_match, args)
                print(result)
        except TypeError:
            self._parsed = self.value

    def parse_trial(self):
        if False:
            while True:
                i = 10
        '\n        Try alternate parsing methods.\n        :return:\n        '
        parts = []
        for c in self.value:
            pass

class EnvironmentValues:
    """
    A class to hold all the environment variables
    """

    def __init__(self, **kw):
        if False:
            while True:
                i = 10
        self._dict = {}
        for k in kw:
            self._dict[k] = EnvironmentValue(kw[k])