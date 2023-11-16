import sys
import re
import b2.util.regex
options = {}

def set(name, value=None):
    if False:
        return 10
    global options
    options[name] = value

def get(name, default_value=None, implied_value=None):
    if False:
        print('Hello World!')
    global options
    matches = b2.util.regex.transform(sys.argv, '--' + re.escape(name) + '=(.*)')
    if matches:
        return matches[-1]
    else:
        m = b2.util.regex.transform(sys.argv, '--(' + re.escape(name) + ')')
        if m and implied_value:
            return implied_value
        elif options.get(name) is not None:
            return options[name]
        else:
            return default_value