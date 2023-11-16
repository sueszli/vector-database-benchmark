"""Implements a umask command for xonsh."""
import os
import re
import xonsh.lazyasd as xl

@xl.lazyobject
def symbolic_matcher():
    if False:
        while True:
            i = 10
    return re.compile('([ugo]*|a)([+-=])([^\\s,]*)')
order = 'rwx'
name_to_value = {'x': 1, 'w': 2, 'r': 4}
value_to_name = {v: k for (k, v) in name_to_value.items()}
class_to_loc = {'u': 6, 'g': 3, 'o': 0}
loc_to_class = {v: k for (k, v) in class_to_loc.items()}
function_map = {'+': lambda orig, new: orig | new, '-': lambda orig, new: orig & ~new, '=': lambda orig, new: new}

def current_mask():
    if False:
        return 10
    out = os.umask(0)
    os.umask(out)
    return out

def invert(perms):
    if False:
        for i in range(10):
            print('nop')
    return 511 - perms

def get_oct_digits(mode):
    if False:
        return 10
    '\n    Separate a given integer into its three components\n    '
    if not 0 <= mode <= 511:
        raise ValueError('expected a value between 000 and 777')
    return {'u': (mode & 448) >> 6, 'g': (mode & 56) >> 3, 'o': mode & 7}

def from_oct_digits(digits):
    if False:
        while True:
            i = 10
    o = 0
    for (c, m) in digits.items():
        o |= m << class_to_loc[c]
    return o

def get_symbolic_rep_single(digit):
    if False:
        return 10
    '\n    Given a single octal digit, return the appropriate string representation.\n    For example, 6 becomes "rw".\n    '
    o = ''
    for sym in 'rwx':
        num = name_to_value[sym]
        if digit & num:
            o += sym
            digit -= num
    return o

def get_symbolic_rep(number):
    if False:
        print('Hello World!')
    digits = get_oct_digits(number)
    return ','.join((f'{class_}={get_symbolic_rep_single(digits[class_])}' for class_ in 'ugo'))

def get_numeric_rep_single(rep):
    if False:
        i = 10
        return i + 15
    '\n    Given a string representation, return the appropriate octal digit.\n    For example, "rw" becomes 6.\n    '
    o = 0
    for sym in set(rep):
        o += name_to_value[sym]
    return o

def single_symbolic_arg(arg, old=None):
    if False:
        for i in range(10):
            print('nop')
    if old is None:
        old = invert(current_mask())
    match = symbolic_matcher.match(arg)
    if not match:
        raise ValueError('could not parse argument %r' % arg)
    (class_, op, mask) = match.groups()
    if class_ == 'a':
        class_ = 'ugo'
    invalid_chars = [i for i in mask if i not in name_to_value]
    if invalid_chars:
        raise ValueError('invalid mask %r' % mask)
    digits = get_oct_digits(old)
    new_num = get_numeric_rep_single(mask)
    for c in set(class_):
        digits[c] = function_map[op](digits[c], new_num)
    return from_oct_digits(digits)

def valid_numeric_argument(x):
    if False:
        while True:
            i = 10
    try:
        return len(x) == 3 and all((0 <= int(i) <= 7 for i in x))
    except:
        return False

def umask(args, stdin, stdout, stderr):
    if False:
        for i in range(10):
            print('nop')
    if '-h' in args:
        print(UMASK_HELP, file=stdout)
        return 0
    symbolic = False
    while '-S' in args:
        symbolic = True
        args.remove('-S')
    cur = current_mask()
    if len(args) == 0:
        if symbolic:
            to_print = get_symbolic_rep(invert(cur))
        else:
            to_print = oct(cur)[2:]
            while len(to_print) < 3:
                to_print = '0%s' % to_print
        print(to_print, file=stdout)
        return 0
    else:
        num = [valid_numeric_argument(i) for i in args]
        if any(num):
            if not all(num):
                print("error: can't mix numeric and symbolic arguments", file=stderr)
                return 1
            if len(num) != 1:
                print("error: can't have more than one numeric argument", file=stderr)
                return 1
        for (arg, isnum) in zip(args, num):
            if isnum:
                cur = int(arg, 8)
            else:
                cur = invert(cur)
                for subarg in arg.split(','):
                    try:
                        cur = single_symbolic_arg(subarg, cur)
                    except:
                        print('error: could not parse argument: %r' % subarg, file=stderr)
                        return 1
                cur = invert(cur)
            os.umask(cur)
UMASK_HELP = 'Usage: umask [-S] [mode]...\nView or set the file creation mask.\n\n  -S             when printing, show output in symbolic format\n  -h  --help     display this message and exit\n\nThis version of umask was written in Python for tako: https://takoshell.org\nBased on the umask command from Bash:\nhttps://www.gnu.org/software/bash/manual/html_node/Bourne-Shell-Builtins.html'
if __name__ == '__main__':
    import sys
    umask(sys.argv, sys.stdin, sys.stdout, sys.stderr)