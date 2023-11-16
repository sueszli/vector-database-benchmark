from __future__ import absolute_import
from __future__ import division
import argparse
import functools
import re
import pwnlib.args
pwnlib.args.free_form = False
from pwn import *
from pwnlib.commandline import common
p = common.parser_commands.add_parser('constgrep', help="Looking up constants from header files.\n\nExample: constgrep -c freebsd -m  ^PROT_ '3 + 4'", description="Looking up constants from header files.\n\nExample: constgrep -c freebsd -m  ^PROT_ '3 + 4'", formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('-e', '--exact', action='store_true', help='Do an exact match for a constant instead of searching for a regex')
p.add_argument('regex', help='The regex matching constant you want to find')
p.add_argument('constant', nargs='?', default=None, type=safeeval.expr, help='The constant to find')
p.add_argument('-i', '--case-insensitive', action='store_true', help='Search case insensitive')
p.add_argument('-m', '--mask-mode', action='store_true', help='Instead of searching for a specific constant value, search for values not containing strictly less bits that the given value.')
p.add_argument('-c', '--context', metavar='arch_or_os', action='append', type=common.context_arg, choices=common.choices, help='The os/architecture/endianness/bits the shellcode will run in (default: linux/i386), choose from: %s' % common.choices)

def main(args):
    if False:
        while True:
            i = 10
    if args.exact:
        print(cpp(args.regex).strip())
    else:
        if context.os == 'freebsd':
            mod = constants.freebsd
        else:
            mod = getattr(getattr(constants, context.os), context.arch)
        if args.case_insensitive:
            matcher = re.compile(args.regex, re.IGNORECASE)
        else:
            matcher = re.compile(args.regex)
        out = []
        maxlen = 0
        constant = args.constant
        for k in dir(mod):
            if k.endswith('__') and k.startswith('__'):
                continue
            if not matcher.search(k):
                continue
            if constant is not None:
                val = getattr(mod, k)
                if args.mask_mode:
                    if constant & val != val:
                        continue
                elif constant != val:
                    continue
            out.append((getattr(mod, k), k))
            maxlen = max(len(k), maxlen)
        for (_, k) in sorted(out):
            print('#define %s %s' % (k.ljust(maxlen), cpp(k).strip()))
        if constant and args.mask_mode:
            mask = constant
            good = []
            out = [(v, k) for (v, k) in out if v != 0]
            while mask and out:
                cur = out.pop()
                mask &= ~cur[0]
                good.append(cur)
                out = [(v, k) for (v, k) in out if mask & v == v]
            if functools.reduce(lambda x, cur: x | cur[0], good, 0) == constant:
                print('')
                print('(%s) == %s' % (' | '.join((k for (v, k) in good)), args.constant))
if __name__ == '__main__':
    pwnlib.commandline.common.main(__file__)