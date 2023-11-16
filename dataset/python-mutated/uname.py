"""
Provides a cross-platform way to figure out the system uname.

This version of uname was written in Python for the xonsh project: http://xon.sh

Based on cat from GNU coreutils: http://www.gnu.org/software/coreutils/
"""
import platform
import sys
from xonsh.cli_utils import ArgParserAlias

def uname_fn(all=False, kernel_name=False, node_name=False, kernel_release=False, kernel_version=False, machine=False, processor=False, hardware_platform=False, operating_system=False):
    if False:
        for i in range(10):
            print('nop')
    'This version of uname was written in Python for the xonsh project: https://xon.sh\n\n    Based on uname from GNU coreutils: http://www.gnu.org/software/coreutils/\n\n\n    Parameters\n    ----------\n    all : -a, --all\n        print all information, in the following order, except omit -p and -i if unknown\n    kernel_name : -s, --kernel-name\n        print the kernel name\n    node_name : -n, --nodename\n        print the network node hostname\n    kernel_release : -r, --kernel-release\n        print the kernel release\n    kernel_version : -v, --kernel-version\n        print the kernel version\n    machine : -m, --machine\n        print the machine hardware name\n    processor : -p, --processor\n        print the processor type (non-portable)\n    hardware_platform : -i, --hardware-platform\n        print the hardware platform (non-portable)\n    operating_system : -o, --operating-system\n        print the operating system\n    '
    info = platform.uname()

    def gen_lines():
        if False:
            for i in range(10):
                print('nop')
        if all or node_name:
            yield info.node
        if all or kernel_release:
            yield info.release
        if all or kernel_version:
            yield info.version
        if all or machine:
            yield info.machine
        if all or processor:
            yield (info.processor or 'unknown')
        if all or hardware_platform:
            yield 'unknown'
        if all or operating_system:
            yield sys.platform
    lines = list(gen_lines())
    if all or kernel_name or (not lines):
        lines.insert(0, info.system)
    line = ' '.join(lines)
    return line
uname = ArgParserAlias(func=uname_fn, has_args=True, prog='uname')

def main(args=None):
    if False:
        print('Hello World!')
    from xonsh.xoreutils.util import run_alias
    run_alias('uname', args)
if __name__ == '__main__':
    main()