from __future__ import print_function
import optparse
from .utils import setup_logging
from ..arctic import Arctic
print = print

def main():
    if False:
        for i in range(10):
            print('nop')
    usage = "usage: %prog [options] [prefix ...]\n\n    Lists the libraries available in a user's database.   If any prefix parameters\n    are given, list only libraries with names that start with one of the prefixes.\n\n    Example:\n        %prog --host=hostname rgautier\n    "
    setup_logging()
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--host', default='localhost', help='Hostname, or clustername. Default: localhost')
    (opts, args) = parser.parse_args()
    store = Arctic(opts.host)
    for name in sorted(store.list_libraries()):
        if not args or [n for n in args if name.startswith(n)]:
            print(name)
if __name__ == '__main__':
    main()