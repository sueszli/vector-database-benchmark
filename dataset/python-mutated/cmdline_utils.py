"""
    Gmvault: a tool to backup and restore your gmail account.
    Copyright (C) <2011>  <guillaume Aubert (guillaume dot aubert at gmail do com)>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import argparse
import sys
import gmv.log_utils as log_utils
LOG = log_utils.LoggerFactory.get_logger('cmdline_utils')

class CmdLineParser(argparse.ArgumentParser):
    """ 
        Added service to OptionParser.
       
        Comments regarding usability of the lib. 
        By default you want to print the default in the help if you had them so the default formatter should print them
        Also new lines are eaten in the epilogue strings. You would use an epilogue to show examples most of the time so you
        want to have the possiblity to go to a new line. There should be a way to format the epilogue differently from  the rest  

    """
    BOOL_TRUE = ['yes', 'true', '1']
    BOOL_FALSE = ['no', 'false', '0']
    BOOL_VALS = BOOL_TRUE + BOOL_FALSE

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ' constructor '
        argparse.ArgumentParser.__init__(self, *args, **kwargs)
        self.epilogue = None

    @classmethod
    def convert_to_boolean(cls, val):
        if False:
            for i in range(10):
                print('nop')
        '\n           Convert yes, True, true, YES to boolean True and\n           no, False, false, NO to boolean NO\n        '
        lower_val = val.lower()
        if lower_val in cls.BOOL_TRUE:
            return True
        elif lower_val in cls.BOOL_FALSE:
            return False
        else:
            raise Exception('val %s should be in %s to be convertible to a boolean.' % (val, cls.BOOL_VALS))

    def print_help(self, out=sys.stderr):
        if False:
            return 10
        ' \n          Print the help message, followed by the epilogue (if set), to the \n          specified output file. You can define an epilogue by setting the \n          ``epilogue`` field. \n           \n          :param out: file desc where to write the usage message\n         \n        '
        super(CmdLineParser, self).print_help(out)
        if self.epilogue:
            (print >> out, '\n%s' % self.epilogue)
            out.flush()

    def show_usage(self, msg=None):
        if False:
            for i in range(10):
                print('nop')
        '\n           Print usage message          \n        '
        self.die_with_usage(msg)

    def die_with_usage(self, msg=None, exit_code=2):
        if False:
            while True:
                i = 10
        ' \n          Display a usage message and exit. \n   \n          :Parameters: \n              msg : str \n                  If not set to ``None`` (the default), this message will be \n                  displayed before the usage message \n                   \n              exit_code : int \n                  The process exit code. Defaults to 2. \n        '
        if msg != None:
            (print >> sys.stderr, msg)
        self.print_help(sys.stderr)
        sys.exit(exit_code)

    def error(self, msg):
        if False:
            i = 10
            return i + 15
        " \n          Overrides parent ``OptionParser`` class's ``error()`` method and \n          forces the full usage message on error. \n        "
        self.die_with_usage('%s: error: %s\n' % (self.prog, msg))

    def message(self, msg):
        if False:
            for i in range(10):
                print('nop')
        '\n           Print a message \n        '
        print('%s: %s\n' % (self.prog, msg))
SYNC_HELP_EPILOGUE = "Examples:\n\na) full synchronisation with email and password login\n\n#> gmvault --email foo.bar@gmail.com --passwd vrysecrtpasswd \n\nb) full synchronisation for german users that have to use googlemail instead of gmail\n\n#> gmvault --imap-server imap.googlemail.com --email foo.bar@gmail.com --passwd sosecrtpasswd\n\nc) restrict synchronisation with an IMAP request\n\n#> gmvault --imap-request 'Since 1-Nov-2011 Before 10-Nov-2011' --email foo.bar@gmail.com --passwd sosecrtpasswd \n\n"

def test_command_parser():
    if False:
        i = 10
        return i + 15
    '\n       Test the command parser\n    '
    parser = CmdLineParser()
    subparsers = parser.add_subparsers(help='commands')
    sync_parser = subparsers.add_parser('sync', formatter_class=argparse.ArgumentDefaultsHelpFormatter, help='synchronize with given gmail account')
    sync_parser.add_argument('-l', '--email', action='store', dest='email', help='email to sync with')
    sync_parser.add_argument('-t', '--type', action='store', default='full-sync', help='type of synchronisation')
    sync_parser.add_argument('-i', '--imap-server', metavar='HOSTNAME', help='Gmail imap server hostname. (default: imap.gmail.com)', dest='host', default='imap.gmail.com')
    sync_parser.add_argument('-p', '--imap-port', metavar='PORT', help='Gmail imap server port. (default: 993)', dest='port', default=993)
    sync_parser.set_defaults(verb='sync')
    sync_parser.epilogue = SYNC_HELP_EPILOGUE
    restore_parser = subparsers.add_parser('restore', help='restore email to a given email account')
    restore_parser.add_argument('email', action='store', help='email to sync with')
    restore_parser.add_argument('--recursive', '-r', default=False, action='store_true', help='Remove the contents of the directory, too')
    restore_parser.set_defaults(verb='restore')
    config_parser = subparsers.add_parser('config', help='add/delete/modify properties in configuration')
    config_parser.add_argument('dirname', action='store', help='New directory to create')
    config_parser.add_argument('--read-only', default=False, action='store_true', help='Set permissions to prevent writing to the directory')
    config_parser.set_defaults(verb='config')
    sys.argv = ['gmvault.py']
    print(parser.parse_args())
if __name__ == '__main__':
    test_command_parser()