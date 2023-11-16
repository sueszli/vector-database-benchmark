"""%(prog)s - generate information from built-in bzr help

%(prog)s creates a file with information on bzr in one of
several different output formats:

    man              man page
    bash_completion  bash completion script
    ...

Examples: 

    python2.4 generated-docs.py man
    python2.4 generated-docs.py bash_completion

Run "%(prog)s --help" for the option reference.
"""
import os
import sys
from optparse import OptionParser
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import bzrlib
from bzrlib import commands, branch, doc_generate

def main(argv):
    if False:
        return 10
    parser = OptionParser(usage='%prog [options] OUTPUT_FORMAT\n\nAvailable OUTPUT_FORMAT:\n\n    man              man page\n    rstx             man page in ReStructuredText format\n    bash_completion  bash completion script')
    parser.add_option('-s', '--show-filename', action='store_true', dest='show_filename', default=False, help='print default filename on stdout')
    parser.add_option('-o', '--output', dest='filename', metavar='FILE', help='write output to FILE')
    parser.add_option('-b', '--bzr-name', dest='bzr_name', default='bzr', metavar='EXEC_NAME', help='name of bzr executable')
    parser.add_option('-e', '--examples', action='callback', callback=print_extended_help, help='Examples of ways to call generate_doc')
    (options, args) = parser.parse_args(argv)
    if len(args) != 2:
        parser.print_help()
        sys.exit(1)
    with bzrlib.initialize():
        commands.install_bzr_command_hooks()
        infogen_type = args[1]
        infogen_mod = doc_generate.get_module(infogen_type)
        if options.filename:
            outfilename = options.filename
        else:
            outfilename = infogen_mod.get_filename(options)
        if outfilename == '-':
            outfile = sys.stdout
        else:
            outfile = open(outfilename, 'w')
        if options.show_filename and outfilename != '-':
            sys.stdout.write(outfilename)
            sys.stdout.write('\n')
        infogen_mod.infogen(options, outfile)

def print_extended_help(option, opt, value, parser):
    if False:
        return 10
    ' Program help examples\n\n    Prints out the examples stored in the docstring. \n\n    '
    sys.stdout.write(__doc__ % {'prog': sys.argv[0]})
    sys.stdout.write('\n')
    sys.exit(0)
if __name__ == '__main__':
    main(sys.argv)