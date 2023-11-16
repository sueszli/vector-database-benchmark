"""A command line tool for diffing files.

This demonstrates the simplest possible way to turn a module into a command line
interface with Python Fire. It exemplifies the power and shortcomings of relying
on Python Fire's simplicity.

See //fire/examples/diff/diff.py for another way of turning
difflib into a CLI that requires more code, but gives the developer more control
over the CLI's API.

Use the help flag to see usage for all the things this CLI can do. For example:

difffull -- -h
difffull HtmlDiff -- -h  # Help for the HtmlDiff class
difffull HtmlDiff - -- -h  # Help for an HtmlDiff object, not the HtmlDiff class

Here are some of the diff commands available:

difffull ndiff A B [LINEJUNK] [CHARJUNK]
difffull context-diff A B [FROMFILE] [TOFILE] [FROMFILEDATE] [TOFILEDATE] [N]
difffull unified-diff A B [FROMFILE] [TOFILE] [FROMFILEDATE] [TOFILEDATE] [N]
difffull HtmlDiff - make-file FROMLINES TOLINES [FROMDESC] [TODESC] [CONTEXT]

For more useful versions of those last four commands using Python Fire, see
//fire/examples/diff:diff.par
"""
import difflib
import fire

def main():
    if False:
        while True:
            i = 10
    fire.Fire(difflib, name='difffull')
if __name__ == '__main__':
    main()