"""
This script reformats the output of `myprog --help` to decent rst.

There are two sphinx plugins which could replace this eventually:
  - https://github.com/ashb/sphinx-argparse
  - https://github.com/gaborbernat/sphinx-argparse-cli
The former's output looks really nice, but lacks support for cross-referencing arguments. The latter supports
cross-referencing, but its output looks hideous. Hopefully either one of them will eventually evolve into something
we can use.

The main functions here are parser_to_rst() and help_to_rst(). Throughout this code the, **cross_references** option
controls whether or not section headings and options should be given cross reference targets.
"""
from textwrap import indent, wrap, dedent
import re
from argparse import ArgumentParser
from functools import partial

def parser_to_rst(parser: ArgumentParser, cross_references=True):
    if False:
        while True:
            i = 10
    '\n    Extract the ``--help`` output from an argparse parser and convert it to restructured text.\n    '
    help = parser.format_help()
    return help_to_rst(help, cross_references)
SECTION_REGEX = re.compile('\n    # A non-empty line with no indentation.\n    ^\\S.*\\n\n\n    # Followed by a non-zero number of either blank lines or indented lines.\n    (?:(?:\\ +.*)?\\n)+\n', re.MULTILINE | re.VERBOSE)

def help_to_rst(help: str, cross_references=True):
    if False:
        return 10
    '\n    Convert the output of a ``cli --help`` call to rst.\n    '
    (summary, *sections) = SECTION_REGEX.findall(help)
    sections = '\n'.join((section_to_rst(section, cross_references) for section in sections))
    return sections

def section_to_rst(section: str, cross_references=True) -> str:
    if False:
        while True:
            i = 10
    "\n    Convert a single option group's ``--help`` output to rst.\n\n    This generates a heading for the option group followed by each option within that group.\n    "
    (title, body) = section.split('\n', maxsplit=1)
    rst_title = rst_headerise(title, cross_references)
    rst_body = OPTION_REGEX.sub(partial(option_to_rst, cross_references=cross_references), body)
    return rst_title + rst_body
OPTION_REGEX = re.compile('\n    # Matches:\n    #   --name, --other-name, -n VALUE  Some description\n    #                                   and some more description.\n\n    # An option name prefixed with at least 1 space.\n    ^(\\ +)(.*?)\n    # Optionally followed by at least 2 spaces and the start of the description.\n    (?:\\ {2,}(.*))?\\n\n\n    # More lines of description.\n    # Each line starts with more spaces than which prefixed the option name (so as to avoid picking up the next option).\n    # Blank lines are allowed.\n    (((?:\\1\\ +(.*))?\\n)*)\n\n', re.MULTILINE | re.VERBOSE)

def option_to_rst(m: re.Match, cross_references=True) -> str:
    if False:
        while True:
            i = 10
    '\n    Convert a single option to rst.\n\n    The output should look like::\n\n        .. option:: --option-name -n\n\n            The help for that option nicely text-wrapped.\n    '
    name = m.group(2)
    assert name
    body = ' '.join((i for i in m.group(3, 4) if i))
    body = body.replace('*', '\\*')
    body = '\n'.join(wrap(dedent(body), width=75, break_on_hyphens=False, break_long_words=False))
    template = '.. option:: {}\n\n{}\n\n' if cross_references else '{}\n\n{}\n\n'
    return template.format(name, indent(body, '    '))

def rst_headerise(title: str, cross_references=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a title with the correct length '---' underline.\n    "
    title = title.strip(' \n:').title()
    out = f"{title}\n{'-' * len(title)}\n\n"
    if cross_references:
        out = f'.. _`{title}`:\n\n' + out
    return out