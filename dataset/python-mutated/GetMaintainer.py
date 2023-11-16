from __future__ import print_function
from collections import defaultdict
from collections import OrderedDict
import argparse
import os
import re
import SetupGit
EXPRESSIONS = {'exclude': re.compile('^X:\\s*(?P<exclude>.*?)\\r*$'), 'file': re.compile('^F:\\s*(?P<file>.*?)\\r*$'), 'list': re.compile('^L:\\s*(?P<list>.*?)\\r*$'), 'maintainer': re.compile('^M:\\s*(?P<maintainer>.*?)\\r*$'), 'reviewer': re.compile('^R:\\s*(?P<reviewer>.*?)\\r*$'), 'status': re.compile('^S:\\s*(?P<status>.*?)\\r*$'), 'tree': re.compile('^T:\\s*(?P<tree>.*?)\\r*$'), 'webpage': re.compile('^W:\\s*(?P<webpage>.*?)\\r*$')}

def printsection(section):
    if False:
        for i in range(10):
            print('nop')
    'Prints out the dictionary describing a Maintainers.txt section.'
    print('===')
    for key in section.keys():
        print('Key: %s' % key)
        for item in section[key]:
            print('  %s' % item)

def pattern_to_regex(pattern):
    if False:
        print('Hello World!')
    'Takes a string containing regular UNIX path wildcards\n       and returns a string suitable for matching with regex.'
    pattern = pattern.replace('.', '\\.')
    pattern = pattern.replace('?', '.')
    pattern = pattern.replace('*', '.*')
    if pattern.endswith('/'):
        pattern += '.*'
    elif pattern.endswith('.*'):
        pattern = pattern[:-2]
        pattern += '(?!.*?/.*?)'
    return pattern

def path_in_section(path, section):
    if False:
        return 10
    'Returns True of False indicating whether the path is covered by\n       the current section.'
    if not 'file' in section:
        return False
    for pattern in section['file']:
        regex = pattern_to_regex(pattern)
        match = re.match(regex, path)
        if match:
            for pattern in section['exclude']:
                regex = pattern_to_regex(pattern)
                match = re.match(regex, path)
                if match:
                    return False
            return True
    return False

def get_section_maintainers(path, section):
    if False:
        print('Hello World!')
    'Returns a list with email addresses to any M: and R: entries\n       matching the provided path in the provided section.'
    maintainers = []
    reviewers = []
    lists = []
    nowarn_status = ['Supported', 'Maintained']
    if path_in_section(path, section):
        for status in section['status']:
            if status not in nowarn_status:
                print('WARNING: Maintained status for "%s" is \'%s\'!' % (path, status))
        for address in section['maintainer']:
            if isinstance(address, list):
                maintainers += address
            else:
                maintainers += [address]
        for address in section['reviewer']:
            if isinstance(address, list):
                reviewers += address
            else:
                reviewers += [address]
        for address in section['list']:
            if isinstance(address, list):
                lists += address
            else:
                lists += [address]
    return {'maintainers': maintainers, 'reviewers': reviewers, 'lists': lists}

def get_maintainers(path, sections, level=0):
    if False:
        return 10
    "For 'path', iterates over all sections, returning maintainers\n       for matching ones."
    maintainers = []
    reviewers = []
    lists = []
    for section in sections:
        recipients = get_section_maintainers(path, section)
        maintainers += recipients['maintainers']
        reviewers += recipients['reviewers']
        lists += recipients['lists']
    if not maintainers:
        print('"%s": no maintainers found, looking for default' % path)
        if level == 0:
            recipients = get_maintainers('<default>', sections, level=level + 1)
            maintainers += recipients['maintainers']
            reviewers += recipients['reviewers']
            lists += recipients['lists']
        else:
            print('No <default> maintainers set for project.')
        if not maintainers:
            return None
    return {'maintainers': maintainers, 'reviewers': reviewers, 'lists': lists}

def parse_maintainers_line(line):
    if False:
        i = 10
        return i + 15
    'Parse one line of Maintainers.txt, returning any match group and its key.'
    for (key, expression) in EXPRESSIONS.items():
        match = expression.match(line)
        if match:
            return (key, match.group(key))
    return (None, None)

def parse_maintainers_file(filename):
    if False:
        while True:
            i = 10
    'Parse the Maintainers.txt from top-level of repo and\n       return a list containing dictionaries of all sections.'
    with open(filename, 'r') as text:
        line = text.readline()
        sectionlist = []
        section = defaultdict(list)
        while line:
            (key, value) = parse_maintainers_line(line)
            if key and value:
                section[key].append(value)
            line = text.readline()
            if not key or not value or (not line):
                if section:
                    sectionlist.append(section.copy())
                    section.clear()
        return sectionlist

def get_modified_files(repo, args):
    if False:
        while True:
            i = 10
    "Returns a list of the files modified by the commit specified in 'args'."
    commit = repo.commit(args.commit)
    return commit.stats.files
if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(description='Retrieves information on who to cc for review on a given commit')
    PARSER.add_argument('commit', action='store', help='git revision to examine (default: HEAD)', nargs='?', default='HEAD')
    PARSER.add_argument('-l', '--lookup', help='Find section matches for path LOOKUP', required=False)
    ARGS = PARSER.parse_args()
    REPO = SetupGit.locate_repo()
    CONFIG_FILE = os.path.join(REPO.working_dir, 'Maintainers.txt')
    SECTIONS = parse_maintainers_file(CONFIG_FILE)
    if ARGS.lookup:
        FILES = [ARGS.lookup.replace('\\', '/')]
    else:
        FILES = get_modified_files(REPO, ARGS)
    ADDRESSES = set([])
    for file in FILES:
        print(file)
        recipients = get_maintainers(file, SECTIONS)
        ADDRESSES |= set(recipients['maintainers'] + recipients['reviewers'] + recipients['lists'])
    ADDRESSES = list(ADDRESSES)
    ADDRESSES.sort()
    for address in ADDRESSES:
        if '<' in address and '>' in address:
            address = address.split('>', 1)[0] + '>'
        print('  %s' % address)