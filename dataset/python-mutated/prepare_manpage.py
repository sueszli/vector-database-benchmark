import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path
import re
from devscripts.utils import compose_functions, get_filename_args, read_file, write_file
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README_FILE = os.path.join(ROOT_DIR, 'README.md')
PREFIX = '%yt-dlp(1)\n\n# NAME\n\nyt\\-dlp \\- A youtube-dl fork with additional features and patches\n\n# SYNOPSIS\n\n**yt-dlp** \\[OPTIONS\\] URL [URL...]\n\n# DESCRIPTION\n\n'

def filter_excluded_sections(readme):
    if False:
        for i in range(10):
            print('nop')
    EXCLUDED_SECTION_BEGIN_STRING = re.escape('<!-- MANPAGE: BEGIN EXCLUDED SECTION -->')
    EXCLUDED_SECTION_END_STRING = re.escape('<!-- MANPAGE: END EXCLUDED SECTION -->')
    return re.sub(f'(?s){EXCLUDED_SECTION_BEGIN_STRING}.+?{EXCLUDED_SECTION_END_STRING}\\n', '', readme)

def move_sections(readme):
    if False:
        i = 10
        return i + 15
    MOVE_TAG_TEMPLATE = '<!-- MANPAGE: MOVE "%s" SECTION HERE -->'
    sections = re.findall('(?m)^%s$' % (re.escape(MOVE_TAG_TEMPLATE).replace('\\%', '%') % '(.+)'), readme)
    for section_name in sections:
        move_tag = MOVE_TAG_TEMPLATE % section_name
        if readme.count(move_tag) > 1:
            raise Exception(f'There is more than one occurrence of "{move_tag}". This is unexpected')
        sections = re.findall(f'(?sm)(^# {re.escape(section_name)}.+?)(?=^# )', readme)
        if len(sections) < 1:
            raise Exception(f'The section {section_name} does not exist')
        elif len(sections) > 1:
            raise Exception(f'There are multiple occurrences of section {section_name}, this is unhandled')
        readme = readme.replace(sections[0], '', 1).replace(move_tag, sections[0], 1)
    return readme

def filter_options(readme):
    if False:
        i = 10
        return i + 15
    section = re.search('(?sm)^# USAGE AND OPTIONS\\n.+?(?=^# )', readme).group(0)
    options = '# OPTIONS\n'
    for line in section.split('\n')[1:]:
        mobj = re.fullmatch('(?x)\n                \\s{4}(?P<opt>-(?:,\\s|[^\\s])+)\n                (?:\\s(?P<meta>(?:[^\\s]|\\s(?!\\s))+))?\n                (\\s{2,}(?P<desc>.+))?\n            ', line)
        if not mobj:
            options += f'{line.lstrip()}\n'
            continue
        (option, metavar, description) = mobj.group('opt', 'meta', 'desc')
        option = f'{option} *{metavar}*' if metavar else option
        description = f'{description}\n' if description else ''
        options += f'\n{option}\n:   {description}'
        continue
    return readme.replace(section, options, 1)
TRANSFORM = compose_functions(filter_excluded_sections, move_sections, filter_options)

def main():
    if False:
        while True:
            i = 10
    write_file(get_filename_args(), PREFIX + TRANSFORM(read_file(README_FILE)))
if __name__ == '__main__':
    main()