import unittest
import re
from os import path, environ
from copy import copy
DIR = path.realpath(path.dirname(__file__))

class TestDocs(unittest.TestCase):

    def test_readmetoc(self):
        if False:
            print('Hello World!')
        start_marker = '<!-- START doctoc generated TOC please keep comment here to allow auto update -->\n'
        end_marker = '<!-- END doctoc generated TOC please keep comment here to allow auto update -->\n'
        template = "{prologue}{start_marker}<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->\n\n\n{toc}\n\n{end_marker}{epilogue}"
        prologue = ''
        toc = []
        epilogue = ''
        state = 'prologue'
        with open(path.join(path.dirname(DIR), 'README.md')) as f:
            contents = f.readlines()
            old_readme = copy(contents)
            state = 'prologue'
            for line in contents:
                if state == 'prologue':
                    if line == start_marker:
                        state = 'toc'
                    else:
                        prologue = prologue + line
                elif state == 'toc':
                    if line == end_marker:
                        state = 'epilogue'
                elif state == 'epilogue':
                    epilogue = epilogue + line
                    m = re.search('^([#]{1,6}) (.*)$', line)
                    if m is not None and m.groups():
                        header = m.group(1)
                        header_text = m.group(2)
                        header_text_strip = re.sub('[^a-zA-Z0-9-_ ]', '', header_text)
                        if header_text_strip == '':
                            continue
                        header_text_no_spaces = header_text_strip.replace(' ', '-').lower()
                        toc_line = '  ' * (len(header) - 2) + '- [%s](#%s)' % (header_text, header_text_no_spaces.lower())
                        toc.append(toc_line)
        new_readme = template.format(toc='\n'.join(toc), start_marker=start_marker, end_marker=end_marker, prologue=prologue, epilogue=epilogue)
        if environ.get('ZAPPA_TEST_SAVE_README_NEW'):
            with open(path.join(path.dirname(DIR), 'README.test.md'), 'w') as f:
                f.write(new_readme)
            msg = 'README.test.md written so you can manually compare.'
        else:
            msg = 'You can set environ[ZAPPA_TEST_SAVE_README_NEW]=1 to generate\n  README.test.md to manually compare.'
        self.assertEquals(''.join(old_readme), new_readme, "README doesn't match after regenerating TOC\n\nYou need to run doctoc after a heading change.\n{}".format(msg))