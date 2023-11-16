"""

Linkfix - a companion to sphinx's linkcheck builder.

Uses the linkcheck's output file to fix links in docs.

Originally created for this issue:
https://github.com/scrapy/scrapy/issues/606

Author: dufferzafar
"""
import re
import sys
from pathlib import Path

def main():
    if False:
        i = 10
        return i + 15
    _filename = None
    _contents = None
    line_re = re.compile('(.*)\\:\\d+\\:\\s\\[(.*)\\]\\s(?:(.*)\\sto\\s(.*)|(.*))')
    try:
        with Path('build/linkcheck/output.txt').open(encoding='utf-8') as out:
            output_lines = out.readlines()
    except OSError:
        print('linkcheck output not found; please run linkcheck first.')
        sys.exit(1)
    for line in output_lines:
        match = re.match(line_re, line)
        if match:
            newfilename = match.group(1)
            errortype = match.group(2)
            if errortype.lower() in ['broken', 'local']:
                print('Not Fixed: ' + line)
            else:
                if newfilename != _filename:
                    if _filename:
                        Path(_filename).write_text(_contents, encoding='utf-8')
                    _filename = newfilename
                    _contents = Path(_filename).read_text(encoding='utf-8')
                _contents = _contents.replace(match.group(3), match.group(4))
        else:
            print('Not Understood: ' + line)
if __name__ == '__main__':
    main()