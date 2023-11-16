import re
from picard.disc.utils import TocEntry, calculate_mb_toc_numbers
from picard.util import detect_unicode_encoding
RE_TOC_TABLE_HEADER = re.compile(' \\s*\n    \\s*.+\\s+ \\| # track\n    \\s+.+\\s+ \\| # start\n    \\s+.+\\s+ \\| # length\n    \\s+.+\\s+ \\| # start sector\n    \\s+.+\\s*$   # end sector\n    ', re.VERBOSE)
RE_TOC_TABLE_LINE = re.compile('\n    \\s*\n    (?P<num>\\d+)\n    \\s*\\|\\s*\n    (?P<start_time>[0-9:.]+)\n    \\s*\\|\\s*\n    (?P<length_time>[0-9:.]+)\n    \\s*\\|\\s*\n    (?P<start_sector>\\d+)\n    \\s*\\|\\s*\n    (?P<end_sector>\\d+)\n    \\s*$', re.VERBOSE)

def filter_toc_entries(lines):
    if False:
        for i in range(10):
            print('nop')
    '\n    Take iterator of lines, return iterator of toc entries\n    '
    for line in lines:
        if RE_TOC_TABLE_HEADER.match(line):
            next(lines)
            break
    for line in lines:
        m = RE_TOC_TABLE_LINE.search(line)
        if not m:
            break
        yield TocEntry(int(m['num']), int(m['start_sector']), int(m['end_sector']))

def toc_from_file(path):
    if False:
        i = 10
        return i + 15
    'Reads EAC / XLD / fre:ac log files, generates MusicBrainz disc TOC listing for use as discid.\n\n    Warning: may work wrong for discs having data tracks. May generate wrong\n    results on other non-standard cases.'
    encoding = detect_unicode_encoding(path)
    with open(path, 'r', encoding=encoding) as f:
        return calculate_mb_toc_numbers(filter_toc_entries(f))