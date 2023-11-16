import re
from picard.disc.utils import NotSupportedTOCError, TocEntry, calculate_mb_toc_numbers
from picard.util import detect_unicode_encoding
RE_TOC_ENTRY = re.compile('^Track (?P<num>\\d+):\\s+Ripped LBA (?P<start_sector>\\d+) to (?P<end_sector>\\d+)')

def filter_toc_entries(lines):
    if False:
        return 10
    '\n    Take iterator of lines, return iterator of toc entries\n    '
    last_track_num = 0
    for line in lines:
        m = RE_TOC_ENTRY.match(line)
        if m:
            track_num = int(m['num'])
            if last_track_num + 1 != track_num:
                raise NotSupportedTOCError(f'Non consecutive track numbers ({last_track_num} => {track_num}) in dBPoweramp log. Likely a partial rip, disc ID cannot be calculated')
            last_track_num = track_num
            yield TocEntry(track_num, int(m['start_sector']), int(m['end_sector']) - 1)

def toc_from_file(path):
    if False:
        while True:
            i = 10
    'Reads dBpoweramp log files, generates MusicBrainz disc TOC listing for use as discid.'
    encoding = detect_unicode_encoding(path)
    with open(path, 'r', encoding=encoding) as f:
        return calculate_mb_toc_numbers(filter_toc_entries(f))