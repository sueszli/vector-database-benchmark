import yaml
from picard.disc.utils import TocEntry, calculate_mb_toc_numbers

def toc_from_file(path):
    if False:
        print('Hello World!')
    'Reads whipper log files, generates musicbrainz disc TOC listing for use as discid.\n\n    Warning: may work wrong for discs having data tracks. May generate wrong\n    results on other non-standard cases.'
    with open(path, encoding='utf-8') as f:
        data = yaml.safe_load(f)
        toc_entries = (TocEntry(num, t['Start sector'], t['End sector']) for (num, t) in data['TOC'].items())
        return calculate_mb_toc_numbers(toc_entries)