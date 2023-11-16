from collections import Counter
from typing import TYPE_CHECKING
from jrnl.plugins.text_exporter import TextExporter
if TYPE_CHECKING:
    from jrnl.journals import Entry
    from jrnl.journals import Journal

class DatesExporter(TextExporter):
    """This Exporter lists dates and their respective counts, for heatingmapping etc."""
    names = ['dates']
    extension = 'dates'

    @classmethod
    def export_entry(cls, entry: 'Entry'):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @classmethod
    def export_journal(cls, journal: 'Journal') -> str:
        if False:
            i = 10
            return i + 15
        'Returns dates and their frequencies for an entire journal.'
        date_counts = Counter()
        for entry in journal.entries:
            date = str(entry.date.date())
            date_counts[date] += 1
        result = '\n'.join((f'{date}, {count}' for (date, count) in date_counts.items()))
        return result