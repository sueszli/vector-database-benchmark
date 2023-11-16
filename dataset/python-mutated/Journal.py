import datetime
import logging
import os
import re
from jrnl import time
from jrnl.config import validate_journal_name
from jrnl.encryption import determine_encryption_method
from jrnl.messages import Message
from jrnl.messages import MsgStyle
from jrnl.messages import MsgText
from jrnl.output import print_msg
from jrnl.path import expand_path
from jrnl.prompt import yesno
from .Entry import Entry

class Tag:

    def __init__(self, name, count=0):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.count = count

    def __str__(self):
        if False:
            return 10
        return self.name

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f"<Tag '{self.name}'>"

class Journal:

    def __init__(self, name='default', **kwargs):
        if False:
            i = 10
            return i + 15
        self.config = {'journal': 'journal.txt', 'encrypt': False, 'default_hour': 9, 'default_minute': 0, 'timeformat': '%Y-%m-%d %H:%M', 'tagsymbols': '@', 'highlight': True, 'linewrap': 80, 'indent_character': '|'}
        self.config.update(kwargs)
        self.search_tags = None
        self.name = name
        self.entries = []
        self.encryption_method = None
        self.added_entry_count = 0
        self.deleted_entry_count = 0

    def __len__(self):
        if False:
            print('Hello World!')
        'Returns the number of entries'
        return len(self.entries)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        "Iterates over the journal's entries."
        return (entry for entry in self.entries)

    @classmethod
    def from_journal(cls, other: 'Journal') -> 'Journal':
        if False:
            for i in range(10):
                print('nop')
        'Creates a new journal by copying configuration and entries from\n        another journal object'
        new_journal = cls(other.name, **other.config)
        new_journal.entries = other.entries
        logging.debug('Imported %d entries from %s to %s', len(new_journal), other.__class__.__name__, cls.__name__)
        return new_journal

    def import_(self, other_journal_txt: str) -> None:
        if False:
            print('Hello World!')
        imported_entries = self._parse(other_journal_txt)
        for entry in imported_entries:
            entry.modified = True
        self.entries = list(frozenset(self.entries) | frozenset(imported_entries))
        self.sort()

    def _get_encryption_method(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        encryption_method = determine_encryption_method(self.config['encrypt'])
        self.encryption_method = encryption_method(self.name, self.config)

    def _decrypt(self, text: bytes) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.encryption_method is None:
            self._get_encryption_method()
        return self.encryption_method.decrypt(text)

    def _encrypt(self, text: str) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        if self.encryption_method is None:
            self._get_encryption_method()
        return self.encryption_method.encrypt(text)

    def open(self, filename: str | None=None) -> 'Journal':
        if False:
            return 10
        'Opens the journal file and parses it into a list of Entries\n        Entries have the form (date, title, body).'
        filename = filename or self.config['journal']
        dirname = os.path.dirname(filename)
        if not os.path.exists(filename):
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
                print_msg(Message(MsgText.DirectoryCreated, MsgStyle.NORMAL, {'directory_name': dirname}))
            self.create_file(filename)
            print_msg(Message(MsgText.JournalCreated, MsgStyle.NORMAL, {'journal_name': self.name, 'filename': filename}))
            self.write()
        text = self._load(filename)
        text = self._decrypt(text)
        self.entries = self._parse(text)
        self.sort()
        logging.debug('opened %s with %d entries', self.__class__.__name__, len(self))
        return self

    def write(self, filename: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        'Dumps the journal into the config file, overwriting it'
        filename = filename or self.config['journal']
        text = self._to_text()
        text = self._encrypt(text)
        self._store(filename, text)

    def validate_parsing(self) -> bool:
        if False:
            return 10
        'Confirms that the jrnl is still parsed correctly after conversion to text.'
        new_entries = self._parse(self._to_text())
        return all((entry == new_entries[i] for (i, entry) in enumerate(self.entries)))

    @staticmethod
    def create_file(filename: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        with open(filename, 'w'):
            pass

    def _to_text(self) -> str:
        if False:
            return 10
        return '\n'.join([str(e) for e in self.entries])

    def _load(self, filename: str) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        with open(filename, 'rb') as f:
            return f.read()

    def _store(self, filename: str, text: bytes) -> None:
        if False:
            print('Hello World!')
        with open(filename, 'wb') as f:
            f.write(text)

    def _parse(self, journal_txt: str) -> list[Entry]:
        if False:
            while True:
                i = 10
        "Parses a journal that's stored in a string and returns a list of entries"
        if not journal_txt:
            return []
        entries = []
        date_blob_re = re.compile('(?:^|\n)\\[([^\\]]+)\\] ')
        last_entry_pos = 0
        for match in date_blob_re.finditer(journal_txt):
            date_blob = match.groups()[0]
            try:
                new_date = datetime.datetime.strptime(date_blob, self.config['timeformat'])
            except ValueError:
                new_date = time.parse(date_blob, bracketed=True)
            if new_date:
                if entries:
                    entries[-1].text = journal_txt[last_entry_pos:match.start()]
                last_entry_pos = match.end()
                entries.append(Entry(self, date=new_date))
        if not entries:
            entries.append(Entry(self, date=time.parse('now')))
        entries[-1].text = journal_txt[last_entry_pos:]
        for entry in entries:
            entry._parse_text()
        return entries

    def pprint(self, short: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Prettyprints the journal's entries"
        return '\n'.join([e.pprint(short=short) for e in self.entries])

    def __str__(self):
        if False:
            return 10
        return self.pprint()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'<Journal with {len(self.entries)} entries>'

    def sort(self) -> None:
        if False:
            return 10
        "Sorts the Journal's entries by date"
        self.entries = sorted(self.entries, key=lambda entry: entry.date)

    def limit(self, n: int | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Removes all but the last n entries'
        if n:
            self.entries = self.entries[-n:]

    @property
    def tags(self) -> list[Tag]:
        if False:
            i = 10
            return i + 15
        'Returns a set of tuples (count, tag) for all tags present in the journal.'
        tags = [tag for entry in self.entries for tag in set(entry.tags)]
        tag_counts = {(tags.count(tag), tag) for tag in tags}
        return [Tag(tag, count=count) for (count, tag) in sorted(tag_counts)]

    def filter(self, tags=[], month=None, day=None, year=None, start_date=None, end_date=None, starred=False, tagged=False, exclude_starred=False, exclude_tagged=False, strict=False, contains=None, exclude=[]):
        if False:
            i = 10
            return i + 15
        'Removes all entries from the journal that don\'t match the filter.\n\n        tags is a list of tags, each being a string that starts with one of the\n        tag symbols defined in the config, e.g. ["@John", "#WorldDomination"].\n\n        start_date and end_date define a timespan by which to filter.\n\n        starred limits journal to starred entries\n\n        If strict is True, all tags must be present in an entry. If false, the\n\n        exclude is a list of the tags which should not appear in the results.\n        entry is kept if any tag is present, unless they appear in exclude.'
        self.search_tags = {tag.lower() for tag in tags}
        excluded_tags = {tag.lower() for tag in exclude}
        end_date = time.parse(end_date, inclusive=True)
        start_date = time.parse(start_date)
        has_tags = self.search_tags.issubset if strict else self.search_tags.intersection

        def excluded(tags):
            if False:
                for i in range(10):
                    print('nop')
            return 0 < len([tag for tag in tags if tag in excluded_tags])
        if contains:
            contains_lower = contains.casefold()
        if month or day or year:
            compare_d = time.parse(f'{month or 1}.{day or 1}.{year or 1}')
        result = [entry for entry in self.entries if (not tags or has_tags(entry.tags)) and (not (starred or exclude_starred) or entry.starred == starred) and (not (tagged or exclude_tagged) or bool(entry.tags) == tagged) and (not month or entry.date.month == compare_d.month) and (not day or entry.date.day == compare_d.day) and (not year or entry.date.year == compare_d.year) and (not start_date or entry.date >= start_date) and (not end_date or entry.date <= end_date) and (not exclude or not excluded(entry.tags)) and (not contains or (contains_lower in entry.title.casefold() or contains_lower in entry.body.casefold()))]
        self.entries = result

    def delete_entries(self, entries_to_delete: list[Entry]) -> None:
        if False:
            i = 10
            return i + 15
        'Deletes specific entries from a journal.'
        for entry in entries_to_delete:
            self.entries.remove(entry)
            self.deleted_entry_count += 1

    def change_date_entries(self, date: datetime.datetime, entries_to_change: list[Entry]) -> None:
        if False:
            print('Hello World!')
        'Changes entry dates to given date.'
        date = time.parse(date)
        for entry in entries_to_change:
            entry.date = date
            entry.modified = True

    def prompt_action_entries(self, msg: MsgText) -> list[Entry]:
        if False:
            while True:
                i = 10
        'Prompts for action for each entry in a journal, using given message.\n        Returns the entries the user wishes to apply the action on.'
        to_act = []

        def ask_action(entry):
            if False:
                while True:
                    i = 10
            return yesno(Message(msg, params={'entry_title': entry.pprint(short=True)}), default=False)
        for entry in self.entries:
            if ask_action(entry):
                to_act.append(entry)
        return to_act

    def new_entry(self, raw: str, date=None, sort: bool=True) -> Entry:
        if False:
            return 10
        'Constructs a new entry from some raw text input.\n        If a date is given, it will parse and use this, otherwise scan for a date in\n        the input first.\n        '
        raw = raw.replace('\\n ', '\n').replace('\\n', '\n')
        sep = re.search('\\n|[?!.]+ +\\n?', raw)
        first_line = raw[:sep.end()].strip() if sep else raw
        starred = False
        if not date:
            colon_pos = first_line.find(': ')
            if colon_pos > 0:
                date = time.parse(raw[:colon_pos], default_hour=self.config['default_hour'], default_minute=self.config['default_minute'])
                if date:
                    starred = raw[:colon_pos].strip().endswith('*')
                    raw = raw[colon_pos + 1:].strip()
        starred = starred or first_line.startswith('*') or first_line.endswith('*') or raw.startswith('*')
        if not date:
            date = time.parse('now')
        entry = Entry(self, date, raw, starred=starred)
        entry.modified = True
        self.entries.append(entry)
        if sort:
            self.sort()
        return entry

    def editable_str(self) -> str:
        if False:
            return 10
        'Turns the journal into a string of entries that can be edited\n        manually and later be parsed with self.parse_editable_str.'
        return '\n'.join([str(e) for e in self.entries])

    def parse_editable_str(self, edited: str) -> None:
        if False:
            i = 10
            return i + 15
        "Parses the output of self.editable_str and updates it's entries."
        mod_entries = self._parse(edited)
        for entry in mod_entries:
            entry.modified = not any((entry == old_entry for old_entry in self.entries))
        self.increment_change_counts_by_edit(mod_entries)
        self.entries = mod_entries

    def increment_change_counts_by_edit(self, mod_entries: Entry) -> None:
        if False:
            while True:
                i = 10
        if len(mod_entries) > len(self.entries):
            self.added_entry_count += len(mod_entries) - len(self.entries)
        else:
            self.deleted_entry_count += len(self.entries) - len(mod_entries)

    def get_change_counts(self) -> dict:
        if False:
            return 10
        return {'added': self.added_entry_count, 'deleted': self.deleted_entry_count, 'modified': len([e for e in self.entries if e.modified])}

class LegacyJournal(Journal):
    """Legacy class to support opening journals formatted with the jrnl 1.x
    standard. Main difference here is that in 1.x, timestamps were not cuddled
    by square brackets. You'll not be able to save these journals anymore."""

    def _parse(self, journal_txt: str) -> list[Entry]:
        if False:
            for i in range(10):
                print('nop')
        "Parses a journal that's stored in a string and returns a list of entries"
        date_length = len(datetime.datetime.today().strftime(self.config['timeformat']))
        entries = []
        current_entry = None
        new_date_format_regex = re.compile('(^\\[[^\\]]+\\].*?$)')
        for line in journal_txt.splitlines():
            line = line.rstrip()
            try:
                new_date = datetime.datetime.strptime(line[:date_length], self.config['timeformat'])
                if new_date and current_entry:
                    entries.append(current_entry)
                if line.endswith('*'):
                    starred = True
                    line = line[:-1]
                else:
                    starred = False
                current_entry = Entry(self, date=new_date, text=line[date_length + 1:], starred=starred)
            except ValueError:
                line = new_date_format_regex.sub(' \\1', line)
                if current_entry:
                    current_entry.text += line + '\n'
        if current_entry:
            entries.append(current_entry)
        for entry in entries:
            entry._parse_text()
        return entries

def open_journal(journal_name: str, config: dict, legacy: bool=False) -> Journal:
    if False:
        i = 10
        return i + 15
    '\n    Creates a normal, encrypted or DayOne journal based on the passed config.\n    If legacy is True, it will open Journals with legacy classes build for\n    backwards compatibility with jrnl 1.x\n    '
    logging.debug(f"open_journal '{journal_name}'")
    validate_journal_name(journal_name, config)
    config = config.copy()
    config['journal'] = expand_path(config['journal'])
    if os.path.isdir(config['journal']):
        if config['encrypt']:
            print_msg(Message(MsgText.ConfigEncryptedForUnencryptableJournalType, MsgStyle.WARNING, {'journal_name': journal_name}))
        if config['journal'].strip('/').endswith('.dayone') or 'entries' in os.listdir(config['journal']):
            from jrnl.journals import DayOne
            return DayOne(**config).open()
        else:
            from jrnl.journals import Folder
            return Folder(journal_name, **config).open()
    if not config['encrypt']:
        if legacy:
            return LegacyJournal(journal_name, **config).open()
        if config['journal'].endswith(os.sep):
            from jrnl.journals import Folder
            return Folder(journal_name, **config).open()
        return Journal(journal_name, **config).open()
    if legacy:
        config['encrypt'] = 'jrnlv1'
        return LegacyJournal(journal_name, **config).open()
    return Journal(journal_name, **config).open()