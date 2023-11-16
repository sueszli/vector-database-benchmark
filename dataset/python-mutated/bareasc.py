"""Provides a bare-ASCII matching query."""
from unidecode import unidecode
from beets import ui
from beets.dbcore.query import StringFieldQuery
from beets.plugins import BeetsPlugin
from beets.ui import decargs, print_

class BareascQuery(StringFieldQuery):
    """Compare items using bare ASCII, without accents etc."""

    @classmethod
    def string_match(cls, pattern, val):
        if False:
            return 10
        'Convert both pattern and string to plain ASCII before matching.\n\n        If pattern is all lower case, also convert string to lower case so\n        match is also case insensitive\n        '
        if pattern.islower():
            val = val.lower()
        pattern = unidecode(pattern)
        val = unidecode(val)
        return pattern in val

    def col_clause(self):
        if False:
            print('Hello World!')
        'Compare ascii version of the pattern.'
        clause = f'unidecode({self.field})'
        if self.pattern.islower():
            clause = f'lower({clause})'
        return (f"{clause} LIKE ? ESCAPE '\\'", [f'%{unidecode(self.pattern)}%'])

class BareascPlugin(BeetsPlugin):
    """Plugin to provide bare-ASCII option for beets matching."""

    def __init__(self):
        if False:
            return 10
        'Default prefix for selecting bare-ASCII matching is #.'
        super().__init__()
        self.config.add({'prefix': '#'})

    def queries(self):
        if False:
            while True:
                i = 10
        'Register bare-ASCII matching.'
        prefix = self.config['prefix'].as_str()
        return {prefix: BareascQuery}

    def commands(self):
        if False:
            for i in range(10):
                print('nop')
        "Add bareasc command as unidecode version of 'list'."
        cmd = ui.Subcommand('bareasc', help='unidecode version of beet list command')
        cmd.parser.usage += "\nExample: %prog -f '$album: $title' artist:beatles"
        cmd.parser.add_all_common_options()
        cmd.func = self.unidecode_list
        return [cmd]

    def unidecode_list(self, lib, opts, args):
        if False:
            i = 10
            return i + 15
        "Emulate normal 'list' command but with unidecode output."
        query = decargs(args)
        album = opts.album
        if album:
            for album in lib.albums(query):
                bare = unidecode(str(album))
                print_(bare)
        else:
            for item in lib.items(query):
                bare = unidecode(str(item))
                print_(bare)