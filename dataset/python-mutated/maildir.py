from __future__ import annotations
import mailbox
import os.path
from libqtile.widget import base

class Maildir(base.ThreadPoolText):
    """A simple widget showing the number of new mails in maildir mailboxes"""
    defaults = [('maildir_path', '~/Mail', 'path to the Maildir folder'), ('sub_folders', [{'path': 'INBOX', 'label': 'Home mail'}, {'path': 'spam', 'label': 'Home junk'}], 'List of subfolders to scan. Each subfolder is a dict of `path` and `label`.'), ('separator', ' ', 'the string to put between the subfolder strings.'), ('total', False, 'Whether or not to sum subfolders into a grand             total. The first label will be used.'), ('hide_when_empty', False, 'Whether not to display anything if the subfolder has no new mail'), ('empty_color', None, 'Display color when no new mail is available'), ('nonempty_color', None, 'Display color when new mail is available'), ('subfolder_fmt', '{label}: {value}', 'Display format for one subfolder')]

    def __init__(self, **config):
        if False:
            for i in range(10):
                print('nop')
        base.ThreadPoolText.__init__(self, '', **config)
        self.add_defaults(Maildir.defaults)
        if isinstance(self.sub_folders[0], str):
            self.sub_folders = [{'path': folder, 'label': folder} for folder in self.sub_folders]

    def poll(self):
        if False:
            i = 10
            return i + 15
        'Scans the mailbox for new messages\n\n        Returns\n        =======\n        A string representing the current mailbox state\n        '
        state = {}

        def to_maildir_fmt(paths):
            if False:
                print('Hello World!')
            for path in iter(paths):
                yield path.rsplit(':')[0]
        for sub_folder in self.sub_folders:
            path = os.path.join(os.path.expanduser(self.maildir_path), sub_folder['path'])
            maildir = mailbox.Maildir(path)
            state[sub_folder['label']] = 0
            for file in to_maildir_fmt(os.listdir(os.path.join(path, 'new'))):
                if file in maildir:
                    state[sub_folder['label']] += 1
        return self.format_text(state)

    def _format_one(self, label: str, value: int) -> str:
        if False:
            i = 10
            return i + 15
        if value == 0 and self.hide_when_empty:
            return ''
        s = self.subfolder_fmt.format(label=label, value=value)
        color = self.empty_color if value == 0 else self.nonempty_color
        if color is None:
            return s
        return s.join(('<span foreground="{}">'.format(color), '</span>'))

    def format_text(self, state: dict[str, int]) -> str:
        if False:
            while True:
                i = 10
        'Converts the state of the subfolders to a string\n\n        Parameters\n        ==========\n        state: dict[str, int]\n            a dictionary mapping subfolder labels to new mail values\n\n        Returns\n        =======\n        a string representation of the given state\n        '
        if self.total:
            return self._format_one(self.sub_folders[0]['label'], sum(state.values()))
        else:
            return self.separator.join((self._format_one(*item) for item in state.items()))