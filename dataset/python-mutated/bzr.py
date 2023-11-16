"""Bazaar-related utilities."""
import csv
import re
from io import StringIO
from readthedocs.projects.exceptions import RepositoryError
from readthedocs.vcs_support.base import BaseVCS, VCSVersion

class Backend(BaseVCS):
    """Bazaar VCS backend."""
    supports_tags = True
    fallback_branch = ''

    def clone(self):
        if False:
            print('Hello World!')
        self.make_clean_working_dir()
        try:
            self.run('bzr', 'checkout', self.repo_url, '.')
        except RepositoryError:
            raise RepositoryError(RepositoryError.CLONE_ERROR())

    @property
    def tags(self):
        if False:
            while True:
                i = 10
        try:
            (code, stdout, stderr) = self.run('bzr', 'tags', record_as_success=True)
            return self.parse_tags(stdout)
        except RepositoryError:
            return []

    def parse_tags(self, data):
        if False:
            for i in range(10):
                print('nop')
        "\n        Parses output of bzr tags.\n\n        Example:\n\n            0.1.0                171\n            0.1.1                173\n            0.1.2                174\n            0.2.0-pre-alpha      177\n\n        Can't forget about poorly formatted tags or tags that lack revisions,\n        such as:\n\n            3.3.0-rc1            ?\n            tag with spaces      123\n        "
        if not isinstance(data, str):
            data = str(data)
        squashed_data = re.sub(' +', ' ', data)
        raw_tags = csv.reader(StringIO(squashed_data), delimiter=' ')
        vcs_tags = []
        for row in raw_tags:
            name = ' '.join(row[:-1])
            commit = row[-1]
            if commit != '?':
                vcs_tags.append(VCSVersion(self, commit, name))
        return vcs_tags

    @property
    def commit(self):
        if False:
            return 10
        (_, stdout, _) = self.run('bzr', 'revno')
        return stdout.strip()

    def checkout(self, identifier=None):
        if False:
            print('Hello World!')
        super().checkout()
        if not identifier:
            return self.up()
        try:
            (code, stdout, stderr) = self.run('bzr', 'switch', identifier)
            return (code, stdout, stderr)
        except RepositoryError:
            raise RepositoryError(RepositoryError.FAILED_TO_CHECKOUT.format(identifier))