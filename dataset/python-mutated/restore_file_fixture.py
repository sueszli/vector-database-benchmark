import os
from tests.fake_trash_dir import trashinfo_content_default_date
from tests.support.files import make_file

class RestoreFileFixture:

    def __init__(self, XDG_DATA_HOME):
        if False:
            i = 10
            return i + 15
        self.XDG_DATA_HOME = XDG_DATA_HOME

    def having_a_trashed_file(self, path):
        if False:
            while True:
                i = 10
        self.make_file('%s/info/foo.trashinfo' % self._trash_dir(), trashinfo_content_default_date(path))
        self.make_file('%s/files/foo' % self._trash_dir())

    def make_file(self, filename, contents=''):
        if False:
            print('Hello World!')
        return make_file(filename, contents)

    def make_empty_file(self, filename):
        if False:
            i = 10
            return i + 15
        return self.make_file(filename)

    def _trash_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return '%s/Trash' % self.XDG_DATA_HOME

    def file_should_have_been_restored(self, filename):
        if False:
            i = 10
            return i + 15
        assert os.path.exists(filename)