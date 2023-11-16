import datetime
import os
import uuid
from trashcli.put.format_trash_info import format_original_location
from .support.files import make_file, make_parent_for, make_unreadable_file

def a_default_datetime():
    if False:
        for i in range(10):
            print('nop')
    return datetime.datetime(2000, 1, 1, 0, 0, 1)

class FakeTrashDir:

    def __init__(self, path):
        if False:
            while True:
                i = 10
        self.info_path = os.path.join(path, 'info')
        self.files_path = os.path.join(path, 'files')

    def add_unreadable_trashinfo(self, basename):
        if False:
            i = 10
            return i + 15
        path = self.a_trashinfo_path(basename)
        make_unreadable_file(path)

    def add_trashed_file(self, basename, path, content, date=a_default_datetime()):
        if False:
            while True:
                i = 10
        self.add_trashinfo3(basename, path, date)
        make_file(self.file_path(basename), content)

    def a_trashinfo_path(self, basename):
        if False:
            return 10
        return os.path.join(self.info_path, '%s.trashinfo' % basename)

    def file_path(self, basename):
        if False:
            while True:
                i = 10
        return os.path.join(self.files_path, basename)

    def add_trashinfo_basename_path(self, basename, path):
        if False:
            return 10
        self.add_trashinfo3(basename, path, a_default_datetime())

    def add_trashinfo2(self, path, deletion_date):
        if False:
            print('Hello World!')
        basename = str(uuid.uuid4())
        self.add_trashinfo3(basename, path, deletion_date)

    def add_trashinfo3(self, basename, path, deletion_date):
        if False:
            return 10
        content = trashinfo_content(path, deletion_date)
        self.add_trashinfo_content(basename, content)

    def add_trashinfo_with_date(self, basename, deletion_date):
        if False:
            return 10
        content = trashinfo_content2([('DeletionDate', deletion_date.strftime('%Y-%m-%dT%H:%M:%S'))])
        self.add_trashinfo_content(basename, content)

    def add_trashinfo_with_invalid_date(self, basename, invalid_date):
        if False:
            print('Hello World!')
        content = trashinfo_content2([('DeletionDate', invalid_date)])
        self.add_trashinfo_content(basename, content)

    def add_trashinfo_without_path(self, basename):
        if False:
            i = 10
            return i + 15
        deletion_date = a_default_datetime()
        content = trashinfo_content2([('DeletionDate', deletion_date.strftime('%Y-%m-%dT%H:%M:%S'))])
        self.add_trashinfo_content(basename, content)

    def add_trashinfo_without_date(self, path):
        if False:
            for i in range(10):
                print('nop')
        basename = str(uuid.uuid4())
        content = trashinfo_content2([('Path', format_original_location(path))])
        self.add_trashinfo_content(basename, content)

    def add_trashinfo_wrong_date(self, path, wrong_date):
        if False:
            for i in range(10):
                print('nop')
        basename = str(uuid.uuid4())
        content = trashinfo_content2([('Path', format_original_location(path)), ('DeletionDate', wrong_date)])
        self.add_trashinfo_content(basename, content)

    def add_trashinfo_content(self, basename, content):
        if False:
            for i in range(10):
                print('nop')
        trashinfo_path = self.a_trashinfo_path(basename)
        make_parent_for(trashinfo_path)
        make_file(trashinfo_path, content)

    def ls_info(self):
        if False:
            while True:
                i = 10
        return os.listdir(self.info_path)

def trashinfo_content_default_date(path):
    if False:
        i = 10
        return i + 15
    return trashinfo_content(path, a_default_datetime())

def trashinfo_content(path, deletion_date):
    if False:
        print('Hello World!')
    return trashinfo_content2([('Path', format_original_location(path)), ('DeletionDate', deletion_date.strftime('%Y-%m-%dT%H:%M:%S'))])

def trashinfo_content2(values):
    if False:
        while True:
            i = 10
    return '[Trash Info]\n' + ''.join(('%s=%s\n' % (name, value) for (name, value) in values))