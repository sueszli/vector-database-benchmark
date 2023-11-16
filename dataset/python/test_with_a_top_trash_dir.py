# Copyright (C) 2011 Andrea Francia Trivolzio(PV) Italy

import os
from datetime import datetime

from tests.test_list.cmd.setup import Setup

from tests.support.asserts import assert_equals_with_unidiff
from tests.fake_trash_dir import FakeTrashDir
from tests.support.files import  make_sticky_dir, make_unsticky_dir


class TestWithATopTrashDir(Setup):
    def setUp(self):
        super(type(self), self).setUp()
        self.top_trashdir1 = FakeTrashDir(self.top_dir / '.Trash/123')
        self.user.set_fake_uid(123)
        self.user.add_volume(self.top_dir)

    def test_should_list_its_contents_if_parent_is_sticky(self):
        make_sticky_dir(self.top_dir / '.Trash')
        self.and_contains_a_valid_trashinfo()

        self.user.run_trash_list()

        assert_equals_with_unidiff("2000-01-01 00:00:00 %s/file1\n" % self.top_dir,
                                   self.user.output())

    def test_and_should_warn_if_parent_is_not_sticky(self):
        make_unsticky_dir(self.top_dir / '.Trash')
        self.and_dir_exists(self.top_dir / '.Trash/123')

        self.user.run_trash_list()

        assert_equals_with_unidiff(
            "TrashDir skipped because parent not sticky: %s/.Trash/123\n" %
            self.top_dir,
            self.user.error()
        )

    def test_but_it_should_not_warn_when_the_parent_is_unsticky_but_there_is_no_trashdir(self):
        make_unsticky_dir(self.top_dir / '.Trash')
        self.but_does_not_exists_any(self.top_dir / '.Trash/123')

        self.user.run_trash_list()

        assert_equals_with_unidiff("", self.user.error())

    def test_should_ignore_trash_from_a_unsticky_topdir(self):
        make_unsticky_dir(self.top_dir / '.Trash')
        self.and_contains_a_valid_trashinfo()

        self.user.run_trash_list()

        assert_equals_with_unidiff('', self.user.output())

    def test_it_should_ignore_Trash_is_a_symlink(self):
        self.when_is_a_symlink_to_a_dir(self.top_dir / '.Trash')
        self.and_contains_a_valid_trashinfo()

        self.user.run_trash_list()

        assert_equals_with_unidiff('', self.user.output())

    def test_and_should_warn_about_it(self):
        self.when_is_a_symlink_to_a_dir(self.top_dir / '.Trash')
        self.and_contains_a_valid_trashinfo()

        self.user.run_trash_list()

        assert_equals_with_unidiff(
            'TrashDir skipped because parent not sticky: %s/.Trash/123\n' %
            self.top_dir,
            self.user.error()
        )

    def but_does_not_exists_any(self, path):
        assert not os.path.exists(path)

    def and_dir_exists(self, path):
        os.mkdir(path)
        assert os.path.isdir(path)

    def and_contains_a_valid_trashinfo(self):
        self.top_trashdir1.add_trashinfo2('file1', datetime(2000, 1, 1, 0, 0, 0))

    def when_is_a_symlink_to_a_dir(self, path):
        dest = "%s-dest" % path
        os.mkdir(dest)
        rel_dest = os.path.basename(dest)
        os.symlink(rel_dest, path)
