"""Tests for generating docs (man pages).

This test checks that generation will be successful
and produce non-empty output.
"""
from cStringIO import StringIO
import bzrlib.commands
from bzrlib.tests import TestCase

class Options:
    """Simply container"""
    pass

class TestGenerateDocs(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestGenerateDocs, self).setUp()
        self.sio = StringIO()
        self.options = Options()
        self.options.bzr_name = 'bzr'
        bzrlib.commands.install_bzr_command_hooks()

    def test_man_page(self):
        if False:
            while True:
                i = 10
        from bzrlib.doc_generate import autodoc_man
        autodoc_man.infogen(self.options, self.sio)
        self.assertNotEqual('', self.sio.getvalue())

    def test_rstx_man(self):
        if False:
            while True:
                i = 10
        from bzrlib.doc_generate import autodoc_rstx
        autodoc_rstx.infogen(self.options, self.sio)
        self.assertNotEqual('', self.sio.getvalue())