import os
from bzrlib import merge, tests
from bzrlib.tests import features, script
from bzrlib.plugins import po_merge

class BlackboxTestPoMerger(script.TestCaseWithTransportAndScript):
    _test_needs_features = [features.msgmerge_feature]

    def setUp(self):
        if False:
            print('Hello World!')
        super(BlackboxTestPoMerger, self).setUp()
        self.builder = make_adduser_branch(self, 'adduser')
        merge.Merger.hooks.install_named_hook('merge_file_content', po_merge.po_merge_hook, '.po file merge')

    def test_merge_with_hook_gives_unexpected_results(self):
        if False:
            return 10
        self.run_script('$ bzr branch adduser -rrevid:this work\n2>Branched 2 revisions.\n$ cd work\n$ bzr merge ../adduser -rrevid:other\n2> M  po/adduser.pot\n2> M  po/fr.po\n2>Text conflict in po/adduser.pot\n2>1 conflicts encountered.\n')

    def test_called_on_remerge(self):
        if False:
            while True:
                i = 10
        self.run_script('$ bzr branch adduser -rrevid:this work\n2>Branched 2 revisions.\n$ cd work\n# set po_dirs to an empty list\n$ bzr merge ../adduser -rrevid:other -Opo_merge.po_dirs=\n2> M  po/adduser.pot\n2> M  po/fr.po\n2>Text conflict in po/adduser.pot\n2>Text conflict in po/fr.po\n2>2 conflicts encountered.\n')
        with open('po/adduser.pot', 'w') as f:
            f.write(_Adduser['resolved_pot'])
        self.run_script('$ bzr resolve po/adduser.pot\n2>1 conflict resolved, 1 remaining\n# Use remerge to trigger the hook, we use the default config options here\n$ bzr remerge po/*.po\n2>All changes applied successfully.\n# There should be no conflicts anymore\n$ bzr conflicts\n')

def make_adduser_branch(test, relpath):
    if False:
        print('Hello World!')
    'Helper for po_merge blackbox tests.\n\n    This creates a branch containing the needed base revisions so tests can\n    attempt merges and conflict resolutions.\n    '
    builder = test.make_branch_builder(relpath)
    builder.start_series()
    builder.build_snapshot('base', None, [('add', ('', 'root-id', 'directory', '')), ('add', ('po', 'dir-id', 'directory', None)), ('add', ('po/adduser.pot', 'pot-id', 'file', _Adduser['base_pot'])), ('add', ('po/fr.po', 'po-id', 'file', _Adduser['base_po']))])
    builder.build_snapshot('other', ['base'], [('modify', ('pot-id', _Adduser['other_pot'])), ('modify', ('po-id', _Adduser['other_po']))])
    builder.build_snapshot('this', ['base'], [('modify', ('pot-id', _Adduser['this_pot'])), ('modify', ('po-id', _Adduser['this_po']))])
    builder.finish_series()
    return builder

class TestAdduserBranch(script.TestCaseWithTransportAndScript):
    """Sanity checks on the adduser branch content."""

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestAdduserBranch, self).setUp()
        self.builder = make_adduser_branch(self, 'adduser')

    def assertAdduserBranchContent(self, revid):
        if False:
            while True:
                i = 10
        env = dict(revid=revid, branch_name=revid)
        self.run_script('$ bzr branch adduser -rrevid:%(revid)s %(branch_name)s\n' % env, null_output_matches_anything=True)
        self.assertFileEqual(_Adduser['%(revid)s_pot' % env], '%(branch_name)s/po/adduser.pot' % env)
        self.assertFileEqual(_Adduser['%(revid)s_po' % env], '%(branch_name)s/po/fr.po' % env)

    def test_base(self):
        if False:
            print('Hello World!')
        self.assertAdduserBranchContent('base')

    def test_this(self):
        if False:
            i = 10
            return i + 15
        self.assertAdduserBranchContent('this')

    def test_other(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAdduserBranchContent('other')
_Adduser = dict(base_pot='# SOME DESCRIPTIVE TITLE.\n# Copyright (C) YEAR THE PACKAGE\'S COPYRIGHT HOLDER\n# This file is distributed under the same license as the PACKAGE package.\n# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.\n#\n#, fuzzy\nmsgid ""\nmsgstr ""\n"Project-Id-Version: PACKAGE VERSION\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2007-01-17 21:50+0100\\n"\n"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"\n"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"\n"Language-Team: LANGUAGE <LL@example.com>\\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=CHARSET\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:135\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n\n#: ../adduser:188\nmsgid "Warning: The home dir you specified already exists.\\n"\nmsgstr ""\n\n', this_pot='# SOME DESCRIPTIVE TITLE.\n# Copyright (C) YEAR THE PACKAGE\'S COPYRIGHT HOLDER\n# This file is distributed under the same license as the PACKAGE package.\n# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.\n#\n#, fuzzy\nmsgid ""\nmsgstr ""\n"Project-Id-Version: PACKAGE VERSION\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2011-01-06 21:06+0000\\n"\n"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"\n"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"\n"Language-Team: LANGUAGE <LL@example.com>\\n"\n"Language: \\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=CHARSET\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:152\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n\n#: ../adduser:208\n#, perl-format\nmsgid "Warning: The home dir %s you specified already exists.\\n"\nmsgstr ""\n\n#: ../adduser:210\n#, perl-format\nmsgid "Warning: The home dir %s you specified can\'t be accessed: %s\\n"\nmsgstr ""\n\n', other_pot='# SOME DESCRIPTIVE TITLE.\n# Copyright (C) YEAR THE PACKAGE\'S COPYRIGHT HOLDER\n# This file is distributed under the same license as the PACKAGE package.\n# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.\n#\n#, fuzzy\nmsgid ""\nmsgstr ""\n"Project-Id-Version: PACKAGE VERSION\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2010-11-21 17:13-0400\\n"\n"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"\n"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"\n"Language-Team: LANGUAGE <LL@example.com>\\n"\n"Language: \\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=CHARSET\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:150\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n\n#: ../adduser:206\n#, perl-format\nmsgid "Warning: The home dir %s you specified already exists.\\n"\nmsgstr ""\n\n#: ../adduser:208\n#, perl-format\nmsgid "Warning: The home dir %s you specified can\'t be accessed: %s\\n"\nmsgstr ""\n\n', resolved_pot='# SOME DESCRIPTIVE TITLE.\n# Copyright (C) YEAR THE PACKAGE\'S COPYRIGHT HOLDER\n# This file is distributed under the same license as the PACKAGE package.\n# FIRST AUTHOR <EMAIL@ADDRESS>, YEAR.\n#\n#, fuzzy\nmsgid ""\nmsgstr ""\n"Project-Id-Version: PACKAGE VERSION\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2011-10-19 12:50-0700\\n"\n"PO-Revision-Date: YEAR-MO-DA HO:MI+ZONE\\n"\n"Last-Translator: FULL NAME <EMAIL@ADDRESS>\\n"\n"Language-Team: LANGUAGE <LL@example.com>\\n"\n"Language: \\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=CHARSET\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:152\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n\n#: ../adduser:208\n#, perl-format\nmsgid "Warning: The home dir %s you specified already exists.\\n"\nmsgstr ""\n\n#: ../adduser:210\n#, perl-format\nmsgid "Warning: The home dir %s you specified can\'t be accessed: %s\\n"\nmsgstr ""\n\n', base_po='# adduser\'s manpages translation to French\n# Copyright (C) 2004 Software in the Public Interest\n# This file is distributed under the same license as the adduser package\n#\n# Translators:\n# Jean-Baka Domelevo Entfellner <domelevo@example.com>, 2009.\n#\nmsgid ""\nmsgstr ""\n"Project-Id-Version: adduser 3.111\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2007-01-17 21:50+0100\\n"\n"PO-Revision-Date: 2010-01-21 10:36+0100\\n"\n"Last-Translator: Jean-Baka Domelevo Entfellner <domelevo@example.com>\\n"\n"Language-Team: Debian French Team <debian-l10n-french@example.com>\\n"\n"Language: \\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=UTF-8\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n"X-Poedit-Language: French\\n"\n"X-Poedit-Country: FRANCE\\n"\n\n# type: Plain text\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:135\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n"Seul le superutilisateur est autorisé à ajouter un utilisateur ou un groupe "\n"au système.\\n"\n\n#: ../adduser:188\nmsgid "Warning: The home dir you specified already exists.\\n"\nmsgstr ""\n"Attention\xa0! Le répertoire personnel que vous avez indiqué existe déjà.\\n"\n\n', this_po='# adduser\'s manpages translation to French\n# Copyright (C) 2004 Software in the Public Interest\n# This file is distributed under the same license as the adduser package\n#\n# Translators:\n# Jean-Baka Domelevo Entfellner <domelevo@example.com>, 2009.\n#\nmsgid ""\nmsgstr ""\n"Project-Id-Version: adduser 3.111\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2010-10-12 15:48+0200\\n"\n"PO-Revision-Date: 2010-01-21 10:36+0100\\n"\n"Last-Translator: Jean-Baka Domelevo Entfellner <domelevo@example.com>\\n"\n"Language-Team: Debian French Team <debian-l10n-french@example.com>\\n"\n"Language: \\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=UTF-8\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n"X-Poedit-Language: French\\n"\n"X-Poedit-Country: FRANCE\\n"\n\n# type: Plain text\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:152\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n"Seul le superutilisateur est autorisé à ajouter un utilisateur ou un groupe "\n"au système.\\n"\n\n#: ../adduser:208\n#, fuzzy, perl-format\nmsgid "Warning: The home dir %s you specified already exists.\\n"\nmsgstr ""\n"Attention\xa0! Le répertoire personnel que vous avez indiqué existe déjà.\\n"\n\n#: ../adduser:210\n#, fuzzy, perl-format\nmsgid "Warning: The home dir %s you specified can\'t be accessed: %s\\n"\nmsgstr ""\n"Attention\xa0! Le répertoire personnel que vous avez indiqué existe déjà.\\n"\n\n', other_po='# adduser\'s manpages translation to French\n# Copyright (C) 2004 Software in the Public Interest\n# This file is distributed under the same license as the adduser package\n#\n# Translators:\n# Jean-Baka Domelevo Entfellner <domelevo@example.com>, 2009, 2010.\n#\nmsgid ""\nmsgstr ""\n"Project-Id-Version: adduser 3.112+nmu2\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2010-11-21 17:13-0400\\n"\n"PO-Revision-Date: 2010-11-10 11:08+0100\\n"\n"Last-Translator: Jean-Baka Domelevo-Entfellner <domelevo@example.com>\\n"\n"Language-Team: Debian French Team <debian-l10n-french@example.com>\\n"\n"Language: \\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=UTF-8\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n"X-Poedit-Country: FRANCE\\n"\n\n# type: Plain text\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:150\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n"Seul le superutilisateur est autorisé à ajouter un utilisateur ou un groupe "\n"au système.\\n"\n\n#: ../adduser:206\n#, perl-format\nmsgid "Warning: The home dir %s you specified already exists.\\n"\nmsgstr ""\n"Attention\xa0! Le répertoire personnel que vous avez indiqué (%s) existe déjà.\\n"\n\n#: ../adduser:208\n#, perl-format\nmsgid "Warning: The home dir %s you specified can\'t be accessed: %s\\n"\nmsgstr ""\n"Attention\xa0! Impossible d\'accéder au répertoire personnel que vous avez "\n"indiqué (%s)\xa0: %s.\\n"\n\n', resolved_po='# adduser\'s manpages translation to French\n# Copyright (C) 2004 Software in the Public Interest\n# This file is distributed under the same license as the adduser package\n#\n# Translators:\n# Jean-Baka Domelevo Entfellner <domelevo@example.com>, 2009, 2010.\n#\nmsgid ""\nmsgstr ""\n"Project-Id-Version: adduser 3.112+nmu2\\n"\n"Report-Msgid-Bugs-To: adduser-devel@example.com\\n"\n"POT-Creation-Date: 2011-10-19 12:50-0700\\n"\n"PO-Revision-Date: 2010-11-10 11:08+0100\\n"\n"Last-Translator: Jean-Baka Domelevo-Entfellner <domelevo@example.com>\\n"\n"Language-Team: Debian French Team <debian-l10n-french@example.com>\\n"\n"Language: \\n"\n"MIME-Version: 1.0\\n"\n"Content-Type: text/plain; charset=UTF-8\\n"\n"Content-Transfer-Encoding: 8bit\\n"\n"X-Poedit-Country: FRANCE\\n"\n\n# type: Plain text\n#. everyone can issue "--help" and "--version", but only root can go on\n#: ../adduser:152\nmsgid "Only root may add a user or group to the system.\\n"\nmsgstr ""\n"Seul le superutilisateur est autorisé à ajouter un utilisateur ou un groupe "\n"au système.\\n"\n\n#: ../adduser:208\n#, perl-format\nmsgid "Warning: The home dir %s you specified already exists.\\n"\nmsgstr ""\n"Attention\xa0! Le répertoire personnel que vous avez indiqué (%s) existe déjà.\\n"\n\n#: ../adduser:210\n#, perl-format\nmsgid "Warning: The home dir %s you specified can\'t be accessed: %s\\n"\nmsgstr ""\n"Attention\xa0! Impossible d\'accéder au répertoire personnel que vous avez "\n"indiqué (%s)\xa0: %s.\\n"\n\n')