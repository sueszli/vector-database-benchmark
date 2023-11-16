"""build_mo command for setup.py"""
from __future__ import absolute_import
from distutils import log
from distutils.core import Command
from distutils.dep_util import newer
from distutils.spawn import find_executable
import os
import re

class build_mo(Command):
    """Subcommand of build command: build_mo"""
    description = 'compile po files to mo files'
    user_options = [('build-dir=', 'd', 'Directory to build locale files'), ('output-base=', 'o', 'mo-files base name'), ('source-dir=', None, 'Directory with sources po files'), ('force', 'f', 'Force creation of mo files'), ('lang=', None, 'Comma-separated list of languages to process')]
    boolean_options = ['force']

    def initialize_options(self):
        if False:
            print('Hello World!')
        self.build_dir = None
        self.output_base = None
        self.source_dir = None
        self.force = None
        self.lang = None

    def finalize_options(self):
        if False:
            while True:
                i = 10
        self.set_undefined_options('build', ('force', 'force'))
        self.prj_name = self.distribution.get_name()
        if self.build_dir is None:
            self.build_dir = 'bzrlib/locale'
        if not self.output_base:
            self.output_base = self.prj_name or 'messages'
        if self.source_dir is None:
            self.source_dir = 'po'
        if self.lang is None:
            re_po = re.compile('^([a-zA-Z_]+)\\.po$')
            self.lang = []
            for i in os.listdir(self.source_dir):
                mo = re_po.match(i)
                if mo:
                    self.lang.append(mo.group(1))
        else:
            self.lang = [i.strip() for i in self.lang.split(',') if i.strip()]

    def run(self):
        if False:
            return 10
        'Run msgfmt for each language'
        if not self.lang:
            return
        if find_executable('msgfmt') is None:
            log.warn('GNU gettext msgfmt utility not found!')
            log.warn('Skip compiling po files.')
            return
        if 'en' in self.lang:
            if find_executable('msginit') is None:
                log.warn('GNU gettext msginit utility not found!')
                log.warn('Skip creating English PO file.')
            else:
                log.info('Creating English PO file...')
                pot = (self.prj_name or 'messages') + '.pot'
                en_po = 'en.po'
                self.spawn(['msginit', '--no-translator', '-l', 'en', '-i', os.path.join(self.source_dir, pot), '-o', os.path.join(self.source_dir, en_po)])
        basename = self.output_base
        if not basename.endswith('.mo'):
            basename += '.mo'
        for lang in self.lang:
            po = os.path.join('po', lang + '.po')
            if not os.path.isfile(po):
                po = os.path.join('po', lang + '.po')
            dir_ = os.path.join(self.build_dir, lang, 'LC_MESSAGES')
            self.mkpath(dir_)
            mo = os.path.join(dir_, basename)
            if self.force or newer(po, mo):
                log.info('Compile: %s -> %s' % (po, mo))
                self.spawn(['msgfmt', '-o', mo, po])