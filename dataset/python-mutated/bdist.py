"""distutils.command.bdist

Implements the Distutils 'bdist' command (create a built [binary]
distribution)."""
import os
from distutils.core import Command
from distutils.errors import *
from distutils.util import get_platform

def show_formats():
    if False:
        for i in range(10):
            print('nop')
    'Print list of available formats (arguments to "--format" option).\n    '
    from distutils.fancy_getopt import FancyGetopt
    formats = []
    for format in bdist.format_commands:
        formats.append(('formats=' + format, None, bdist.format_command[format][1]))
    pretty_printer = FancyGetopt(formats)
    pretty_printer.print_help('List of available distribution formats:')

class bdist(Command):
    description = 'create a built (binary) distribution'
    user_options = [('bdist-base=', 'b', 'temporary directory for creating built distributions'), ('plat-name=', 'p', 'platform name to embed in generated filenames (default: %s)' % get_platform()), ('formats=', None, 'formats for distribution (comma-separated list)'), ('dist-dir=', 'd', 'directory to put final built distributions in [default: dist]'), ('skip-build', None, 'skip rebuilding everything (for testing/debugging)'), ('owner=', 'u', 'Owner name used when creating a tar file [default: current user]'), ('group=', 'g', 'Group name used when creating a tar file [default: current group]')]
    boolean_options = ['skip-build']
    help_options = [('help-formats', None, 'lists available distribution formats', show_formats)]
    no_format_option = ('bdist_rpm',)
    default_format = {'posix': 'gztar', 'nt': 'zip'}
    format_commands = ['rpm', 'gztar', 'bztar', 'xztar', 'ztar', 'tar', 'zip', 'msi']
    format_command = {'rpm': ('bdist_rpm', 'RPM distribution'), 'gztar': ('bdist_dumb', "gzip'ed tar file"), 'bztar': ('bdist_dumb', "bzip2'ed tar file"), 'xztar': ('bdist_dumb', "xz'ed tar file"), 'ztar': ('bdist_dumb', 'compressed tar file'), 'tar': ('bdist_dumb', 'tar file'), 'zip': ('bdist_dumb', 'ZIP file'), 'msi': ('bdist_msi', 'Microsoft Installer')}

    def initialize_options(self):
        if False:
            return 10
        self.bdist_base = None
        self.plat_name = None
        self.formats = None
        self.dist_dir = None
        self.skip_build = 0
        self.group = None
        self.owner = None

    def finalize_options(self):
        if False:
            i = 10
            return i + 15
        if self.plat_name is None:
            if self.skip_build:
                self.plat_name = get_platform()
            else:
                self.plat_name = self.get_finalized_command('build').plat_name
        if self.bdist_base is None:
            build_base = self.get_finalized_command('build').build_base
            self.bdist_base = os.path.join(build_base, 'bdist.' + self.plat_name)
        self.ensure_string_list('formats')
        if self.formats is None:
            try:
                self.formats = [self.default_format[os.name]]
            except KeyError:
                raise DistutilsPlatformError("don't know how to create built distributions on platform %s" % os.name)
        if self.dist_dir is None:
            self.dist_dir = 'dist'

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        commands = []
        for format in self.formats:
            try:
                commands.append(self.format_command[format][0])
            except KeyError:
                raise DistutilsOptionError("invalid format '%s'" % format)
        for i in range(len(self.formats)):
            cmd_name = commands[i]
            sub_cmd = self.reinitialize_command(cmd_name)
            if cmd_name not in self.no_format_option:
                sub_cmd.format = self.formats[i]
            if cmd_name == 'bdist_dumb':
                sub_cmd.owner = self.owner
                sub_cmd.group = self.group
            if cmd_name in commands[i + 1:]:
                sub_cmd.keep_temp = 1
            self.run_command(cmd_name)