"""Commands for generating snapshot information about a bzr tree."""
from __future__ import absolute_import
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nfrom bzrlib import (\n    branch,\n    errors,\n    version_info_formats,\n    workingtree,\n    )\nfrom bzrlib.i18n import gettext\n')
from bzrlib.commands import Command
from bzrlib.option import Option, RegistryOption

def _parse_version_info_format(format):
    if False:
        return 10
    'Convert a string passed by the user into a VersionInfoFormat.\n\n    This looks in the version info format registry, and if the format\n    cannot be found, generates a useful error exception.\n    '
    try:
        return version_info_formats.get_builder(format)
    except KeyError:
        formats = version_info_formats.get_builder_formats()
        raise errors.BzrCommandError(gettext('No known version info format {0}. Supported types are: {1}').format(format, formats))

class cmd_version_info(Command):
    __doc__ = 'Show version information about this tree.\n\n    You can use this command to add information about version into\n    source code of an application. The output can be in one of the\n    supported formats or in a custom format based on a template.\n\n    For example::\n\n      bzr version-info --custom \\\n        --template="#define VERSION_INFO \\"Project 1.2.3 (r{revno})\\"\\n"\n\n    will produce a C header file with formatted string containing the\n    current revision number. Other supported variables in templates are:\n\n      * {date} - date of the last revision\n      * {build_date} - current date\n      * {revno} - revision number\n      * {revision_id} - revision id\n      * {branch_nick} - branch nickname\n      * {clean} - 0 if the source tree contains uncommitted changes,\n                  otherwise 1\n    '
    takes_options = [RegistryOption('format', 'Select the output format.', value_switches=True, lazy_registry=('bzrlib.version_info_formats', 'format_registry')), Option('all', help='Include all possible information.'), Option('check-clean', help='Check if tree is clean.'), Option('include-history', help='Include the revision-history.'), Option('include-file-revisions', help='Include the last revision for each file.'), Option('template', type=str, help='Template for the output.'), 'revision']
    takes_args = ['location?']
    encoding_type = 'exact'

    def run(self, location=None, format=None, all=False, check_clean=False, include_history=False, include_file_revisions=False, template=None, revision=None):
        if False:
            for i in range(10):
                print('nop')
        if revision and len(revision) > 1:
            raise errors.BzrCommandError(gettext('bzr version-info --revision takes exactly one revision specifier'))
        if location is None:
            location = '.'
        if format is None:
            format = version_info_formats.format_registry.get()
        try:
            wt = workingtree.WorkingTree.open_containing(location)[0]
        except errors.NoWorkingTree:
            b = branch.Branch.open(location)
            wt = None
        else:
            b = wt.branch
        if all:
            include_history = True
            check_clean = True
            include_file_revisions = True
        if template:
            include_history = True
            include_file_revisions = True
            if '{clean}' in template:
                check_clean = True
        if revision is not None:
            revision_id = revision[0].as_revision_id(b)
        else:
            revision_id = None
        builder = format(b, working_tree=wt, check_for_clean=check_clean, include_revision_history=include_history, include_file_revisions=include_file_revisions, template=template, revision_id=revision_id)
        builder.generate(self.outf)