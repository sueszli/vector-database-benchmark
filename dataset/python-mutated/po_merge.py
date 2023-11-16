"""Merge logic for po_merge plugin."""
from __future__ import absolute_import
from bzrlib import config, merge
from bzrlib.lazy_import import lazy_import
lazy_import(globals(), '\nimport fnmatch\nimport subprocess\nimport tempfile\nimport sys\n\nfrom bzrlib import (\n    cmdline,\n    osutils,\n    trace,\n    )\n')
command_option = config.Option('po_merge.command', default='msgmerge -N "{other}" "{pot_file}" -C "{this}" -o "{result}"', help='Command used to create a conflict-free .po file during merge.\n\nThe following parameters are provided by the hook:\n``this`` is the ``.po`` file content before the merge in the current branch,\n``other`` is the ``.po`` file content in the branch merged from,\n``pot_file`` is the path to the ``.pot`` file corresponding to the ``.po``\nfile being merged.\n``result`` is the path where ``msgmerge`` will output its result. The hook will\nuse the content of this file to produce the resulting ``.po`` file.\n\nAll paths are absolute.\n')
po_dirs_option = config.ListOption('po_merge.po_dirs', default='po,debian/po', help='List of dirs containing .po files that the hook applies to.')
po_glob_option = config.Option('po_merge.po_glob', default='*.po', help='Glob matching all ``.po`` files in one of ``po_merge.po_dirs``.')
pot_glob_option = config.Option('po_merge.pot_glob', default='*.pot', help='Glob matching the ``.pot`` file in one of ``po_merge.po_dirs``.')

class PoMerger(merge.PerFileMerger):
    """Merge .po files."""

    def __init__(self, merger):
        if False:
            while True:
                i = 10
        super(merge.PerFileMerger, self).__init__(merger)
        self.conf = merger.this_branch.get_config_stack()
        self.po_dirs = self.conf.get('po_merge.po_dirs')
        self.po_glob = self.conf.get('po_merge.po_glob')
        self.pot_glob = self.conf.get('po_merge.pot_glob')
        self.command = self.conf.get('po_merge.command', expand=False)
        self.pot_file_abspath = None
        trace.mutter('PoMerger created')

    def file_matches(self, params):
        if False:
            i = 10
            return i + 15
        'Return True if merge_matching should be called on this file.'
        if not self.po_dirs or not self.command:
            return False
        po_dir = None
        po_path = self.get_filepath(params, self.merger.this_tree)
        for po_dir in self.po_dirs:
            glob = osutils.pathjoin(po_dir, self.po_glob)
            if fnmatch.fnmatch(po_path, glob):
                trace.mutter('po %s matches: %s' % (po_path, glob))
                break
        else:
            trace.mutter('PoMerger did not match for %s and %s' % (self.po_dirs, self.po_glob))
            return False
        for inv_entry in self.merger.this_tree.list_files(from_dir=po_dir, recursive=False):
            trace.mutter('inv_entry: %r' % (inv_entry,))
            (pot_name, pot_file_id) = (inv_entry[0], inv_entry[3])
            if fnmatch.fnmatch(pot_name, self.pot_glob):
                relpath = osutils.pathjoin(po_dir, pot_name)
                self.pot_file_abspath = self.merger.this_tree.abspath(relpath)
                trace.mutter('will msgmerge %s using %s' % (po_path, self.pot_file_abspath))
                return True
        else:
            return False

    def _invoke(self, command):
        if False:
            print('Hello World!')
        trace.mutter('Will msgmerge: %s' % (command,))
        proc = subprocess.Popen(cmdline.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        (out, err) = proc.communicate()
        return (proc.returncode, out, err)

    def merge_matching(self, params):
        if False:
            while True:
                i = 10
        return self.merge_text(params)

    def merge_text(self, params):
        if False:
            for i in range(10):
                print('nop')
        'Calls msgmerge when .po files conflict.\n\n        This requires a valid .pot file to reconcile both sides.\n        '
        tmpdir = tempfile.mkdtemp(prefix='po_merge')
        env = {}
        env['this'] = osutils.pathjoin(tmpdir, 'this')
        env['other'] = osutils.pathjoin(tmpdir, 'other')
        env['result'] = osutils.pathjoin(tmpdir, 'result')
        env['pot_file'] = self.pot_file_abspath
        try:
            with osutils.open_file(env['this'], 'wb') as f:
                f.writelines(params.this_lines)
            with osutils.open_file(env['other'], 'wb') as f:
                f.writelines(params.other_lines)
            command = self.conf.expand_options(self.command, env)
            (retcode, out, err) = self._invoke(command)
            with osutils.open_file(env['result']) as f:
                return ('success', list(f.readlines()))
        finally:
            osutils.rmtree(tmpdir)
        return ('not applicable', [])