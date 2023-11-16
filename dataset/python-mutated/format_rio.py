"""A generator which creates a rio stanza of the current tree info"""
from __future__ import absolute_import
from bzrlib import hooks
from bzrlib.revision import NULL_REVISION
from bzrlib.rio import RioWriter, Stanza
from bzrlib.version_info_formats import create_date_str, VersionInfoBuilder

class RioVersionInfoBuilder(VersionInfoBuilder):
    """This writes a rio stream out."""

    def generate(self, to_file):
        if False:
            i = 10
            return i + 15
        info = Stanza()
        revision_id = self._get_revision_id()
        if revision_id != NULL_REVISION:
            info.add('revision-id', revision_id)
            rev = self._branch.repository.get_revision(revision_id)
            info.add('date', create_date_str(rev.timestamp, rev.timezone))
            revno = self._get_revno_str(revision_id)
            for hook in RioVersionInfoBuilder.hooks['revision']:
                hook(rev, info)
        else:
            revno = '0'
        info.add('build-date', create_date_str())
        info.add('revno', revno)
        if self._branch.nick is not None:
            info.add('branch-nick', self._branch.nick)
        if self._check or self._include_file_revs:
            self._extract_file_revisions()
        if self._check:
            if self._clean:
                info.add('clean', 'True')
            else:
                info.add('clean', 'False')
        if self._include_history:
            log = Stanza()
            for (revision_id, message, timestamp, timezone) in self._iter_revision_history():
                log.add('id', revision_id)
                log.add('message', message)
                log.add('date', create_date_str(timestamp, timezone))
            info.add('revisions', log.to_unicode())
        if self._include_file_revs:
            files = Stanza()
            for path in sorted(self._file_revisions.keys()):
                files.add('path', path)
                files.add('revision', self._file_revisions[path])
            info.add('file-revisions', files.to_unicode())
        writer = RioWriter(to_file=to_file)
        writer.write_stanza(info)

class RioVersionInfoBuilderHooks(hooks.Hooks):
    """Hooks for rio-formatted version-info output."""

    def __init__(self):
        if False:
            return 10
        super(RioVersionInfoBuilderHooks, self).__init__('bzrlib.version_info_formats.format_rio', 'RioVersionInfoBuilder.hooks')
        self.add_hook('revision', 'Invoked when adding information about a revision to the RIO stanza that is printed. revision is called with a revision object and a RIO stanza.', (1, 15))
RioVersionInfoBuilder.hooks = RioVersionInfoBuilderHooks()