"""A generator which creates a template-based output from the current
   tree info."""
from __future__ import absolute_import
from bzrlib import errors
from bzrlib.revision import NULL_REVISION
from bzrlib.lazy_regex import lazy_compile
from bzrlib.version_info_formats import create_date_str, VersionInfoBuilder

class Template(object):
    """A simple template engine.

    >>> t = Template()
    >>> t.add('test', 'xxx')
    >>> print list(t.process('{test}'))
    ['xxx']
    >>> print list(t.process('{test} test'))
    ['xxx', ' test']
    >>> print list(t.process('test {test}'))
    ['test ', 'xxx']
    >>> print list(t.process('test {test} test'))
    ['test ', 'xxx', ' test']
    >>> print list(t.process('{test}\\\\n'))
    ['xxx', '\\n']
    >>> print list(t.process('{test}\\n'))
    ['xxx', '\\n']
    """
    _tag_re = lazy_compile('{(\\w+)}')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._data = {}

    def add(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        self._data[name] = value

    def process(self, tpl):
        if False:
            print('Hello World!')
        tpl = tpl.decode('string_escape')
        pos = 0
        while True:
            match = self._tag_re.search(tpl, pos)
            if not match:
                if pos < len(tpl):
                    yield tpl[pos:]
                break
            (start, end) = match.span()
            if start > 0:
                yield tpl[pos:start]
            pos = end
            name = match.group(1)
            try:
                data = self._data[name]
            except KeyError:
                raise errors.MissingTemplateVariable(name)
            if not isinstance(data, basestring):
                data = str(data)
            yield data

class CustomVersionInfoBuilder(VersionInfoBuilder):
    """Create a version file based on a custom template."""

    def generate(self, to_file):
        if False:
            print('Hello World!')
        if self._template is None:
            raise errors.NoTemplate()
        info = Template()
        info.add('build_date', create_date_str())
        info.add('branch_nick', self._branch.nick)
        revision_id = self._get_revision_id()
        if revision_id == NULL_REVISION:
            info.add('revno', 0)
        else:
            info.add('revno', self._get_revno_str(revision_id))
            info.add('revision_id', revision_id)
            rev = self._branch.repository.get_revision(revision_id)
            info.add('date', create_date_str(rev.timestamp, rev.timezone))
        if self._check:
            self._extract_file_revisions()
        if self._check:
            if self._clean:
                info.add('clean', 1)
            else:
                info.add('clean', 0)
        to_file.writelines(info.process(self._template))