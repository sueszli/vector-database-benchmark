"""
This module provides an object oriented interface for pattern matching
of files.
"""
from . import util
from .compat import Collection, iterkeys, izip_longest, string_types, unicode

class PathSpec(object):
    """
	The :class:`PathSpec` class is a wrapper around a list of compiled
	:class:`.Pattern` instances.
	"""

    def __init__(self, patterns):
        if False:
            return 10
        '\n\t\tInitializes the :class:`PathSpec` instance.\n\n\t\t*patterns* (:class:`~collections.abc.Collection` or :class:`~collections.abc.Iterable`)\n\t\tyields each compiled pattern (:class:`.Pattern`).\n\t\t'
        self.patterns = patterns if isinstance(patterns, Collection) else list(patterns)
        '\n\t\t*patterns* (:class:`~collections.abc.Collection` of :class:`.Pattern`)\n\t\tcontains the compiled patterns.\n\t\t'

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        '\n\t\tTests the equality of this path-spec with *other* (:class:`PathSpec`)\n\t\tby comparing their :attr:`~PathSpec.patterns` attributes.\n\t\t'
        if isinstance(other, PathSpec):
            paired_patterns = izip_longest(self.patterns, other.patterns)
            return all((a == b for (a, b) in paired_patterns))
        else:
            return NotImplemented

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n\t\tReturns the number of compiled patterns this path-spec contains\n\t\t(:class:`int`).\n\t\t'
        return len(self.patterns)

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        '\n\t\tCombines the :attr:`Pathspec.patterns` patterns from two\n\t\t:class:`PathSpec` instances.\n\t\t'
        if isinstance(other, PathSpec):
            return PathSpec(self.patterns + other.patterns)
        else:
            return NotImplemented

    def __iadd__(self, other):
        if False:
            return 10
        '\n\t\tAdds the :attr:`Pathspec.patterns` patterns from one :class:`PathSpec`\n\t\tinstance to this instance.\n\t\t'
        if isinstance(other, PathSpec):
            self.patterns += other.patterns
            return self
        else:
            return NotImplemented

    @classmethod
    def from_lines(cls, pattern_factory, lines):
        if False:
            print('Hello World!')
        '\n\t\tCompiles the pattern lines.\n\n\t\t*pattern_factory* can be either the name of a registered pattern\n\t\tfactory (:class:`str`), or a :class:`~collections.abc.Callable` used\n\t\tto compile patterns. It must accept an uncompiled pattern (:class:`str`)\n\t\tand return the compiled pattern (:class:`.Pattern`).\n\n\t\t*lines* (:class:`~collections.abc.Iterable`) yields each uncompiled\n\t\tpattern (:class:`str`). This simply has to yield each line so it can\n\t\tbe a :class:`file` (e.g., from :func:`open` or :class:`io.StringIO`)\n\t\tor the result from :meth:`str.splitlines`.\n\n\t\tReturns the :class:`PathSpec` instance.\n\t\t'
        if isinstance(pattern_factory, string_types):
            pattern_factory = util.lookup_pattern(pattern_factory)
        if not callable(pattern_factory):
            raise TypeError('pattern_factory:{!r} is not callable.'.format(pattern_factory))
        if not util._is_iterable(lines):
            raise TypeError('lines:{!r} is not an iterable.'.format(lines))
        lines = [pattern_factory(line) for line in lines if line]
        return cls(lines)

    def match_file(self, file, separators=None):
        if False:
            print('Hello World!')
        '\n\t\tMatches the file to this path-spec.\n\n\t\t*file* (:class:`str` or :class:`~pathlib.PurePath`) is the file path\n\t\tto be matched against :attr:`self.patterns <PathSpec.patterns>`.\n\n\t\t*separators* (:class:`~collections.abc.Collection` of :class:`str`)\n\t\toptionally contains the path separators to normalize. See\n\t\t:func:`~pathspec.util.normalize_file` for more information.\n\n\t\tReturns :data:`True` if *file* matched; otherwise, :data:`False`.\n\t\t'
        norm_file = util.normalize_file(file, separators=separators)
        return util.match_file(self.patterns, norm_file)

    def match_entries(self, entries, separators=None):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tMatches the entries to this path-spec.\n\n\t\t*entries* (:class:`~collections.abc.Iterable` of :class:`~util.TreeEntry`)\n\t\tcontains the entries to be matched against :attr:`self.patterns <PathSpec.patterns>`.\n\n\t\t*separators* (:class:`~collections.abc.Collection` of :class:`str`;\n\t\tor :data:`None`) optionally contains the path separators to\n\t\tnormalize. See :func:`~pathspec.util.normalize_file` for more\n\t\tinformation.\n\n\t\tReturns the matched entries (:class:`~collections.abc.Iterable` of\n\t\t:class:`~util.TreeEntry`).\n\t\t'
        if not util._is_iterable(entries):
            raise TypeError('entries:{!r} is not an iterable.'.format(entries))
        entry_map = util._normalize_entries(entries, separators=separators)
        match_paths = util.match_files(self.patterns, iterkeys(entry_map))
        for path in match_paths:
            yield entry_map[path]

    def match_files(self, files, separators=None):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tMatches the files to this path-spec.\n\n\t\t*files* (:class:`~collections.abc.Iterable` of :class:`str; or\n\t\t:class:`pathlib.PurePath`) contains the file paths to be matched\n\t\tagainst :attr:`self.patterns <PathSpec.patterns>`.\n\n\t\t*separators* (:class:`~collections.abc.Collection` of :class:`str`;\n\t\tor :data:`None`) optionally contains the path separators to\n\t\tnormalize. See :func:`~pathspec.util.normalize_file` for more\n\t\tinformation.\n\n\t\tReturns the matched files (:class:`~collections.abc.Iterable` of\n\t\t:class:`str`).\n\t\t'
        if not util._is_iterable(files):
            raise TypeError('files:{!r} is not an iterable.'.format(files))
        file_map = util.normalize_files(files, separators=separators)
        matched_files = util.match_files(self.patterns, iterkeys(file_map))
        for path in matched_files:
            yield file_map[path]

    def match_tree_entries(self, root, on_error=None, follow_links=None):
        if False:
            return 10
        '\n\t\tWalks the specified root path for all files and matches them to this\n\t\tpath-spec.\n\n\t\t*root* (:class:`str`; or :class:`pathlib.PurePath`) is the root\n\t\tdirectory to search.\n\n\t\t*on_error* (:class:`~collections.abc.Callable` or :data:`None`)\n\t\toptionally is the error handler for file-system exceptions. See\n\t\t:func:`~pathspec.util.iter_tree_entries` for more information.\n\n\t\t*follow_links* (:class:`bool` or :data:`None`) optionally is whether\n\t\tto walk symbolic links that resolve to directories. See\n\t\t:func:`~pathspec.util.iter_tree_files` for more information.\n\n\t\tReturns the matched files (:class:`~collections.abc.Iterable` of\n\t\t:class:`str`).\n\t\t'
        entries = util.iter_tree_entries(root, on_error=on_error, follow_links=follow_links)
        return self.match_entries(entries)

    def match_tree_files(self, root, on_error=None, follow_links=None):
        if False:
            return 10
        '\n\t\tWalks the specified root path for all files and matches them to this\n\t\tpath-spec.\n\n\t\t*root* (:class:`str`; or :class:`pathlib.PurePath`) is the root\n\t\tdirectory to search for files.\n\n\t\t*on_error* (:class:`~collections.abc.Callable` or :data:`None`)\n\t\toptionally is the error handler for file-system exceptions. See\n\t\t:func:`~pathspec.util.iter_tree_files` for more information.\n\n\t\t*follow_links* (:class:`bool` or :data:`None`) optionally is whether\n\t\tto walk symbolic links that resolve to directories. See\n\t\t:func:`~pathspec.util.iter_tree_files` for more information.\n\n\t\tReturns the matched files (:class:`~collections.abc.Iterable` of\n\t\t:class:`str`).\n\t\t'
        files = util.iter_tree_files(root, on_error=on_error, follow_links=follow_links)
        return self.match_files(files)
    match_tree = match_tree_files