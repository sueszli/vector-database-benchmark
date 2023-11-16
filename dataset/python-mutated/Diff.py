import copy
import difflib
import logging
from unidiff import PatchSet
from coalib.results.LineDiff import LineDiff, ConflictError
from coalib.results.SourceRange import SourceRange
from coalib.results.TextRange import TextRange
from coala_utils.decorators import enforce_signature, generate_eq

@generate_eq('_file', 'modified', 'rename', 'delete')
class Diff:
    """
    A Diff result represents a difference for one file.
    """

    def __init__(self, file_list, rename=False, delete=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates an empty diff for the given file.\n\n        :param file_list: The original (unmodified) file as a list of its\n                          lines.\n        :param rename:    False or str containing new name of file.\n        :param delete:    True if file is set to be deleted.\n        '
        self._changes = {}
        self._file = list(file_list)
        self._original = self._generate_linebreaks(self._file)
        self.rename = rename
        self.delete = delete

    @classmethod
    def from_string_arrays(cls, file_array_1, file_array_2, rename=False):
        if False:
            while True:
                i = 10
        '\n        Creates a Diff object from two arrays containing strings.\n\n        If this Diff is applied to the original array, the second array will be\n        created.\n\n        :param file_array_1: Original array\n        :param file_array_2: Array to compare\n        :param rename:       False or str containing new name of file.\n        '
        result = cls(file_array_1, rename=rename)
        matcher = difflib.SequenceMatcher(None, file_array_1, file_array_2)
        for change_group in matcher.get_grouped_opcodes(1):
            for (tag, a_index_1, a_index_2, b_index_1, b_index_2) in change_group:
                if tag == 'delete':
                    for index in range(a_index_1 + 1, a_index_2 + 1):
                        result.delete_line(index)
                elif tag == 'insert':
                    result.add_lines(a_index_1, file_array_2[b_index_1:b_index_2])
                elif tag == 'replace':
                    result.modify_line(a_index_1 + 1, file_array_2[b_index_1])
                    result.add_lines(a_index_1 + 1, file_array_2[b_index_1 + 1:b_index_2])
                    for index in range(a_index_1 + 2, a_index_2 + 1):
                        result.delete_line(index)
        return result

    @classmethod
    def from_unified_diff(cls, unified_diff, original_file):
        if False:
            i = 10
            return i + 15
        '\n        Creates a ``Diff`` object from given unified diff.\n\n        If the provided unified diff does not contain any patch,\n        the ``Diff`` object initialized from the original file is\n        returned.\n\n        :param unified_diff:  Unified diff string.\n        :param original_file: The contents of the original file\n                              (line-splitted).\n        :raises RuntimeError: Raised when the context lines or the\n                              lines to be removed do not match in\n                              the original file and the unified diff.\n        '
        patch_set = PatchSet(unified_diff.splitlines())
        diff = Diff(original_file)
        if not patch_set:
            return diff
        file = patch_set[0]
        for hunk in file:
            file_line = hunk.source_start
            hunk_iterator = iter(hunk)
            try:
                while True:
                    line = next(hunk_iterator)
                    source_code = str(line)[1:]
                    if line.is_added:
                        add_set = []
                        try:
                            while line.is_added:
                                add_set.append(source_code)
                                line = next(hunk_iterator)
                                source_code = str(line)[1:]
                            diff.add_lines(file_line - 1, add_set)
                        except StopIteration:
                            diff.add_lines(file_line - 1, add_set)
                            break
                    original_line = original_file[file_line - 1].rstrip('\n')
                    if line.is_removed:
                        if source_code != original_line:
                            raise RuntimeError(f'The line to delete does not match with the line in the original file. Line to delete: {source_code!r}, Original line #{file_line!r}: {original_line!r}')
                        diff.delete_line(file_line)
                    elif source_code != original_line:
                        raise RuntimeError(f'Context lines do not match. Line from unified diff: {source_code!r}, Original line #{file_line!r}: {original_line!r}')
                    file_line += 1
            except StopIteration:
                pass
        return diff

    def _get_change(self, line_nr, min_line=1):
        if False:
            while True:
                i = 10
        if not isinstance(line_nr, int):
            raise TypeError('line_nr needs to be an integer.')
        if line_nr < min_line:
            raise IndexError('The given line number is not allowed.')
        return self._changes.get(line_nr, LineDiff())

    def stats(self):
        if False:
            while True:
                i = 10
        '\n        Returns tuple containing number of additions and deletions in the diff.\n        '
        additions = 0
        deletions = 0
        for line_diff in self._changes.values():
            if line_diff.change:
                additions += 1
                deletions += 1
            elif line_diff.delete:
                deletions += 1
            if line_diff.add_after:
                additions += len(line_diff.add_after)
        return (additions, deletions)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns total number of additions and deletions in diff.\n        '
        return sum(self.stats())

    @property
    def rename(self):
        if False:
            return 10
        '\n        :return: string containing new name of the file.\n        '
        return self._rename

    @rename.setter
    @enforce_signature
    def rename(self, rename: (str, False)):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param rename: False or string containing new name of file.\n        '
        self._rename = rename

    @property
    def delete(self):
        if False:
            return 10
        '\n        :return: True if file is set to be deleted.\n        '
        return self._delete

    @delete.setter
    @enforce_signature
    def delete(self, delete: bool):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param delete: True if file is set to be deleted, False otherwise.\n        '
        self._delete = delete

    @property
    def original(self):
        if False:
            i = 10
            return i + 15
        '\n        Retrieves the original file.\n        '
        return self._original

    def _raw_modified(self):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the modified file, after applying the Diff to the original.\n        '
        result = []
        if self.delete:
            return result
        current_line = 0
        for line_nr in sorted(self._changes):
            result.extend(self._file[current_line:max(line_nr - 1, 0)])
            linediff = self._changes[line_nr]
            if not linediff.delete and (not linediff.change) and (line_nr > 0):
                result.append(self._file[line_nr - 1])
            elif linediff.change:
                result.append(linediff.change[1])
            if linediff.add_after:
                result.extend(linediff.add_after)
            current_line = line_nr
        result.extend(self._file[current_line:])
        return result

    @property
    def modified(self):
        if False:
            i = 10
            return i + 15
        '\n        Calculates the modified file, after applying the Diff to the original.\n\n        This property also adds linebreaks at the end of each line.\n        If no newline was present at the end of file before, this state will\n        be preserved, except if the last line is deleted.\n        '
        return self._generate_linebreaks(self._raw_modified())

    @property
    def unified_diff(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates a unified diff corresponding to this patch.\n\n        Each change will be displayed on its own line. Additionally, the\n        unified diff preserves the EOF-state of the original file. This\n        means that the ``Diff`` will only have a linebreak on the last line,\n        if that was also present in the original file.\n\n        Note that the unified diff is not deterministic and thus not suitable\n        for equality comparison.\n        '
        list_unified_diff = list(difflib.unified_diff(self._file, self._raw_modified(), tofile=self.rename if isinstance(self.rename, str) else ''))
        return ''.join(self._generate_linebreaks(list_unified_diff))

    def __json__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Override JSON export, using the unified diff is the easiest thing for\n        the users.\n        '
        return self.unified_diff

    def affected_code(self, filename):
        if False:
            return 10
        "\n        Creates a list of SourceRange objects which point to the related code.\n        Changes on continuous lines will be put into one SourceRange.\n\n        :param filename: The filename to associate the SourceRange's to.\n        :return:         A list of all related SourceRange objects.\n        "
        return list((diff.range(filename) for diff in self.split_diff(distance=0)))

    def split_diff(self, distance=1):
        if False:
            print('Hello World!')
        "\n        Splits this diff into small pieces, such that several continuously\n        altered lines are still together in one diff. All subdiffs will be\n        yielded.\n\n        A diff like this with changes being together closely won't be splitted:\n\n        >>> diff = Diff.from_string_arrays([     'b', 'c', 'e'],\n        ...                                ['a', 'b', 'd', 'f'])\n        >>> len(list(diff.split_diff()))\n        1\n\n        If we set the distance to 0, it will be splitted:\n\n        >>> len(list(diff.split_diff(distance=0)))\n        2\n\n        If ``-1`` is given, every change will be yielded as an own\n        diff, even if they are right beneath each other:\n\n        >>> len(list(diff.split_diff(distance=-1)))\n        3\n\n        If a file gets renamed or deleted only, it will be yielded as is:\n\n        >>> len(list(Diff([], rename='test').split_diff()))\n        1\n\n        An empty diff will not yield any diffs:\n\n        >>> len(list(Diff([]).split_diff()))\n        0\n\n        :param distance: Number of unchanged lines that are allowed in between\n                         two changed lines so they get yielded as one diff.\n        "
        if not self:
            return
        last_line = -1
        this_diff = Diff(self._file, rename=self.rename, delete=self.delete)
        for line in sorted(self._changes.keys()):
            if line > last_line + distance + 1 and len(this_diff._changes) > 0:
                yield this_diff
                this_diff = Diff(self._file, rename=self.rename, delete=self.delete)
            last_line = line
            this_diff._changes[line] = self._changes[line]
        yield this_diff

    def range(self, filename):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates a SourceRange spanning over the whole Diff. If something is\n        added after the 0th line (i.e. before the first line) the first line\n        will be included in the SourceRange.\n\n        The range of an empty diff will only affect the filename:\n\n        >>> range = Diff([]).range("file")\n        >>> range.file is None\n        False\n        >>> print(range.start.line)\n        None\n\n        :param filename: The filename to associate the SourceRange with.\n        :return:         A SourceRange object.\n        '
        if len(self._changes) == 0:
            return SourceRange.from_values(filename)
        start = min(self._changes.keys())
        end = max(self._changes.keys())
        return SourceRange.from_values(filename, start_line=max(1, start), end_line=max(1, end))

    def __add__(self, other):
        if False:
            while True:
                i = 10
        '\n        Adds another diff to this one. Will throw an exception if this is not\n        possible. (This will *not* be done in place.)\n        '
        if not isinstance(other, Diff):
            raise TypeError('Only diffs can be added to a diff.')
        if self.rename != other.rename and False not in (self.rename, other.rename):
            raise ConflictError('Diffs contain conflicting renamings.')
        result = copy.deepcopy(self)
        result.rename = self.rename or other.rename
        result.delete = self.delete or other.delete
        for line_nr in other._changes:
            change = other._changes[line_nr]
            if change.delete is True:
                result.delete_line(line_nr)
            if change.add_after is not False:
                result.add_lines(line_nr, change.add_after)
            if change.change is not False:
                result.modify_line(line_nr, change.change[1])
        return result

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        '\n        >>> bool(Diff([]))\n        False\n        >>> bool(Diff([], rename="some"))\n        True\n        >>> bool(Diff([], delete=True))\n        True\n        >>> bool(Diff.from_string_arrays([\'1\'], []))\n        True\n\n        :return: False if the patch has no effect at all when applied.\n        '
        return self.rename is not False or self.delete is True or self.modified != self.original

    def delete_line(self, line_nr):
        if False:
            i = 10
            return i + 15
        "\n        Mark the given line nr as deleted. The first line is line number 1.\n\n        Raises an exception if line number doesn't exist in the diff.\n        "
        if line_nr > len(self._file):
            raise IndexError('The given line number is out of bounds.')
        linediff = self._get_change(line_nr)
        linediff.delete = True
        self._changes[line_nr] = linediff

    def delete_lines(self, line_nr_start, line_nr_end):
        if False:
            return 10
        '\n        Delete lines in a specified range, inclusively.\n\n        The range must be valid, i.e. lines must exist in diff, else an\n        exception is raised.\n        '
        for line_nr in range(line_nr_start, line_nr_end + 1):
            self.delete_line(line_nr)

    def add_lines(self, line_nr_before, lines):
        if False:
            return 10
        '\n        Adds lines after the given line number.\n\n        :param line_nr_before: Line number of the line before the additions.\n                               Use 0 for insert lines before everything.\n        :param lines:          A list of lines to add.\n        '
        if lines == []:
            return
        linediff = self._get_change(line_nr_before, min_line=0)
        if linediff.add_after is not False:
            raise ConflictError('Cannot add lines after the given line since there are already lines.')
        linediff.add_after = lines
        self._changes[line_nr_before] = linediff

    def add_line(self, line_nr_before, line):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds line after the given line number.\n\n        :param line_nr_before: Line number of the line before the addition.\n                               Use 0 to insert line before everything.\n        :param line:           Line to add.\n        '
        return self.add_lines(line_nr_before, [line])

    def modify_line(self, line_nr, replacement):
        if False:
            print('Hello World!')
        '\n        Changes the given line with the given line number. The replacement will\n        be there instead.\n\n        Given an empty diff object:\n\n        >>> diff = Diff([\'Hey there! Gorgeous.\\n\',\n        ...              "It\'s nice that we\'re here.\\n"])\n\n        We can change a line easily:\n\n        >>> diff.modify_line(1,\n        ...                  \'Hey there! This is sad.\\n\')\n        >>> diff.modified\n        [\'Hey there! This is sad.\\n\', "It\'s nice that we\'re here.\\n"]\n\n        We can even merge changes within one line:\n\n        >>> diff.modify_line(1,\n        ...                  \'Hello. :( Gorgeous.\\n\')\n        >>> diff.modified\n        [\'Hello. :( This is sad.\\n\', "It\'s nice that we\'re here.\\n"]\n\n        However, if we change something that has been changed before, we\'ll get\n        a conflict:\n\n        >>> diff.modify_line(1, \'Hello. This is not ok. Gorgeous.\\n\')\n        Traceback (most recent call last):\n         ...\n        coalib.results.LineDiff.ConflictError: ...\n        '
        linediff = self._get_change(line_nr)
        if linediff.change is not False and linediff.change[1] != replacement:
            if len(replacement) == len(linediff.change[1]) == 1:
                raise ConflictError('Cannot merge the given line changes.')
            orig_diff = Diff.from_string_arrays(linediff.change[0], linediff.change[1])
            new_diff = Diff.from_string_arrays(linediff.change[0], replacement)
            replacement = ''.join((orig_diff + new_diff)._raw_modified())
        linediff.change = (self._file[line_nr - 1], replacement)
        self._changes[line_nr] = linediff

    def change_line(self, line_nr, original_line, replacement):
        if False:
            return 10
        logging.debug('Use of change_line method is deprecated. Instead use modify_line method, without the original_line argument')
        self.modify_line(line_nr, replacement)

    def replace(self, range, replacement):
        if False:
            for i in range(10):
                print('nop')
        "\n        Replaces a part of text. Allows to span multiple lines.\n\n        This function uses ``add_lines`` and ``delete_lines`` accordingly, so\n        calls of those functions on lines given ``range`` affects after usage\n        or vice versa lead to ``ConflictError``.\n\n        >>> from coalib.results.TextRange import TextRange\n        >>> test_text = ['hello\\n', 'world\\n', '4lines\\n', 'done\\n']\n        >>> def replace(range, text):\n        ...     diff = Diff(test_text)\n        ...     diff.replace(range, text)\n        ...     return diff.modified\n        >>> replace(TextRange.from_values(1, 5, 4, 3), '\\nyeah\\ncool\\nno')\n        ['hell\\n', 'yeah\\n', 'cool\\n', 'none\\n']\n        >>> replace(TextRange.from_values(2, 1, 3, 5), 'b')\n        ['hello\\n', 'bes\\n', 'done\\n']\n        >>> replace(TextRange.from_values(1, 6, 4, 3), '')\n        ['hellone\\n']\n\n        :param range:       The ``TextRange`` that gets replaced.\n        :param replacement: The replacement string. Can be multiline.\n        "
        first_part = self._file[range.start.line - 1][:range.start.column - 1]
        last_part = self._file[range.end.line - 1][range.end.column - 1:]
        self.delete_lines(range.start.line, range.end.line)
        self.add_lines(range.start.line - 1, (first_part + replacement + last_part).splitlines(True))

    def insert(self, position, text):
        if False:
            print('Hello World!')
        "\n        Inserts (multiline) text at arbitrary position.\n\n        >>> from coalib.results.TextPosition import TextPosition\n        >>> test_text = ['123\\n', '456\\n', '789\\n']\n        >>> def insert(position, text):\n        ...     diff = Diff(test_text)\n        ...     diff.insert(position, text)\n        ...     return diff.modified\n        >>> insert(TextPosition(2, 3), 'woopy doopy')\n        ['123\\n', '45woopy doopy6\\n', '789\\n']\n        >>> insert(TextPosition(1, 1), 'woopy\\ndoopy')\n        ['woopy\\n', 'doopy123\\n', '456\\n', '789\\n']\n        >>> insert(TextPosition(2, 4), '\\nwoopy\\ndoopy\\n')\n        ['123\\n', '456\\n', 'woopy\\n', 'doopy\\n', '\\n', '789\\n']\n\n        :param position: The ``TextPosition`` where to insert text.\n        :param text:     The text to insert.\n        "
        self.replace(TextRange(position, position), text)

    def remove(self, range):
        if False:
            while True:
                i = 10
        "\n        Removes a piece of text in a given range.\n\n        >>> from coalib.results.TextRange import TextRange\n        >>> test_text = ['nice\\n', 'try\\n', 'bro\\n']\n        >>> def remove(range):\n        ...     diff = Diff(test_text)\n        ...     diff.remove(range)\n        ...     return diff.modified\n        >>> remove(TextRange.from_values(1, 1, 1, 4))\n        ['e\\n', 'try\\n', 'bro\\n']\n        >>> remove(TextRange.from_values(1, 5, 2, 1))\n        ['nicetry\\n', 'bro\\n']\n        >>> remove(TextRange.from_values(1, 3, 3, 2))\n        ['niro\\n']\n        >>> remove(TextRange.from_values(2, 1, 2, 1))\n        ['nice\\n', 'try\\n', 'bro\\n']\n\n        :param range: The range to delete.\n        "
        self.replace(range, '')

    @staticmethod
    def _add_linebreaks(lines):
        if False:
            print('Hello World!')
        '\n        Validate that each line in lines ends with a\n        newline character and appends one if that is not the case.\n\n        :param lines: A list of strings, representing lines.\n        '
        return [line if line.endswith('\n') else line + '\n' for line in lines]

    @staticmethod
    def _generate_linebreaks(lines):
        if False:
            return 10
        '\n        Validate that each line in lines ends with a\n        newline character and appends one if that is not the case.\n        Exception is the last line in the list.\n\n        :param lines: A list of strings, representing lines.\n        '
        if lines == []:
            return []
        return Diff._add_linebreaks(lines[:-1]) + [lines[-1]]