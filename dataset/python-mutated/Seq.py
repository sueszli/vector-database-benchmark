"""Provide objects to represent biological sequences.

See also the Seq_ wiki and the chapter in our tutorial:
 - `HTML Tutorial`_
 - `PDF Tutorial`_

.. _Seq: http://biopython.org/wiki/Seq
.. _`HTML Tutorial`: http://biopython.org/DIST/docs/tutorial/Tutorial.html
.. _`PDF Tutorial`: http://biopython.org/DIST/docs/tutorial/Tutorial.pdf

"""
import array
import collections
import numbers
import warnings
from abc import ABC
from abc import abstractmethod
from typing import overload, Optional, Union, Dict
from Bio import BiopythonDeprecationWarning
from Bio import BiopythonWarning
from Bio.Data import CodonTable
from Bio.Data import IUPACData

def _maketrans(complement_mapping):
    if False:
        print('Hello World!')
    "Make a python string translation table (PRIVATE).\n\n    Arguments:\n     - complement_mapping - a dictionary such as ambiguous_dna_complement\n       and ambiguous_rna_complement from Data.IUPACData.\n\n    Returns a translation table (a bytes object of length 256) for use with\n    the python string's translate method to use in a (reverse) complement.\n\n    Compatible with lower case and upper case sequences.\n\n    For internal use only.\n    "
    keys = ''.join(complement_mapping.keys()).encode('ASCII')
    values = ''.join(complement_mapping.values()).encode('ASCII')
    return bytes.maketrans(keys + keys.lower(), values + values.lower())
ambiguous_dna_complement = dict(IUPACData.ambiguous_dna_complement)
ambiguous_dna_complement['U'] = ambiguous_dna_complement['T']
_dna_complement_table = _maketrans(ambiguous_dna_complement)
del ambiguous_dna_complement
ambiguous_rna_complement = dict(IUPACData.ambiguous_rna_complement)
ambiguous_rna_complement['T'] = ambiguous_rna_complement['U']
_rna_complement_table = _maketrans(ambiguous_rna_complement)
del ambiguous_rna_complement

class SequenceDataAbstractBaseClass(ABC):
    """Abstract base class for sequence content providers.

    Most users will not need to use this class. It is used internally as a base
    class for sequence content provider classes such as _UndefinedSequenceData
    defined in this module, and _TwoBitSequenceData in Bio.SeqIO.TwoBitIO.
    Instances of these classes can be used instead of a ``bytes`` object as the
    data argument when creating a Seq object, and provide the sequence content
    only when requested via ``__getitem__``. This allows lazy parsers to load
    and parse sequence data from a file only for the requested sequence regions,
    and _UndefinedSequenceData instances to raise an exception when undefined
    sequence data are requested.

    Future implementations of lazy parsers that similarly provide on-demand
    parsing of sequence data should use a subclass of this abstract class and
    implement the abstract methods ``__len__`` and ``__getitem__``:

    * ``__len__`` must return the sequence length;
    * ``__getitem__`` must return

      * a ``bytes`` object for the requested region; or
      * a new instance of the subclass for the requested region; or
      * raise an ``UndefinedSequenceError``.

      Calling ``__getitem__`` for a sequence region of size zero should always
      return an empty ``bytes`` object.
      Calling ``__getitem__`` for the full sequence (as in data[:]) should
      either return a ``bytes`` object with the full sequence, or raise an
      ``UndefinedSequenceError``.

    Subclasses of SequenceDataAbstractBaseClass must call ``super().__init__()``
    as part of their ``__init__`` method.
    """
    __slots__ = ()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if ``__getitem__`` returns a bytes-like object.'
        assert self[:0] == b''

    @abstractmethod
    def __len__(self):
        if False:
            print('Hello World!')
        pass

    @abstractmethod
    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        pass

    def __bytes__(self):
        if False:
            i = 10
            return i + 15
        return self[:]

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(bytes(self))

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return bytes(self) == other

    def __lt__(self, other):
        if False:
            return 10
        return bytes(self) < other

    def __le__(self, other):
        if False:
            while True:
                i = 10
        return bytes(self) <= other

    def __gt__(self, other):
        if False:
            return 10
        return bytes(self) > other

    def __ge__(self, other):
        if False:
            i = 10
            return i + 15
        return bytes(self) >= other

    def __add__(self, other):
        if False:
            while True:
                i = 10
        try:
            return bytes(self) + bytes(other)
        except UndefinedSequenceError:
            return NotImplemented

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return other + bytes(self)

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return other * bytes(self)

    def __contains__(self, item):
        if False:
            print('Hello World!')
        return bytes(self).__contains__(item)

    def decode(self, encoding='utf-8'):
        if False:
            print('Hello World!')
        'Decode the data as bytes using the codec registered for encoding.\n\n        encoding\n          The encoding with which to decode the bytes.\n        '
        return bytes(self).decode(encoding)

    def count(self, sub, start=None, end=None):
        if False:
            while True:
                i = 10
        'Return the number of non-overlapping occurrences of sub in data[start:end].\n\n        Optional arguments start and end are interpreted as in slice notation.\n        This method behaves as the count method of Python strings.\n        '
        return bytes(self).count(sub, start, end)

    def find(self, sub, start=None, end=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the lowest index in data where subsection sub is found.\n\n        Return the lowest index in data where subsection sub is found,\n        such that sub is contained within data[start,end].  Optional\n        arguments start and end are interpreted as in slice notation.\n\n        Return -1 on failure.\n        '
        return bytes(self).find(sub, start, end)

    def rfind(self, sub, start=None, end=None):
        if False:
            i = 10
            return i + 15
        'Return the highest index in data where subsection sub is found.\n\n        Return the highest index in data where subsection sub is found,\n        such that sub is contained within data[start,end].  Optional\n        arguments start and end are interpreted as in slice notation.\n\n        Return -1 on failure.\n        '
        return bytes(self).rfind(sub, start, end)

    def index(self, sub, start=None, end=None):
        if False:
            print('Hello World!')
        'Return the lowest index in data where subsection sub is found.\n\n        Return the lowest index in data where subsection sub is found,\n        such that sub is contained within data[start,end].  Optional\n        arguments start and end are interpreted as in slice notation.\n\n        Raises ValueError when the subsection is not found.\n        '
        return bytes(self).index(sub, start, end)

    def rindex(self, sub, start=None, end=None):
        if False:
            print('Hello World!')
        'Return the highest index in data where subsection sub is found.\n\n        Return the highest index in data where subsection sub is found,\n        such that sub is contained within data[start,end].  Optional\n        arguments start and end are interpreted as in slice notation.\n\n        Raise ValueError when the subsection is not found.\n        '
        return bytes(self).rindex(sub, start, end)

    def startswith(self, prefix, start=None, end=None):
        if False:
            return 10
        'Return True if data starts with the specified prefix, False otherwise.\n\n        With optional start, test data beginning at that position.\n        With optional end, stop comparing data at that position.\n        prefix can also be a tuple of bytes to try.\n        '
        return bytes(self).startswith(prefix, start, end)

    def endswith(self, suffix, start=None, end=None):
        if False:
            for i in range(10):
                print('nop')
        'Return True if data ends with the specified suffix, False otherwise.\n\n        With optional start, test data beginning at that position.\n        With optional end, stop comparing data at that position.\n        suffix can also be a tuple of bytes to try.\n        '
        return bytes(self).endswith(suffix, start, end)

    def split(self, sep=None, maxsplit=-1):
        if False:
            while True:
                i = 10
        'Return a list of the sections in the data, using sep as the delimiter.\n\n        sep\n          The delimiter according which to split the data.\n          None (the default value) means split on ASCII whitespace characters\n          (space, tab, return, newline, formfeed, vertical tab).\n        maxsplit\n          Maximum number of splits to do.\n          -1 (the default value) means no limit.\n        '
        return bytes(self).split(sep, maxsplit)

    def rsplit(self, sep=None, maxsplit=-1):
        if False:
            return 10
        'Return a list of the sections in the data, using sep as the delimiter.\n\n        sep\n          The delimiter according which to split the data.\n          None (the default value) means split on ASCII whitespace characters\n          (space, tab, return, newline, formfeed, vertical tab).\n        maxsplit\n          Maximum number of splits to do.\n          -1 (the default value) means no limit.\n\n        Splitting is done starting at the end of the data and working to the front.\n        '
        return bytes(self).rsplit(sep, maxsplit)

    def strip(self, chars=None):
        if False:
            for i in range(10):
                print('nop')
        'Strip leading and trailing characters contained in the argument.\n\n        If the argument is omitted or None, strip leading and trailing ASCII whitespace.\n        '
        return bytes(self).strip(chars)

    def lstrip(self, chars=None):
        if False:
            return 10
        'Strip leading characters contained in the argument.\n\n        If the argument is omitted or None, strip leading ASCII whitespace.\n        '
        return bytes(self).lstrip(chars)

    def rstrip(self, chars=None):
        if False:
            return 10
        'Strip trailing characters contained in the argument.\n\n        If the argument is omitted or None, strip trailing ASCII whitespace.\n        '
        return bytes(self).rstrip(chars)

    def removeprefix(self, prefix):
        if False:
            while True:
                i = 10
        'Remove the prefix if present.'
        data = bytes(self)
        try:
            return data.removeprefix(prefix)
        except AttributeError:
            if data.startswith(prefix):
                return data[len(prefix):]
            else:
                return data

    def removesuffix(self, suffix):
        if False:
            i = 10
            return i + 15
        'Remove the suffix if present.'
        data = bytes(self)
        try:
            return data.removesuffix(suffix)
        except AttributeError:
            if data.startswith(suffix):
                return data[:-len(suffix)]
            else:
                return data

    def upper(self):
        if False:
            i = 10
            return i + 15
        'Return a copy of data with all ASCII characters converted to uppercase.'
        return bytes(self).upper()

    def lower(self):
        if False:
            while True:
                i = 10
        'Return a copy of data with all ASCII characters converted to lowercase.'
        return bytes(self).lower()

    def isupper(self):
        if False:
            print('Hello World!')
        'Return True if all ASCII characters in data are uppercase.\n\n        If there are no cased characters, the method returns False.\n        '
        return bytes(self).isupper()

    def islower(self):
        if False:
            i = 10
            return i + 15
        'Return True if all ASCII characters in data are lowercase.\n\n        If there are no cased characters, the method returns False.\n        '
        return bytes(self).islower()

    def replace(self, old, new):
        if False:
            print('Hello World!')
        'Return a copy with all occurrences of substring old replaced by new.'
        return bytes(self).replace(old, new)

    def translate(self, table, delete=b''):
        if False:
            print('Hello World!')
        'Return a copy with each character mapped by the given translation table.\n\n          table\n            Translation table, which must be a bytes object of length 256.\n\n        All characters occurring in the optional argument delete are removed.\n        The remaining characters are mapped through the given translation table.\n        '
        return bytes(self).translate(table, delete)

    @property
    def defined(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if the sequence is defined, False if undefined or partially defined.\n\n        Zero-length sequences are always considered to be defined.\n        '
        return True

    @property
    def defined_ranges(self):
        if False:
            return 10
        'Return a tuple of the ranges where the sequence contents is defined.\n\n        The return value has the format ((start1, end1), (start2, end2), ...).\n        '
        length = len(self)
        if length > 0:
            return ((0, length),)
        else:
            return ()

class _SeqAbstractBaseClass(ABC):
    """Abstract base class for the Seq and MutableSeq classes (PRIVATE).

    Most users will not need to use this class. It is used internally as an
    abstract base class for Seq and MutableSeq, as most of their methods are
    identical.
    """
    __slots__ = ('_data',)
    __array_ufunc__ = None

    @abstractmethod
    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __bytes__(self):
        if False:
            print('Hello World!')
        return bytes(self._data)

    def __repr__(self):
        if False:
            print('Hello World!')
        'Return (truncated) representation of the sequence.'
        data = self._data
        if isinstance(data, _UndefinedSequenceData):
            return f'Seq(None, length={len(self)})'
        if isinstance(data, _PartiallyDefinedSequenceData):
            d = {}
            for (position, seq) in data._data.items():
                if len(seq) > 60:
                    start = seq[:54].decode('ASCII')
                    end = seq[-3:].decode('ASCII')
                    seq = f'{start}...{end}'
                else:
                    seq = seq.decode('ASCII')
                d[position] = seq
            return 'Seq(%r, length=%d)' % (d, len(self))
        if len(data) > 60:
            start = data[:54].decode('ASCII')
            end = data[-3:].decode('ASCII')
            return f"{self.__class__.__name__}('{start}...{end}')"
        else:
            data = data.decode('ASCII')
            return f"{self.__class__.__name__}('{data}')"

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Return the full sequence as a python string.'
        return self._data.decode('ASCII')

    def __eq__(self, other):
        if False:
            print('Hello World!')
        'Compare the sequence to another sequence or a string.\n\n        Sequences are equal to each other if their sequence contents is\n        identical:\n\n        >>> from Bio.Seq import Seq, MutableSeq\n        >>> seq1 = Seq("ACGT")\n        >>> seq2 = Seq("ACGT")\n        >>> mutable_seq = MutableSeq("ACGT")\n        >>> seq1 == seq2\n        True\n        >>> seq1 == mutable_seq\n        True\n        >>> seq1 == "ACGT"\n        True\n\n        Note that the sequence objects themselves are not identical to each\n        other:\n\n        >>> id(seq1) == id(seq2)\n        False\n        >>> seq1 is seq2\n        False\n\n        Sequences can also be compared to strings, ``bytes``, and ``bytearray``\n        objects:\n\n        >>> seq1 == "ACGT"\n        True\n        >>> seq1 == b"ACGT"\n        True\n        >>> seq1 == bytearray(b"ACGT")\n        True\n        '
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data == other._data
        elif isinstance(other, str):
            return self._data == other.encode('ASCII')
        else:
            return self._data == other

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        'Implement the less-than operand.'
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data < other._data
        elif isinstance(other, str):
            return self._data < other.encode('ASCII')
        else:
            return self._data < other

    def __le__(self, other):
        if False:
            print('Hello World!')
        'Implement the less-than or equal operand.'
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data <= other._data
        elif isinstance(other, str):
            return self._data <= other.encode('ASCII')
        else:
            return self._data <= other

    def __gt__(self, other):
        if False:
            print('Hello World!')
        'Implement the greater-than operand.'
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data > other._data
        elif isinstance(other, str):
            return self._data > other.encode('ASCII')
        else:
            return self._data > other

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        'Implement the greater-than or equal operand.'
        if isinstance(other, _SeqAbstractBaseClass):
            return self._data >= other._data
        elif isinstance(other, str):
            return self._data >= other.encode('ASCII')
        else:
            return self._data >= other

    def __len__(self):
        if False:
            return 10
        'Return the length of the sequence.'
        return len(self._data)

    def __iter__(self):
        if False:
            print('Hello World!')
        'Return an iterable of the sequence.'
        return self._data.decode('ASCII').__iter__()

    @overload
    def __getitem__(self, index: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def __getitem__(self, index: slice) -> 'Seq':
        if False:
            while True:
                i = 10
        ...

    def __getitem__(self, index):
        if False:
            return 10
        "Return a subsequence as a single letter or as a sequence object.\n\n        If the index is an integer, a single letter is returned as a Python\n        string:\n\n        >>> seq = Seq('ACTCGACGTCG')\n        >>> seq[5]\n        'A'\n\n        Otherwise, a new sequence object of the same class is returned:\n\n        >>> seq[5:8]\n        Seq('ACG')\n        >>> mutable_seq = MutableSeq('ACTCGACGTCG')\n        >>> mutable_seq[5:8]\n        MutableSeq('ACG')\n        "
        if isinstance(index, numbers.Integral):
            return chr(self._data[index])
        else:
            return self.__class__(self._data[index])

    def __add__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Add a sequence or string to this sequence.\n\n        >>> from Bio.Seq import Seq, MutableSeq\n        >>> Seq("MELKI") + "LV"\n        Seq(\'MELKILV\')\n        >>> MutableSeq("MELKI") + "LV"\n        MutableSeq(\'MELKILV\')\n        '
        if isinstance(other, _SeqAbstractBaseClass):
            return self.__class__(self._data + other._data)
        elif isinstance(other, str):
            return self.__class__(self._data + other.encode('ASCII'))
        else:
            return NotImplemented

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        'Add a sequence string on the left.\n\n        >>> from Bio.Seq import Seq, MutableSeq\n        >>> "LV" + Seq("MELKI")\n        Seq(\'LVMELKI\')\n        >>> "LV" + MutableSeq("MELKI")\n        MutableSeq(\'LVMELKI\')\n\n        Adding two sequence objects is handled via the __add__ method.\n        '
        if isinstance(other, str):
            return self.__class__(other.encode('ASCII') + self._data)
        else:
            return NotImplemented

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        "Multiply sequence by integer.\n\n        >>> from Bio.Seq import Seq, MutableSeq\n        >>> Seq('ATG') * 2\n        Seq('ATGATG')\n        >>> MutableSeq('ATG') * 2\n        MutableSeq('ATGATG')\n        "
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"can't multiply {self.__class__.__name__} by non-int type")
        data = self._data.__mul__(other)
        return self.__class__(data)

    def __rmul__(self, other):
        if False:
            return 10
        "Multiply integer by sequence.\n\n        >>> from Bio.Seq import Seq\n        >>> 2 * Seq('ATG')\n        Seq('ATGATG')\n        "
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"can't multiply {self.__class__.__name__} by non-int type")
        data = self._data.__mul__(other)
        return self.__class__(data)

    def __imul__(self, other):
        if False:
            print('Hello World!')
        "Multiply the sequence object by other and assign.\n\n        >>> from Bio.Seq import Seq\n        >>> seq = Seq('ATG')\n        >>> seq *= 2\n        >>> seq\n        Seq('ATGATG')\n\n        Note that this is different from in-place multiplication. The ``seq``\n        variable is reassigned to the multiplication result, but any variable\n        pointing to ``seq`` will remain unchanged:\n\n        >>> seq = Seq('ATG')\n        >>> seq2 = seq\n        >>> id(seq) == id(seq2)\n        True\n        >>> seq *= 2\n        >>> seq\n        Seq('ATGATG')\n        >>> seq2\n        Seq('ATG')\n        >>> id(seq) == id(seq2)\n        False\n        "
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"can't multiply {self.__class__.__name__} by non-int type")
        data = self._data.__mul__(other)
        return self.__class__(data)

    def count(self, sub, start=None, end=None):
        if False:
            while True:
                i = 10
        'Return a non-overlapping count, like that of a python string.\n\n        The number of occurrences of substring argument sub in the\n        (sub)sequence given by [start:end] is returned as an integer.\n        Optional arguments start and end are interpreted as in slice\n        notation.\n\n        Arguments:\n         - sub - a string or another Seq object to look for\n         - start - optional integer, slice start\n         - end - optional integer, slice end\n\n        e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> my_seq = Seq("AAAATGA")\n        >>> print(my_seq.count("A"))\n        5\n        >>> print(my_seq.count("ATG"))\n        1\n        >>> print(my_seq.count(Seq("AT")))\n        1\n        >>> print(my_seq.count("AT", 2, -1))\n        1\n\n        HOWEVER, please note because the ``count`` method of Seq and MutableSeq\n        objects, like that of Python strings, do a non-overlapping search, this\n        may not give the answer you expect:\n\n        >>> "AAAA".count("AA")\n        2\n        >>> print(Seq("AAAA").count("AA"))\n        2\n\n        For an overlapping search, use the ``count_overlap`` method:\n\n        >>> print(Seq("AAAA").count_overlap("AA"))\n        3\n        '
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.count(sub, start, end)

    def count_overlap(self, sub, start=None, end=None):
        if False:
            print('Hello World!')
        'Return an overlapping count.\n\n        Returns an integer, the number of occurrences of substring\n        argument sub in the (sub)sequence given by [start:end].\n        Optional arguments start and end are interpreted as in slice\n        notation.\n\n        Arguments:\n         - sub - a string or another Seq object to look for\n         - start - optional integer, slice start\n         - end - optional integer, slice end\n\n        e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> print(Seq("AAAA").count_overlap("AA"))\n        3\n        >>> print(Seq("ATATATATA").count_overlap("ATA"))\n        4\n        >>> print(Seq("ATATATATA").count_overlap("ATA", 3, -1))\n        1\n\n        For a non-overlapping search, use the ``count`` method:\n\n        >>> print(Seq("AAAA").count("AA"))\n        2\n\n        Where substrings do not overlap, ``count_overlap`` behaves the same as\n        the ``count`` method:\n\n        >>> from Bio.Seq import Seq\n        >>> my_seq = Seq("AAAATGA")\n        >>> print(my_seq.count_overlap("A"))\n        5\n        >>> my_seq.count_overlap("A") == my_seq.count("A")\n        True\n        >>> print(my_seq.count_overlap("ATG"))\n        1\n        >>> my_seq.count_overlap("ATG") == my_seq.count("ATG")\n        True\n        >>> print(my_seq.count_overlap(Seq("AT")))\n        1\n        >>> my_seq.count_overlap(Seq("AT")) == my_seq.count(Seq("AT"))\n        True\n        >>> print(my_seq.count_overlap("AT", 2, -1))\n        1\n        >>> my_seq.count_overlap("AT", 2, -1) == my_seq.count("AT", 2, -1)\n        True\n\n        HOWEVER, do not use this method for such cases because the\n        count() method is much for efficient.\n        '
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        data = self._data
        overlap_count = 0
        while True:
            start = data.find(sub, start, end) + 1
            if start != 0:
                overlap_count += 1
            else:
                return overlap_count

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Return True if item is a subsequence of the sequence, and False otherwise.\n\n        e.g.\n\n        >>> from Bio.Seq import Seq, MutableSeq\n        >>> my_dna = Seq("ATATGAAATTTGAAAA")\n        >>> "AAA" in my_dna\n        True\n        >>> Seq("AAA") in my_dna\n        True\n        >>> MutableSeq("AAA") in my_dna\n        True\n        '
        if isinstance(item, _SeqAbstractBaseClass):
            item = bytes(item)
        elif isinstance(item, str):
            item = item.encode('ASCII')
        return item in self._data

    def find(self, sub, start=None, end=None):
        if False:
            return 10
        'Return the lowest index in the sequence where subsequence sub is found.\n\n        With optional arguments start and end, return the lowest index in the\n        sequence such that the subsequence sub is contained within the sequence\n        region [start:end].\n\n        Arguments:\n         - sub - a string or another Seq or MutableSeq object to search for\n         - start - optional integer, slice start\n         - end - optional integer, slice end\n\n        Returns -1 if the subsequence is NOT found.\n\n        e.g. Locating the first typical start codon, AUG, in an RNA sequence:\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_rna.find("AUG")\n        3\n\n        The next typical start codon can then be found by starting the search\n        at position 4:\n\n        >>> my_rna.find("AUG", 4)\n        15\n\n        See the ``search`` method to find the locations of multiple subsequences\n        at the same time.\n        '
        if isinstance(sub, _SeqAbstractBaseClass):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.find(sub, start, end)

    def rfind(self, sub, start=None, end=None):
        if False:
            print('Hello World!')
        'Return the highest index in the sequence where subsequence sub is found.\n\n        With optional arguments start and end, return the highest index in the\n        sequence such that the subsequence sub is contained within the sequence\n        region [start:end].\n\n        Arguments:\n         - sub - a string or another Seq or MutableSeq object to search for\n         - start - optional integer, slice start\n         - end - optional integer, slice end\n\n        Returns -1 if the subsequence is NOT found.\n\n        e.g. Locating the last typical start codon, AUG, in an RNA sequence:\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_rna.rfind("AUG")\n        15\n\n        The location of the typical start codon before that can be found by\n        ending the search at position 15:\n\n        >>> my_rna.rfind("AUG", end=15)\n        3\n\n        See the ``search`` method to find the locations of multiple subsequences\n        at the same time.\n        '
        if isinstance(sub, _SeqAbstractBaseClass):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.rfind(sub, start, end)

    def index(self, sub, start=None, end=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the lowest index in the sequence where subsequence sub is found.\n\n        With optional arguments start and end, return the lowest index in the\n        sequence such that the subsequence sub is contained within the sequence\n        region [start:end].\n\n        Arguments:\n         - sub - a string or another Seq or MutableSeq object to search for\n         - start - optional integer, slice start\n         - end - optional integer, slice end\n\n        Raises a ValueError if the subsequence is NOT found.\n\n        e.g. Locating the first typical start codon, AUG, in an RNA sequence:\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_rna.index("AUG")\n        3\n\n        The next typical start codon can then be found by starting the search\n        at position 4:\n\n        >>> my_rna.index("AUG", 4)\n        15\n\n        This method performs the same search as the ``find`` method.  However,\n        if the subsequence is not found, ``find`` returns -1 while ``index``\n        raises a ValueError:\n\n        >>> my_rna.index("T")\n        Traceback (most recent call last):\n                   ...\n        ValueError: ...\n        >>> my_rna.find("T")\n        -1\n\n        See the ``search`` method to find the locations of multiple subsequences\n        at the same time.\n        '
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.index(sub, start, end)

    def rindex(self, sub, start=None, end=None):
        if False:
            while True:
                i = 10
        'Return the highest index in the sequence where subsequence sub is found.\n\n        With optional arguments start and end, return the highest index in the\n        sequence such that the subsequence sub is contained within the sequence\n        region [start:end].\n\n        Arguments:\n         - sub - a string or another Seq or MutableSeq object to search for\n         - start - optional integer, slice start\n         - end - optional integer, slice end\n\n        Returns -1 if the subsequence is NOT found.\n\n        e.g. Locating the last typical start codon, AUG, in an RNA sequence:\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_rna.rindex("AUG")\n        15\n\n        The location of the typical start codon before that can be found by\n        ending the search at position 15:\n\n        >>> my_rna.rindex("AUG", end=15)\n        3\n\n        This method performs the same search as the ``rfind`` method.  However,\n        if the subsequence is not found, ``rfind`` returns -1 which ``rindex``\n        raises a ValueError:\n\n        >>> my_rna.rindex("T")\n        Traceback (most recent call last):\n                   ...\n        ValueError: ...\n        >>> my_rna.rfind("T")\n        -1\n\n        See the ``search`` method to find the locations of multiple subsequences\n        at the same time.\n        '
        if isinstance(sub, MutableSeq):
            sub = sub._data
        elif isinstance(sub, Seq):
            sub = bytes(sub)
        elif isinstance(sub, str):
            sub = sub.encode('ASCII')
        elif not isinstance(sub, (bytes, bytearray)):
            raise TypeError("a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % type(sub))
        return self._data.rindex(sub, start, end)

    def search(self, subs):
        if False:
            print('Hello World!')
        'Search the substrings subs in self and yield the index and substring found.\n\n        Arguments:\n         - subs - a list of strings, Seq, MutableSeq, bytes, or bytearray\n           objects containing the substrings to search for.\n\n        >>> from Bio.Seq import Seq\n        >>> dna = Seq("GTCATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAGTTG")\n        >>> matches = dna.search(["CC", Seq("ATTG"), "ATTG", Seq("CCC")])\n        >>> for index, substring in matches:\n        ...     print(index, substring)\n        ...\n        7 CC\n        9 ATTG\n        20 CC\n        34 CC\n        34 CCC\n        35 CC\n        '
        subdict = collections.defaultdict(set)
        for (index, sub) in enumerate(subs):
            if isinstance(sub, (_SeqAbstractBaseClass, bytearray)):
                sub = bytes(sub)
            elif isinstance(sub, str):
                sub = sub.encode('ASCII')
            elif not isinstance(sub, bytes):
                raise TypeError("subs[%d]: a Seq, MutableSeq, str, bytes, or bytearray object is required, not '%s'" % (index, type(sub)))
            length = len(sub)
            subdict[length].add(sub)
        for start in range(len(self) - 1):
            for (length, subs) in subdict.items():
                stop = start + length
                for sub in subs:
                    if self._data[start:stop] == sub:
                        yield (start, sub.decode())
                        break

    def startswith(self, prefix, start=None, end=None):
        if False:
            while True:
                i = 10
        'Return True if the sequence starts with the given prefix, False otherwise.\n\n        Return True if the sequence starts with the specified prefix\n        (a string or another Seq object), False otherwise.\n        With optional start, test sequence beginning at that position.\n        With optional end, stop comparing sequence at that position.\n        prefix can also be a tuple of strings to try.  e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_rna.startswith("GUC")\n        True\n        >>> my_rna.startswith("AUG")\n        False\n        >>> my_rna.startswith("AUG", 3)\n        True\n        >>> my_rna.startswith(("UCC", "UCA", "UCG"), 1)\n        True\n        '
        if isinstance(prefix, tuple):
            prefix = tuple((bytes(p) if isinstance(p, _SeqAbstractBaseClass) else p.encode('ASCII') for p in prefix))
        elif isinstance(prefix, _SeqAbstractBaseClass):
            prefix = bytes(prefix)
        elif isinstance(prefix, str):
            prefix = prefix.encode('ASCII')
        return self._data.startswith(prefix, start, end)

    def endswith(self, suffix, start=None, end=None):
        if False:
            return 10
        'Return True if the sequence ends with the given suffix, False otherwise.\n\n        Return True if the sequence ends with the specified suffix\n        (a string or another Seq object), False otherwise.\n        With optional start, test sequence beginning at that position.\n        With optional end, stop comparing sequence at that position.\n        suffix can also be a tuple of strings to try.  e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_rna.endswith("UUG")\n        True\n        >>> my_rna.endswith("AUG")\n        False\n        >>> my_rna.endswith("AUG", 0, 18)\n        True\n        >>> my_rna.endswith(("UCC", "UCA", "UUG"))\n        True\n        '
        if isinstance(suffix, tuple):
            suffix = tuple((bytes(p) if isinstance(p, _SeqAbstractBaseClass) else p.encode('ASCII') for p in suffix))
        elif isinstance(suffix, _SeqAbstractBaseClass):
            suffix = bytes(suffix)
        elif isinstance(suffix, str):
            suffix = suffix.encode('ASCII')
        return self._data.endswith(suffix, start, end)

    def split(self, sep=None, maxsplit=-1):
        if False:
            while True:
                i = 10
        'Return a list of subsequences when splitting the sequence by separator sep.\n\n        Return a list of the subsequences in the sequence (as Seq objects),\n        using sep as the delimiter string.  If maxsplit is given, at\n        most maxsplit splits are done.  If maxsplit is omitted, all\n        splits are made.\n\n        For consistency with the ``split`` method of Python strings, any\n        whitespace (tabs, spaces, newlines) is a separator if sep is None, the\n        default value\n\n        e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_aa = my_rna.translate()\n        >>> my_aa\n        Seq(\'VMAIVMGR*KGAR*L\')\n        >>> for pep in my_aa.split("*"):\n        ...     pep\n        Seq(\'VMAIVMGR\')\n        Seq(\'KGAR\')\n        Seq(\'L\')\n        >>> for pep in my_aa.split("*", 1):\n        ...     pep\n        Seq(\'VMAIVMGR\')\n        Seq(\'KGAR*L\')\n\n        See also the rsplit method, which splits the sequence starting from the\n        end:\n\n        >>> for pep in my_aa.rsplit("*", 1):\n        ...     pep\n        Seq(\'VMAIVMGR*KGAR\')\n        Seq(\'L\')\n        '
        if isinstance(sep, _SeqAbstractBaseClass):
            sep = bytes(sep)
        elif isinstance(sep, str):
            sep = sep.encode('ASCII')
        return [Seq(part) for part in self._data.split(sep, maxsplit)]

    def rsplit(self, sep=None, maxsplit=-1):
        if False:
            i = 10
            return i + 15
        'Return a list of subsequences by splitting the sequence from the right.\n\n        Return a list of the subsequences in the sequence (as Seq objects),\n        using sep as the delimiter string.  If maxsplit is given, at\n        most maxsplit splits are done.  If maxsplit is omitted, all\n        splits are made.\n\n        For consistency with the ``rsplit`` method of Python strings, any\n        whitespace (tabs, spaces, newlines) is a separator if sep is None, the\n        default value\n\n        e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> my_rna = Seq("GUCAUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAGUUG")\n        >>> my_aa = my_rna.translate()\n        >>> my_aa\n        Seq(\'VMAIVMGR*KGAR*L\')\n        >>> for pep in my_aa.rsplit("*"):\n        ...     pep\n        Seq(\'VMAIVMGR\')\n        Seq(\'KGAR\')\n        Seq(\'L\')\n        >>> for pep in my_aa.rsplit("*", 1):\n        ...     pep\n        Seq(\'VMAIVMGR*KGAR\')\n        Seq(\'L\')\n\n        See also the split method, which splits the sequence starting from the\n        beginning:\n\n        >>> for pep in my_aa.split("*", 1):\n        ...     pep\n        Seq(\'VMAIVMGR\')\n        Seq(\'KGAR*L\')\n        '
        if isinstance(sep, _SeqAbstractBaseClass):
            sep = bytes(sep)
        elif isinstance(sep, str):
            sep = sep.encode('ASCII')
        return [Seq(part) for part in self._data.rsplit(sep, maxsplit)]

    def strip(self, chars=None, inplace=False):
        if False:
            i = 10
            return i + 15
        'Return a sequence object with leading and trailing ends stripped.\n\n        With default arguments, leading and trailing whitespace is removed:\n\n        >>> seq = Seq(" ACGT ")\n        >>> seq.strip()\n        Seq(\'ACGT\')\n        >>> seq\n        Seq(\' ACGT \')\n\n        If ``chars`` is given and not ``None``, remove characters in ``chars``\n        instead.  The order of the characters to be removed is not important:\n\n        >>> Seq("ACGTACGT").strip("TGCA")\n        Seq(\'\')\n\n        A copy of the sequence is returned if ``inplace`` is ``False`` (the\n        default value).  If ``inplace`` is ``True``, the sequence is stripped\n        in-place and returned.\n\n        >>> seq = MutableSeq(" ACGT ")\n        >>> seq.strip(inplace=False)\n        MutableSeq(\'ACGT\')\n        >>> seq\n        MutableSeq(\' ACGT \')\n        >>> seq.strip(inplace=True)\n        MutableSeq(\'ACGT\')\n        >>> seq\n        MutableSeq(\'ACGT\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if ``strip``\n        is called on a ``Seq`` object with ``inplace=True``.\n\n        See also the lstrip and rstrip methods.\n        '
        if isinstance(chars, _SeqAbstractBaseClass):
            chars = bytes(chars)
        elif isinstance(chars, str):
            chars = chars.encode('ASCII')
        try:
            data = self._data.strip(chars)
        except TypeError:
            raise TypeError('argument must be None or a string, Seq, MutableSeq, or bytes-like object') from None
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def lstrip(self, chars=None, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        'Return a sequence object with leading and trailing ends stripped.\n\n        With default arguments, leading whitespace is removed:\n\n        >>> seq = Seq(" ACGT ")\n        >>> seq.lstrip()\n        Seq(\'ACGT \')\n        >>> seq\n        Seq(\' ACGT \')\n\n        If ``chars`` is given and not ``None``, remove characters in ``chars``\n        from the leading end instead.  The order of the characters to be removed\n        is not important:\n\n        >>> Seq("ACGACGTTACG").lstrip("GCA")\n        Seq(\'TTACG\')\n\n        A copy of the sequence is returned if ``inplace`` is ``False`` (the\n        default value).  If ``inplace`` is ``True``, the sequence is stripped\n        in-place and returned.\n\n        >>> seq = MutableSeq(" ACGT ")\n        >>> seq.lstrip(inplace=False)\n        MutableSeq(\'ACGT \')\n        >>> seq\n        MutableSeq(\' ACGT \')\n        >>> seq.lstrip(inplace=True)\n        MutableSeq(\'ACGT \')\n        >>> seq\n        MutableSeq(\'ACGT \')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``lstrip`` is called on a ``Seq`` object with ``inplace=True``.\n\n        See also the strip and rstrip methods.\n        '
        if isinstance(chars, _SeqAbstractBaseClass):
            chars = bytes(chars)
        elif isinstance(chars, str):
            chars = chars.encode('ASCII')
        try:
            data = self._data.lstrip(chars)
        except TypeError:
            raise TypeError('argument must be None or a string, Seq, MutableSeq, or bytes-like object') from None
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def rstrip(self, chars=None, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        'Return a sequence object with trailing ends stripped.\n\n        With default arguments, trailing whitespace is removed:\n\n        >>> seq = Seq(" ACGT ")\n        >>> seq.rstrip()\n        Seq(\' ACGT\')\n        >>> seq\n        Seq(\' ACGT \')\n\n        If ``chars`` is given and not ``None``, remove characters in ``chars``\n        from the trailing end instead.  The order of the characters to be\n        removed is not important:\n\n        >>> Seq("ACGACGTTACG").rstrip("GCA")\n        Seq(\'ACGACGTT\')\n\n        A copy of the sequence is returned if ``inplace`` is ``False`` (the\n        default value).  If ``inplace`` is ``True``, the sequence is stripped\n        in-place and returned.\n\n        >>> seq = MutableSeq(" ACGT ")\n        >>> seq.rstrip(inplace=False)\n        MutableSeq(\' ACGT\')\n        >>> seq\n        MutableSeq(\' ACGT \')\n        >>> seq.rstrip(inplace=True)\n        MutableSeq(\' ACGT\')\n        >>> seq\n        MutableSeq(\' ACGT\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``rstrip`` is called on a ``Seq`` object with ``inplace=True``.\n\n        See also the strip and lstrip methods.\n        '
        if isinstance(chars, _SeqAbstractBaseClass):
            chars = bytes(chars)
        elif isinstance(chars, str):
            chars = chars.encode('ASCII')
        try:
            data = self._data.rstrip(chars)
        except TypeError:
            raise TypeError('argument must be None or a string, Seq, MutableSeq, or bytes-like object') from None
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def removeprefix(self, prefix, inplace=False):
        if False:
            return 10
        'Return a new Seq object with prefix (left) removed.\n\n        This behaves like the python string method of the same name.\n\n        e.g. Removing a start Codon:\n\n        >>> from Bio.Seq import Seq\n        >>> my_seq = Seq("ATGGTGTGTGT")\n        >>> my_seq\n        Seq(\'ATGGTGTGTGT\')\n        >>> my_seq.removeprefix(\'ATG\')\n        Seq(\'GTGTGTGT\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``removeprefix`` is called on a ``Seq`` object with ``inplace=True``.\n\n        See also the removesuffix method.\n        '
        if isinstance(prefix, _SeqAbstractBaseClass):
            prefix = bytes(prefix)
        elif isinstance(prefix, str):
            prefix = prefix.encode('ASCII')
        try:
            data = self._data.removeprefix(prefix)
        except TypeError:
            raise TypeError('argument must be a string, Seq, MutableSeq, or bytes-like object') from None
        except AttributeError:
            data = self._data
            if data.startswith(prefix):
                data = data[len(prefix):]
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def removesuffix(self, suffix, inplace=False):
        if False:
            while True:
                i = 10
        'Return a new Seq object with suffix (right) removed.\n\n        This behaves like the python string method of the same name.\n\n        e.g. Removing a stop codon:\n\n        >>> from Bio.Seq import Seq\n        >>> my_seq = Seq("GTGTGTGTTAG")\n        >>> my_seq\n        Seq(\'GTGTGTGTTAG\')\n        >>> stop_codon = Seq("TAG")\n        >>> my_seq.removesuffix(stop_codon)\n        Seq(\'GTGTGTGT\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``removesuffix`` is called on a ``Seq`` object with ``inplace=True``.\n\n        See also the removeprefix method.\n        '
        if isinstance(suffix, _SeqAbstractBaseClass):
            suffix = bytes(suffix)
        elif isinstance(suffix, str):
            suffix = suffix.encode('ASCII')
        try:
            data = self._data.removesuffix(suffix)
        except TypeError:
            raise TypeError('argument must be a string, Seq, MutableSeq, or bytes-like object') from None
        except AttributeError:
            data = self._data
            if data.endswith(suffix):
                data = data[:-len(suffix)]
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def upper(self, inplace=False):
        if False:
            return 10
        'Return the sequence in upper case.\n\n        An upper-case copy of the sequence is returned if inplace is False,\n        the default value:\n\n        >>> from Bio.Seq import Seq, MutableSeq\n        >>> my_seq = Seq("VHLTPeeK*")\n        >>> my_seq\n        Seq(\'VHLTPeeK*\')\n        >>> my_seq.lower()\n        Seq(\'vhltpeek*\')\n        >>> my_seq.upper()\n        Seq(\'VHLTPEEK*\')\n        >>> my_seq\n        Seq(\'VHLTPeeK*\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> my_seq = MutableSeq("VHLTPeeK*")\n        >>> my_seq\n        MutableSeq(\'VHLTPeeK*\')\n        >>> my_seq.lower()\n        MutableSeq(\'vhltpeek*\')\n        >>> my_seq.upper()\n        MutableSeq(\'VHLTPEEK*\')\n        >>> my_seq\n        MutableSeq(\'VHLTPeeK*\')\n\n        >>> my_seq.lower(inplace=True)\n        MutableSeq(\'vhltpeek*\')\n        >>> my_seq\n        MutableSeq(\'vhltpeek*\')\n        >>> my_seq.upper(inplace=True)\n        MutableSeq(\'VHLTPEEK*\')\n        >>> my_seq\n        MutableSeq(\'VHLTPEEK*\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``upper`` is called on a ``Seq`` object with ``inplace=True``.\n\n        See also the ``lower`` method.\n        '
        data = self._data.upper()
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def lower(self, inplace=False):
        if False:
            while True:
                i = 10
        'Return the sequence in lower case.\n\n        An lower-case copy of the sequence is returned if inplace is False,\n        the default value:\n\n        >>> from Bio.Seq import Seq, MutableSeq\n        >>> my_seq = Seq("VHLTPeeK*")\n        >>> my_seq\n        Seq(\'VHLTPeeK*\')\n        >>> my_seq.lower()\n        Seq(\'vhltpeek*\')\n        >>> my_seq.upper()\n        Seq(\'VHLTPEEK*\')\n        >>> my_seq\n        Seq(\'VHLTPeeK*\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> my_seq = MutableSeq("VHLTPeeK*")\n        >>> my_seq\n        MutableSeq(\'VHLTPeeK*\')\n        >>> my_seq.lower()\n        MutableSeq(\'vhltpeek*\')\n        >>> my_seq.upper()\n        MutableSeq(\'VHLTPEEK*\')\n        >>> my_seq\n        MutableSeq(\'VHLTPeeK*\')\n\n        >>> my_seq.lower(inplace=True)\n        MutableSeq(\'vhltpeek*\')\n        >>> my_seq\n        MutableSeq(\'vhltpeek*\')\n        >>> my_seq.upper(inplace=True)\n        MutableSeq(\'VHLTPEEK*\')\n        >>> my_seq\n        MutableSeq(\'VHLTPEEK*\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``lower`` is called on a ``Seq`` object with ``inplace=True``.\n\n        See also the ``upper`` method.\n        '
        data = self._data.lower()
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        else:
            return self.__class__(data)

    def isupper(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if all ASCII characters in data are uppercase.\n\n        If there are no cased characters, the method returns False.\n        '
        return self._data.isupper()

    def islower(self):
        if False:
            print('Hello World!')
        'Return True if all ASCII characters in data are lowercase.\n\n        If there are no cased characters, the method returns False.\n        '
        return self._data.islower()

    def translate(self, table='Standard', stop_symbol='*', to_stop=False, cds=False, gap='-'):
        if False:
            i = 10
            return i + 15
        'Turn a nucleotide sequence into a protein sequence by creating a new sequence object.\n\n        This method will translate DNA or RNA sequences. It should not\n        be used on protein sequences as any result will be biologically\n        meaningless.\n\n        Arguments:\n         - table - Which codon table to use?  This can be either a name\n           (string), an NCBI identifier (integer), or a CodonTable\n           object (useful for non-standard genetic codes).  This\n           defaults to the "Standard" table.\n         - stop_symbol - Single character string, what to use for\n           terminators.  This defaults to the asterisk, "*".\n         - to_stop - Boolean, defaults to False meaning do a full\n           translation continuing on past any stop codons (translated as the\n           specified stop_symbol).  If True, translation is terminated at\n           the first in frame stop codon (and the stop_symbol is not\n           appended to the returned protein sequence).\n         - cds - Boolean, indicates this is a complete CDS.  If True,\n           this checks the sequence starts with a valid alternative start\n           codon (which will be translated as methionine, M), that the\n           sequence length is a multiple of three, and that there is a\n           single in frame stop codon at the end (this will be excluded\n           from the protein sequence, regardless of the to_stop option).\n           If these tests fail, an exception is raised.\n         - gap - Single character string to denote symbol used for gaps.\n           Defaults to the minus sign.\n\n        A ``Seq`` object is returned if ``translate`` is called on a ``Seq``\n        object; a ``MutableSeq`` object is returned if ``translate`` is called\n        pn a ``MutableSeq`` object.\n\n        e.g. Using the standard table:\n\n        >>> coding_dna = Seq("GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")\n        >>> coding_dna.translate()\n        Seq(\'VAIVMGR*KGAR*\')\n        >>> coding_dna.translate(stop_symbol="@")\n        Seq(\'VAIVMGR@KGAR@\')\n        >>> coding_dna.translate(to_stop=True)\n        Seq(\'VAIVMGR\')\n\n        Now using NCBI table 2, where TGA is not a stop codon:\n\n        >>> coding_dna.translate(table=2)\n        Seq(\'VAIVMGRWKGAR*\')\n        >>> coding_dna.translate(table=2, to_stop=True)\n        Seq(\'VAIVMGRWKGAR\')\n\n        In fact, GTG is an alternative start codon under NCBI table 2, meaning\n        this sequence could be a complete CDS:\n\n        >>> coding_dna.translate(table=2, cds=True)\n        Seq(\'MAIVMGRWKGAR\')\n\n        It isn\'t a valid CDS under NCBI table 1, due to both the start codon\n        and also the in frame stop codons:\n\n        >>> coding_dna.translate(table=1, cds=True)\n        Traceback (most recent call last):\n            ...\n        Bio.Data.CodonTable.TranslationError: First codon \'GTG\' is not a start codon\n\n        If the sequence has no in-frame stop codon, then the to_stop argument\n        has no effect:\n\n        >>> coding_dna2 = Seq("TTGGCCATTGTAATGGGCCGC")\n        >>> coding_dna2.translate()\n        Seq(\'LAIVMGR\')\n        >>> coding_dna2.translate(to_stop=True)\n        Seq(\'LAIVMGR\')\n\n        NOTE - Ambiguous codons like "TAN" or "NNN" could be an amino acid\n        or a stop codon.  These are translated as "X".  Any invalid codon\n        (e.g. "TA?" or "T-A") will throw a TranslationError.\n\n        NOTE - This does NOT behave like the python string\'s translate\n        method.  For that use str(my_seq).translate(...) instead\n        '
        try:
            data = str(self)
        except UndefinedSequenceError:
            n = len(self)
            if n % 3 != 0:
                warnings.warn('Partial codon, len(sequence) not a multiple of three. This may become an error in future.', BiopythonWarning)
            return Seq(None, n // 3)
        return self.__class__(_translate_str(str(self), table, stop_symbol, to_stop, cds, gap=gap))

    def complement(self, inplace=None):
        if False:
            print('Hello World!')
        'Return the complement as a DNA sequence.\n\n        >>> Seq("CGA").complement()\n        Seq(\'GCT\')\n\n        Any U in the sequence is treated as a T:\n\n        >>> Seq("CGAUT").complement(inplace=False)\n        Seq(\'GCTAA\')\n\n        In contrast, ``complement_rna`` returns an RNA sequence:\n\n        >>> Seq("CGAUT").complement_rna()\n        Seq(\'GCUAA\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> my_seq = MutableSeq("CGA")\n        >>> my_seq\n        MutableSeq(\'CGA\')\n        >>> my_seq.complement(inplace=False)\n        MutableSeq(\'GCT\')\n        >>> my_seq\n        MutableSeq(\'CGA\')\n\n        >>> my_seq.complement(inplace=True)\n        MutableSeq(\'GCT\')\n        >>> my_seq\n        MutableSeq(\'GCT\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``complement_rna`` is called on a ``Seq`` object with ``inplace=True``.\n        '
        ttable = _dna_complement_table
        try:
            if inplace is None:
                if isinstance(self._data, bytearray):
                    warnings.warn('mutable_seq.complement() will change in the near future and will no longer change the sequence in-place by default. Please use\n\nmutable_seq.complement(inplace=True)\n\nif you want to continue to use this method to change a mutable sequence in-place.', BiopythonDeprecationWarning)
                    inplace = True
                if isinstance(self._data, _PartiallyDefinedSequenceData):
                    for seq in self._data._data.values():
                        if b'U' in seq or b'u' in seq:
                            warnings.warn('seq.complement() will change in the near future to always return DNA nucleotides only. Please use\n\nseq.complement_rna()\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
                            for seq in self._data._data.values():
                                if b't' in seq or b'T' in seq:
                                    raise ValueError('Mixed RNA/DNA found')
                            ttable = _rna_complement_table
                            break
                elif b'U' in self._data or b'u' in self._data:
                    warnings.warn('seq.complement() will change in the near future to always return DNA nucleotides only. Please use\n\nseq.complement_rna()\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
                    if b't' in self._data or b'T' in self._data:
                        raise ValueError('Mixed RNA/DNA found')
                    ttable = _rna_complement_table
            data = self._data.translate(ttable)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def complement_rna(self, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        'Return the complement as an RNA sequence.\n\n        >>> Seq("CGA").complement_rna()\n        Seq(\'GCU\')\n\n        Any T in the sequence is treated as a U:\n\n        >>> Seq("CGAUT").complement_rna()\n        Seq(\'GCUAA\')\n\n        In contrast, ``complement`` returns a DNA sequence by default:\n\n        >>> Seq("CGA").complement()\n        Seq(\'GCT\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> my_seq = MutableSeq("CGA")\n        >>> my_seq\n        MutableSeq(\'CGA\')\n        >>> my_seq.complement_rna()\n        MutableSeq(\'GCU\')\n        >>> my_seq\n        MutableSeq(\'CGA\')\n\n        >>> my_seq.complement_rna(inplace=True)\n        MutableSeq(\'GCU\')\n        >>> my_seq\n        MutableSeq(\'GCU\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``complement_rna`` is called on a ``Seq`` object with ``inplace=True``.\n        '
        try:
            data = self._data.translate(_rna_complement_table)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def reverse_complement(self, inplace=None):
        if False:
            for i in range(10):
                print('nop')
        'Return the reverse complement as a DNA sequence.\n\n        >>> Seq("CGA").reverse_complement(inplace=False)\n        Seq(\'TCG\')\n\n        Any U in the sequence is treated as a T:\n\n        >>> Seq("CGAUT").reverse_complement(inplace=False)\n        Seq(\'AATCG\')\n\n        In contrast, ``reverse_complement_rna`` returns an RNA sequence:\n\n        >>> Seq("CGA").reverse_complement_rna()\n        Seq(\'UCG\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> my_seq = MutableSeq("CGA")\n        >>> my_seq\n        MutableSeq(\'CGA\')\n        >>> my_seq.reverse_complement(inplace=False)\n        MutableSeq(\'TCG\')\n        >>> my_seq\n        MutableSeq(\'CGA\')\n\n        >>> my_seq.reverse_complement(inplace=True)\n        MutableSeq(\'TCG\')\n        >>> my_seq\n        MutableSeq(\'TCG\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``reverse_complement`` is called on a ``Seq`` object with\n        ``inplace=True``.\n        '
        try:
            if inplace is None:
                if isinstance(self._data, bytearray):
                    warnings.warn('mutable_seq.reverse_complement() will change in the near future and will no longer change the sequence in-place by default. Please use\n\nmutable_seq.reverse_complement(inplace=True)\n\nif you want to continue to use this method to change a mutable sequence in-place.', BiopythonDeprecationWarning)
                    inplace = True
                else:
                    inplace = False
                if isinstance(self._data, _PartiallyDefinedSequenceData):
                    for seq in self._data._data.values():
                        if b'U' in seq or b'u' in seq:
                            warnings.warn('seq.reverse_complement() will change in the near future to always return DNA nucleotides only. Please use\n\nseq.reverse_complement_rna()\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
                            for seq in self._data._data.values():
                                if b't' in seq or b'T' in seq:
                                    raise ValueError('Mixed RNA/DNA found')
                            return self.reverse_complement_rna(inplace=inplace)
                elif b'U' in self._data or b'u' in self._data:
                    warnings.warn('seq.reverse_complement() will change in the near future to always return DNA nucleotides only. Please use\n\nseq.reverse_complement_rna()\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
                    if b't' in self._data or b'T' in self._data:
                        raise ValueError('Mixed RNA/DNA found')
                    return self.reverse_complement_rna(inplace=inplace)
            data = self._data.translate(_dna_complement_table)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[::-1] = data
            return self
        return self.__class__(data[::-1])

    def reverse_complement_rna(self, inplace=False):
        if False:
            for i in range(10):
                print('nop')
        'Return the reverse complement as an RNA sequence.\n\n        >>> Seq("CGA").reverse_complement_rna()\n        Seq(\'UCG\')\n\n        Any T in the sequence is treated as a U:\n\n        >>> Seq("CGAUT").reverse_complement_rna()\n        Seq(\'AAUCG\')\n\n        In contrast, ``reverse_complement`` returns a DNA sequence:\n\n        >>> Seq("CGA").reverse_complement(inplace=False)\n        Seq(\'TCG\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> my_seq = MutableSeq("CGA")\n        >>> my_seq\n        MutableSeq(\'CGA\')\n        >>> my_seq.reverse_complement_rna()\n        MutableSeq(\'UCG\')\n        >>> my_seq\n        MutableSeq(\'CGA\')\n\n        >>> my_seq.reverse_complement_rna(inplace=True)\n        MutableSeq(\'UCG\')\n        >>> my_seq\n        MutableSeq(\'UCG\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``reverse_complement_rna`` is called on a ``Seq`` object with\n        ``inplace=True``.\n        '
        try:
            data = self._data.translate(_rna_complement_table)
        except UndefinedSequenceError:
            return self
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[::-1] = data
            return self
        return self.__class__(data[::-1])

    def transcribe(self, inplace=False):
        if False:
            while True:
                i = 10
        'Transcribe a DNA sequence into RNA and return the RNA sequence as a new Seq object.\n\n        Following the usual convention, the sequence is interpreted as the\n        coding strand of the DNA double helix, not the template strand. This\n        means we can get the RNA sequence just by switching T to U.\n\n        >>> from Bio.Seq import Seq\n        >>> coding_dna = Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")\n        >>> coding_dna\n        Seq(\'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\')\n        >>> coding_dna.transcribe()\n        Seq(\'AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> sequence = MutableSeq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")\n        >>> sequence\n        MutableSeq(\'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\')\n        >>> sequence.transcribe()\n        MutableSeq(\'AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG\')\n        >>> sequence\n        MutableSeq(\'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\')\n\n        >>> sequence.transcribe(inplace=True)\n        MutableSeq(\'AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG\')\n        >>> sequence\n        MutableSeq(\'AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``transcribe`` is called on a ``Seq`` object with ``inplace=True``.\n\n        Trying to transcribe an RNA sequence has no effect.\n        If you have a nucleotide sequence which might be DNA or RNA\n        (or even a mixture), calling the transcribe method will ensure\n        any T becomes U.\n\n        Trying to transcribe a protein sequence will replace any\n        T for Threonine with U for Selenocysteine, which has no\n        biologically plausible rational.\n\n        >>> from Bio.Seq import Seq\n        >>> my_protein = Seq("MAIVMGRT")\n        >>> my_protein.transcribe()\n        Seq(\'MAIVMGRU\')\n        '
        data = self._data.replace(b'T', b'U').replace(b't', b'u')
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def back_transcribe(self, inplace=False):
        if False:
            while True:
                i = 10
        'Return the DNA sequence from an RNA sequence by creating a new Seq object.\n\n        >>> from Bio.Seq import Seq\n        >>> messenger_rna = Seq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")\n        >>> messenger_rna\n        Seq(\'AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG\')\n        >>> messenger_rna.back_transcribe()\n        Seq(\'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\')\n\n        The sequence is modified in-place and returned if inplace is True:\n\n        >>> sequence = MutableSeq("AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG")\n        >>> sequence\n        MutableSeq(\'AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG\')\n        >>> sequence.back_transcribe()\n        MutableSeq(\'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\')\n        >>> sequence\n        MutableSeq(\'AUGGCCAUUGUAAUGGGCCGCUGAAAGGGUGCCCGAUAG\')\n\n        >>> sequence.back_transcribe(inplace=True)\n        MutableSeq(\'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\')\n        >>> sequence\n        MutableSeq(\'ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``transcribe`` is called on a ``Seq`` object with ``inplace=True``.\n\n        Trying to back-transcribe DNA has no effect, If you have a nucleotide\n        sequence which might be DNA or RNA (or even a mixture), calling the\n        back-transcribe method will ensure any U becomes T.\n\n        Trying to back-transcribe a protein sequence will replace any U for\n        Selenocysteine with T for Threonine, which is biologically meaningless.\n\n        >>> from Bio.Seq import Seq\n        >>> my_protein = Seq("MAIVMGRU")\n        >>> my_protein.back_transcribe()\n        Seq(\'MAIVMGRT\')\n        '
        data = self._data.replace(b'U', b'T').replace(b'u', b't')
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    def join(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Return a merge of the sequences in other, spaced by the sequence from self.\n\n        Accepts a Seq object, MutableSeq object, or string (and iterates over\n        the letters), or an iterable containing Seq, MutableSeq, or string\n        objects. These arguments will be concatenated with the calling sequence\n        as the spacer:\n\n        >>> concatenated = Seq(\'NNNNN\').join([Seq("AAA"), Seq("TTT"), Seq("PPP")])\n        >>> concatenated\n        Seq(\'AAANNNNNTTTNNNNNPPP\')\n\n        Joining the letters of a single sequence:\n\n        >>> Seq(\'NNNNN\').join(Seq("ACGT"))\n        Seq(\'ANNNNNCNNNNNGNNNNNT\')\n        >>> Seq(\'NNNNN\').join("ACGT")\n        Seq(\'ANNNNNCNNNNNGNNNNNT\')\n        '
        if isinstance(other, _SeqAbstractBaseClass):
            return self.__class__(str(self).join(str(other)))
        elif isinstance(other, str):
            return self.__class__(str(self).join(other))
        from Bio.SeqRecord import SeqRecord
        if isinstance(other, SeqRecord):
            raise TypeError('Iterable cannot be a SeqRecord')
        for c in other:
            if isinstance(c, SeqRecord):
                raise TypeError('Iterable cannot contain SeqRecords')
            elif not isinstance(c, (str, _SeqAbstractBaseClass)):
                raise TypeError('Input must be an iterable of Seq objects, MutableSeq objects, or strings')
        return self.__class__(str(self).join([str(_) for _ in other]))

    def replace(self, old, new, inplace=False):
        if False:
            while True:
                i = 10
        'Return a copy with all occurrences of subsequence old replaced by new.\n\n        >>> s = Seq("ACGTAACCGGTT")\n        >>> t = s.replace("AC", "XYZ")\n        >>> s\n        Seq(\'ACGTAACCGGTT\')\n        >>> t\n        Seq(\'XYZGTAXYZCGGTT\')\n\n        For mutable sequences, passing inplace=True will modify the sequence in place:\n\n        >>> m = MutableSeq("ACGTAACCGGTT")\n        >>> t = m.replace("AC", "XYZ")\n        >>> m\n        MutableSeq(\'ACGTAACCGGTT\')\n        >>> t\n        MutableSeq(\'XYZGTAXYZCGGTT\')\n\n        >>> m = MutableSeq("ACGTAACCGGTT")\n        >>> t = m.replace("AC", "XYZ", inplace=True)\n        >>> m\n        MutableSeq(\'XYZGTAXYZCGGTT\')\n        >>> t\n        MutableSeq(\'XYZGTAXYZCGGTT\')\n\n        As ``Seq`` objects are immutable, a ``TypeError`` is raised if\n        ``replace`` is called on a ``Seq`` object with ``inplace=True``.\n        '
        if isinstance(old, _SeqAbstractBaseClass):
            old = bytes(old)
        elif isinstance(old, str):
            old = old.encode('ASCII')
        if isinstance(new, _SeqAbstractBaseClass):
            new = bytes(new)
        elif isinstance(new, str):
            new = new.encode('ASCII')
        data = self._data.replace(old, new)
        if inplace:
            if not isinstance(self._data, bytearray):
                raise TypeError('Sequence is immutable')
            self._data[:] = data
            return self
        return self.__class__(data)

    @property
    def defined(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if the sequence is defined, False if undefined or partially defined.\n\n        Zero-length sequences are always considered to be defined.\n        '
        if isinstance(self._data, (bytes, bytearray)):
            return True
        else:
            return self._data.defined

    @property
    def defined_ranges(self):
        if False:
            i = 10
            return i + 15
        'Return a tuple of the ranges where the sequence contents is defined.\n\n        The return value has the format ((start1, end1), (start2, end2), ...).\n        '
        if isinstance(self._data, (bytes, bytearray)):
            length = len(self)
            if length > 0:
                return ((0, length),)
            else:
                return ()
        else:
            return self._data.defined_ranges

class Seq(_SeqAbstractBaseClass):
    """Read-only sequence object (essentially a string with biological methods).

    Like normal python strings, our basic sequence object is immutable.
    This prevents you from doing my_seq[5] = "A" for example, but does allow
    Seq objects to be used as dictionary keys.

    The Seq object provides a number of string like methods (such as count,
    find, split and strip).

    The Seq object also provides some biological methods, such as complement,
    reverse_complement, transcribe, back_transcribe and translate (which are
    not applicable to protein sequences).
    """
    _data: Union[bytes, SequenceDataAbstractBaseClass]

    def __init__(self, data: Union[str, bytes, bytearray, _SeqAbstractBaseClass, SequenceDataAbstractBaseClass, dict, None], length: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a Seq object.\n\n        Arguments:\n         - data - Sequence, required (string)\n         - length - Sequence length, used only if data is None or a dictionary (integer)\n\n        You will typically use Bio.SeqIO to read in sequences from files as\n        SeqRecord objects, whose sequence will be exposed as a Seq object via\n        the seq property.\n\n        However, you can also create a Seq object directly:\n\n        >>> from Bio.Seq import Seq\n        >>> my_seq = Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF")\n        >>> my_seq\n        Seq(\'MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\')\n        >>> print(my_seq)\n        MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\n\n        To create a Seq object with for a sequence of known length but\n        unknown sequence contents, use None for the data argument and pass\n        the sequence length for the length argument. Trying to access the\n        sequence contents of a Seq object created in this way will raise\n        an UndefinedSequenceError:\n\n        >>> my_undefined_sequence = Seq(None, 20)\n        >>> my_undefined_sequence\n        Seq(None, length=20)\n        >>> len(my_undefined_sequence)\n        20\n        >>> print(my_undefined_sequence)\n        Traceback (most recent call last):\n        ...\n        Bio.Seq.UndefinedSequenceError: Sequence content is undefined\n\n        If the sequence contents is known for parts of the sequence only, use\n        a dictionary for the data argument to pass the known sequence segments:\n\n        >>> my_partially_defined_sequence = Seq({3: "ACGT"}, 10)\n        >>> my_partially_defined_sequence\n        Seq({3: \'ACGT\'}, length=10)\n        >>> len(my_partially_defined_sequence)\n        10\n        >>> print(my_partially_defined_sequence)\n        Traceback (most recent call last):\n        ...\n        Bio.Seq.UndefinedSequenceError: Sequence content is only partially defined\n        >>> my_partially_defined_sequence[3:7]\n        Seq(\'ACGT\')\n        >>> print(my_partially_defined_sequence[3:7])\n        ACGT\n        '
        if data is None:
            if length is None:
                raise ValueError('length must not be None if data is None')
            elif length == 0:
                self._data = b''
            elif length < 0:
                raise ValueError('length must not be negative.')
            else:
                self._data = _UndefinedSequenceData(length)
        elif isinstance(data, (bytes, SequenceDataAbstractBaseClass)):
            self._data = data
        elif isinstance(data, (bytearray, _SeqAbstractBaseClass)):
            self._data = bytes(data)
        elif isinstance(data, str):
            self._data = bytes(data, encoding='ASCII')
        elif isinstance(data, dict):
            if length is None:
                raise ValueError('length must not be None if data is a dictionary')
            elif length == 0:
                self._data = b''
            elif length < 0:
                raise ValueError('length must not be negative.')
            else:
                current = 0
                end = -1
                starts = sorted(data.keys())
                _data: Dict[int, bytes] = {}
                for start in starts:
                    seq = data[start]
                    if isinstance(seq, str):
                        seq = bytes(seq, encoding='ASCII')
                    else:
                        try:
                            seq = bytes(seq)
                        except Exception:
                            raise ValueError('Expected bytes-like objects or strings')
                    if start < end:
                        raise ValueError('Sequence data are overlapping.')
                    elif start == end:
                        _data[current] += seq
                    else:
                        _data[start] = seq
                        current = start
                    end = start + len(seq)
                if end > length:
                    raise ValueError('Provided sequence data extend beyond sequence length.')
                elif end == length and current == 0:
                    self._data = _data[current]
                else:
                    self._data = _PartiallyDefinedSequenceData(length, _data)
        else:
            raise TypeError('data should be a string, bytes, bytearray, Seq, or MutableSeq object')

    def __hash__(self):
        if False:
            print('Hello World!')
        'Hash of the sequence as a string for comparison.\n\n        See Seq object comparison documentation (method ``__eq__`` in\n        particular) as this has changed in Biopython 1.65. Older versions\n        would hash on object identity.\n        '
        return hash(self._data)

    def ungap(self, gap='-'):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the sequence without the gap character(s) (DEPRECATED).\n\n        The gap character now defaults to the minus sign, and can only\n        be specified via the method argument. This is no longer possible\n        via the sequence\'s alphabet (as was possible up to Biopython 1.77):\n\n        >>> from Bio.Seq import Seq\n        >>> my_dna = Seq("-ATA--TGAAAT-TTGAAAA")\n        >>> my_dna\n        Seq(\'-ATA--TGAAAT-TTGAAAA\')\n        >>> my_dna.ungap("-")\n        Seq(\'ATATGAAATTTGAAAA\')\n\n        This method is DEPRECATED; please use my_dna.replace(gap, "") instead.\n        '
        warnings.warn('myseq.ungap(gap) is deprecated; please use myseq.replace(gap, "") instead.', BiopythonDeprecationWarning)
        if not gap:
            raise ValueError('Gap character required.')
        elif len(gap) != 1 or not isinstance(gap, str):
            raise ValueError(f'Unexpected gap character, {gap!r}')
        return self.replace(gap, b'')

class MutableSeq(_SeqAbstractBaseClass):
    """An editable sequence object.

    Unlike normal python strings and our basic sequence object (the Seq class)
    which are immutable, the MutableSeq lets you edit the sequence in place.
    However, this means you cannot use a MutableSeq object as a dictionary key.

    >>> from Bio.Seq import MutableSeq
    >>> my_seq = MutableSeq("ACTCGTCGTCG")
    >>> my_seq
    MutableSeq('ACTCGTCGTCG')
    >>> my_seq[5]
    'T'
    >>> my_seq[5] = "A"
    >>> my_seq
    MutableSeq('ACTCGACGTCG')
    >>> my_seq[5]
    'A'
    >>> my_seq[5:8] = "NNN"
    >>> my_seq
    MutableSeq('ACTCGNNNTCG')
    >>> len(my_seq)
    11

    Note that the MutableSeq object does not support as many string-like
    or biological methods as the Seq object.
    """

    def __init__(self, data):
        if False:
            while True:
                i = 10
        'Create a MutableSeq object.'
        if isinstance(data, bytearray):
            self._data = data
        elif isinstance(data, bytes):
            self._data = bytearray(data)
        elif isinstance(data, str):
            self._data = bytearray(data, 'ASCII')
        elif isinstance(data, MutableSeq):
            self._data = data._data[:]
        elif isinstance(data, Seq):
            self._data = bytearray(bytes(data))
        else:
            raise TypeError('data should be a string, bytearray object, Seq object, or a MutableSeq object')

    def __setitem__(self, index, value):
        if False:
            while True:
                i = 10
        "Set a subsequence of single letter via value parameter.\n\n        >>> my_seq = MutableSeq('ACTCGACGTCG')\n        >>> my_seq[0] = 'T'\n        >>> my_seq\n        MutableSeq('TCTCGACGTCG')\n        "
        if isinstance(index, numbers.Integral):
            self._data[index] = ord(value)
        elif isinstance(value, MutableSeq):
            self._data[index] = value._data
        elif isinstance(value, Seq):
            self._data[index] = bytes(value)
        elif isinstance(value, str):
            self._data[index] = value.encode('ASCII')
        else:
            raise TypeError(f"received unexpected type '{type(value).__name__}'")

    def __delitem__(self, index):
        if False:
            i = 10
            return i + 15
        "Delete a subsequence of single letter.\n\n        >>> my_seq = MutableSeq('ACTCGACGTCG')\n        >>> del my_seq[0]\n        >>> my_seq\n        MutableSeq('CTCGACGTCG')\n        "
        del self._data[index]

    def append(self, c):
        if False:
            for i in range(10):
                print('nop')
        "Add a subsequence to the mutable sequence object.\n\n        >>> my_seq = MutableSeq('ACTCGACGTCG')\n        >>> my_seq.append('A')\n        >>> my_seq\n        MutableSeq('ACTCGACGTCGA')\n\n        No return value.\n        "
        self._data.append(ord(c.encode('ASCII')))

    def insert(self, i, c):
        if False:
            for i in range(10):
                print('nop')
        "Add a subsequence to the mutable sequence object at a given index.\n\n        >>> my_seq = MutableSeq('ACTCGACGTCG')\n        >>> my_seq.insert(0,'A')\n        >>> my_seq\n        MutableSeq('AACTCGACGTCG')\n        >>> my_seq.insert(8,'G')\n        >>> my_seq\n        MutableSeq('AACTCGACGGTCG')\n\n        No return value.\n        "
        self._data.insert(i, ord(c.encode('ASCII')))

    def pop(self, i=-1):
        if False:
            while True:
                i = 10
        "Remove a subsequence of a single letter at given index.\n\n        >>> my_seq = MutableSeq('ACTCGACGTCG')\n        >>> my_seq.pop()\n        'G'\n        >>> my_seq\n        MutableSeq('ACTCGACGTC')\n        >>> my_seq.pop()\n        'C'\n        >>> my_seq\n        MutableSeq('ACTCGACGT')\n\n        Returns the last character of the sequence.\n        "
        c = self._data[i]
        del self._data[i]
        return chr(c)

    def remove(self, item):
        if False:
            i = 10
            return i + 15
        "Remove a subsequence of a single letter from mutable sequence.\n\n        >>> my_seq = MutableSeq('ACTCGACGTCG')\n        >>> my_seq.remove('C')\n        >>> my_seq\n        MutableSeq('ATCGACGTCG')\n        >>> my_seq.remove('A')\n        >>> my_seq\n        MutableSeq('TCGACGTCG')\n\n        No return value.\n        "
        codepoint = ord(item)
        try:
            self._data.remove(codepoint)
        except ValueError:
            raise ValueError('value not found in MutableSeq') from None

    def reverse(self):
        if False:
            return 10
        'Modify the mutable sequence to reverse itself.\n\n        No return value.\n        '
        self._data.reverse()

    def extend(self, other):
        if False:
            while True:
                i = 10
        "Add a sequence to the original mutable sequence object.\n\n        >>> my_seq = MutableSeq('ACTCGACGTCG')\n        >>> my_seq.extend('A')\n        >>> my_seq\n        MutableSeq('ACTCGACGTCGA')\n        >>> my_seq.extend('TTT')\n        >>> my_seq\n        MutableSeq('ACTCGACGTCGATTT')\n\n        No return value.\n        "
        if isinstance(other, MutableSeq):
            self._data.extend(other._data)
        elif isinstance(other, Seq):
            self._data.extend(bytes(other))
        elif isinstance(other, str):
            self._data.extend(other.encode('ASCII'))
        else:
            raise TypeError('expected a string, Seq or MutableSeq')

class UndefinedSequenceError(ValueError):
    """Sequence contents is undefined."""

class _UndefinedSequenceData(SequenceDataAbstractBaseClass):
    """Stores the length of a sequence with an undefined sequence contents (PRIVATE).

    Objects of this class can be used to create a Seq object to represent
    sequences with a known length, but an unknown sequence contents.
    Calling __len__ returns the sequence length, calling __getitem__ raises an
    UndefinedSequenceError except for requests of zero size, for which it
    returns an empty bytes object.
    """
    __slots__ = ('_length',)

    def __init__(self, length):
        if False:
            return 10
        'Initialize the object with the sequence length.\n\n        The calling function is responsible for ensuring that the length is\n        greater than zero.\n        '
        self._length = length
        super().__init__()

    def __getitem__(self, key: slice) -> Union[bytes, '_UndefinedSequenceData']:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(key, slice):
            (start, end, step) = key.indices(self._length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
            return _UndefinedSequenceData(size)
        else:
            raise UndefinedSequenceError('Sequence content is undefined')

    def __len__(self):
        if False:
            print('Hello World!')
        return self._length

    def __bytes__(self):
        if False:
            for i in range(10):
                print('nop')
        raise UndefinedSequenceError('Sequence content is undefined')

    def __add__(self, other):
        if False:
            print('Hello World!')
        length = len(self) + len(other)
        try:
            other = bytes(other)
        except UndefinedSequenceError:
            if isinstance(other, _UndefinedSequenceData):
                return _UndefinedSequenceData(length)
            else:
                return NotImplemented
        else:
            data = {len(self): other}
            return _PartiallyDefinedSequenceData(length, data)

    def __radd__(self, other):
        if False:
            while True:
                i = 10
        data = {0: bytes(other)}
        length = len(other) + len(self)
        return _PartiallyDefinedSequenceData(length, data)

    def upper(self):
        if False:
            print('Hello World!')
        'Return an upper case copy of the sequence.'
        return _UndefinedSequenceData(self._length)

    def lower(self):
        if False:
            i = 10
            return i + 15
        'Return a lower case copy of the sequence.'
        return _UndefinedSequenceData(self._length)

    def isupper(self):
        if False:
            return 10
        'Return True if all ASCII characters in data are uppercase.\n\n        If there are no cased characters, the method returns False.\n        '
        raise UndefinedSequenceError('Sequence content is undefined')

    def islower(self):
        if False:
            i = 10
            return i + 15
        'Return True if all ASCII characters in data are lowercase.\n\n        If there are no cased characters, the method returns False.\n        '
        raise UndefinedSequenceError('Sequence content is undefined')

    def replace(self, old, new):
        if False:
            while True:
                i = 10
        'Return a copy with all occurrences of substring old replaced by new.'
        if len(old) != len(new):
            raise UndefinedSequenceError('Sequence content is undefined')
        return _UndefinedSequenceData(self._length)

    @property
    def defined(self):
        if False:
            print('Hello World!')
        'Return False, as the sequence is not defined and has a non-zero length.'
        return False

    @property
    def defined_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a tuple of the ranges where the sequence contents is defined.\n\n        As the sequence contents of an _UndefinedSequenceData object is fully\n        undefined, the return value is always an empty tuple.\n        '
        return ()

class _PartiallyDefinedSequenceData(SequenceDataAbstractBaseClass):
    """Stores the length of a sequence with an undefined sequence contents (PRIVATE).

    Objects of this class can be used to create a Seq object to represent
    sequences with a known length, but with a sequence contents that is only
    partially known.
    Calling __len__ returns the sequence length, calling __getitem__ returns
    the sequence contents if known, otherwise an UndefinedSequenceError is
    raised.
    """
    __slots__ = ('_length', '_data')

    def __init__(self, length, data):
        if False:
            for i in range(10):
                print('nop')
        'Initialize with the sequence length and defined sequence segments.\n\n        The calling function is responsible for ensuring that the length is\n        greater than zero.\n        '
        self._length = length
        self._data = data
        super().__init__()

    def __getitem__(self, key: Union[slice, int]) -> Union[bytes, SequenceDataAbstractBaseClass]:
        if False:
            print('Hello World!')
        if isinstance(key, slice):
            (start, end, step) = key.indices(self._length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
            data = {}
            for (s, d) in self._data.items():
                indices = range(-s, -s + self._length)[key]
                e: Optional[int] = indices.stop
                assert e is not None
                if step > 0:
                    if e <= 0:
                        continue
                    if indices.start < 0:
                        s = indices.start % step
                    else:
                        s = indices.start
                else:
                    if e < 0:
                        e = None
                    end = len(d) - 1
                    if indices.start > end:
                        s = end + (indices.start - end) % step
                    else:
                        s = indices.start
                    if s < 0:
                        continue
                start = (s - indices.start) // step
                d = d[s:e:step]
                if d:
                    data[start] = d
            if len(data) == 0:
                return _UndefinedSequenceData(size)
            end = -1
            previous = 0
            items = data.items()
            data = {}
            for (start, seq) in items:
                if end == start:
                    data[previous] += seq
                else:
                    data[start] = seq
                    previous = start
                end = start + len(seq)
            if len(data) == 1:
                seq = data.get(0)
                if seq is not None and len(seq) == size:
                    return seq
            if step < 0:
                data = {start: data[start] for start in reversed(list(data.keys()))}
            return _PartiallyDefinedSequenceData(size, data)
        elif self._length <= key:
            raise IndexError('sequence index out of range')
        else:
            for (start, seq) in self._data.items():
                if start <= key and key < start + len(seq):
                    return seq[key - start]
            raise UndefinedSequenceError('Sequence at position %d is undefined' % key)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._length

    def __bytes__(self):
        if False:
            i = 10
            return i + 15
        raise UndefinedSequenceError('Sequence content is only partially defined')

    def __add__(self, other):
        if False:
            print('Hello World!')
        length = len(self) + len(other)
        data = dict(self._data)
        items = list(self._data.items())
        (start, seq) = items[-1]
        end = start + len(seq)
        try:
            other = bytes(other)
        except UndefinedSequenceError:
            if isinstance(other, _UndefinedSequenceData):
                pass
            elif isinstance(other, _PartiallyDefinedSequenceData):
                other_items = list(other._data.items())
                if end == len(self):
                    (other_start, other_seq) = other_items.pop(0)
                    if other_start == 0:
                        data[start] += other_seq
                    else:
                        data[len(self) + other_start] = other_seq
                for (other_start, other_seq) in other_items:
                    data[len(self) + other_start] = other_seq
        else:
            if end == len(self):
                data[start] += other
            else:
                data[len(self)] = other
        return _PartiallyDefinedSequenceData(length, data)

    def __radd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        length = len(other) + len(self)
        try:
            other = bytes(other)
        except UndefinedSequenceError:
            data = {len(other) + start: seq for (start, seq) in self._data.items()}
        else:
            data = {0: other}
            items = list(self._data.items())
            (start, seq) = items.pop(0)
            if start == 0:
                data[0] += seq
            else:
                data[len(other) + start] = seq
            for (start, seq) in items:
                data[len(other) + start] = seq
        return _PartiallyDefinedSequenceData(length, data)

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        length = self._length
        items = self._data.items()
        data = {}
        end = -1
        previous = 0
        for i in range(other):
            for (start, seq) in items:
                start += i * length
                if end == start:
                    data[previous] += seq
                else:
                    data[start] = seq
                    previous = start
            end = start + len(seq)
        return _PartiallyDefinedSequenceData(length * other, data)

    def upper(self):
        if False:
            return 10
        'Return an upper case copy of the sequence.'
        data = {start: seq.upper() for (start, seq) in self._data.items()}
        return _PartiallyDefinedSequenceData(self._length, data)

    def lower(self):
        if False:
            while True:
                i = 10
        'Return a lower case copy of the sequence.'
        data = {start: seq.lower() for (start, seq) in self._data.items()}
        return _PartiallyDefinedSequenceData(self._length, data)

    def isupper(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True if all ASCII characters in data are uppercase.\n\n        If there are no cased characters, the method returns False.\n        '
        raise UndefinedSequenceError('Sequence content is only partially defined')

    def islower(self):
        if False:
            return 10
        'Return True if all ASCII characters in data are lowercase.\n\n        If there are no cased characters, the method returns False.\n        '
        raise UndefinedSequenceError('Sequence content is only partially defined')

    def translate(self, table, delete=b''):
        if False:
            while True:
                i = 10
        'Return a copy with each character mapped by the given translation table.\n\n          table\n            Translation table, which must be a bytes object of length 256.\n\n        All characters occurring in the optional argument delete are removed.\n        The remaining characters are mapped through the given translation table.\n        '
        items = self._data.items()
        data = {start: seq.translate(table, delete) for (start, seq) in items}
        return _PartiallyDefinedSequenceData(self._length, data)

    def replace(self, old, new):
        if False:
            while True:
                i = 10
        'Return a copy with all occurrences of substring old replaced by new.'
        if len(old) != len(new):
            raise UndefinedSequenceError('Sequence content is only partially defined; substring \nreplacement cannot be performed reliably')
        items = self._data.items()
        data = {start: seq.replace(old, new) for (start, seq) in items}
        return _PartiallyDefinedSequenceData(self._length, data)

    @property
    def defined(self):
        if False:
            while True:
                i = 10
        'Return False, as the sequence is not fully defined and has a non-zero length.'
        return False

    @property
    def defined_ranges(self):
        if False:
            return 10
        'Return a tuple of the ranges where the sequence contents is defined.\n\n        The return value has the format ((start1, end1), (start2, end2), ...).\n        '
        return tuple(((start, start + len(seq)) for (start, seq) in self._data.items()))

def transcribe(dna):
    if False:
        for i in range(10):
            print('nop')
    'Transcribe a DNA sequence into RNA.\n\n    Following the usual convention, the sequence is interpreted as the\n    coding strand of the DNA double helix, not the template strand. This\n    means we can get the RNA sequence just by switching T to U.\n\n    If given a string, returns a new string object.\n\n    Given a Seq or MutableSeq, returns a new Seq object.\n\n    e.g.\n\n    >>> transcribe("ACTGN")\n    \'ACUGN\'\n    '
    if isinstance(dna, Seq):
        return dna.transcribe()
    elif isinstance(dna, MutableSeq):
        return Seq(dna).transcribe()
    else:
        return dna.replace('T', 'U').replace('t', 'u')

def back_transcribe(rna):
    if False:
        i = 10
        return i + 15
    'Return the RNA sequence back-transcribed into DNA.\n\n    If given a string, returns a new string object.\n\n    Given a Seq or MutableSeq, returns a new Seq object.\n\n    e.g.\n\n    >>> back_transcribe("ACUGN")\n    \'ACTGN\'\n    '
    if isinstance(rna, Seq):
        return rna.back_transcribe()
    elif isinstance(rna, MutableSeq):
        return Seq(rna).back_transcribe()
    else:
        return rna.replace('U', 'T').replace('u', 't')

def _translate_str(sequence, table, stop_symbol='*', to_stop=False, cds=False, pos_stop='X', gap=None):
    if False:
        for i in range(10):
            print('nop')
    'Translate nucleotide string into a protein string (PRIVATE).\n\n    Arguments:\n     - sequence - a string\n     - table - Which codon table to use?  This can be either a name (string),\n       an NCBI identifier (integer), or a CodonTable object (useful for\n       non-standard genetic codes).  This defaults to the "Standard" table.\n     - stop_symbol - a single character string, what to use for terminators.\n     - to_stop - boolean, should translation terminate at the first\n       in frame stop codon?  If there is no in-frame stop codon\n       then translation continues to the end.\n     - pos_stop - a single character string for a possible stop codon\n       (e.g. TAN or NNN)\n     - cds - Boolean, indicates this is a complete CDS.  If True, this\n       checks the sequence starts with a valid alternative start\n       codon (which will be translated as methionine, M), that the\n       sequence length is a multiple of three, and that there is a\n       single in frame stop codon at the end (this will be excluded\n       from the protein sequence, regardless of the to_stop option).\n       If these tests fail, an exception is raised.\n     - gap - Single character string to denote symbol used for gaps.\n       Defaults to None.\n\n    Returns a string.\n\n    e.g.\n\n    >>> from Bio.Data import CodonTable\n    >>> table = CodonTable.ambiguous_dna_by_id[1]\n    >>> _translate_str("AAA", table)\n    \'K\'\n    >>> _translate_str("TAR", table)\n    \'*\'\n    >>> _translate_str("TAN", table)\n    \'X\'\n    >>> _translate_str("TAN", table, pos_stop="@")\n    \'@\'\n    >>> _translate_str("TA?", table)\n    Traceback (most recent call last):\n       ...\n    Bio.Data.CodonTable.TranslationError: Codon \'TA?\' is invalid\n\n    In a change to older versions of Biopython, partial codons are now\n    always regarded as an error (previously only checked if cds=True)\n    and will trigger a warning (likely to become an exception in a\n    future release).\n\n    If **cds=True**, the start and stop codons are checked, and the start\n    codon will be translated at methionine. The sequence must be an\n    while number of codons.\n\n    >>> _translate_str("ATGCCCTAG", table, cds=True)\n    \'MP\'\n    >>> _translate_str("AAACCCTAG", table, cds=True)\n    Traceback (most recent call last):\n       ...\n    Bio.Data.CodonTable.TranslationError: First codon \'AAA\' is not a start codon\n    >>> _translate_str("ATGCCCTAGCCCTAG", table, cds=True)\n    Traceback (most recent call last):\n       ...\n    Bio.Data.CodonTable.TranslationError: Extra in frame stop codon \'TAG\' found.\n    '
    try:
        table_id = int(table)
    except ValueError:
        try:
            codon_table = CodonTable.ambiguous_generic_by_name[table]
        except KeyError:
            if isinstance(table, str):
                raise ValueError("The Bio.Seq translate methods and function DO NOT take a character string mapping table like the python string object's translate method. Use str(my_seq).translate(...) instead.") from None
            else:
                raise TypeError('table argument must be integer or string') from None
    except (AttributeError, TypeError):
        if isinstance(table, CodonTable.CodonTable):
            codon_table = table
        else:
            raise ValueError('Bad table argument') from None
    else:
        codon_table = CodonTable.ambiguous_generic_by_id[table_id]
    sequence = sequence.upper()
    amino_acids = []
    forward_table = codon_table.forward_table
    stop_codons = codon_table.stop_codons
    if codon_table.nucleotide_alphabet is not None:
        valid_letters = set(codon_table.nucleotide_alphabet.upper())
    else:
        valid_letters = set(IUPACData.ambiguous_dna_letters.upper() + IUPACData.ambiguous_rna_letters.upper())
    n = len(sequence)
    dual_coding = [c for c in stop_codons if c in forward_table]
    if dual_coding:
        c = dual_coding[0]
        if to_stop:
            raise ValueError(f"You cannot use 'to_stop=True' with this table as it contains {len(dual_coding)} codon(s) which can be both STOP and an amino acid (e.g. '{c}' -> '{forward_table[c]}' or STOP).")
        warnings.warn(f"This table contains {len(dual_coding)} codon(s) which code(s) for both STOP and an amino acid (e.g. '{c}' -> '{forward_table[c]}' or STOP). Such codons will be translated as amino acid.", BiopythonWarning)
    if cds:
        if str(sequence[:3]).upper() not in codon_table.start_codons:
            raise CodonTable.TranslationError(f"First codon '{sequence[:3]}' is not a start codon")
        if n % 3 != 0:
            raise CodonTable.TranslationError(f'Sequence length {n} is not a multiple of three')
        if str(sequence[-3:]).upper() not in stop_codons:
            raise CodonTable.TranslationError(f"Final codon '{sequence[-3:]}' is not a stop codon")
        sequence = sequence[3:-3]
        n -= 6
        amino_acids = ['M']
    elif n % 3 != 0:
        warnings.warn('Partial codon, len(sequence) not a multiple of three. Explicitly trim the sequence or add trailing N before translation. This may become an error in future.', BiopythonWarning)
    if gap is not None:
        if not isinstance(gap, str):
            raise TypeError('Gap character should be a single character string.')
        elif len(gap) > 1:
            raise ValueError('Gap character should be a single character string.')
    for i in range(0, n - n % 3, 3):
        codon = sequence[i:i + 3]
        try:
            amino_acids.append(forward_table[codon])
        except (KeyError, CodonTable.TranslationError):
            if codon in codon_table.stop_codons:
                if cds:
                    raise CodonTable.TranslationError(f"Extra in frame stop codon '{codon}' found.") from None
                if to_stop:
                    break
                amino_acids.append(stop_symbol)
            elif valid_letters.issuperset(set(codon)):
                amino_acids.append(pos_stop)
            elif gap is not None and codon == gap * 3:
                amino_acids.append(gap)
            else:
                raise CodonTable.TranslationError(f"Codon '{codon}' is invalid") from None
    return ''.join(amino_acids)

def translate(sequence, table='Standard', stop_symbol='*', to_stop=False, cds=False, gap=None):
    if False:
        return 10
    'Translate a nucleotide sequence into amino acids.\n\n    If given a string, returns a new string object. Given a Seq or\n    MutableSeq, returns a Seq object.\n\n    Arguments:\n     - table - Which codon table to use?  This can be either a name\n       (string), an NCBI identifier (integer), or a CodonTable object\n       (useful for non-standard genetic codes).  Defaults to the "Standard"\n       table.\n     - stop_symbol - Single character string, what to use for any\n       terminators, defaults to the asterisk, "*".\n     - to_stop - Boolean, defaults to False meaning do a full\n       translation continuing on past any stop codons\n       (translated as the specified stop_symbol).  If\n       True, translation is terminated at the first in\n       frame stop codon (and the stop_symbol is not\n       appended to the returned protein sequence).\n     - cds - Boolean, indicates this is a complete CDS.  If True, this\n       checks the sequence starts with a valid alternative start\n       codon (which will be translated as methionine, M), that the\n       sequence length is a multiple of three, and that there is a\n       single in frame stop codon at the end (this will be excluded\n       from the protein sequence, regardless of the to_stop option).\n       If these tests fail, an exception is raised.\n     - gap - Single character string to denote symbol used for gaps.\n       Defaults to None.\n\n    A simple string example using the default (standard) genetic code:\n\n    >>> coding_dna = "GTGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"\n    >>> translate(coding_dna)\n    \'VAIVMGR*KGAR*\'\n    >>> translate(coding_dna, stop_symbol="@")\n    \'VAIVMGR@KGAR@\'\n    >>> translate(coding_dna, to_stop=True)\n    \'VAIVMGR\'\n\n    Now using NCBI table 2, where TGA is not a stop codon:\n\n    >>> translate(coding_dna, table=2)\n    \'VAIVMGRWKGAR*\'\n    >>> translate(coding_dna, table=2, to_stop=True)\n    \'VAIVMGRWKGAR\'\n\n    In fact this example uses an alternative start codon valid under NCBI\n    table 2, GTG, which means this example is a complete valid CDS which\n    when translated should really start with methionine (not valine):\n\n    >>> translate(coding_dna, table=2, cds=True)\n    \'MAIVMGRWKGAR\'\n\n    Note that if the sequence has no in-frame stop codon, then the to_stop\n    argument has no effect:\n\n    >>> coding_dna2 = "GTGGCCATTGTAATGGGCCGC"\n    >>> translate(coding_dna2)\n    \'VAIVMGR\'\n    >>> translate(coding_dna2, to_stop=True)\n    \'VAIVMGR\'\n\n    NOTE - Ambiguous codons like "TAN" or "NNN" could be an amino acid\n    or a stop codon.  These are translated as "X".  Any invalid codon\n    (e.g. "TA?" or "T-A") will throw a TranslationError.\n\n    It will however translate either DNA or RNA.\n\n    NOTE - Since version 1.71 Biopython contains codon tables with \'ambiguous\n    stop codons\'. These are stop codons with unambiguous sequence but which\n    have a context dependent coding as STOP or as amino acid. With these tables\n    \'to_stop\' must be False (otherwise a ValueError is raised). The dual\n    coding codons will always be translated as amino acid, except for\n    \'cds=True\', where the last codon will be translated as STOP.\n\n    >>> coding_dna3 = "ATGGCACGGAAGTGA"\n    >>> translate(coding_dna3)\n    \'MARK*\'\n\n    >>> translate(coding_dna3, table=27)  # Table 27: TGA -> STOP or W\n    \'MARKW\'\n\n    It will however raise a BiopythonWarning (not shown).\n\n    >>> translate(coding_dna3, table=27, cds=True)\n    \'MARK\'\n\n    >>> translate(coding_dna3, table=27, to_stop=True)\n    Traceback (most recent call last):\n       ...\n    ValueError: You cannot use \'to_stop=True\' with this table ...\n    '
    if isinstance(sequence, Seq):
        return sequence.translate(table, stop_symbol, to_stop, cds)
    elif isinstance(sequence, MutableSeq):
        return Seq(sequence).translate(table, stop_symbol, to_stop, cds)
    else:
        return _translate_str(sequence, table, stop_symbol, to_stop, cds, gap=gap)

def reverse_complement(sequence, inplace=None):
    if False:
        return 10
    'Return the reverse complement as a DNA sequence.\n\n    If given a string, returns a new string object.\n    Given a Seq object, returns a new Seq object.\n    Given a MutableSeq, returns a new MutableSeq object.\n    Given a SeqRecord object, returns a new SeqRecord object.\n\n    >>> my_seq = "CGA"\n    >>> reverse_complement(my_seq, inplace=False)\n    \'TCG\'\n    >>> my_seq = Seq("CGA")\n    >>> reverse_complement(my_seq, inplace=False)\n    Seq(\'TCG\')\n    >>> my_seq = MutableSeq("CGA")\n    >>> reverse_complement(my_seq, inplace=False)\n    MutableSeq(\'TCG\')\n    >>> my_seq\n    MutableSeq(\'CGA\')\n\n    Any U in the sequence is treated as a T:\n\n    >>> reverse_complement(Seq("CGAUT"), inplace=False)\n    Seq(\'AATCG\')\n\n    In contrast, ``reverse_complement_rna`` returns an RNA sequence:\n\n    >>> reverse_complement_rna(Seq("CGAUT"))\n    Seq(\'AAUCG\')\n\n    Supports and lower- and upper-case characters, and unambiguous and\n    ambiguous nucleotides. All other characters are not converted:\n\n    >>> reverse_complement("ACGTUacgtuXYZxyz", inplace=False)\n    \'zrxZRXaacgtAACGT\'\n\n    The sequence is modified in-place and returned if inplace is True:\n\n    >>> my_seq = MutableSeq("CGA")\n    >>> reverse_complement(my_seq, inplace=True)\n    MutableSeq(\'TCG\')\n    >>> my_seq\n    MutableSeq(\'TCG\')\n\n    As strings and ``Seq`` objects are immutable, a ``TypeError`` is\n    raised if ``reverse_complement`` is called on a ``Seq`` object with\n    ``inplace=True``.\n    '
    from Bio.SeqRecord import SeqRecord
    if inplace is None:
        if isinstance(sequence, Seq):
            if b'U' in sequence._data or b'u' in sequence._data:
                warnings.warn('reverse_complement(sequence) will change in the near future to always return DNA nucleotides only. Please use\n\nreverse_complement_rna(sequence)\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
                if b'T' in sequence._data or b't' in sequence._data:
                    raise ValueError('Mixed RNA/DNA found')
                return sequence.reverse_complement_rna()
        elif isinstance(sequence, MutableSeq):
            warnings.warn('reverse_complement(mutable_seq) will change in the near future to return a MutableSeq object instead of a Seq object.', BiopythonDeprecationWarning)
            return Seq(sequence).reverse_complement()
        elif 'U' in sequence or 'u' in sequence:
            warnings.warn('reverse_complement(sequence) will change in the near future to always return DNA nucleotides only. Please use\n\nreverse_complement_rna(sequence)\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
            if 'T' in sequence or 't' in sequence:
                raise ValueError('Mixed RNA/DNA found')
            sequence = sequence.encode('ASCII')
            sequence = sequence.translate(_rna_complement_table)
            return sequence.decode('ASCII')[::-1]
    if isinstance(sequence, (Seq, MutableSeq)):
        return sequence.reverse_complement(inplace)
    if isinstance(sequence, SeqRecord):
        if inplace:
            raise TypeError('SeqRecords are immutable')
        return sequence.reverse_complement()
    if inplace:
        raise TypeError('strings are immutable')
    sequence = sequence.encode('ASCII')
    sequence = sequence.translate(_dna_complement_table)
    sequence = sequence.decode('ASCII')
    return sequence[::-1]

def reverse_complement_rna(sequence, inplace=False):
    if False:
        return 10
    'Return the reverse complement as an RNA sequence.\n\n    If given a string, returns a new string object.\n    Given a Seq object, returns a new Seq object.\n    Given a MutableSeq, returns a new MutableSeq object.\n    Given a SeqRecord object, returns a new SeqRecord object.\n\n    >>> my_seq = "CGA"\n    >>> reverse_complement_rna(my_seq)\n    \'UCG\'\n    >>> my_seq = Seq("CGA")\n    >>> reverse_complement_rna(my_seq)\n    Seq(\'UCG\')\n    >>> my_seq = MutableSeq("CGA")\n    >>> reverse_complement_rna(my_seq)\n    MutableSeq(\'UCG\')\n    >>> my_seq\n    MutableSeq(\'CGA\')\n\n    Any T in the sequence is treated as a U:\n\n    >>> reverse_complement_rna(Seq("CGAUT"))\n    Seq(\'AAUCG\')\n\n    In contrast, ``reverse_complement`` returns a DNA sequence:\n\n    >>> reverse_complement(Seq("CGAUT"), inplace=False)\n    Seq(\'AATCG\')\n\n    Supports and lower- and upper-case characters, and unambiguous and\n    ambiguous nucleotides. All other characters are not converted:\n\n    >>> reverse_complement_rna("ACGTUacgtuXYZxyz")\n    \'zrxZRXaacguAACGU\'\n\n    The sequence is modified in-place and returned if inplace is True:\n\n    >>> my_seq = MutableSeq("CGA")\n    >>> reverse_complement_rna(my_seq, inplace=True)\n    MutableSeq(\'UCG\')\n    >>> my_seq\n    MutableSeq(\'UCG\')\n\n    As strings and ``Seq`` objects are immutable, a ``TypeError`` is\n    raised if ``reverse_complement`` is called on a ``Seq`` object with\n    ``inplace=True``.\n    '
    from Bio.SeqRecord import SeqRecord
    if isinstance(sequence, (Seq, MutableSeq)):
        return sequence.reverse_complement_rna(inplace)
    if isinstance(sequence, SeqRecord):
        if inplace:
            raise TypeError('SeqRecords are immutable')
        return sequence.reverse_complement_rna()
    if inplace:
        raise TypeError('strings are immutable')
    sequence = sequence.encode('ASCII')
    sequence = sequence.translate(_rna_complement_table)
    sequence = sequence.decode('ASCII')
    return sequence[::-1]

def complement(sequence, inplace=None):
    if False:
        while True:
            i = 10
    'Return the complement as a DNA sequence.\n\n    If given a string, returns a new string object.\n    Given a Seq object, returns a new Seq object.\n    Given a MutableSeq, returns a new MutableSeq object.\n    Given a SeqRecord object, returns a new SeqRecord object.\n\n    >>> my_seq = "CGA"\n    >>> complement(my_seq, inplace=False)\n    \'GCT\'\n    >>> my_seq = Seq("CGA")\n    >>> complement(my_seq, inplace=False)\n    Seq(\'GCT\')\n    >>> my_seq = MutableSeq("CGA")\n    >>> complement(my_seq, inplace=False)\n    MutableSeq(\'GCT\')\n    >>> my_seq\n    MutableSeq(\'CGA\')\n\n    Any U in the sequence is treated as a T:\n\n    >>> complement(Seq("CGAUT"), inplace=False)\n    Seq(\'GCTAA\')\n\n    In contrast, ``complement_rna`` returns an RNA sequence:\n\n    >>> complement_rna(Seq("CGAUT"))\n    Seq(\'GCUAA\')\n\n    Supports and lower- and upper-case characters, and unambiguous and\n    ambiguous nucleotides. All other characters are not converted:\n\n    >>> complement("ACGTUacgtuXYZxyz", inplace=False)\n    \'TGCAAtgcaaXRZxrz\'\n\n    The sequence is modified in-place and returned if inplace is True:\n\n    >>> my_seq = MutableSeq("CGA")\n    >>> complement(my_seq, inplace=True)\n    MutableSeq(\'GCT\')\n    >>> my_seq\n    MutableSeq(\'GCT\')\n\n    As strings and ``Seq`` objects are immutable, a ``TypeError`` is\n    raised if ``reverse_complement`` is called on a ``Seq`` object with\n    ``inplace=True``.\n    '
    from Bio.SeqRecord import SeqRecord
    if inplace is None:
        if isinstance(sequence, Seq):
            if b'U' in sequence._data or b'u' in sequence._data:
                warnings.warn('complement(sequence) will change in the near future to always return DNA nucleotides only. Please use\n\ncomplement_rna(sequence)\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
                if b'T' in sequence._data or b't' in sequence._data:
                    raise ValueError('Mixed RNA/DNA found')
                return sequence.complement_rna()
        elif isinstance(sequence, MutableSeq):
            warnings.warn('complement(mutable_seq) will change in the near futureto return a MutableSeq object instead of a Seq object.', BiopythonDeprecationWarning)
            return Seq(sequence).complement()
        elif 'U' in sequence or 'u' in sequence:
            warnings.warn('complement(sequence) will change in the near future to always return DNA nucleotides only. Please use\n\ncomplement_rna(sequence)\n\nif you want to receive an RNA sequence instead.', BiopythonDeprecationWarning)
            if 'T' in sequence or 't' in sequence:
                raise ValueError('Mixed RNA/DNA found')
            ttable = _rna_complement_table
            sequence = sequence.encode('ASCII')
            sequence = sequence.translate(ttable)
            return sequence.decode('ASCII')
    if isinstance(sequence, (Seq, MutableSeq)):
        return sequence.complement(inplace)
    if isinstance(sequence, SeqRecord):
        if inplace:
            raise TypeError('SeqRecords are immutable')
        return sequence.complement()
    if inplace:
        raise TypeError('strings are immutable')
    sequence = sequence.encode('ASCII')
    sequence = sequence.translate(_dna_complement_table)
    return sequence.decode('ASCII')

def complement_rna(sequence, inplace=False):
    if False:
        print('Hello World!')
    'Return the complement as an RNA sequence.\n\n    If given a string, returns a new string object.\n    Given a Seq object, returns a new Seq object.\n    Given a MutableSeq, returns a new MutableSeq object.\n    Given a SeqRecord object, returns a new SeqRecord object.\n\n    >>> my_seq = "CGA"\n    >>> complement_rna(my_seq)\n    \'GCU\'\n    >>> my_seq = Seq("CGA")\n    >>> complement_rna(my_seq)\n    Seq(\'GCU\')\n    >>> my_seq = MutableSeq("CGA")\n    >>> complement_rna(my_seq)\n    MutableSeq(\'GCU\')\n    >>> my_seq\n    MutableSeq(\'CGA\')\n\n    Any T in the sequence is treated as a U:\n\n    >>> complement_rna(Seq("CGAUT"))\n    Seq(\'GCUAA\')\n\n    In contrast, ``complement`` returns a DNA sequence:\n\n    >>> complement(Seq("CGAUT"),inplace=False)\n    Seq(\'GCTAA\')\n\n    Supports and lower- and upper-case characters, and unambiguous and\n    ambiguous nucleotides. All other characters are not converted:\n\n    >>> complement_rna("ACGTUacgtuXYZxyz")\n    \'UGCAAugcaaXRZxrz\'\n\n    The sequence is modified in-place and returned if inplace is True:\n\n    >>> my_seq = MutableSeq("CGA")\n    >>> complement(my_seq, inplace=True)\n    MutableSeq(\'GCT\')\n    >>> my_seq\n    MutableSeq(\'GCT\')\n\n    As strings and ``Seq`` objects are immutable, a ``TypeError`` is\n    raised if ``reverse_complement`` is called on a ``Seq`` object with\n    ``inplace=True``.\n    '
    from Bio.SeqRecord import SeqRecord
    if isinstance(sequence, (Seq, MutableSeq)):
        return sequence.complement_rna(inplace)
    if isinstance(sequence, SeqRecord):
        if inplace:
            raise TypeError('SeqRecords are immutable')
        return sequence.complement_rna()
    if inplace:
        raise TypeError('strings are immutable')
    sequence = sequence.encode('ASCII')
    sequence = sequence.translate(_rna_complement_table)
    return sequence.decode('ASCII')

def _test():
    if False:
        i = 10
        return i + 15
    "Run the Bio.Seq module's doctests (PRIVATE)."
    print('Running doctests...')
    import doctest
    doctest.testmod(optionflags=doctest.IGNORE_EXCEPTION_DETAIL)
    print('Done')
if __name__ == '__main__':
    _test()