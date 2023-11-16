"""Restriction Enzyme classes.

Notes about the diverses class of the restriction enzyme implementation::

            RestrictionType is the type of all restriction enzymes.
        -----------------------------------------------------------------------
            AbstractCut implements some methods that are common to all enzymes.
        -----------------------------------------------------------------------
            NoCut, OneCut,TwoCuts   represent the number of double strand cuts
                                    produced by the enzyme.
                                    they correspond to the 4th field of the
                                    rebase record emboss_e.NNN.
                    0->NoCut    : the enzyme is not characterised.
                    2->OneCut   : the enzyme produce one double strand cut.
                    4->TwoCuts  : two double strand cuts.
        -----------------------------------------------------------------------
            Meth_Dep, Meth_Undep    represent the methylation susceptibility to
                                    the enzyme.
                                    Not implemented yet.
        -----------------------------------------------------------------------
            Palindromic,            if the site is palindromic or not.
            NotPalindromic          allow some optimisations of the code.
                                    No need to check the reverse strand
                                    with palindromic sites.
        -----------------------------------------------------------------------
            Unknown, Blunt,         represent the overhang.
            Ov5, Ov3                Unknown is here for symmetry reasons and
                                    correspond to enzymes that are not
                                    characterised in rebase.
        -----------------------------------------------------------------------
            Defined, Ambiguous,     represent the sequence of the overhang.
            NotDefined
                                    NotDefined is for enzymes not characterised
                                    in rebase.

                                    Defined correspond to enzymes that display
                                    a constant overhang whatever the sequence.
                                    ex : EcoRI. G^AATTC -> overhang :AATT
                                                CTTAA^G

                                    Ambiguous : the overhang varies with the
                                    sequence restricted.
                                    Typically enzymes which cut outside their
                                    restriction site or (but not always)
                                    inside an ambiguous site.
                                    ex:
                                    AcuI CTGAAG(22/20)  -> overhang : NN
                                    AasI GACNNN^NNNGTC  -> overhang : NN
                                         CTGN^NNNNNCAG

                note : these 3 classes refers to the overhang not the site.
                   So the enzyme ApoI (RAATTY) is defined even if its
                   restriction site is ambiguous.

                        ApoI R^AATTY -> overhang : AATT -> Defined
                             YTTAA^R
                   Accordingly, blunt enzymes are always Defined even
                   when they cut outside their restriction site.
        -----------------------------------------------------------------------
            Not_available,          as found in rebase file emboss_r.NNN files.
            Commercially_available
                                    allow the selection of the enzymes
                                    according to their suppliers to reduce the
                                    quantity of results.
                                    Also will allow the implementation of
                                    buffer compatibility tables. Not
                                    implemented yet.

                                    the list of suppliers is extracted from
                                    emboss_s.NNN
        -----------------------------------------------------------------------

"""
import warnings
import re
import string
import itertools
from Bio.Seq import Seq, MutableSeq
from Bio.Restriction.Restriction_Dictionary import rest_dict as enzymedict
from Bio.Restriction.Restriction_Dictionary import typedict
from Bio.Restriction.Restriction_Dictionary import suppliers as suppliers_dict
from Bio.Restriction.PrintFormat import PrintFormat
from Bio import BiopythonWarning
matching = {'A': 'ARWMHVDN', 'C': 'CYSMHBVN', 'G': 'GRSKBVDN', 'T': 'TYWKHBDN', 'R': 'ABDGHKMNSRWV', 'Y': 'CBDHKMNSTWVY', 'W': 'ABDHKMNRTWVY', 'S': 'CBDGHKMNSRVY', 'M': 'ACBDHMNSRWVY', 'K': 'BDGHKNSRTWVY', 'H': 'ACBDHKMNSRTWVY', 'B': 'CBDGHKMNSRTWVY', 'V': 'ACBDGHKMNSRWVY', 'D': 'ABDGHKMNSRTWVY', 'N': 'ACBDGHKMNSRTWVY'}
DNA = Seq

def _make_FormattedSeq_table() -> bytes:
    if False:
        return 10
    table = bytearray(256)
    upper_to_lower = ord('A') - ord('a')
    for c in b'ABCDGHKMNRSTVWY':
        table[c] = c
        table[c - upper_to_lower] = c
    return bytes(table)

class FormattedSeq:
    """A linear or circular sequence object for restriction analysis.

    Translates a Bio.Seq into a formatted sequence to be used with Restriction.

    Roughly: remove anything which is not IUPAC alphabet and then add a space
             in front of the sequence to get a biological index instead of a
             python index (i.e. index of the first base is 1 not 0).

    Retains information about the shape of the molecule linear (default) or
    circular. Restriction sites are search over the edges of circular sequence.
    """
    _remove_chars = string.whitespace.encode() + string.digits.encode()
    _table = _make_FormattedSeq_table()

    def __init__(self, seq, linear=True):
        if False:
            for i in range(10):
                print('nop')
        'Initialize ``FormattedSeq`` with sequence and topology (optional).\n\n        ``seq`` is either a ``Bio.Seq``, ``Bio.MutableSeq`` or a\n        ``FormattedSeq``. If ``seq`` is a ``FormattedSeq``, ``linear``\n        will have no effect on the shape of the sequence.\n        '
        if isinstance(seq, (Seq, MutableSeq)):
            self.lower = seq.islower()
            data = bytes(seq)
            self.data = data.translate(self._table, delete=self._remove_chars)
            if 0 in self.data:
                raise TypeError(f'Invalid character found in {data.decode()}')
            self.data = ' ' + self.data.decode('ASCII')
            self.linear = linear
            self.klass = seq.__class__
        elif isinstance(seq, FormattedSeq):
            self.lower = seq.lower
            self.data = seq.data
            self.linear = seq.linear
            self.klass = seq.klass
        else:
            raise TypeError(f'expected Seq or MutableSeq, got {type(seq)}')

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return length of ``FormattedSeq``.\n\n        ``FormattedSeq`` has a leading space, thus subtract 1.\n        '
        return len(self.data) - 1

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Represent ``FormattedSeq`` class as a string.'
        return f'FormattedSeq({self[1:]!r}, linear={self.linear!r})'

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        'Implement equality operator for ``FormattedSeq`` object.'
        if isinstance(other, FormattedSeq):
            if repr(self) == repr(other):
                return True
            else:
                return False
        return False

    def circularise(self):
        if False:
            i = 10
            return i + 15
        'Circularise sequence in place.'
        self.linear = False

    def linearise(self):
        if False:
            i = 10
            return i + 15
        'Linearise sequence in place.'
        self.linear = True

    def to_linear(self):
        if False:
            while True:
                i = 10
        'Make a new instance of sequence as linear.'
        new = self.__class__(self)
        new.linear = True
        return new

    def to_circular(self):
        if False:
            return 10
        'Make a new instance of sequence as circular.'
        new = self.__class__(self)
        new.linear = False
        return new

    def is_linear(self):
        if False:
            for i in range(10):
                print('nop')
        'Return if sequence is linear (True) or circular (False).'
        return self.linear

    def finditer(self, pattern, size):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of a given pattern which occurs in the sequence.\n\n        The list is made of tuple (location, pattern.group).\n        The latter is used with non palindromic sites.\n        Pattern is the regular expression pattern corresponding to the\n        enzyme restriction site.\n        Size is the size of the restriction enzyme recognition-site size.\n        '
        if self.is_linear():
            data = self.data
        else:
            data = self.data + self.data[1:size]
        return [(i.start(), i.group) for i in re.finditer(pattern, data)]

    def __getitem__(self, i):
        if False:
            i = 10
            return i + 15
        "Return substring of ``FormattedSeq``.\n\n        The class of the returned object is the class of the respective\n        sequence. Note that due to the leading space, indexing is 1-based:\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.Restriction.Restriction import FormattedSeq\n        >>> f_seq = FormattedSeq(Seq('ATGCATGC'))\n        >>> f_seq[1]\n        Seq('A')\n\n        "
        if self.lower:
            return self.klass(self.data[i].lower())
        return self.klass(self.data[i])

class RestrictionType(type):
    """RestrictionType. Type from which all enzyme classes are derived.

    Implement the operator methods.
    """

    def __init__(cls, name='', bases=(), dct=None):
        if False:
            i = 10
            return i + 15
        'Initialize RestrictionType instance.\n\n        Not intended to be used in normal operation. The enzymes are\n        instantiated when importing the module.\n        See below.\n        '
        if '-' in name:
            raise ValueError(f'Problem with hyphen in {name!r} as enzyme name')
        try:
            cls.compsite = re.compile(cls.compsite)
        except AttributeError:
            pass
        except Exception:
            raise ValueError(f'Problem with regular expression, re.compiled({cls.compsite!r})') from None

    def __add__(cls, other):
        if False:
            print('Hello World!')
        'Add restriction enzyme to a RestrictionBatch().\n\n        If other is an enzyme returns a batch of the two enzymes.\n        If other is already a RestrictionBatch add enzyme to it.\n        '
        if isinstance(other, RestrictionType):
            return RestrictionBatch([cls, other])
        elif isinstance(other, RestrictionBatch):
            return other.add_nocheck(cls)
        else:
            raise TypeError

    def __truediv__(cls, other):
        if False:
            for i in range(10):
                print('nop')
        "Override '/' operator to use as search method.\n\n        >>> from Bio.Restriction import EcoRI\n        >>> EcoRI/Seq('GAATTC')\n        [2]\n\n        Returns RE.search(other).\n        "
        return cls.search(other)

    def __rtruediv__(cls, other):
        if False:
            for i in range(10):
                print('nop')
        "Override division with reversed operands to use as search method.\n\n        >>> from Bio.Restriction import EcoRI\n        >>> Seq('GAATTC')/EcoRI\n        [2]\n\n        Returns RE.search(other).\n        "
        return cls.search(other)

    def __floordiv__(cls, other):
        if False:
            i = 10
            return i + 15
        "Override '//' operator to use as catalyse method.\n\n        >>> from Bio.Restriction import EcoRI\n        >>> EcoRI//Seq('GAATTC')\n        (Seq('G'), Seq('AATTC'))\n\n        Returns RE.catalyse(other).\n        "
        return cls.catalyse(other)

    def __rfloordiv__(cls, other):
        if False:
            print('Hello World!')
        "As __floordiv__, with reversed operands.\n\n        >>> from Bio.Restriction import EcoRI\n        >>> Seq('GAATTC')//EcoRI\n        (Seq('G'), Seq('AATTC'))\n\n        Returns RE.catalyse(other).\n        "
        return cls.catalyse(other)

    def __str__(cls):
        if False:
            while True:
                i = 10
        'Return the name of the enzyme as string.'
        return cls.__name__

    def __repr__(cls):
        if False:
            i = 10
            return i + 15
        'Implement repr method.\n\n        Used with eval or exec will instantiate the enzyme.\n        '
        return f'{cls.__name__}'

    def __len__(cls):
        if False:
            return 10
        'Return length of recognition site of enzyme as int.'
        try:
            return cls.size
        except AttributeError:
            return 0

    def __hash__(cls):
        if False:
            return 10
        'Implement ``hash()`` method for ``RestrictionType``.\n\n        Python default is to use ``id(...)``\n        This is consistent with the ``__eq__`` implementation\n        '
        return id(cls)

    def __eq__(cls, other):
        if False:
            print('Hello World!')
        "Override '==' operator.\n\n        True if RE and other are the same enzyme.\n\n        Specifically this checks they are the same Python object.\n        "
        return id(cls) == id(other)

    def __ne__(cls, other):
        if False:
            print('Hello World!')
        "Override '!=' operator.\n\n        Isoschizomer strict (same recognition site, same restriction) -> False\n        All the other-> True\n\n        WARNING - This is not the inverse of the __eq__ method\n\n        >>> from Bio.Restriction import SacI, SstI\n        >>> SacI != SstI  # true isoschizomers\n        False\n        >>> SacI == SstI\n        False\n        "
        if not isinstance(other, RestrictionType):
            return True
        elif cls.charac == other.charac:
            return False
        else:
            return True

    def __rshift__(cls, other):
        if False:
            print('Hello World!')
        "Override '>>' operator to test for neoschizomers.\n\n        neoschizomer : same recognition site, different restriction. -> True\n        all the others :                                             -> False\n\n        >>> from Bio.Restriction import SmaI, XmaI\n        >>> SmaI >> XmaI\n        True\n        "
        if not isinstance(other, RestrictionType):
            return False
        elif cls.site == other.site and cls.charac != other.charac:
            return True
        else:
            return False

    def __mod__(cls, other):
        if False:
            for i in range(10):
                print('nop')
        "Override '%' operator to test for compatible overhangs.\n\n        True if a and b have compatible overhang.\n\n        >>> from Bio.Restriction import XhoI, SalI\n        >>> XhoI % SalI\n        True\n        "
        if not isinstance(other, RestrictionType):
            raise TypeError(f'expected RestrictionType, got {type(other)} instead')
        return cls._mod1(other)

    def __ge__(cls, other):
        if False:
            for i in range(10):
                print('nop')
        "Compare length of recognition site of two enzymes.\n\n        Override '>='. a is greater or equal than b if the a site is longer\n        than b site. If their site have the same length sort by alphabetical\n        order of their names.\n\n        >>> from Bio.Restriction import EcoRI, EcoRV\n        >>> EcoRI.size\n        6\n        >>> EcoRV.size\n        6\n        >>> EcoRI >= EcoRV\n        False\n        "
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        if len(cls) > len(other):
            return True
        elif cls.size == len(other) and cls.__name__ >= other.__name__:
            return True
        else:
            return False

    def __gt__(cls, other):
        if False:
            i = 10
            return i + 15
        "Compare length of recognition site of two enzymes.\n\n        Override '>'. Sorting order:\n\n        1. size of the recognition site.\n        2. if equal size, alphabetical order of the names.\n\n        "
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        if len(cls) > len(other):
            return True
        elif cls.size == len(other) and cls.__name__ > other.__name__:
            return True
        else:
            return False

    def __le__(cls, other):
        if False:
            return 10
        "Compare length of recognition site of two enzymes.\n\n        Override '<='. Sorting order:\n\n        1. size of the recognition site.\n        2. if equal size, alphabetical order of the names.\n\n        "
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        elif len(cls) < len(other):
            return True
        elif len(cls) == len(other) and cls.__name__ <= other.__name__:
            return True
        else:
            return False

    def __lt__(cls, other):
        if False:
            for i in range(10):
                print('nop')
        "Compare length of recognition site of two enzymes.\n\n        Override '<'. Sorting order:\n\n        1. size of the recognition site.\n        2. if equal size, alphabetical order of the names.\n\n        "
        if not isinstance(other, RestrictionType):
            raise NotImplementedError
        elif len(cls) < len(other):
            return True
        elif len(cls) == len(other) and cls.__name__ < other.__name__:
            return True
        else:
            return False

class AbstractCut(RestrictionType):
    """Implement the methods that are common to all restriction enzymes.

    All the methods are classmethod.

    For internal use only. Not meant to be instantiated.
    """

    @classmethod
    def search(cls, dna, linear=True):
        if False:
            while True:
                i = 10
        "Return a list of cutting sites of the enzyme in the sequence.\n\n        Compensate for circular sequences and so on.\n\n        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.\n\n        If linear is False, the restriction sites that span over the boundaries\n        will be included.\n\n        The positions are the first base of the 3' fragment,\n        i.e. the first base after the position the enzyme will cut.\n        "
        if isinstance(dna, FormattedSeq):
            cls.dna = dna
            return cls._search()
        else:
            cls.dna = FormattedSeq(dna, linear)
            return cls._search()

    @classmethod
    def all_suppliers(cls):
        if False:
            i = 10
            return i + 15
        'Print all the suppliers of restriction enzyme.'
        supply = sorted((x[0] for x in suppliers_dict.values()))
        print(',\n'.join(supply))

    @classmethod
    def is_equischizomer(cls, other):
        if False:
            return 10
        'Test for real isoschizomer.\n\n        True if other is an isoschizomer of RE, but not an neoschizomer,\n        else False.\n\n        Equischizomer: same site, same position of restriction.\n\n        >>> from Bio.Restriction import SacI, SstI, SmaI, XmaI\n        >>> SacI.is_equischizomer(SstI)\n        True\n        >>> SmaI.is_equischizomer(XmaI)\n        False\n\n        '
        return not cls != other

    @classmethod
    def is_neoschizomer(cls, other):
        if False:
            for i in range(10):
                print('nop')
        'Test for neoschizomer.\n\n        True if other is an isoschizomer of RE, else False.\n        Neoschizomer: same site, different position of restriction.\n        '
        return cls >> other

    @classmethod
    def is_isoschizomer(cls, other):
        if False:
            return 10
        'Test for same recognition site.\n\n        True if other has the same recognition site, else False.\n\n        Isoschizomer: same site.\n\n        >>> from Bio.Restriction import SacI, SstI, SmaI, XmaI\n        >>> SacI.is_isoschizomer(SstI)\n        True\n        >>> SmaI.is_isoschizomer(XmaI)\n        True\n\n        '
        return not cls != other or cls >> other

    @classmethod
    def equischizomers(cls, batch=None):
        if False:
            while True:
                i = 10
        'List equischizomers of the enzyme.\n\n        Return a tuple of all the isoschizomers of RE.\n        If batch is supplied it is used instead of the default AllEnzymes.\n\n        Equischizomer: same site, same position of restriction.\n        '
        if not batch:
            batch = AllEnzymes
        r = [x for x in batch if not cls != x]
        i = r.index(cls)
        del r[i]
        r.sort()
        return r

    @classmethod
    def neoschizomers(cls, batch=None):
        if False:
            i = 10
            return i + 15
        'List neoschizomers of the enzyme.\n\n        Return a tuple of all the neoschizomers of RE.\n        If batch is supplied it is used instead of the default AllEnzymes.\n\n        Neoschizomer: same site, different position of restriction.\n        '
        if not batch:
            batch = AllEnzymes
        r = sorted((x for x in batch if cls >> x))
        return r

    @classmethod
    def isoschizomers(cls, batch=None):
        if False:
            while True:
                i = 10
        'List all isoschizomers of the enzyme.\n\n        Return a tuple of all the equischizomers and neoschizomers of RE.\n        If batch is supplied it is used instead of the default AllEnzymes.\n        '
        if not batch:
            batch = AllEnzymes
        r = [x for x in batch if cls >> x or not cls != x]
        i = r.index(cls)
        del r[i]
        r.sort()
        return r

    @classmethod
    def frequency(cls):
        if False:
            return 10
        "Return the theoretically cutting frequency of the enzyme.\n\n        Frequency of the site, given as 'one cut per x bases' (int).\n        "
        return cls.freq

class NoCut(AbstractCut):
    """Implement the methods specific to the enzymes that do not cut.

    These enzymes are generally enzymes that have been only partially
    characterised and the way they cut the DNA is unknown or enzymes for
    which the pattern of cut is to complex to be recorded in Rebase
    (ncuts values of 0 in emboss_e.###).

    When using search() with these enzymes the values returned are at the start
    of the restriction site.

    Their catalyse() method returns a TypeError.

    Unknown and NotDefined are also part of the base classes of these enzymes.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def cut_once(cls):
        if False:
            return 10
        'Return if the cutting pattern has one cut.\n\n        True if the enzyme cut the sequence one time on each strand.\n        '
        return False

    @classmethod
    def cut_twice(cls):
        if False:
            while True:
                i = 10
        'Return if the cutting pattern has two cuts.\n\n        True if the enzyme cut the sequence twice on each strand.\n        '
        return False

    @classmethod
    def _modify(cls, location):
        if False:
            i = 10
            return i + 15
        'Return a generator that moves the cutting position by 1 (PRIVATE).\n\n        For internal use only.\n\n        location is an integer corresponding to the location of the match for\n        the enzyme pattern in the sequence.\n        _modify returns the real place where the enzyme will cut.\n\n        Example::\n\n            EcoRI pattern : GAATTC\n            EcoRI will cut after the G.\n            so in the sequence:\n                     ______\n            GAATACACGGAATTCGA\n                     |\n                     10\n            dna.finditer(GAATTC, 6) will return 10 as G is the 10th base\n            EcoRI cut after the G so:\n            EcoRI._modify(10) -> 11.\n\n        If the enzyme cut twice _modify will returns two integer corresponding\n        to each cutting site.\n        '
        yield location

    @classmethod
    def _rev_modify(cls, location):
        if False:
            return 10
        'Return a generator that moves the cutting position by 1 (PRIVATE).\n\n        For internal use only.\n\n        As _modify for site situated on the antiparallel strand when the\n        enzyme is not palindromic.\n        '
        yield location

    @classmethod
    def characteristic(cls):
        if False:
            while True:
                i = 10
        "Return a list of the enzyme's characteristics as tuple.\n\n        the tuple contains the attributes:\n\n        - fst5 -> first 5' cut ((current strand) or None\n        - fst3 -> first 3' cut (complementary strand) or None\n        - scd5 -> second 5' cut (current strand) or None\n        - scd5 -> second 3' cut (complementary strand) or None\n        - site -> recognition site.\n\n        "
        return (None, None, None, None, cls.site)

class OneCut(AbstractCut):
    """Implement the methods for enzymes that cut the DNA only once.

    Correspond to ncuts values of 2 in emboss_e.###

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def cut_once(cls):
        if False:
            return 10
        'Return if the cutting pattern has one cut.\n\n        True if the enzyme cut the sequence one time on each strand.\n        '
        return True

    @classmethod
    def cut_twice(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return if the cutting pattern has two cuts.\n\n        True if the enzyme cut the sequence twice on each strand.\n        '
        return False

    @classmethod
    def _modify(cls, location):
        if False:
            for i in range(10):
                print('nop')
        'Return a generator that moves the cutting position by 1 (PRIVATE).\n\n        For internal use only.\n\n        location is an integer corresponding to the location of the match for\n        the enzyme pattern in the sequence.\n        _modify returns the real place where the enzyme will cut.\n\n        Example::\n\n            EcoRI pattern : GAATTC\n            EcoRI will cut after the G.\n            so in the sequence:\n                     ______\n            GAATACACGGAATTCGA\n                     |\n                     10\n            dna.finditer(GAATTC, 6) will return 10 as G is the 10th base\n            EcoRI cut after the G so:\n            EcoRI._modify(10) -> 11.\n\n        if the enzyme cut twice _modify will returns two integer corresponding\n        to each cutting site.\n        '
        yield (location + cls.fst5)

    @classmethod
    def _rev_modify(cls, location):
        if False:
            i = 10
            return i + 15
        'Return a generator that moves the cutting position by 1 (PRIVATE).\n\n        For internal use only.\n\n        As _modify for site situated on the antiparallel strand when the\n        enzyme is not palindromic\n        '
        yield (location - cls.fst3)

    @classmethod
    def characteristic(cls):
        if False:
            print('Hello World!')
        "Return a list of the enzyme's characteristics as tuple.\n\n        The tuple contains the attributes:\n\n        - fst5 -> first 5' cut ((current strand) or None\n        - fst3 -> first 3' cut (complementary strand) or None\n        - scd5 -> second 5' cut (current strand) or None\n        - scd5 -> second 3' cut (complementary strand) or None\n        - site -> recognition site.\n\n        "
        return (cls.fst5, cls.fst3, None, None, cls.site)

class TwoCuts(AbstractCut):
    """Implement the methods for enzymes that cut the DNA twice.

    Correspond to ncuts values of 4 in emboss_e.###

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def cut_once(cls):
        if False:
            while True:
                i = 10
        'Return if the cutting pattern has one cut.\n\n        True if the enzyme cut the sequence one time on each strand.\n        '
        return False

    @classmethod
    def cut_twice(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return if the cutting pattern has two cuts.\n\n        True if the enzyme cut the sequence twice on each strand.\n        '
        return True

    @classmethod
    def _modify(cls, location):
        if False:
            for i in range(10):
                print('nop')
        'Return a generator that moves the cutting position by 1 (PRIVATE).\n\n        For internal use only.\n\n        location is an integer corresponding to the location of the match for\n        the enzyme pattern in the sequence.\n        _modify returns the real place where the enzyme will cut.\n\n        example::\n\n            EcoRI pattern : GAATTC\n            EcoRI will cut after the G.\n            so in the sequence:\n                     ______\n            GAATACACGGAATTCGA\n                     |\n                     10\n            dna.finditer(GAATTC, 6) will return 10 as G is the 10th base\n            EcoRI cut after the G so:\n            EcoRI._modify(10) -> 11.\n\n        if the enzyme cut twice _modify will returns two integer corresponding\n        to each cutting site.\n        '
        yield (location + cls.fst5)
        yield (location + cls.scd5)

    @classmethod
    def _rev_modify(cls, location):
        if False:
            return 10
        'Return a generator that moves the cutting position by 1 (PRIVATE).\n\n        for internal use only.\n\n        as _modify for site situated on the antiparallel strand when the\n        enzyme is not palindromic\n        '
        yield (location - cls.fst3)
        yield (location - cls.scd3)

    @classmethod
    def characteristic(cls):
        if False:
            return 10
        "Return a list of the enzyme's characteristics as tuple.\n\n        the tuple contains the attributes:\n\n        - fst5 -> first 5' cut ((current strand) or None\n        - fst3 -> first 3' cut (complementary strand) or None\n        - scd5 -> second 5' cut (current strand) or None\n        - scd5 -> second 3' cut (complementary strand) or None\n        - site -> recognition site.\n\n        "
        return (cls.fst5, cls.fst3, cls.scd5, cls.scd3, cls.site)

class Meth_Dep(AbstractCut):
    """Implement the information about methylation.

    Enzymes of this class possess a site which is methylable.
    """

    @classmethod
    def is_methylable(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return if recognition site can be methylated.\n\n        True if the recognition site is a methylable.\n        '
        return True

class Meth_Undep(AbstractCut):
    """Implement information about methylation sensitibility.

    Enzymes of this class are not sensible to methylation.
    """

    @classmethod
    def is_methylable(cls):
        if False:
            return 10
        'Return if recognition site can be methylated.\n\n        True if the recognition site is a methylable.\n        '
        return False

class Palindromic(AbstractCut):
    """Implement methods for enzymes with palindromic recognition sites.

    palindromic means : the recognition site and its reverse complement are
                        identical.
    Remarks     : an enzyme with a site CGNNCG is palindromic even if some
                  of the sites that it will recognise are not.
                  for example here : CGAACG

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def _search(cls):
        if False:
            return 10
        'Return a list of cutting sites of the enzyme in the sequence (PRIVATE).\n\n        For internal use only.\n\n        Implement the search method for palindromic enzymes.\n        '
        siteloc = cls.dna.finditer(cls.compsite, cls.size)
        cls.results = [r for (s, g) in siteloc for r in cls._modify(s)]
        if cls.results:
            cls._drop()
        return cls.results

    @classmethod
    def is_palindromic(cls):
        if False:
            print('Hello World!')
        'Return if the enzyme has a palindromic recoginition site.'
        return True

class NonPalindromic(AbstractCut):
    """Implement methods for enzymes with non-palindromic recognition sites.

    Palindromic means : the recognition site and its reverse complement are
                        identical.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def _search(cls):
        if False:
            print('Hello World!')
        'Return a list of cutting sites of the enzyme in the sequence (PRIVATE).\n\n        For internal use only.\n\n        Implement the search method for non palindromic enzymes.\n        '
        iterator = cls.dna.finditer(cls.compsite, cls.size)
        cls.results = []
        modif = cls._modify
        revmodif = cls._rev_modify
        s = str(cls)
        cls.on_minus = []
        for (start, group) in iterator:
            if group(s):
                cls.results += list(modif(start))
            else:
                cls.on_minus += list(revmodif(start))
        cls.results += cls.on_minus
        if cls.results:
            cls.results.sort()
            cls._drop()
        return cls.results

    @classmethod
    def is_palindromic(cls):
        if False:
            print('Hello World!')
        'Return if the enzyme has a palindromic recoginition site.'
        return False

class Unknown(AbstractCut):
    """Implement methods for enzymes that produce unknown overhangs.

    These enzymes are also NotDefined and NoCut.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def catalyse(cls, dna, linear=True):
        if False:
            while True:
                i = 10
        'List the sequence fragments after cutting dna with enzyme.\n\n        Return a tuple of dna as will be produced by using RE to restrict the\n        dna.\n\n        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.\n\n        If linear is False, the sequence is considered to be circular and the\n        output will be modified accordingly.\n        '
        raise NotImplementedError(f'{cls.__name__} restriction is unknown.')
    catalyze = catalyse

    @classmethod
    def is_blunt(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return if the enzyme produces blunt ends.\n\n        True if the enzyme produces blunt end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_5overhang()\n        - RE.is_unknown()\n\n        '
        return False

    @classmethod
    def is_5overhang(cls):
        if False:
            for i in range(10):
                print('nop')
        "Return if the enzymes produces 5' overhanging ends.\n\n        True if the enzyme produces 5' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return False

    @classmethod
    def is_3overhang(cls):
        if False:
            print('Hello World!')
        "Return if the enzyme produces 3' overhanging ends.\n\n        True if the enzyme produces 3' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_5overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return False

    @classmethod
    def overhang(cls):
        if False:
            i = 10
            return i + 15
        'Return the type of the enzyme\'s overhang as string.\n\n        Can be "3\' overhang", "5\' overhang", "blunt", "unknown".\n        '
        return 'unknown'

    @classmethod
    def compatible_end(cls):
        if False:
            return 10
        'List all enzymes that produce compatible ends for the enzyme.'
        return []

    @classmethod
    def _mod1(cls, other):
        if False:
            while True:
                i = 10
        'Test if other enzyme produces compatible ends for enzyme (PRIVATE).\n\n        For internal use only.\n\n        Test for the compatibility of restriction ending of RE and other.\n        '
        return False

class Blunt(AbstractCut):
    """Implement methods for enzymes that produce blunt ends.

    The enzyme cuts the + strand and the - strand of the DNA at the same
    place.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def catalyse(cls, dna, linear=True):
        if False:
            return 10
        'List the sequence fragments after cutting dna with enzyme.\n\n        Return a tuple of dna as will be produced by using RE to restrict the\n        dna.\n\n        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.\n\n        If linear is False, the sequence is considered to be circular and the\n        output will be modified accordingly.\n        '
        r = cls.search(dna, linear)
        d = cls.dna
        if not r:
            return (d[1:],)
        fragments = []
        length = len(r) - 1
        if d.is_linear():
            fragments.append(d[1:r[0]])
            if length:
                fragments += [d[r[x]:r[x + 1]] for x in range(length)]
            fragments.append(d[r[-1]:])
        else:
            fragments.append(d[r[-1]:] + d[1:r[0]])
            if not length:
                return tuple(fragments)
            fragments += [d[r[x]:r[x + 1]] for x in range(length)]
        return tuple(fragments)
    catalyze = catalyse

    @classmethod
    def is_blunt(cls):
        if False:
            print('Hello World!')
        'Return if the enzyme produces blunt ends.\n\n        True if the enzyme produces blunt end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_5overhang()\n        - RE.is_unknown()\n\n        '
        return True

    @classmethod
    def is_5overhang(cls):
        if False:
            for i in range(10):
                print('nop')
        "Return if the enzymes produces 5' overhanging ends.\n\n        True if the enzyme produces 5' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return False

    @classmethod
    def is_3overhang(cls):
        if False:
            print('Hello World!')
        "Return if the enzyme produces 3' overhanging ends.\n\n        True if the enzyme produces 3' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_5overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return False

    @classmethod
    def overhang(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return the type of the enzyme\'s overhang as string.\n\n        Can be "3\' overhang", "5\' overhang", "blunt", "unknown".\n        '
        return 'blunt'

    @classmethod
    def compatible_end(cls, batch=None):
        if False:
            for i in range(10):
                print('nop')
        'List all enzymes that produce compatible ends for the enzyme.'
        if not batch:
            batch = AllEnzymes
        r = sorted((x for x in iter(AllEnzymes) if x.is_blunt()))
        return r

    @staticmethod
    def _mod1(other):
        if False:
            return 10
        'Test if other enzyme produces compatible ends for enzyme (PRIVATE).\n\n        For internal use only\n\n        Test for the compatibility of restriction ending of RE and other.\n        '
        return issubclass(other, Blunt)

class Ov5(AbstractCut):
    """Implement methods for enzymes that produce 5' overhanging ends.

    The enzyme cuts the + strand after the - strand of the DNA.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def catalyse(cls, dna, linear=True):
        if False:
            i = 10
            return i + 15
        'List the sequence fragments after cutting dna with enzyme.\n\n        Return a tuple of dna as will be produced by using RE to restrict the\n        dna.\n\n        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.\n\n        If linear is False, the sequence is considered to be circular and the\n        output will be modified accordingly.\n        '
        r = cls.search(dna, linear)
        d = cls.dna
        if not r:
            return (d[1:],)
        length = len(r) - 1
        fragments = []
        if d.is_linear():
            fragments.append(d[1:r[0]])
            if length:
                fragments += [d[r[x]:r[x + 1]] for x in range(length)]
            fragments.append(d[r[-1]:])
        else:
            fragments.append(d[r[-1]:] + d[1:r[0]])
            if not length:
                return tuple(fragments)
            fragments += [d[r[x]:r[x + 1]] for x in range(length)]
        return tuple(fragments)
    catalyze = catalyse

    @classmethod
    def is_blunt(cls):
        if False:
            while True:
                i = 10
        'Return if the enzyme produces blunt ends.\n\n        True if the enzyme produces blunt end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_5overhang()\n        - RE.is_unknown()\n\n        '
        return False

    @classmethod
    def is_5overhang(cls):
        if False:
            while True:
                i = 10
        "Return if the enzymes produces 5' overhanging ends.\n\n        True if the enzyme produces 5' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return True

    @classmethod
    def is_3overhang(cls):
        if False:
            i = 10
            return i + 15
        "Return if the enzyme produces 3' overhanging ends.\n\n        True if the enzyme produces 3' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_5overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return False

    @classmethod
    def overhang(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return the type of the enzyme\'s overhang as string.\n\n        Can be "3\' overhang", "5\' overhang", "blunt", "unknown".\n        '
        return "5' overhang"

    @classmethod
    def compatible_end(cls, batch=None):
        if False:
            while True:
                i = 10
        'List all enzymes that produce compatible ends for the enzyme.'
        if not batch:
            batch = AllEnzymes
        r = sorted((x for x in iter(AllEnzymes) if x.is_5overhang() and x % cls))
        return r

    @classmethod
    def _mod1(cls, other):
        if False:
            print('Hello World!')
        'Test if other enzyme produces compatible ends for enzyme (PRIVATE).\n\n        For internal use only.\n\n        Test for the compatibility of restriction ending of RE and other.\n        '
        if issubclass(other, Ov5):
            return cls._mod2(other)
        else:
            return False

class Ov3(AbstractCut):
    """Implement methods for enzymes that produce 3' overhanging ends.

    The enzyme cuts the - strand after the + strand of the DNA.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def catalyse(cls, dna, linear=True):
        if False:
            print('Hello World!')
        'List the sequence fragments after cutting dna with enzyme.\n\n        Return a tuple of dna as will be produced by using RE to restrict the\n        dna.\n\n        dna must be a Bio.Seq.Seq instance or a Bio.Seq.MutableSeq instance.\n\n        If linear is False, the sequence is considered to be circular and the\n        output will be modified accordingly.\n        '
        r = cls.search(dna, linear)
        d = cls.dna
        if not r:
            return (d[1:],)
        fragments = []
        length = len(r) - 1
        if d.is_linear():
            fragments.append(d[1:r[0]])
            if length:
                fragments += [d[r[x]:r[x + 1]] for x in range(length)]
            fragments.append(d[r[-1]:])
        else:
            fragments.append(d[r[-1]:] + d[1:r[0]])
            if not length:
                return tuple(fragments)
            fragments += [d[r[x]:r[x + 1]] for x in range(length)]
        return tuple(fragments)
    catalyze = catalyse

    @classmethod
    def is_blunt(cls):
        if False:
            return 10
        'Return if the enzyme produces blunt ends.\n\n        True if the enzyme produces blunt end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_5overhang()\n        - RE.is_unknown()\n\n        '
        return False

    @classmethod
    def is_5overhang(cls):
        if False:
            return 10
        "Return if the enzymes produces 5' overhanging ends.\n\n        True if the enzyme produces 5' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_3overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return False

    @classmethod
    def is_3overhang(cls):
        if False:
            return 10
        "Return if the enzyme produces 3' overhanging ends.\n\n        True if the enzyme produces 3' overhang sticky end.\n\n        Related methods:\n\n        - RE.is_5overhang()\n        - RE.is_blunt()\n        - RE.is_unknown()\n\n        "
        return True

    @classmethod
    def overhang(cls):
        if False:
            i = 10
            return i + 15
        'Return the type of the enzyme\'s overhang as string.\n\n        Can be "3\' overhang", "5\' overhang", "blunt", "unknown".\n        '
        return "3' overhang"

    @classmethod
    def compatible_end(cls, batch=None):
        if False:
            while True:
                i = 10
        'List all enzymes that produce compatible ends for the enzyme.'
        if not batch:
            batch = AllEnzymes
        r = sorted((x for x in iter(AllEnzymes) if x.is_3overhang() and x % cls))
        return r

    @classmethod
    def _mod1(cls, other):
        if False:
            i = 10
            return i + 15
        'Test if other enzyme produces compatible ends for enzyme (PRIVATE).\n\n        For internal use only.\n\n        Test for the compatibility of restriction ending of RE and other.\n        '
        if issubclass(other, Ov3):
            return cls._mod2(other)
        else:
            return False

class Defined(AbstractCut):
    """Implement methods for enzymes with defined recognition site and cut.

    Typical example : EcoRI -> G^AATT_C
                      The overhang will always be AATT
    Notes:
        Blunt enzymes are always defined. Even if their site is GGATCCNNN^_N
        Their overhang is always the same : blunt!

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def _drop(cls):
        if False:
            return 10
        'Remove cuts that are outsite of the sequence (PRIVATE).\n\n        For internal use only.\n\n        Drop the site that are situated outside the sequence in linear\n        sequence. Modify the index for site in circular sequences.\n        '
        length = len(cls.dna)
        drop = itertools.dropwhile
        take = itertools.takewhile
        if cls.dna.is_linear():
            cls.results = list(drop(lambda x: x <= 1, cls.results))
            cls.results = list(take(lambda x: x <= length, cls.results))
        else:
            for (index, location) in enumerate(cls.results):
                if location < 1:
                    cls.results[index] += length
                else:
                    break
            for (index, location) in enumerate(cls.results[::-1]):
                if location > length:
                    cls.results[-(index + 1)] -= length
                else:
                    break

    @classmethod
    def is_defined(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return if recognition sequence and cut are defined.\n\n        True if the sequence recognised and cut is constant,\n        i.e. the recognition site is not degenerated AND the enzyme cut inside\n        the site.\n\n        Related methods:\n\n        - RE.is_ambiguous()\n        - RE.is_unknown()\n\n        '
        return True

    @classmethod
    def is_ambiguous(cls):
        if False:
            print('Hello World!')
        'Return if recognition sequence and cut may be ambiguous.\n\n        True if the sequence recognised and cut is ambiguous,\n        i.e. the recognition site is degenerated AND/OR the enzyme cut outside\n        the site.\n\n        Related methods:\n\n        - RE.is_defined()\n        - RE.is_unknown()\n\n        '
        return False

    @classmethod
    def is_unknown(cls):
        if False:
            return 10
        'Return if recognition sequence is unknown.\n\n        True if the sequence is unknown,\n        i.e. the recognition site has not been characterised yet.\n\n        Related methods:\n\n        - RE.is_defined()\n        - RE.is_ambiguous()\n\n        '
        return False

    @classmethod
    def elucidate(cls):
        if False:
            for i in range(10):
                print('nop')
        "Return a string representing the recognition site and cuttings.\n\n        Return a representation of the site with the cut on the (+) strand\n        represented as '^' and the cut on the (-) strand as '_'.\n        ie:\n\n        >>> from Bio.Restriction import EcoRI, KpnI, EcoRV, SnaI\n        >>> EcoRI.elucidate()   # 5' overhang\n        'G^AATT_C'\n        >>> KpnI.elucidate()    # 3' overhang\n        'G_GTAC^C'\n        >>> EcoRV.elucidate()   # blunt\n        'GAT^_ATC'\n        >>> SnaI.elucidate()    # NotDefined, cut profile unknown.\n        '? GTATAC ?'\n        >>>\n\n        "
        f5 = cls.fst5
        f3 = cls.fst3
        site = cls.site
        if cls.cut_twice():
            re = 'cut twice, not yet implemented sorry.'
        elif cls.is_5overhang():
            if f5 == f3 == 0:
                re = 'N^' + cls.site + '_N'
            elif f3 == 0:
                re = site[:f5] + '^' + site[f5:] + '_N'
            else:
                re = site[:f5] + '^' + site[f5:f3] + '_' + site[f3:]
        elif cls.is_blunt():
            re = site[:f5] + '^_' + site[f5:]
        elif f5 == f3 == 0:
            re = 'N_' + site + '^N'
        else:
            re = site[:f3] + '_' + site[f3:f5] + '^' + site[f5:]
        return re

    @classmethod
    def _mod2(cls, other):
        if False:
            print('Hello World!')
        'Test if other enzyme produces compatible ends for enzyme (PRIVATE).\n\n        For internal use only.\n\n        Test for the compatibility of restriction ending of RE and other.\n        '
        if other.ovhgseq == cls.ovhgseq:
            return True
        elif issubclass(other, Ambiguous):
            return other._mod2(cls)
        else:
            return False

class Ambiguous(AbstractCut):
    """Implement methods for enzymes that produce variable overhangs.

    Typical example : BstXI -> CCAN_NNNN^NTGG
                      The overhang can be any sequence of 4 bases.

    Notes:
        Blunt enzymes are always defined. Even if their site is GGATCCNNN^_N
        Their overhang is always the same : blunt!

    Internal use only. Not meant to be instantiated.

    """

    @classmethod
    def _drop(cls):
        if False:
            print('Hello World!')
        'Remove cuts that are outsite of the sequence (PRIVATE).\n\n        For internal use only.\n\n        Drop the site that are situated outside the sequence in linear\n        sequence. Modify the index for site in circular sequences.\n        '
        length = len(cls.dna)
        drop = itertools.dropwhile
        take = itertools.takewhile
        if cls.dna.is_linear():
            cls.results = list(drop(lambda x: x <= 1, cls.results))
            cls.results = list(take(lambda x: x <= length, cls.results))
        else:
            for (index, location) in enumerate(cls.results):
                if location < 1:
                    cls.results[index] += length
                else:
                    break
            for (index, location) in enumerate(cls.results[::-1]):
                if location > length:
                    cls.results[-(index + 1)] -= length
                else:
                    break

    @classmethod
    def is_defined(cls):
        if False:
            i = 10
            return i + 15
        'Return if recognition sequence and cut are defined.\n\n        True if the sequence recognised and cut is constant,\n        i.e. the recognition site is not degenerated AND the enzyme cut inside\n        the site.\n\n        Related methods:\n\n        - RE.is_ambiguous()\n        - RE.is_unknown()\n\n        '
        return False

    @classmethod
    def is_ambiguous(cls):
        if False:
            i = 10
            return i + 15
        'Return if recognition sequence and cut may be ambiguous.\n\n        True if the sequence recognised and cut is ambiguous,\n        i.e. the recognition site is degenerated AND/OR the enzyme cut outside\n        the site.\n\n        Related methods:\n\n        - RE.is_defined()\n        - RE.is_unknown()\n\n        '
        return True

    @classmethod
    def is_unknown(cls):
        if False:
            print('Hello World!')
        'Return if recognition sequence is unknown.\n\n        True if the sequence is unknown,\n        i.e. the recognition site has not been characterised yet.\n\n        Related methods:\n\n        - RE.is_defined()\n        - RE.is_ambiguous()\n\n        '
        return False

    @classmethod
    def _mod2(cls, other):
        if False:
            i = 10
            return i + 15
        'Test if other enzyme produces compatible ends for enzyme (PRIVATE).\n\n        For internal use only.\n\n        Test for the compatibility of restriction ending of RE and other.\n        '
        if len(cls.ovhgseq) != len(other.ovhgseq):
            return False
        else:
            se = cls.ovhgseq
            for base in se:
                if base in 'ATCG':
                    pass
                if base in 'N':
                    se = '.'.join(se.split('N'))
                if base in 'RYWMSKHDBV':
                    expand = '[' + matching[base] + ']'
                    se = expand.join(se.split(base))
            if re.match(se, other.ovhgseq):
                return True
            else:
                return False

    @classmethod
    def elucidate(cls):
        if False:
            for i in range(10):
                print('nop')
        "Return a string representing the recognition site and cuttings.\n\n        Return a representation of the site with the cut on the (+) strand\n        represented as '^' and the cut on the (-) strand as '_'.\n        ie:\n\n        >>> from Bio.Restriction import EcoRI, KpnI, EcoRV, SnaI\n        >>> EcoRI.elucidate()   # 5' overhang\n        'G^AATT_C'\n        >>> KpnI.elucidate()    # 3' overhang\n        'G_GTAC^C'\n        >>> EcoRV.elucidate()   # blunt\n        'GAT^_ATC'\n        >>> SnaI.elucidate()     # NotDefined, cut profile unknown.\n        '? GTATAC ?'\n        >>>\n\n        "
        f5 = cls.fst5
        f3 = cls.fst3
        length = len(cls)
        site = cls.site
        if cls.cut_twice():
            re = 'cut twice, not yet implemented sorry.'
        elif cls.is_5overhang():
            if f3 == f5 == 0:
                re = 'N^' + site + '_N'
            elif 0 <= f5 <= length and 0 <= f3 + length <= length:
                re = site[:f5] + '^' + site[f5:f3] + '_' + site[f3:]
            elif 0 <= f5 <= length:
                re = site[:f5] + '^' + site[f5:] + f3 * 'N' + '_N'
            elif 0 <= f3 + length <= length:
                re = 'N^' + abs(f5) * 'N' + site[:f3] + '_' + site[f3:]
            elif f3 + length < 0:
                re = 'N^' + abs(f5) * 'N' + '_' + abs(length + f3) * 'N' + site
            elif f5 > length:
                re = site + (f5 - length) * 'N' + '^' + (length + f3 - f5) * 'N' + '_N'
            else:
                re = 'N^' + abs(f5) * 'N' + site + f3 * 'N' + '_N'
        elif cls.is_blunt():
            if f5 < 0:
                re = 'N^_' + abs(f5) * 'N' + site
            elif f5 > length:
                re = site + (f5 - length) * 'N' + '^_N'
            else:
                raise ValueError('%s.easyrepr() : error f5=%i' % (cls.name, f5))
        elif f3 == 0:
            if f5 == 0:
                re = 'N_' + site + '^N'
            else:
                re = site + '_' + (f5 - length) * 'N' + '^N'
        elif 0 < f3 + length <= length and 0 <= f5 <= length:
            re = site[:f3] + '_' + site[f3:f5] + '^' + site[f5:]
        elif 0 < f3 + length <= length:
            re = site[:f3] + '_' + site[f3:] + (f5 - length) * 'N' + '^N'
        elif 0 <= f5 <= length:
            re = 'N_' + 'N' * (f3 + length) + site[:f5] + '^' + site[f5:]
        elif f3 > 0:
            re = site + f3 * 'N' + '_' + (f5 - f3 - length) * 'N' + '^N'
        elif f5 < 0:
            re = 'N_' + abs(f3 - f5 + length) * 'N' + '^' + abs(f5) * 'N' + site
        else:
            re = 'N_' + abs(f3 + length) * 'N' + site + (f5 - length) * 'N' + '^N'
        return re

class NotDefined(AbstractCut):
    """Implement methods for enzymes with non-characterized overhangs.

    Correspond to NoCut and Unknown.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def _drop(cls):
        if False:
            while True:
                i = 10
        'Remove cuts that are outsite of the sequence (PRIVATE).\n\n        For internal use only.\n\n        Drop the site that are situated outside the sequence in linear\n        sequence. Modify the index for site in circular sequences.\n        '
        if cls.dna.is_linear():
            return
        else:
            length = len(cls.dna)
            for (index, location) in enumerate(cls.results):
                if location < 1:
                    cls.results[index] += length
                else:
                    break
            for (index, location) in enumerate(cls.results[:-1]):
                if location > length:
                    cls.results[-(index + 1)] -= length
                else:
                    break

    @classmethod
    def is_defined(cls):
        if False:
            print('Hello World!')
        'Return if recognition sequence and cut are defined.\n\n        True if the sequence recognised and cut is constant,\n        i.e. the recognition site is not degenerated AND the enzyme cut inside\n        the site.\n\n        Related methods:\n\n        - RE.is_ambiguous()\n        - RE.is_unknown()\n\n        '
        return False

    @classmethod
    def is_ambiguous(cls):
        if False:
            return 10
        'Return if recognition sequence and cut may be ambiguous.\n\n        True if the sequence recognised and cut is ambiguous,\n        i.e. the recognition site is degenerated AND/OR the enzyme cut outside\n        the site.\n\n        Related methods:\n\n        - RE.is_defined()\n        - RE.is_unknown()\n\n        '
        return False

    @classmethod
    def is_unknown(cls):
        if False:
            return 10
        'Return if recognition sequence is unknown.\n\n        True if the sequence is unknown,\n        i.e. the recognition site has not been characterised yet.\n\n        Related methods:\n\n        - RE.is_defined()\n        - RE.is_ambiguous()\n\n        '
        return True

    @classmethod
    def _mod2(cls, other):
        if False:
            print('Hello World!')
        'Test if other enzyme produces compatible ends for enzyme (PRIVATE).\n\n        For internal use only.\n\n        Test for the compatibility of restriction ending of RE and other.\n        '
        raise ValueError('%s.mod2(%s), %s : NotDefined. pas glop pas glop!' % (str(cls), str(other), str(cls)))

    @classmethod
    def elucidate(cls):
        if False:
            while True:
                i = 10
        "Return a string representing the recognition site and cuttings.\n\n        Return a representation of the site with the cut on the (+) strand\n        represented as '^' and the cut on the (-) strand as '_'.\n        ie:\n\n        >>> from Bio.Restriction import EcoRI, KpnI, EcoRV, SnaI\n        >>> EcoRI.elucidate()   # 5' overhang\n        'G^AATT_C'\n        >>> KpnI.elucidate()    # 3' overhang\n        'G_GTAC^C'\n        >>> EcoRV.elucidate()   # blunt\n        'GAT^_ATC'\n        >>> SnaI.elucidate()     # NotDefined, cut profile unknown.\n        '? GTATAC ?'\n        >>>\n\n        "
        return f'? {cls.site} ?'

class Commercially_available(AbstractCut):
    """Implement methods for enzymes which are commercially available.

    Internal use only. Not meant to be instantiated.
    """

    @classmethod
    def suppliers(cls):
        if False:
            print('Hello World!')
        'Print a list of suppliers of the enzyme.'
        for s in cls.suppl:
            print(suppliers_dict[s][0] + ',')

    @classmethod
    def supplier_list(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return a list of suppliers of the enzyme.'
        return [v[0] for (k, v) in suppliers_dict.items() if k in cls.suppl]

    @classmethod
    def buffers(cls, supplier):
        if False:
            i = 10
            return i + 15
        'Return the recommended buffer of the supplier for this enzyme.\n\n        Not implemented yet.\n        '

    @classmethod
    def is_comm(cls):
        if False:
            for i in range(10):
                print('nop')
        'Return if enzyme is commercially available.\n\n        True if RE has suppliers.\n        '
        return True

class Not_available(AbstractCut):
    """Implement methods for enzymes which are not commercially available.

    Internal use only. Not meant to be instantiated.
    """

    @staticmethod
    def suppliers():
        if False:
            return 10
        'Print a list of suppliers of the enzyme.'
        return None

    @classmethod
    def supplier_list(cls):
        if False:
            i = 10
            return i + 15
        'Return a list of suppliers of the enzyme.'
        return []

    @classmethod
    def buffers(cls, supplier):
        if False:
            while True:
                i = 10
        'Return the recommended buffer of the supplier for this enzyme.\n\n        Not implemented yet.\n        '
        raise TypeError('Enzyme not commercially available.')

    @classmethod
    def is_comm(cls):
        if False:
            return 10
        'Return if enzyme is commercially available.\n\n        True if RE has suppliers.\n        '
        return False

class RestrictionBatch(set):
    """Class for operations on more than one enzyme."""

    def __init__(self, first=(), suppliers=()):
        if False:
            while True:
                i = 10
        'Initialize empty RB or pre-fill with enzymes (from supplier).'
        first = [self.format(x) for x in first]
        first += [eval(x) for n in suppliers for x in suppliers_dict[n][1]]
        set.__init__(self, first)
        self.mapping = dict.fromkeys(self)
        self.already_mapped = None
        self.suppliers = [x for x in suppliers if x in suppliers_dict]

    def __str__(self):
        if False:
            return 10
        'Return a readable representation of the ``RestrictionBatch``.'
        if len(self) < 5:
            return '+'.join(self.elements())
        else:
            return '...'.join(('+'.join(self.elements()[:2]), '+'.join(self.elements()[-2:])))

    def __repr__(self):
        if False:
            return 10
        'Represent ``RestrictionBatch`` class as a string for debugging.'
        return f'RestrictionBatch({self.elements()})'

    def __contains__(self, other):
        if False:
            i = 10
            return i + 15
        'Implement ``in`` for ``RestrictionBatch``.'
        try:
            other = self.format(other)
        except ValueError:
            return False
        return set.__contains__(self, other)

    def __div__(self, other):
        if False:
            i = 10
            return i + 15
        "Override '/' operator to use as search method."
        return self.search(other)

    def __rdiv__(self, other):
        if False:
            i = 10
            return i + 15
        'Override division with reversed operands to use as search method.'
        return self.search(other)

    def __truediv__(self, other):
        if False:
            i = 10
            return i + 15
        'Override Python 3 division operator to use as search method.\n\n        Like __div__.\n        '
        return self.search(other)

    def __rtruediv__(self, other):
        if False:
            while True:
                i = 10
        'As __truediv___, with reversed operands.\n\n        Like __rdiv__.\n        '
        return self.search(other)

    def get(self, enzyme, add=False):
        if False:
            print('Hello World!')
        'Check if enzyme is in batch and return it.\n\n        If add is True and enzyme is not in batch add enzyme to batch.\n        If add is False (which is the default) only return enzyme.\n        If enzyme is not a RestrictionType or can not be evaluated to\n        a RestrictionType, raise a ValueError.\n        '
        e = self.format(enzyme)
        if e in self:
            return e
        elif add:
            self.add(e)
            return e
        else:
            raise ValueError(f'enzyme {e.__name__} is not in RestrictionBatch')

    def lambdasplit(self, func):
        if False:
            return 10
        'Filter enzymes in batch with supplied function.\n\n        The new batch will contain only the enzymes for which\n        func return True.\n        '
        d = list(filter(func, self))
        new = RestrictionBatch()
        new._data = dict(zip(d, [True] * len(d)))
        return new

    def add_supplier(self, letter):
        if False:
            while True:
                i = 10
        'Add all enzymes from a given supplier to batch.\n\n        letter represents the suppliers as defined in the dictionary\n        RestrictionDictionary.suppliers\n        Returns None.\n        Raise a KeyError if letter is not a supplier code.\n        '
        supplier = suppliers_dict[letter]
        self.suppliers.append(letter)
        for x in supplier[1]:
            self.add_nocheck(eval(x))

    def current_suppliers(self):
        if False:
            i = 10
            return i + 15
        'List the current suppliers for the restriction batch.\n\n        Return a sorted list of the suppliers which have been used to\n        create the batch.\n        '
        suppl_list = sorted((suppliers_dict[x][0] for x in self.suppliers))
        return suppl_list

    def __iadd__(self, other):
        if False:
            while True:
                i = 10
        "Override '+=' for use with sets.\n\n        b += other -> add other to b, check the type of other.\n        "
        self.add(other)
        return self

    def __add__(self, other):
        if False:
            return 10
        "Override '+' for use with sets.\n\n        b + other -> new RestrictionBatch.\n        "
        new = self.__class__(self)
        new.add(other)
        return new

    def remove(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Remove enzyme from restriction batch.\n\n        Safe set.remove method. Verify that other is a RestrictionType or can\n        be evaluated to a RestrictionType.\n        Raise a ValueError if other can not be evaluated to a RestrictionType.\n        Raise a KeyError if other is not in B.\n        '
        return set.remove(self, self.format(other))

    def add(self, other):
        if False:
            while True:
                i = 10
        'Add a restriction enzyme to the restriction batch.\n\n        Safe set.add method. Verify that other is a RestrictionType or can be\n        evaluated to a RestrictionType.\n        Raise a ValueError if other can not be evaluated to a RestrictionType.\n        '
        return set.add(self, self.format(other))

    def add_nocheck(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Add restriction enzyme to batch without checking its type.'
        return set.add(self, other)

    def format(self, y):
        if False:
            while True:
                i = 10
        'Evaluate enzyme (name) and return it (as RestrictionType).\n\n        If y is a RestrictionType return y.\n        If y can be evaluated to a RestrictionType return eval(y).\n        Raise a ValueError in all other case.\n        '
        try:
            if isinstance(y, RestrictionType):
                return y
            elif isinstance(eval(str(y)), RestrictionType):
                return eval(y)
        except (NameError, SyntaxError):
            pass
        raise ValueError(f'{y.__class__} is not a RestrictionType')

    def is_restriction(self, y):
        if False:
            i = 10
            return i + 15
        'Return if enzyme (name) is a known enzyme.\n\n        True if y or eval(y) is a RestrictionType.\n        '
        return isinstance(y, RestrictionType) or isinstance(eval(str(y)), RestrictionType)

    def split(self, *classes, **bool):
        if False:
            i = 10
            return i + 15
        'Extract enzymes of a certain class and put in new RestrictionBatch.\n\n        It works but it is slow, so it has really an interest when splitting\n        over multiple conditions.\n        '

        def splittest(element):
            if False:
                while True:
                    i = 10
            for klass in classes:
                b = bool.get(klass.__name__, True)
                if issubclass(element, klass):
                    if b:
                        continue
                    else:
                        return False
                elif b:
                    return False
                else:
                    continue
            return True
        d = list(filter(splittest, self))
        new = RestrictionBatch()
        new._data = dict(zip(d, [True] * len(d)))
        return new

    def elements(self):
        if False:
            while True:
                i = 10
        'List the enzymes of the RestrictionBatch as list of strings.\n\n        Give all the names of the enzymes in B sorted alphabetically.\n        '
        return sorted((str(e) for e in self))

    def as_string(self):
        if False:
            i = 10
            return i + 15
        'List the names of the enzymes of the RestrictionBatch.\n\n        Return a list of the name of the elements of the batch.\n        '
        return [str(e) for e in self]

    @classmethod
    def suppl_codes(cls):
        if False:
            return 10
        'Return a dictionary with supplier codes.\n\n        Letter code for the suppliers.\n        '
        supply = {k: v[0] for (k, v) in suppliers_dict.items()}
        return supply

    @classmethod
    def show_codes(cls):
        if False:
            while True:
                i = 10
        'Print a list of supplier codes.'
        supply = [' = '.join(i) for i in cls.suppl_codes().items()]
        print('\n'.join(supply))

    def search(self, dna, linear=True):
        if False:
            i = 10
            return i + 15
        'Return a dic of cutting sites in the seq for the batch enzymes.'
        if not hasattr(self, 'already_mapped'):
            self.already_mapped = None
        if isinstance(dna, DNA):
            if (str(dna), linear) == self.already_mapped:
                return self.mapping
            else:
                self.already_mapped = (str(dna), linear)
                fseq = FormattedSeq(dna, linear)
                self.mapping = {x: x.search(fseq) for x in self}
                return self.mapping
        elif isinstance(dna, FormattedSeq):
            if (str(dna), dna.linear) == self.already_mapped:
                return self.mapping
            else:
                self.already_mapped = (str(dna), dna.linear)
                self.mapping = {x: x.search(dna) for x in self}
                return self.mapping
        raise TypeError(f'Expected Seq or MutableSeq instance, got {type(dna)} instead')
_empty_DNA = DNA('')
_restrictionbatch = RestrictionBatch()

class Analysis(RestrictionBatch, PrintFormat):
    """Provide methods for enhanced analysis and pretty printing."""

    def __init__(self, restrictionbatch=_restrictionbatch, sequence=_empty_DNA, linear=True):
        if False:
            while True:
                i = 10
        'Initialize an Analysis with RestrictionBatch and sequence.\n\n        For most of the methods of this class if a dictionary is given it will\n        be used as the base to calculate the results.\n        If no dictionary is given a new analysis using the RestrictionBatch\n        which has been given when the Analysis class has been instantiated,\n        will be carried out and used.\n        '
        RestrictionBatch.__init__(self, restrictionbatch)
        self.rb = restrictionbatch
        self.sequence = sequence
        self.linear = linear
        if self.sequence:
            self.search(self.sequence, self.linear)

    def __repr__(self):
        if False:
            print('Hello World!')
        'Represent ``Analysis`` class as a string.'
        return f'Analysis({self.rb!r},{self.sequence!r},{self.linear})'

    def _sub_set(self, wanted):
        if False:
            for i in range(10):
                print('nop')
        'Filter result for keys which are in wanted (PRIVATE).\n\n        Internal use only. Returns a dict.\n\n        Screen the results through wanted set.\n        Keep only the results for which the enzymes is in wanted set.\n        '
        return {k: v for (k, v) in self.mapping.items() if k in wanted}

    def _boundaries(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        'Set boundaries to correct values (PRIVATE).\n\n        Format the boundaries for use with the methods that limit the\n        search to only part of the sequence given to analyse.\n        '
        if not isinstance(start, int):
            raise TypeError(f'expected int, got {type(start)} instead')
        if not isinstance(end, int):
            raise TypeError(f'expected int, got {type(end)} instead')
        if start < 1:
            start += len(self.sequence)
        if end < 1:
            end += len(self.sequence)
        if start < end:
            pass
        else:
            (start, end) = (end, start)
        if start < end:
            return (start, end, self._test_normal)

    def _test_normal(self, start, end, site):
        if False:
            i = 10
            return i + 15
        'Test if site is between start and end (PRIVATE).\n\n        Internal use only\n        '
        return start <= site < end

    def _test_reverse(self, start, end, site):
        if False:
            return 10
        'Test if site is between end and start, for circular sequences (PRIVATE).\n\n        Internal use only.\n        '
        return start <= site <= len(self.sequence) or 1 <= site < end

    def format_output(self, dct=None, title='', s1=''):
        if False:
            for i in range(10):
                print('nop')
        'Collect data and pass to PrintFormat.\n\n        If dct is not given the full dictionary is used.\n        '
        if not dct:
            dct = self.mapping
        return PrintFormat.format_output(self, dct, title, s1)

    def print_that(self, dct=None, title='', s1=''):
        if False:
            i = 10
            return i + 15
        'Print the output of the analysis.\n\n        If dct is not given the full dictionary is used.\n        s1: Title for non-cutting enzymes\n        This method prints the output of A.format_output() and it is here\n        for backwards compatibility.\n        '
        print(self.format_output(dct, title, s1))

    def change(self, **what):
        if False:
            while True:
                i = 10
        'Change parameters of print output.\n\n        It is possible to change the width of the shell by setting\n        self.ConsoleWidth to what you want.\n        self.NameWidth refer to the maximal length of the enzyme name.\n\n        Changing one of these parameters here might not give the results\n        you expect. In which case, you can settle back to a 80 columns shell\n        or try to change self.Cmodulo and self.PrefWidth in PrintFormat until\n        you get it right.\n        '
        for (k, v) in what.items():
            if k in ('NameWidth', 'ConsoleWidth'):
                setattr(self, k, v)
                self.Cmodulo = self.ConsoleWidth % self.NameWidth
                self.PrefWidth = self.ConsoleWidth - self.Cmodulo
            elif k == 'sequence':
                setattr(self, 'sequence', v)
                self.search(self.sequence, self.linear)
            elif k == 'rb':
                self = Analysis.__init__(self, v, self.sequence, self.linear)
            elif k == 'linear':
                setattr(self, 'linear', v)
                self.search(self.sequence, v)
            elif k in ('Indent', 'Maxsize'):
                setattr(self, k, v)
            elif k in ('Cmodulo', 'PrefWidth'):
                raise AttributeError(f'To change {k}, change NameWidth and/or ConsoleWidth')
            else:
                raise AttributeError(f'Analysis has no attribute {k}')

    def full(self, linear=True):
        if False:
            return 10
        'Perform analysis with all enzymes of batch and return all results.\n\n        Full Restriction Map of the sequence, as a dictionary.\n        '
        return self.mapping

    def blunt(self, dct=None):
        if False:
            print('Hello World!')
        'Return only cuts that have blunt ends.'
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if k.is_blunt()}

    def overhang5(self, dct=None):
        if False:
            for i in range(10):
                print('nop')
        "Return only cuts that have 5' overhangs."
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if k.is_5overhang()}

    def overhang3(self, dct=None):
        if False:
            return 10
        "Return only cuts that have 3' overhangs."
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if k.is_3overhang()}

    def defined(self, dct=None):
        if False:
            return 10
        'Return only results from enzymes that produce defined overhangs.'
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if k.is_defined()}

    def with_sites(self, dct=None):
        if False:
            for i in range(10):
                print('nop')
        'Return only results from enzyme with at least one cut.'
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if v}

    def without_site(self, dct=None):
        if False:
            for i in range(10):
                print('nop')
        "Return only results from enzymes that don't cut the sequence."
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if not v}

    def with_N_sites(self, N, dct=None):
        if False:
            print('Hello World!')
        'Return only results from enzymes that cut the sequence N times.'
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if len(v) == N}

    def with_number_list(self, list, dct=None):
        if False:
            while True:
                i = 10
        'Return only results from enzymes that cut (x,y,z,...) times.'
        if not dct:
            dct = self.mapping
        return {k: v for (k, v) in dct.items() if len(v) in list}

    def with_name(self, names, dct=None):
        if False:
            print('Hello World!')
        'Return only results from enzymes which names are listed.'
        for (i, enzyme) in enumerate(names):
            if enzyme not in AllEnzymes:
                warnings.warn(f'no data for the enzyme: {enzyme}', BiopythonWarning)
                del names[i]
        if not dct:
            return RestrictionBatch(names).search(self.sequence, self.linear)
        return {n: dct[n] for n in names if n in dct}

    def with_site_size(self, site_size, dct=None):
        if False:
            print('Hello World!')
        'Return only results form enzymes with a given site size.'
        sites = [name for name in self if name.size == site_size]
        if not dct:
            return RestrictionBatch(sites).search(self.sequence)
        return {k: v for (k, v) in dct.items() if k in site_size}

    def only_between(self, start, end, dct=None):
        if False:
            i = 10
            return i + 15
        'Return only results from enzymes that only cut within start, end.'
        (start, end, test) = self._boundaries(start, end)
        if not dct:
            dct = self.mapping
        d = dict(dct)
        for (key, sites) in dct.items():
            if not sites:
                del d[key]
                continue
            for site in sites:
                if test(start, end, site):
                    continue
                else:
                    del d[key]
                    break
        return d

    def between(self, start, end, dct=None):
        if False:
            for i in range(10):
                print('nop')
        'Return only results from enzymes that cut at least within borders.\n\n        Enzymes that cut the sequence at least in between start and end.\n        They may cut outside as well.\n        '
        (start, end, test) = self._boundaries(start, end)
        d = {}
        if not dct:
            dct = self.mapping
        for (key, sites) in dct.items():
            for site in sites:
                if test(start, end, site):
                    d[key] = sites
                    break
                continue
        return d

    def show_only_between(self, start, end, dct=None):
        if False:
            return 10
        'Return only results from within start, end.\n\n        Enzymes must cut inside start/end and may also cut outside. However,\n        only the cutting positions within start/end will be returned.\n        '
        d = []
        if start <= end:
            d = [(k, [vv for vv in v if start <= vv <= end]) for (k, v) in self.between(start, end, dct).items()]
        else:
            d = [(k, [vv for vv in v if start <= vv or vv <= end]) for (k, v) in self.between(start, end, dct).items()]
        return dict(d)

    def only_outside(self, start, end, dct=None):
        if False:
            for i in range(10):
                print('nop')
        'Return only results from enzymes that only cut outside start, end.\n\n        Enzymes that cut the sequence outside of the region\n        in between start and end but do not cut inside.\n        '
        (start, end, test) = self._boundaries(start, end)
        if not dct:
            dct = self.mapping
        d = dict(dct)
        for (key, sites) in dct.items():
            if not sites:
                del d[key]
                continue
            for site in sites:
                if test(start, end, site):
                    del d[key]
                    break
                else:
                    continue
        return d

    def outside(self, start, end, dct=None):
        if False:
            while True:
                i = 10
        'Return only results from enzymes that at least cut outside borders.\n\n        Enzymes that cut outside the region in between start and end.\n        They may cut inside as well.\n        '
        (start, end, test) = self._boundaries(start, end)
        if not dct:
            dct = self.mapping
        d = {}
        for (key, sites) in dct.items():
            for site in sites:
                if test(start, end, site):
                    continue
                else:
                    d[key] = sites
                    break
        return d

    def do_not_cut(self, start, end, dct=None):
        if False:
            print('Hello World!')
        "Return only results from enzymes that don't cut between borders."
        if not dct:
            dct = self.mapping
        d = self.without_site()
        d.update(self.only_outside(start, end, dct))
        return d
CommOnly = RestrictionBatch()
NonComm = RestrictionBatch()
for (TYPE, (bases, enzymes)) in typedict.items():
    bases2 = tuple((eval(x) for x in bases))
    T = type.__new__(RestrictionType, 'RestrictionType', bases2, {})
    for k in enzymes:
        newenz = T(k, bases2, enzymedict[k])
        if newenz.is_comm():
            CommOnly.add_nocheck(newenz)
        else:
            NonComm.add_nocheck(newenz)
AllEnzymes = RestrictionBatch(CommOnly)
AllEnzymes.update(NonComm)
names = [str(x) for x in AllEnzymes]
locals().update(dict(zip(names, AllEnzymes)))
__all__ = ('FormattedSeq', 'Analysis', 'RestrictionBatch', 'AllEnzymes', 'CommOnly', 'NonComm') + tuple(names)
del k, enzymes, TYPE, bases, bases2, names