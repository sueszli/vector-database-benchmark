"""Represent a Sequence Feature holding info about a part of a sequence.

This is heavily modeled after the Biocorba SeqFeature objects, and
may be pretty biased towards GenBank stuff since I'm writing it
for the GenBank parser output...

What's here:

Base class to hold a Feature
----------------------------

Classes:
 - SeqFeature

Hold information about a Reference
----------------------------------

This is an attempt to create a General class to hold Reference type
information.

Classes:
 - Reference

Specify locations of a feature on a Sequence
--------------------------------------------

This aims to handle, in Ewan Birney's words, 'the dreaded fuzziness issue'.
This has the advantages of allowing us to handle fuzzy stuff in case anyone
needs it, and also be compatible with BioPerl etc and BioSQL.

Classes:
 - Location - abstract base class of SimpleLocation and CompoundLocation.
 - SimpleLocation - Specify the start and end location of a feature.
 - CompoundLocation - Collection of SimpleLocation objects (for joins etc).
 - Position - abstract base class of ExactPosition, WithinPosition,
   BetweenPosition, AfterPosition, OneOfPosition, UncertainPosition, and
   UnknownPosition.
 - ExactPosition - Specify the position as being exact.
 - WithinPosition - Specify a position occurring within some range.
 - BetweenPosition - Specify a position occurring between a range (OBSOLETE?).
 - BeforePosition - Specify the position as being found before some base.
 - AfterPosition - Specify the position as being found after some base.
 - OneOfPosition - Specify a position consisting of multiple alternative positions.
 - UncertainPosition - Specify a specific position which is uncertain.
 - UnknownPosition - Represents missing information like '?' in UniProt.


Exceptions:
 - LocationParserError - Exception indicating a failure to parse a location
   string.

"""
import functools
import re
import warnings
from abc import ABC, abstractmethod
from Bio import BiopythonParserWarning
from Bio import BiopythonDeprecationWarning
from Bio.Seq import MutableSeq
from Bio.Seq import reverse_complement
from Bio.Seq import Seq
_reference = '(?:[a-zA-Z][a-zA-Z0-9_\\.\\|]*[a-zA-Z0-9]?\\:)'
_oneof_position = 'one\\-of\\(\\d+[,\\d+]+\\)'
_oneof_location = f'[<>]?(?:\\d+|{_oneof_position})\\.\\.[<>]?(?:\\d+|{_oneof_position})'
_any_location = f'({_reference}?{_oneof_location}|complement\\({_oneof_location}\\)|[^,]+|complement\\([^,]+\\))'
_split = re.compile(_any_location).split
assert _split('123..145')[1::2] == ['123..145']
assert _split('123..145,200..209')[1::2] == ['123..145', '200..209']
assert _split('one-of(200,203)..300')[1::2] == ['one-of(200,203)..300']
assert _split('complement(123..145),200..209')[1::2] == ['complement(123..145)', '200..209']
assert _split('123..145,one-of(200,203)..209')[1::2] == ['123..145', 'one-of(200,203)..209']
assert _split('123..145,one-of(200,203)..one-of(209,211),300')[1::2] == ['123..145', 'one-of(200,203)..one-of(209,211)', '300']
assert _split('123..145,complement(one-of(200,203)..one-of(209,211)),300')[1::2] == ['123..145', 'complement(one-of(200,203)..one-of(209,211))', '300']
assert _split('123..145,200..one-of(209,211),300')[1::2] == ['123..145', '200..one-of(209,211)', '300']
assert _split('123..145,200..one-of(209,211)')[1::2] == ['123..145', '200..one-of(209,211)']
assert _split('complement(149815..150200),complement(293787..295573),NC_016402.1:6618..6676,181647..181905')[1::2] == ['complement(149815..150200)', 'complement(293787..295573)', 'NC_016402.1:6618..6676', '181647..181905']
_pair_location = '[<>]?-?\\d+\\.\\.[<>]?-?\\d+'
_between_location = '\\d+\\^\\d+'
_within_position = '\\(\\d+\\.\\d+\\)'
_within_location = '([<>]?\\d+|%s)\\.\\.([<>]?\\d+|%s)' % (_within_position, _within_position)
_within_position = '\\((\\d+)\\.(\\d+)\\)'
_re_within_position = re.compile(_within_position)
assert _re_within_position.match('(3.9)')
_oneof_location = '([<>]?\\d+|%s)\\.\\.([<>]?\\d+|%s)' % (_oneof_position, _oneof_position)
_oneof_position = 'one\\-of\\((\\d+[,\\d+]+)\\)'
_re_oneof_position = re.compile(_oneof_position)
assert _re_oneof_position.match('one-of(6,9)')
assert not _re_oneof_position.match('one-of(3)')
assert _re_oneof_position.match('one-of(3,6)')
assert _re_oneof_position.match('one-of(3,6,9)')
_solo_location = '[<>]?\\d+'
_solo_bond = 'bond\\(%s\\)' % _solo_location
_re_location_category = re.compile('^(?P<pair>%s)|(?P<between>%s)|(?P<within>%s)|(?P<oneof>%s)|(?P<bond>%s)|(?P<solo>%s)$' % (_pair_location, _between_location, _within_location, _oneof_location, _solo_bond, _solo_location))

class LocationParserError(ValueError):
    """Could not parse a feature location string."""

class SeqFeature:
    """Represent a Sequence Feature on an object.

    Attributes:
     - location - the location of the feature on the sequence (SimpleLocation)
     - type - the specified type of the feature (ie. CDS, exon, repeat...)
     - location_operator - a string specifying how this SeqFeature may
       be related to others. For example, in the example GenBank feature
       shown below, the location_operator would be "join". This is a proxy
       for feature.location.operator and only applies to compound locations.
     - strand - A value specifying on which strand (of a DNA sequence, for
       instance) the feature deals with. 1 indicates the plus strand, -1
       indicates the minus strand, 0 indicates stranded but unknown (? in GFF3),
       while the default of None indicates that strand doesn't apply (dot in GFF3,
       e.g. features on proteins). Note this is a shortcut for accessing the
       strand property of the feature's location.
     - id - A string identifier for the feature.
     - ref - A reference to another sequence. This could be an accession
       number for some different sequence. Note this is a shortcut for the
       reference property of the feature's location.
     - ref_db - A different database for the reference accession number.
       Note this is a shortcut for the reference property of the location
     - qualifiers - A dictionary of qualifiers on the feature. These are
       analogous to the qualifiers from a GenBank feature table. The keys of
       the dictionary are qualifier names, the values are the qualifier
       values.

    """

    def __init__(self, location=None, type='', location_operator='', strand=None, id='<unknown id>', qualifiers=None, sub_features=None, ref=None, ref_db=None):
        if False:
            while True:
                i = 10
        'Initialize a SeqFeature on a sequence.\n\n        location can either be a SimpleLocation (with strand argument also\n        given if required), or None.\n\n        e.g. With no strand, on the forward strand, and on the reverse strand:\n\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> f1 = SeqFeature(SimpleLocation(5, 10), type="domain")\n        >>> f1.strand == f1.location.strand == None\n        True\n        >>> f2 = SeqFeature(SimpleLocation(7, 110, strand=1), type="CDS")\n        >>> f2.strand == f2.location.strand == +1\n        True\n        >>> f3 = SeqFeature(SimpleLocation(9, 108, strand=-1), type="CDS")\n        >>> f3.strand == f3.location.strand == -1\n        True\n\n        For exact start/end positions, an integer can be used (as shown above)\n        as shorthand for the ExactPosition object. For non-exact locations, the\n        SimpleLocation must be specified via the appropriate position objects.\n\n        Note that the strand, ref and ref_db arguments to the SeqFeature are\n        now deprecated and will later be removed. Set them via the location\n        object instead.\n\n        Note that location_operator and sub_features arguments can no longer\n        be used, instead do this via the CompoundLocation object.\n        '
        if location is not None and (not isinstance(location, SimpleLocation)) and (not isinstance(location, CompoundLocation)):
            raise TypeError('SimpleLocation, CompoundLocation (or None) required for the location')
        self.location = location
        self.type = type
        if location_operator:
            warnings.warn('Using the location_operator argument is deprecated, and will be removed in a future release. Please do this via the CompoundLocation object instead.', BiopythonDeprecationWarning)
            self.location_operator = location_operator
        if strand is not None:
            warnings.warn('Using the strand argument is deprecated, and will be removed in a future release. Please set it via the location object instead.', BiopythonDeprecationWarning)
            self.strand = strand
        self.id = id
        self.qualifiers = {}
        if qualifiers is not None:
            self.qualifiers.update(qualifiers)
        if sub_features is not None:
            raise TypeError('Rather than sub_features, use a CompoundLocation')
        if ref is not None:
            warnings.warn('Using the ref argument is deprecated, and will be removed in a future release. Please set it via the location object instead.', BiopythonDeprecationWarning)
            self.ref = ref
        if ref_db is not None:
            warnings.warn('Using the ref_db argument is deprecated, and will be removed in a future release. Please set it via the location object instead.', BiopythonDeprecationWarning)
            self.ref_db = ref_db

    def _get_strand(self):
        if False:
            i = 10
            return i + 15
        'Get function for the strand property (PRIVATE).'
        return self.location.strand

    def _set_strand(self, value):
        if False:
            return 10
        'Set function for the strand property (PRIVATE).'
        try:
            self.location.strand = value
        except AttributeError:
            if self.location is None:
                if value is not None:
                    raise ValueError("Can't set strand without a location.") from None
            else:
                raise
    strand = property(fget=_get_strand, fset=_set_strand, doc="Feature's strand\n\n                          This is a shortcut for feature.location.strand\n                          ")

    def _get_ref(self):
        if False:
            i = 10
            return i + 15
        'Get function for the reference property (PRIVATE).'
        try:
            return self.location.ref
        except AttributeError:
            return None

    def _set_ref(self, value):
        if False:
            return 10
        'Set function for the reference property (PRIVATE).'
        try:
            self.location.ref = value
        except AttributeError:
            if self.location is None:
                if value is not None:
                    raise ValueError("Can't set ref without a location.") from None
            else:
                raise
    ref = property(fget=_get_ref, fset=_set_ref, doc='Feature location reference (e.g. accession).\n\n                       This is a shortcut for feature.location.ref\n                       ')

    def _get_ref_db(self):
        if False:
            for i in range(10):
                print('nop')
        'Get function for the database reference property (PRIVATE).'
        try:
            return self.location.ref_db
        except AttributeError:
            return None

    def _set_ref_db(self, value):
        if False:
            print('Hello World!')
        'Set function for the database reference property (PRIVATE).'
        self.location.ref_db = value
    ref_db = property(fget=_get_ref_db, fset=_set_ref_db, doc="Feature location reference's database.\n\n                          This is a shortcut for feature.location.ref_db\n                          ")

    def _get_location_operator(self):
        if False:
            for i in range(10):
                print('nop')
        'Get function for the location operator property (PRIVATE).'
        try:
            return self.location.operator
        except AttributeError:
            return None

    def _set_location_operator(self, value):
        if False:
            return 10
        'Set function for the location operator property (PRIVATE).'
        if value:
            if isinstance(self.location, CompoundLocation):
                self.location.operator = value
            elif self.location is None:
                raise ValueError(f"Location is None so can't set its operator (to {value!r})")
            else:
                raise ValueError(f'Only CompoundLocation gets an operator ({value!r})')
    location_operator = property(fget=_get_location_operator, fset=_set_location_operator, doc='Location operator for compound locations (e.g. join).')

    def __eq__(self, other):
        if False:
            print('Hello World!')
        'Check if two SeqFeature objects should be considered equal.'
        return isinstance(other, SeqFeature) and self.id == other.id and (self.type == other.type) and (self.location == other.location) and (self.qualifiers == other.qualifiers)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Represent the feature as a string for debugging.'
        answer = f'{self.__class__.__name__}({self.location!r}'
        if self.type:
            answer += f', type={self.type!r}'
        if self.location_operator:
            answer += f', location_operator={self.location_operator!r}'
        if self.id and self.id != '<unknown id>':
            answer += f', id={self.id!r}'
        if self.qualifiers:
            answer += ', qualifiers=...'
        if self.ref:
            answer += f', ref={self.ref!r}'
        if self.ref_db:
            answer += f', ref_db={self.ref_db!r}'
        answer += ')'
        return answer

    def __str__(self):
        if False:
            return 10
        'Return the full feature as a python string.'
        out = f'type: {self.type}\n'
        out += f'location: {self.location}\n'
        if self.id and self.id != '<unknown id>':
            out += f'id: {self.id}\n'
        out += 'qualifiers:\n'
        for qual_key in sorted(self.qualifiers):
            out += f'    Key: {qual_key}, Value: {self.qualifiers[qual_key]}\n'
        return out

    def _shift(self, offset):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the feature with its location shifted (PRIVATE).\n\n        The annotation qualifiers are copied.\n        '
        return SeqFeature(location=self.location._shift(offset), type=self.type, id=self.id, qualifiers=self.qualifiers.copy())

    def _flip(self, length):
        if False:
            i = 10
            return i + 15
        'Return a copy of the feature with its location flipped (PRIVATE).\n\n        The argument length gives the length of the parent sequence. For\n        example a location 0..20 (+1 strand) with parent length 30 becomes\n        after flipping 10..30 (-1 strand). Strandless (None) or unknown\n        strand (0) remain like that - just their end points are changed.\n\n        The annotation qualifiers are copied.\n        '
        return SeqFeature(location=self.location._flip(length), type=self.type, id=self.id, qualifiers=self.qualifiers.copy())

    def extract(self, parent_sequence, references=None):
        if False:
            print('Hello World!')
        'Extract the feature\'s sequence from supplied parent sequence.\n\n        The parent_sequence can be a Seq like object or a string, and will\n        generally return an object of the same type. The exception to this is\n        a MutableSeq as the parent sequence will return a Seq object.\n\n        This should cope with complex locations including complements, joins\n        and fuzzy positions. Even mixed strand features should work! This\n        also covers features on protein sequences (e.g. domains), although\n        here reverse strand features are not permitted. If the\n        location refers to other records, they must be supplied in the\n        optional dictionary references.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")\n        >>> f = SeqFeature(SimpleLocation(8, 15), type="domain")\n        >>> f.extract(seq)\n        Seq(\'VALIVIC\')\n\n        If the SimpleLocation is None, e.g. when parsing invalid locus\n        locations in the GenBank parser, extract() will raise a ValueError.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqFeature import SeqFeature\n        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")\n        >>> f = SeqFeature(None, type="domain")\n        >>> f.extract(seq)\n        Traceback (most recent call last):\n           ...\n        ValueError: The feature\'s .location is None. Check the sequence file for a valid location.\n\n        Note - currently only compound features of type "join" are supported.\n        '
        if self.location is None:
            raise ValueError("The feature's .location is None. Check the sequence file for a valid location.")
        return self.location.extract(parent_sequence, references=references)

    def translate(self, parent_sequence, table='Standard', start_offset=None, stop_symbol='*', to_stop=False, cds=None, gap=None):
        if False:
            print('Hello World!')
        'Get a translation of the feature\'s sequence.\n\n        This method is intended for CDS or other features that code proteins\n        and is a shortcut that will both extract the feature and\n        translate it, taking into account the codon_start and transl_table\n        qualifiers, if they are present. If they are not present the\n        value of the arguments "table" and "start_offset" are used.\n\n        The "cds" parameter is set to "True" if the feature is of type\n        "CDS" but can be overridden by giving an explicit argument.\n\n        The arguments stop_symbol, to_stop and gap have the same meaning\n        as Seq.translate, refer to that documentation for further information.\n\n        Arguments:\n         - parent_sequence - A DNA or RNA sequence.\n         - table - Which codon table to use if there is no transl_table\n           qualifier for this feature. This can be either a name\n           (string), an NCBI identifier (integer), or a CodonTable\n           object (useful for non-standard genetic codes).  This\n           defaults to the "Standard" table.\n         - start_offset - offset at which the first complete codon of a\n           coding feature can be found, relative to the first base of\n           that feature. Has a valid value of 0, 1 or 2. NOTE: this\n           uses python\'s 0-based numbering whereas the codon_start\n           qualifier in files from NCBI use 1-based numbering.\n           Will override a codon_start qualifier\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> seq = Seq("GGTTACACTTACCGATAATGTCTCTGATGA")\n        >>> f = SeqFeature(SimpleLocation(0, 30), type="CDS")\n        >>> f.qualifiers[\'transl_table\'] = [11]\n\n        Note that features of type CDS are subject to the usual\n        checks at translation. But you can override this behavior\n        by giving explicit arguments:\n\n        >>> f.translate(seq, cds=False)\n        Seq(\'GYTYR*CL**\')\n\n        Now use the start_offset argument to change the frame. Note\n        this uses python 0-based numbering.\n\n        >>> f.translate(seq, start_offset=1, cds=False)\n        Seq(\'VTLTDNVSD\')\n\n        Alternatively use the codon_start qualifier to do the same\n        thing. Note: this uses 1-based numbering, which is found\n        in files from NCBI.\n\n        >>> f.qualifiers[\'codon_start\'] = [2]\n        >>> f.translate(seq, cds=False)\n        Seq(\'VTLTDNVSD\')\n        '
        if start_offset is None:
            try:
                start_offset = int(self.qualifiers['codon_start'][0]) - 1
            except KeyError:
                start_offset = 0
        if start_offset not in [0, 1, 2]:
            raise ValueError(f'The start_offset must be 0, 1, or 2. The supplied value is {start_offset}. Check the value of either the codon_start qualifier or the start_offset argument')
        feat_seq = self.extract(parent_sequence)[start_offset:]
        codon_table = self.qualifiers.get('transl_table', [table])[0]
        if cds is None:
            cds = self.type == 'CDS'
        return feat_seq.translate(table=codon_table, stop_symbol=stop_symbol, to_stop=to_stop, cds=cds, gap=gap)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        'Boolean value of an instance of this class (True).\n\n        This behavior is for backwards compatibility, since until the\n        __len__ method was added, a SeqFeature always evaluated as True.\n\n        Note that in comparison, Seq objects, strings, lists, etc, will all\n        evaluate to False if they have length zero.\n\n        WARNING: The SeqFeature may in future evaluate to False when its\n        length is zero (in order to better match normal python behavior)!\n        '
        return True

    def __len__(self):
        if False:
            while True:
                i = 10
        'Return the length of the region where the feature is located.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")\n        >>> f = SeqFeature(SimpleLocation(8, 15), type="domain")\n        >>> len(f)\n        7\n        >>> f.extract(seq)\n        Seq(\'VALIVIC\')\n        >>> len(f.extract(seq))\n        7\n\n        This is a proxy for taking the length of the feature\'s location:\n\n        >>> len(f.location)\n        7\n\n        For simple features this is the same as the region spanned (end\n        position minus start position using Pythonic counting). However, for\n        a compound location (e.g. a CDS as the join of several exons) the\n        gaps are not counted (e.g. introns). This ensures that len(f) matches\n        len(f.extract(parent_seq)), and also makes sure things work properly\n        with features wrapping the origin etc.\n        '
        return len(self.location)

    def __iter__(self):
        if False:
            return 10
        'Iterate over the parent positions within the feature.\n\n        The iteration order is strand aware, and can be thought of as moving\n        along the feature using the parent sequence coordinates:\n\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> f = SeqFeature(SimpleLocation(5, 10, strand=-1), type="domain")\n        >>> len(f)\n        5\n        >>> for i in f: print(i)\n        9\n        8\n        7\n        6\n        5\n        >>> list(f)\n        [9, 8, 7, 6, 5]\n\n        This is a proxy for iterating over the location,\n\n        >>> list(f.location)\n        [9, 8, 7, 6, 5]\n        '
        return iter(self.location)

    def __contains__(self, value):
        if False:
            for i in range(10):
                print('nop')
        'Check if an integer position is within the feature.\n\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> f = SeqFeature(SimpleLocation(5, 10, strand=-1), type="domain")\n        >>> len(f)\n        5\n        >>> [i for i in range(15) if i in f]\n        [5, 6, 7, 8, 9]\n\n        For example, to see which features include a SNP position, you could\n        use this:\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("GenBank/NC_000932.gb", "gb")\n        >>> for f in record.features:\n        ...     if 1750 in f:\n        ...         print("%s %s" % (f.type, f.location))\n        source [0:154478](+)\n        gene [1716:4347](-)\n        tRNA join{[4310:4347](-), [1716:1751](-)}\n\n        Note that for a feature defined as a join of several subfeatures (e.g.\n        the union of several exons) the gaps are not checked (e.g. introns).\n        In this example, the tRNA location is defined in the GenBank file as\n        complement(join(1717..1751,4311..4347)), so that position 1760 falls\n        in the gap:\n\n        >>> for f in record.features:\n        ...     if 1760 in f:\n        ...         print("%s %s" % (f.type, f.location))\n        source [0:154478](+)\n        gene [1716:4347](-)\n\n        Note that additional care may be required with fuzzy locations, for\n        example just before a BeforePosition:\n\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> from Bio.SeqFeature import BeforePosition\n        >>> f = SeqFeature(SimpleLocation(BeforePosition(3), 8), type="domain")\n        >>> len(f)\n        5\n        >>> [i for i in range(10) if i in f]\n        [3, 4, 5, 6, 7]\n\n        Note that is is a proxy for testing membership on the location.\n\n        >>> [i for i in range(10) if i in f.location]\n        [3, 4, 5, 6, 7]\n        '
        return value in self.location

class Reference:
    """Represent a Generic Reference object.

    Attributes:
     - location - A list of Location objects specifying regions of
       the sequence that the references correspond to. If no locations are
       specified, the entire sequence is assumed.
     - authors - A big old string, or a list split by author, of authors
       for the reference.
     - title - The title of the reference.
     - journal - Journal the reference was published in.
     - medline_id - A medline reference for the article.
     - pubmed_id - A pubmed reference for the article.
     - comment - A place to stick any comments about the reference.

    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.location = []
        self.authors = ''
        self.consrtm = ''
        self.title = ''
        self.journal = ''
        self.medline_id = ''
        self.pubmed_id = ''
        self.comment = ''

    def __str__(self):
        if False:
            print('Hello World!')
        'Return the full Reference object as a python string.'
        out = ''
        for single_location in self.location:
            out += f'location: {single_location}\n'
        out += f'authors: {self.authors}\n'
        if self.consrtm:
            out += f'consrtm: {self.consrtm}\n'
        out += f'title: {self.title}\n'
        out += f'journal: {self.journal}\n'
        out += f'medline id: {self.medline_id}\n'
        out += f'pubmed id: {self.pubmed_id}\n'
        out += f'comment: {self.comment}\n'
        return out

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        'Represent the Reference object as a string for debugging.'
        return f'{self.__class__.__name__}(title={self.title!r}, ...)'

    def __eq__(self, other):
        if False:
            return 10
        'Check if two Reference objects should be considered equal.\n\n        Note prior to Biopython 1.70 the location was not compared, as\n        until then __eq__ for the SimpleLocation class was not defined.\n        '
        return self.authors == other.authors and self.consrtm == other.consrtm and (self.title == other.title) and (self.journal == other.journal) and (self.medline_id == other.medline_id) and (self.pubmed_id == other.pubmed_id) and (self.comment == other.comment) and (self.location == other.location)

class Location(ABC):
    """Abstract base class representing a location."""

    @abstractmethod
    def __repr__(self):
        if False:
            print('Hello World!')
        'Represent the Location object as a string for debugging.'
        return f'{self.__class__.__name__}(...)'

    def fromstring(text, length=None, circular=False, stranded=True):
        if False:
            for i in range(10):
                print('nop')
        'Create a Location object from a string.\n\n        This should accept any valid location string in the INSDC Feature Table\n        format (https://www.insdc.org/submitting-standards/feature-table/) as\n        used in GenBank, DDBJ and EMBL files.\n\n        Simple examples:\n\n        >>> Location.fromstring("123..456", 1000)\n        SimpleLocation(ExactPosition(122), ExactPosition(456), strand=1)\n        >>> Location.fromstring("complement(<123..>456)", 1000)\n        SimpleLocation(BeforePosition(122), AfterPosition(456), strand=-1)\n\n        A more complex location using within positions,\n\n        >>> Location.fromstring("(9.10)..(20.25)", 1000)\n        SimpleLocation(WithinPosition(8, left=8, right=9), WithinPosition(25, left=20, right=25), strand=1)\n\n        Notice how that will act as though it has overall start 8 and end 25.\n\n        Zero length between feature,\n\n        >>> Location.fromstring("123^124", 1000)\n        SimpleLocation(ExactPosition(123), ExactPosition(123), strand=1)\n\n        The expected sequence length is needed for a special case, a between\n        position at the start/end of a circular genome:\n\n        >>> Location.fromstring("1000^1", 1000)\n        SimpleLocation(ExactPosition(1000), ExactPosition(1000), strand=1)\n\n        Apart from this special case, between positions P^Q must have P+1==Q,\n\n        >>> Location.fromstring("123^456", 1000)\n        Traceback (most recent call last):\n           ...\n        Bio.SeqFeature.LocationParserError: invalid feature location \'123^456\'\n\n        You can optionally provide a reference name:\n\n        >>> Location.fromstring("AL391218.9:105173..108462", 2000000)\n        SimpleLocation(ExactPosition(105172), ExactPosition(108462), strand=1, ref=\'AL391218.9\')\n\n        >>> Location.fromstring("<2644..159", 2868, "circular")\n        CompoundLocation([SimpleLocation(BeforePosition(2643), ExactPosition(2868), strand=1), SimpleLocation(ExactPosition(0), ExactPosition(159), strand=1)], \'join\')\n        '
        if text.startswith('complement('):
            if text[-1] != ')':
                raise ValueError(f"closing bracket missing in '{text}'")
            text = text[11:-1]
            strand = -1
        elif stranded:
            strand = 1
        else:
            strand = None
        if text.startswith('join('):
            operator = 'join'
            parts = _split(text[5:-1])[1::2]
        elif text.startswith('order('):
            operator = 'order'
            parts = _split(text[6:-1])[1::2]
        elif text.startswith('bond('):
            operator = 'bond'
            parts = _split(text[5:-1])[1::2]
        else:
            loc = SimpleLocation.fromstring(text, length, circular)
            loc.strand = strand
            if strand == -1:
                loc.parts.reverse()
            return loc
        locs = []
        for part in parts:
            loc = SimpleLocation.fromstring(part, length, circular)
            if loc is None:
                break
            if loc.strand == -1:
                if strand == -1:
                    raise LocationParserError("double complement in '{text}'?")
            else:
                loc.strand = strand
            locs.extend(loc.parts)
        else:
            if len(locs) == 1:
                return loc
            if strand == -1:
                for loc in locs:
                    assert loc.strand == -1
                locs = locs[::-1]
            return CompoundLocation(locs, operator=operator)
        if 'order' in text and 'join' in text:
            raise LocationParserError(f"failed to parse feature location '{text}' containing a combination of 'join' and 'order' (nested operators) are illegal")
        if ',)' in text:
            warnings.warn('Dropping trailing comma in malformed feature location', BiopythonParserWarning)
            text = text.replace(',)', ')')
            return Location.fromstring(text)
        raise LocationParserError(f"failed to parse feature location '{text}'")

class SimpleLocation(Location):
    """Specify the location of a feature along a sequence.

    The SimpleLocation is used for simple continuous features, which can
    be described as running from a start position to and end position
    (optionally with a strand and reference information).  More complex
    locations made up from several non-continuous parts (e.g. a coding
    sequence made up of several exons) are described using a SeqFeature
    with a CompoundLocation.

    Note that the start and end location numbering follow Python's scheme,
    thus a GenBank entry of 123..150 (one based counting) becomes a location
    of [122:150] (zero based counting).

    >>> from Bio.SeqFeature import SimpleLocation
    >>> f = SimpleLocation(122, 150)
    >>> print(f)
    [122:150]
    >>> print(f.start)
    122
    >>> print(f.end)
    150
    >>> print(f.strand)
    None

    Note the strand defaults to None. If you are working with nucleotide
    sequences you'd want to be explicit if it is the forward strand:

    >>> from Bio.SeqFeature import SimpleLocation
    >>> f = SimpleLocation(122, 150, strand=+1)
    >>> print(f)
    [122:150](+)
    >>> print(f.strand)
    1

    Note that for a parent sequence of length n, the SimpleLocation
    start and end must satisfy the inequality 0 <= start <= end <= n.
    This means even for features on the reverse strand of a nucleotide
    sequence, we expect the 'start' coordinate to be less than the
    'end'.

    >>> from Bio.SeqFeature import SimpleLocation
    >>> r = SimpleLocation(122, 150, strand=-1)
    >>> print(r)
    [122:150](-)
    >>> print(r.start)
    122
    >>> print(r.end)
    150
    >>> print(r.strand)
    -1

    i.e. Rather than thinking of the 'start' and 'end' biologically in a
    strand aware manner, think of them as the 'left most' or 'minimum'
    boundary, and the 'right most' or 'maximum' boundary of the region
    being described. This is particularly important with compound
    locations describing non-continuous regions.

    In the example above we have used standard exact positions, but there
    are also specialised position objects used to represent fuzzy positions
    as well, for example a GenBank location like complement(<123..150)
    would use a BeforePosition object for the start.
    """

    def __init__(self, start, end, strand=None, ref=None, ref_db=None):
        if False:
            print('Hello World!')
        'Initialize the class.\n\n        start and end arguments specify the values where the feature begins\n        and ends. These can either by any of the ``*Position`` objects that\n        inherit from Position, or can just be integers specifying the position.\n        In the case of integers, the values are assumed to be exact and are\n        converted in ExactPosition arguments. This is meant to make it easy\n        to deal with non-fuzzy ends.\n\n        i.e. Short form:\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> loc = SimpleLocation(5, 10, strand=-1)\n        >>> print(loc)\n        [5:10](-)\n\n        Explicit form:\n\n        >>> from Bio.SeqFeature import SimpleLocation, ExactPosition\n        >>> loc = SimpleLocation(ExactPosition(5), ExactPosition(10), strand=-1)\n        >>> print(loc)\n        [5:10](-)\n\n        Other fuzzy positions are used similarly,\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> from Bio.SeqFeature import BeforePosition, AfterPosition\n        >>> loc2 = SimpleLocation(BeforePosition(5), AfterPosition(10), strand=-1)\n        >>> print(loc2)\n        [<5:>10](-)\n\n        For nucleotide features you will also want to specify the strand,\n        use 1 for the forward (plus) strand, -1 for the reverse (negative)\n        strand, 0 for stranded but strand unknown (? in GFF3), or None for\n        when the strand does not apply (dot in GFF3), e.g. features on\n        proteins.\n\n        >>> loc = SimpleLocation(5, 10, strand=+1)\n        >>> print(loc)\n        [5:10](+)\n        >>> print(loc.strand)\n        1\n\n        Normally feature locations are given relative to the parent\n        sequence you are working with, but an explicit accession can\n        be given with the optional ref and db_ref strings:\n\n        >>> loc = SimpleLocation(105172, 108462, ref="AL391218.9", strand=1)\n        >>> print(loc)\n        AL391218.9[105172:108462](+)\n        >>> print(loc.ref)\n        AL391218.9\n\n        '
        if isinstance(start, Position):
            self._start = start
        elif isinstance(start, int):
            self._start = ExactPosition(start)
        else:
            raise TypeError(f'start={start!r} {type(start)}')
        if isinstance(end, Position):
            self._end = end
        elif isinstance(end, int):
            self._end = ExactPosition(end)
        else:
            raise TypeError(f'end={end!r} {type(end)}')
        if isinstance(self.start, int) and isinstance(self.end, int) and (self.start > self.end):
            raise ValueError(f'End location ({self.end}) must be greater than or equal to start location ({self.start})')
        self.strand = strand
        self.ref = ref
        self.ref_db = ref_db

    @staticmethod
    def fromstring(text, length=None, circular=False):
        if False:
            print('Hello World!')
        'Create a SimpleLocation object from a string.'
        if text.startswith('complement('):
            text = text[11:-1]
            strand = -1
        else:
            strand = None
        try:
            (s, e) = text.split('..')
            s = int(s) - 1
            e = int(e)
        except ValueError:
            pass
        else:
            if 0 <= s <= e:
                return SimpleLocation(s, e, strand)
        try:
            (ref, text) = text.split(':')
        except ValueError:
            ref = None
        m = _re_location_category.match(text)
        if m is None:
            raise LocationParserError(f"Could not parse feature location '{text}'")
        for (key, value) in m.groupdict().items():
            if value is not None:
                break
        assert value == text
        if key == 'bond':
            warnings.warn('Dropping bond qualifier in feature location', BiopythonParserWarning)
            text = text[5:-1]
            s_pos = Position.fromstring(text, -1)
            e_pos = Position.fromstring(text)
        elif key == 'solo':
            s_pos = Position.fromstring(text, -1)
            e_pos = Position.fromstring(text)
        elif key in ('pair', 'within', 'oneof'):
            (s, e) = text.split('..')
            s_pos = Position.fromstring(s, -1)
            e_pos = Position.fromstring(e)
            if s_pos > e_pos:
                if not circular:
                    raise LocationParserError(f"it appears that '{text}' is a feature that spans the origin, but the sequence topology is undefined")
                warnings.warn('Attempting to fix invalid location %r as it looks like incorrect origin wrapping. Please fix input file, this could have unintended behavior.' % text, BiopythonParserWarning)
                f1 = SimpleLocation(s_pos, length, strand)
                f2 = SimpleLocation(0, e_pos, strand)
                if strand == -1:
                    return f2 + f1
                else:
                    return f1 + f2
        elif key == 'between':
            (s, e) = text.split('^')
            s = int(s)
            e = int(e)
            if s + 1 == e or (s == length and e == 1):
                s_pos = ExactPosition(s)
                e_pos = s_pos
            else:
                raise LocationParserError(f"invalid feature location '{text}'")
        if s_pos < 0:
            raise LocationParserError(f"negative starting position in feature location '{text}'")
        return SimpleLocation(s_pos, e_pos, strand, ref=ref)

    def _get_strand(self):
        if False:
            print('Hello World!')
        'Get function for the strand property (PRIVATE).'
        return self._strand

    def _set_strand(self, value):
        if False:
            print('Hello World!')
        'Set function for the strand property (PRIVATE).'
        if value not in [+1, -1, 0, None]:
            raise ValueError(f'Strand should be +1, -1, 0 or None, not {value!r}')
        self._strand = value
    strand = property(fget=_get_strand, fset=_set_strand, doc='Strand of the location (+1, -1, 0 or None).')

    def __str__(self):
        if False:
            return 10
        'Return a representation of the SimpleLocation object (with python counting).\n\n        For the simple case this uses the python splicing syntax, [122:150]\n        (zero based counting) which GenBank would call 123..150 (one based\n        counting).\n        '
        answer = f'[{self._start}:{self._end}]'
        if self.ref and self.ref_db:
            answer = f'{self.ref_db}:{self.ref}{answer}'
        elif self.ref:
            answer = self.ref + answer
        if self.strand is None:
            return answer
        elif self.strand == +1:
            return answer + '(+)'
        elif self.strand == -1:
            return answer + '(-)'
        else:
            return answer + '(?)'

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Represent the SimpleLocation object as a string for debugging.'
        optional = ''
        if self.strand is not None:
            optional += f', strand={self.strand!r}'
        if self.ref is not None:
            optional += f', ref={self.ref!r}'
        if self.ref_db is not None:
            optional += f', ref_db={self.ref_db!r}'
        return f'{self.__class__.__name__}({self.start!r}, {self.end!r}{optional})'

    def __add__(self, other):
        if False:
            while True:
                i = 10
        'Combine location with another SimpleLocation object, or shift it.\n\n        You can add two feature locations to make a join CompoundLocation:\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> f1 = SimpleLocation(5, 10)\n        >>> f2 = SimpleLocation(20, 30)\n        >>> combined = f1 + f2\n        >>> print(combined)\n        join{[5:10], [20:30]}\n\n        This is thus equivalent to:\n\n        >>> from Bio.SeqFeature import CompoundLocation\n        >>> join = CompoundLocation([f1, f2])\n        >>> print(join)\n        join{[5:10], [20:30]}\n\n        You can also use sum(...) in this way:\n\n        >>> join = sum([f1, f2])\n        >>> print(join)\n        join{[5:10], [20:30]}\n\n        Furthermore, you can combine a SimpleLocation with a CompoundLocation\n        in this way.\n\n        Separately, adding an integer will give a new SimpleLocation with\n        its start and end offset by that amount. For example:\n\n        >>> print(f1)\n        [5:10]\n        >>> print(f1 + 100)\n        [105:110]\n        >>> print(200 + f1)\n        [205:210]\n\n        This can be useful when editing annotation.\n        '
        if isinstance(other, SimpleLocation):
            return CompoundLocation([self, other])
        elif isinstance(other, int):
            return self._shift(other)
        else:
            return NotImplemented

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        'Return a SimpleLocation object by shifting the location by an integer amount.'
        if isinstance(other, int):
            return self._shift(other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        'Subtracting an integer will shift the start and end by that amount.\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> f1 = SimpleLocation(105, 150)\n        >>> print(f1)\n        [105:150]\n        >>> print(f1 - 100)\n        [5:50]\n\n        This can be useful when editing annotation. You can also add an integer\n        to a feature location (which shifts in the opposite direction).\n        '
        if isinstance(other, int):
            return self._shift(-other)
        else:
            return NotImplemented

    def __nonzero__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return True regardless of the length of the feature.\n\n        This behavior is for backwards compatibility, since until the\n        __len__ method was added, a SimpleLocation always evaluated as True.\n\n        Note that in comparison, Seq objects, strings, lists, etc, will all\n        evaluate to False if they have length zero.\n\n        WARNING: The SimpleLocation may in future evaluate to False when its\n        length is zero (in order to better match normal python behavior)!\n        '
        return True

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the length of the region described by the SimpleLocation object.\n\n        Note that extra care may be needed for fuzzy locations, e.g.\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> from Bio.SeqFeature import BeforePosition, AfterPosition\n        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10))\n        >>> len(loc)\n        5\n        '
        return int(self._end) - int(self._start)

    def __contains__(self, value):
        if False:
            while True:
                i = 10
        'Check if an integer position is within the SimpleLocation object.\n\n        Note that extra care may be needed for fuzzy locations, e.g.\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> from Bio.SeqFeature import BeforePosition, AfterPosition\n        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10))\n        >>> len(loc)\n        5\n        >>> [i for i in range(15) if i in loc]\n        [5, 6, 7, 8, 9]\n        '
        if not isinstance(value, int):
            raise ValueError('Currently we only support checking for integer positions being within a SimpleLocation.')
        if value < self._start or value >= self._end:
            return False
        else:
            return True

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Iterate over the parent positions within the SimpleLocation object.\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> from Bio.SeqFeature import BeforePosition, AfterPosition\n        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10))\n        >>> len(loc)\n        5\n        >>> for i in loc: print(i)\n        5\n        6\n        7\n        8\n        9\n        >>> list(loc)\n        [5, 6, 7, 8, 9]\n        >>> [i for i in range(15) if i in loc]\n        [5, 6, 7, 8, 9]\n\n        Note this is strand aware:\n\n        >>> loc = SimpleLocation(BeforePosition(5), AfterPosition(10), strand = -1)\n        >>> list(loc)\n        [9, 8, 7, 6, 5]\n        '
        if self.strand == -1:
            yield from range(self._end - 1, self._start - 1, -1)
        else:
            yield from range(self._start, self._end)

    def __eq__(self, other):
        if False:
            return 10
        'Implement equality by comparing all the location attributes.'
        if not isinstance(other, SimpleLocation):
            return False
        return self._start == other.start and self._end == other.end and (self._strand == other.strand) and (self.ref == other.ref) and (self.ref_db == other.ref_db)

    def _shift(self, offset):
        if False:
            print('Hello World!')
        'Return a copy of the SimpleLocation shifted by an offset (PRIVATE).\n\n        Returns self when location is relative to an external reference.\n        '
        if self.ref or self.ref_db:
            return self
        return SimpleLocation(start=self._start + offset, end=self._end + offset, strand=self.strand)

    def _flip(self, length):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the location after the parent is reversed (PRIVATE).\n\n        Returns self when location is relative to an external reference.\n        '
        if self.ref or self.ref_db:
            return self
        if self.strand == +1:
            flip_strand = -1
        elif self.strand == -1:
            flip_strand = +1
        else:
            flip_strand = self.strand
        return SimpleLocation(start=self._end._flip(length), end=self._start._flip(length), strand=flip_strand)

    @property
    def parts(self):
        if False:
            for i in range(10):
                print('nop')
        'Read only list of sections (always one, the SimpleLocation object).\n\n        This is a convenience property allowing you to write code handling\n        both SimpleLocation objects (with one part) and more complex\n        CompoundLocation objects (with multiple parts) interchangeably.\n        '
        return [self]

    @property
    def start(self):
        if False:
            print('Hello World!')
        'Start location - left most (minimum) value, regardless of strand.\n\n        Read only, returns an integer like position object, possibly a fuzzy\n        position.\n        '
        return self._start

    @property
    def end(self):
        if False:
            return 10
        'End location - right most (maximum) value, regardless of strand.\n\n        Read only, returns an integer like position object, possibly a fuzzy\n        position.\n        '
        return self._end

    @property
    def nofuzzy_start(self):
        if False:
            for i in range(10):
                print('nop')
        'Start position (integer, approximated if fuzzy, read only) (DEPRECATED).\n\n        This is now an alias for int(feature.start), which should be\n        used in preference -- unless you are trying to support old\n        versions of Biopython.\n        '
        warnings.warn('Use int(feature.start) rather than feature.nofuzzy_start', BiopythonDeprecationWarning)
        try:
            return int(self._start)
        except TypeError:
            if isinstance(self._start, UnknownPosition):
                return None
            raise

    @property
    def nofuzzy_end(self):
        if False:
            print('Hello World!')
        'End position (integer, approximated if fuzzy, read only) (DEPRECATED).\n\n        This is now an alias for int(feature.end), which should be\n        used in preference -- unless you are trying to support old\n        versions of Biopython.\n        '
        warnings.warn('Use int(feature.end) rather than feature.nofuzzy_end', BiopythonDeprecationWarning)
        try:
            return int(self._end)
        except TypeError:
            if isinstance(self._end, UnknownPosition):
                return None
            raise

    def extract(self, parent_sequence, references=None):
        if False:
            i = 10
            return i + 15
        'Extract the sequence from supplied parent sequence using the SimpleLocation object.\n\n        The parent_sequence can be a Seq like object or a string, and will\n        generally return an object of the same type. The exception to this is\n        a MutableSeq as the parent sequence will return a Seq object.\n        If the location refers to other records, they must be supplied\n        in the optional dictionary references.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")\n        >>> feature_loc = SimpleLocation(8, 15)\n        >>> feature_loc.extract(seq)\n        Seq(\'VALIVIC\')\n\n        '
        if self.ref or self.ref_db:
            if not references:
                raise ValueError(f'Feature references another sequence ({self.ref}), references mandatory')
            elif self.ref not in references:
                raise ValueError(f'Feature references another sequence ({self.ref}), not found in references')
            parent_sequence = references[self.ref]
        f_seq = parent_sequence[int(self.start):int(self.end)]
        if isinstance(f_seq, MutableSeq):
            f_seq = Seq(f_seq)
        if self.strand == -1:
            f_seq = reverse_complement(f_seq, inplace=False)
        return f_seq
FeatureLocation = SimpleLocation

class CompoundLocation(Location):
    """For handling joins etc where a feature location has several parts."""

    def __init__(self, parts, operator='join'):
        if False:
            print('Hello World!')
        "Initialize the class.\n\n        >>> from Bio.SeqFeature import SimpleLocation, CompoundLocation\n        >>> f1 = SimpleLocation(10, 40, strand=+1)\n        >>> f2 = SimpleLocation(50, 59, strand=+1)\n        >>> f = CompoundLocation([f1, f2])\n        >>> len(f) == len(f1) + len(f2) == 39 == len(list(f))\n        True\n        >>> print(f.operator)\n        join\n        >>> 5 in f\n        False\n        >>> 15 in f\n        True\n        >>> f.strand\n        1\n\n        Notice that the strand of the compound location is computed\n        automatically - in the case of mixed strands on the sub-locations\n        the overall strand is set to None.\n\n        >>> f = CompoundLocation([SimpleLocation(3, 6, strand=+1),\n        ...                       SimpleLocation(10, 13, strand=-1)])\n        >>> print(f.strand)\n        None\n        >>> len(f)\n        6\n        >>> list(f)\n        [3, 4, 5, 12, 11, 10]\n\n        The example above doing list(f) iterates over the coordinates within the\n        feature. This allows you to use max and min on the location, to find the\n        range covered:\n\n        >>> min(f)\n        3\n        >>> max(f)\n        12\n\n        More generally, you can use the compound location's start and end which\n        give the full span covered, 0 <= start <= end <= full sequence length.\n\n        >>> f.start == min(f)\n        True\n        >>> f.end == max(f) + 1\n        True\n\n        This is consistent with the behavior of the SimpleLocation for a single\n        region, where again the 'start' and 'end' do not necessarily give the\n        biological start and end, but rather the 'minimal' and 'maximal'\n        coordinate boundaries.\n\n        Note that adding locations provides a more intuitive method of\n        construction:\n\n        >>> f = SimpleLocation(3, 6, strand=+1) + SimpleLocation(10, 13, strand=-1)\n        >>> len(f)\n        6\n        >>> list(f)\n        [3, 4, 5, 12, 11, 10]\n        "
        self.operator = operator
        self.parts = list(parts)
        for loc in self.parts:
            if not isinstance(loc, SimpleLocation):
                raise ValueError('CompoundLocation should be given a list of SimpleLocation objects, not %s' % loc.__class__)
        if len(parts) < 2:
            raise ValueError(f'CompoundLocation should have at least 2 parts, not {parts!r}')

    def __str__(self):
        if False:
            print('Hello World!')
        'Return a representation of the CompoundLocation object (with python counting).'
        return '%s{%s}' % (self.operator, ', '.join((str(loc) for loc in self.parts)))

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Represent the CompoundLocation object as string for debugging.'
        return f'{self.__class__.__name__}({self.parts!r}, {self.operator!r})'

    def _get_strand(self):
        if False:
            i = 10
            return i + 15
        'Get function for the strand property (PRIVATE).'
        if len({loc.strand for loc in self.parts}) == 1:
            return self.parts[0].strand
        else:
            return None

    def _set_strand(self, value):
        if False:
            return 10
        'Set function for the strand property (PRIVATE).'
        for loc in self.parts:
            loc.strand = value
    strand = property(fget=_get_strand, fset=_set_strand, doc='Overall strand of the compound location.\n\n        If all the parts have the same strand, that is returned. Otherwise\n        for mixed strands, this returns None.\n\n        >>> from Bio.SeqFeature import SimpleLocation, CompoundLocation\n        >>> f1 = SimpleLocation(15, 17, strand=1)\n        >>> f2 = SimpleLocation(20, 30, strand=-1)\n        >>> f = f1 + f2\n        >>> f1.strand\n        1\n        >>> f2.strand\n        -1\n        >>> f.strand\n        >>> f.strand is None\n        True\n\n        If you set the strand of a CompoundLocation, this is applied to\n        all the parts - use with caution:\n\n        >>> f.strand = 1\n        >>> f1.strand\n        1\n        >>> f2.strand\n        1\n        >>> f.strand\n        1\n\n        ')

    def __add__(self, other):
        if False:
            while True:
                i = 10
        "Combine locations, or shift the location by an integer offset.\n\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> f1 = SimpleLocation(15, 17) + SimpleLocation(20, 30)\n        >>> print(f1)\n        join{[15:17], [20:30]}\n\n        You can add another SimpleLocation:\n\n        >>> print(f1 + SimpleLocation(40, 50))\n        join{[15:17], [20:30], [40:50]}\n        >>> print(SimpleLocation(5, 10) + f1)\n        join{[5:10], [15:17], [20:30]}\n\n        You can also add another CompoundLocation:\n\n        >>> f2 = SimpleLocation(40, 50) + SimpleLocation(60, 70)\n        >>> print(f2)\n        join{[40:50], [60:70]}\n        >>> print(f1 + f2)\n        join{[15:17], [20:30], [40:50], [60:70]}\n\n        Also, as with the SimpleLocation, adding an integer shifts the\n        location's coordinates by that offset:\n\n        >>> print(f1 + 100)\n        join{[115:117], [120:130]}\n        >>> print(200 + f1)\n        join{[215:217], [220:230]}\n        >>> print(f1 + (-5))\n        join{[10:12], [15:25]}\n        "
        if isinstance(other, SimpleLocation):
            return CompoundLocation(self.parts + [other], self.operator)
        elif isinstance(other, CompoundLocation):
            if self.operator != other.operator:
                raise ValueError(f'Mixed operators {self.operator} and {other.operator}')
            return CompoundLocation(self.parts + other.parts, self.operator)
        elif isinstance(other, int):
            return self._shift(other)
        else:
            raise NotImplementedError

    def __radd__(self, other):
        if False:
            return 10
        'Add a feature to the left.'
        if isinstance(other, SimpleLocation):
            return CompoundLocation([other] + self.parts, self.operator)
        elif isinstance(other, int):
            return self._shift(other)
        else:
            raise NotImplementedError

    def __contains__(self, value):
        if False:
            print('Hello World!')
        'Check if an integer position is within the CompoundLocation object.'
        for loc in self.parts:
            if value in loc:
                return True
        return False

    def __nonzero__(self):
        if False:
            i = 10
            return i + 15
        'Return True regardless of the length of the feature.\n\n        This behavior is for backwards compatibility, since until the\n        __len__ method was added, a SimpleLocation always evaluated as True.\n\n        Note that in comparison, Seq objects, strings, lists, etc, will all\n        evaluate to False if they have length zero.\n\n        WARNING: The SimpleLocation may in future evaluate to False when its\n        length is zero (in order to better match normal python behavior)!\n        '
        return True

    def __len__(self):
        if False:
            print('Hello World!')
        'Return the length of the CompoundLocation object.'
        return sum((len(loc) for loc in self.parts))

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over the parent positions within the CompoundLocation object.'
        for loc in self.parts:
            yield from loc

    def __eq__(self, other):
        if False:
            return 10
        'Check if all parts of CompoundLocation are equal to all parts of other CompoundLocation.'
        if not isinstance(other, CompoundLocation):
            return False
        if len(self.parts) != len(other.parts):
            return False
        if self.operator != other.operator:
            return False
        for (self_part, other_part) in zip(self.parts, other.parts):
            if self_part != other_part:
                return False
        return True

    def _shift(self, offset):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the CompoundLocation shifted by an offset (PRIVATE).'
        return CompoundLocation([loc._shift(offset) for loc in self.parts], self.operator)

    def _flip(self, length):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the locations after the parent is reversed (PRIVATE).\n\n        Note that the order of the parts is NOT reversed too. Consider a CDS\n        on the forward strand with exons small, medium and large (in length).\n        Once we change the frame of reference to the reverse complement strand,\n        the start codon is still part of the small exon, and the stop codon\n        still part of the large exon - so the part order remains the same!\n\n        Here is an artificial example, were the features map to the two upper\n        case regions and the lower case runs of n are not used:\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqFeature import SimpleLocation\n        >>> dna = Seq("nnnnnAGCATCCTGCTGTACnnnnnnnnGAGAMTGCCATGCCCCTGGAGTGAnnnnn")\n        >>> small = SimpleLocation(5, 20, strand=1)\n        >>> large = SimpleLocation(28, 52, strand=1)\n        >>> location = small + large\n        >>> print(small)\n        [5:20](+)\n        >>> print(large)\n        [28:52](+)\n        >>> print(location)\n        join{[5:20](+), [28:52](+)}\n        >>> for part in location.parts:\n        ...     print(len(part))\n        ...\n        15\n        24\n\n        As you can see, this is a silly example where each "exon" is a word:\n\n        >>> print(small.extract(dna).translate())\n        SILLY\n        >>> print(large.extract(dna).translate())\n        EXAMPLE*\n        >>> print(location.extract(dna).translate())\n        SILLYEXAMPLE*\n        >>> for part in location.parts:\n        ...     print(part.extract(dna).translate())\n        ...\n        SILLY\n        EXAMPLE*\n\n        Now, let\'s look at this from the reverse strand frame of reference:\n\n        >>> flipped_dna = dna.reverse_complement()\n        >>> flipped_location = location._flip(len(dna))\n        >>> print(flipped_location.extract(flipped_dna).translate())\n        SILLYEXAMPLE*\n        >>> for part in flipped_location.parts:\n        ...     print(part.extract(flipped_dna).translate())\n        ...\n        SILLY\n        EXAMPLE*\n\n        The key point here is the first part of the CompoundFeature is still the\n        small exon, while the second part is still the large exon:\n\n        >>> for part in flipped_location.parts:\n        ...     print(len(part))\n        ...\n        15\n        24\n        >>> print(flipped_location)\n        join{[37:52](-), [5:29](-)}\n\n        Notice the parts are not reversed. However, there was a bug here in older\n        versions of Biopython which would have given join{[5:29](-), [37:52](-)}\n        and the translation would have wrongly been "EXAMPLE*SILLY" instead.\n\n        '
        return CompoundLocation([loc._flip(length) for loc in self.parts], self.operator)

    @property
    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Start location - left most (minimum) value, regardless of strand.\n\n        Read only, returns an integer like position object, possibly a fuzzy\n        position.\n\n        For the special case of a CompoundLocation wrapping the origin of a\n        circular genome, this will return zero.\n        '
        return min((loc.start for loc in self.parts))

    @property
    def end(self):
        if False:
            i = 10
            return i + 15
        'End location - right most (maximum) value, regardless of strand.\n\n        Read only, returns an integer like position object, possibly a fuzzy\n        position.\n\n        For the special case of a CompoundLocation wrapping the origin of\n        a circular genome this will match the genome length.\n        '
        return max((loc.end for loc in self.parts))

    @property
    def nofuzzy_start(self):
        if False:
            for i in range(10):
                print('nop')
        'Start position (integer, approximated if fuzzy, read only) (DEPRECATED).\n\n        This is an alias for int(feature.start), which should be used in\n        preference -- unless you are trying to support old versions of\n        Biopython.\n        '
        warnings.warn('Use int(feature.start) rather than feature.nofuzzy_start', BiopythonDeprecationWarning)
        try:
            return int(self.start)
        except TypeError:
            if isinstance(self.start, UnknownPosition):
                return None
            raise

    @property
    def nofuzzy_end(self):
        if False:
            print('Hello World!')
        'End position (integer, approximated if fuzzy, read only) (DEPRECATED).\n\n        This is an alias for int(feature.end), which should be used in\n        preference -- unless you are trying to support old versions of\n        Biopython.\n        '
        warnings.warn('Use int(feature.end) rather than feature.nofuzzy_end', BiopythonDeprecationWarning)
        try:
            return int(self.end)
        except TypeError:
            if isinstance(self.end, UnknownPosition):
                return None
            raise

    @property
    def ref(self):
        if False:
            print('Hello World!')
        'Not present in CompoundLocation, dummy method for API compatibility.'
        return None

    @property
    def ref_db(self):
        if False:
            i = 10
            return i + 15
        'Not present in CompoundLocation, dummy method for API compatibility.'
        return None

    def extract(self, parent_sequence, references=None):
        if False:
            i = 10
            return i + 15
        'Extract the sequence from supplied parent sequence using the CompoundLocation object.\n\n        The parent_sequence can be a Seq like object or a string, and will\n        generally return an object of the same type. The exception to this is\n        a MutableSeq as the parent sequence will return a Seq object.\n        If the location refers to other records, they must be supplied\n        in the optional dictionary references.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqFeature import SimpleLocation, CompoundLocation\n        >>> seq = Seq("MKQHKAMIVALIVICITAVVAAL")\n        >>> fl1 = SimpleLocation(2, 8)\n        >>> fl2 = SimpleLocation(10, 15)\n        >>> fl3 = CompoundLocation([fl1,fl2])\n        >>> fl3.extract(seq)\n        Seq(\'QHKAMILIVIC\')\n\n        '
        parts = [loc.extract(parent_sequence, references=references) for loc in self.parts]
        f_seq = functools.reduce(lambda x, y: x + y, parts)
        return f_seq

class Position(ABC):
    """Abstract base class representing a position."""

    @abstractmethod
    def __repr__(self):
        if False:
            print('Hello World!')
        'Represent the Position object as a string for debugging.'
        return f'{self.__class__.__name__}(...)'

    @property
    def position(self):
        if False:
            i = 10
            return i + 15
        'Legacy attribute to get (left-most) position as an integer (DEPRECATED).'
        warnings.warn('Alias location.position is deprecated and will be removed in a future release. Use location directly, or int(location). However, that will fail for UnknownPosition, and for OneOfPosition and WithinPosition will give the default rather than left-most value.', BiopythonDeprecationWarning)
        return int(self)

    @property
    def extension(self):
        if False:
            print('Hello World!')
        "Legacy attribute to get the position's 'width' as an integer, typically zero (DEPRECATED)."
        warnings.warn('Alias location.extension is deprecated and will be removed in a future release. It was undefined or zero except for OneOfPosition, WithinPosition and WithinPosition which must now be handled explicitly instead.', BiopythonDeprecationWarning)
        return 0

    @staticmethod
    def fromstring(text, offset=0):
        if False:
            return 10
        'Build a Position object from the text string.\n\n        For an end position, leave offset as zero (default):\n\n        >>> Position.fromstring("5")\n        ExactPosition(5)\n\n        For a start position, set offset to minus one (for Python counting):\n\n        >>> Position.fromstring("5", -1)\n        ExactPosition(4)\n\n        This also covers fuzzy positions:\n\n        >>> p = Position.fromstring("<5")\n        >>> p\n        BeforePosition(5)\n        >>> print(p)\n        <5\n        >>> int(p)\n        5\n\n        >>> Position.fromstring(">5")\n        AfterPosition(5)\n\n        By default assumes an end position, so note the integer behavior:\n\n        >>> p = Position.fromstring("one-of(5,8,11)")\n        >>> p\n        OneOfPosition(11, choices=[ExactPosition(5), ExactPosition(8), ExactPosition(11)])\n        >>> print(p)\n        one-of(5,8,11)\n        >>> int(p)\n        11\n\n        >>> Position.fromstring("(8.10)")\n        WithinPosition(10, left=8, right=10)\n\n        Fuzzy start positions:\n\n        >>> p = Position.fromstring("<5", -1)\n        >>> p\n        BeforePosition(4)\n        >>> print(p)\n        <4\n        >>> int(p)\n        4\n\n        Notice how the integer behavior changes too!\n\n        >>> p = Position.fromstring("one-of(5,8,11)", -1)\n        >>> p\n        OneOfPosition(4, choices=[ExactPosition(4), ExactPosition(7), ExactPosition(10)])\n        >>> print(p)\n        one-of(4,7,10)\n        >>> int(p)\n        4\n\n        '
        if offset != 0 and offset != -1:
            raise ValueError('To convert one-based indices to zero-based indices, offset must be either 0 (for end positions) or -1 (for start positions).')
        if text == '?':
            return UnknownPosition()
        if text.startswith('?'):
            return UncertainPosition(int(text[1:]) + offset)
        if text.startswith('<'):
            return BeforePosition(int(text[1:]) + offset)
        if text.startswith('>'):
            return AfterPosition(int(text[1:]) + offset)
        m = _re_within_position.match(text)
        if m is not None:
            (s, e) = m.groups()
            s = int(s) + offset
            e = int(e) + offset
            if offset == -1:
                default = s
            else:
                default = e
            return WithinPosition(default, left=s, right=e)
        m = _re_oneof_position.match(text)
        if m is not None:
            positions = m.groups()[0]
            parts = [ExactPosition(int(pos) + offset) for pos in positions.split(',')]
            if offset == -1:
                default = min((int(pos) for pos in parts))
            else:
                default = max((int(pos) for pos in parts))
            return OneOfPosition(default, choices=parts)
        return ExactPosition(int(text) + offset)

class ExactPosition(int, Position):
    """Specify the specific position of a boundary.

    Arguments:
     - position - The position of the boundary.
     - extension - An optional argument which must be zero since we don't
       have an extension. The argument is provided so that the same number
       of arguments can be passed to all position types.

    In this case, there is no fuzziness associated with the position.

    >>> p = ExactPosition(5)
    >>> p
    ExactPosition(5)
    >>> print(p)
    5

    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    Integer comparisons and operations should work as expected:

    >>> p == 5
    True
    >>> p < 6
    True
    >>> p <= 5
    True
    >>> p + 10
    ExactPosition(15)

    """

    def __new__(cls, position, extension=0):
        if False:
            print('Hello World!')
        'Create an ExactPosition object.'
        if extension != 0:
            raise AttributeError(f'Non-zero extension {extension} for exact position.')
        return int.__new__(cls, position)

    def __str__(self):
        if False:
            return 10
        'Return a representation of the ExactPosition object (with python counting).'
        return str(int(self))

    def __repr__(self):
        if False:
            print('Hello World!')
        'Represent the ExactPosition object as a string for debugging.'
        return '%s(%i)' % (self.__class__.__name__, int(self))

    def __add__(self, offset):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the position object with its location shifted (PRIVATE).'
        return self.__class__(int(self) + offset)

    def _flip(self, length):
        if False:
            i = 10
            return i + 15
        'Return a copy of the location after the parent is reversed (PRIVATE).'
        return self.__class__(length - int(self))

class UncertainPosition(ExactPosition):
    """Specify a specific position which is uncertain.

    This is used in UniProt, e.g. ?222 for uncertain position 222, or in the
    XML format explicitly marked as uncertain. Does not apply to GenBank/EMBL.
    """

class UnknownPosition(Position):
    """Specify a specific position which is unknown (has no position).

    This is used in UniProt, e.g. ? or in the XML as unknown.
    """

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Represent the UnknownPosition object as a string for debugging.'
        return f'{self.__class__.__name__}()'

    def __hash__(self):
        if False:
            print('Hello World!')
        'Return the hash value of the UnknownPosition object.'
        return hash(None)

    @property
    def position(self):
        if False:
            print('Hello World!')
        'Legacy attribute to get location (None) (DEPRECATED).\n\n        In general you can use the location directly as with the exception of\n        UnknownPosition it subclasses int, or use int(location), rather than\n        this location.position legacy attribute.\n\n        However, the UnknownPosition cannot be cast to an integer, and thus\n        does not subclass int, and int(...) will fail. The legacy attribute\n        would return None instead.\n\n        Note that while None == None, UnknownPosition() != UnknownPosition()\n        which is like the behavour for NaN.\n        '
        warnings.warn('Alias location.position is deprecated and will be removed in a future release. In general use position directly, but not note for UnknownPosition int(location) will fail. Use try/except or isinstance(location, UnknownPosition).', BiopythonDeprecationWarning)
        return None

    def __add__(self, offset):
        if False:
            while True:
                i = 10
        'Return a copy of the position object with its location shifted (PRIVATE).'
        return self

    def _flip(self, length):
        if False:
            return 10
        'Return a copy of the location after the parent is reversed (PRIVATE).'
        return self

class WithinPosition(int, Position):
    """Specify the position of a boundary within some coordinates.

    Arguments:
    - position - The default integer position
    - left - The start (left) position of the boundary
    - right - The end (right) position of the boundary

    This allows dealing with a location like ((11.14)..100). This
    indicates that the start of the sequence is somewhere between 11
    and 14. Since this is a start coordinate, it should act like
    it is at position 11 (or in Python counting, 10).

    >>> p = WithinPosition(10, 10, 13)
    >>> p
    WithinPosition(10, left=10, right=13)
    >>> print(p)
    (10.13)
    >>> int(p)
    10

    Basic integer comparisons and operations should work as though
    this were a plain integer:

    >>> p == 10
    True
    >>> p in [9, 10, 11]
    True
    >>> p < 11
    True
    >>> p + 10
    WithinPosition(20, left=20, right=23)

    >>> isinstance(p, WithinPosition)
    True
    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    Note this also applies for comparison to other position objects,
    where again the integer behavior is used:

    >>> p == 10
    True
    >>> p == ExactPosition(10)
    True
    >>> p == BeforePosition(10)
    True
    >>> p == AfterPosition(10)
    True

    If this were an end point, you would want the position to be 13
    (the right/larger value, not the left/smaller value as above):

    >>> p2 = WithinPosition(13, 10, 13)
    >>> p2
    WithinPosition(13, left=10, right=13)
    >>> print(p2)
    (10.13)
    >>> int(p2)
    13
    >>> p2 == 13
    True
    >>> p2 == ExactPosition(13)
    True

    """

    def __new__(cls, position, left, right):
        if False:
            i = 10
            return i + 15
        'Create a WithinPosition object.'
        if not (position == left or position == right):
            raise RuntimeError('WithinPosition: %r should match left %r or right %r' % (position, left, right))
        obj = int.__new__(cls, position)
        obj._left = left
        obj._right = right
        return obj

    def __getnewargs__(self):
        if False:
            print('Hello World!')
        'Return the arguments accepted by __new__.\n\n        Necessary to allow pickling and unpickling of class instances.\n        '
        return (int(self), self._left, self._right)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Represent the WithinPosition object as a string for debugging.'
        return '%s(%i, left=%i, right=%i)' % (self.__class__.__name__, int(self), self._left, self._right)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a representation of the WithinPosition object (with python counting).'
        return f'({self._left}.{self._right})'

    @property
    def position(self):
        if False:
            for i in range(10):
                print('nop')
        'Legacy attribute to get (left) position as integer (DEPRECATED).'
        warnings.warn('Alias location.position is deprecated and will be removed in a future release. Use location directly, or int(location) which will return the preferred location defined for WithinPosition (which may not be the left-most position).', BiopythonDeprecationWarning)
        return self._left

    @property
    def extension(self):
        if False:
            while True:
                i = 10
        "Legacy attribute to get the within-position's 'width' as an integer (DEPRECATED)."
        warnings.warn('Alias location.extension is deprecated and will be removed in a future release. This is usually zero, but there is no neat replacement for the WithinPosition object.', BiopythonDeprecationWarning)
        return self._right - self._left

    def __add__(self, offset):
        if False:
            while True:
                i = 10
        'Return a copy of the position object with its location shifted.'
        return self.__class__(int(self) + offset, self._left + offset, self._right + offset)

    def _flip(self, length):
        if False:
            while True:
                i = 10
        'Return a copy of the location after the parent is reversed (PRIVATE).'
        return self.__class__(length - int(self), length - self._right, length - self._left)

class BetweenPosition(int, Position):
    """Specify the position of a boundary between two coordinates (OBSOLETE?).

    Arguments:
     - position - The default integer position
     - left - The start (left) position of the boundary
     - right - The end (right) position of the boundary

    This allows dealing with a position like 123^456. This
    indicates that the start of the sequence is somewhere between
    123 and 456. It is up to the parser to set the position argument
    to either boundary point (depending on if this is being used as
    a start or end of the feature). For example as a feature end:

    >>> p = BetweenPosition(456, 123, 456)
    >>> p
    BetweenPosition(456, left=123, right=456)
    >>> print(p)
    (123^456)
    >>> int(p)
    456

    Integer equality and comparison use the given position,

    >>> p == 456
    True
    >>> p in [455, 456, 457]
    True
    >>> p > 300
    True

    The old legacy properties of position and extension give the
    starting/lower/left position as an integer, and the distance
    to the ending/higher/right position as an integer. Note that
    the position object will act like either the left or the right
    end-point depending on how it was created:

    >>> p2 = BetweenPosition(123, left=123, right=456)
    >>> int(p) == int(p2)
    False
    >>> p == 456
    True
    >>> p2 == 123
    True

    Note this potentially surprising behavior:

    >>> BetweenPosition(123, left=123, right=456) == ExactPosition(123)
    True
    >>> BetweenPosition(123, left=123, right=456) == BeforePosition(123)
    True
    >>> BetweenPosition(123, left=123, right=456) == AfterPosition(123)
    True

    i.e. For equality (and sorting) the position objects behave like
    integers.

    """

    def __new__(cls, position, left, right):
        if False:
            while True:
                i = 10
        'Create a new instance in BetweenPosition object.'
        assert position == left or position == right
        obj = int.__new__(cls, position)
        obj._left = left
        obj._right = right
        return obj

    def __getnewargs__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the arguments accepted by __new__.\n\n        Necessary to allow pickling and unpickling of class instances.\n        '
        return (int(self), self._left, self._right)

    def __repr__(self):
        if False:
            return 10
        'Represent the BetweenPosition object as a string for debugging.'
        return '%s(%i, left=%i, right=%i)' % (self.__class__.__name__, int(self), self._left, self._right)

    def __str__(self):
        if False:
            print('Hello World!')
        'Return a representation of the BetweenPosition object (with python counting).'
        return f'({self._left}^{self._right})'

    @property
    def position(self):
        if False:
            print('Hello World!')
        'Legacy attribute to get (left) position as integer (DEPRECATED).'
        warnings.warn('Alias location.position is deprecated and will be removed in a future release. Use location directly, or int(location) which will return the preferred location defined for a BetweenPosition (which may not be the left-most position).', BiopythonDeprecationWarning)
        return self._left

    @property
    def extension(self):
        if False:
            return 10
        "Legacy attribute to get the between-position's 'width' as an integer (DEPRECATED)."
        warnings.warn('Alias location.extension is deprecated and will be removed in a future release. This is usually zero, but there is no neat replacement for the BetweenPosition object.', BiopythonDeprecationWarning)
        return self._right - self._left

    def __add__(self, offset):
        if False:
            print('Hello World!')
        'Return a copy of the position object with its location shifted (PRIVATE).'
        return self.__class__(int(self) + offset, self._left + offset, self._right + offset)

    def _flip(self, length):
        if False:
            return 10
        'Return a copy of the location after the parent is reversed (PRIVATE).'
        return self.__class__(length - int(self), length - self._right, length - self._left)

class BeforePosition(int, Position):
    """Specify a position where the actual location occurs before it.

    Arguments:
     - position - The upper boundary of where the location can occur.
     - extension - An optional argument which must be zero since we don't
       have an extension. The argument is provided so that the same number
       of arguments can be passed to all position types.

    This is used to specify positions like (<10..100) where the location
    occurs somewhere before position 10.

    >>> p = BeforePosition(5)
    >>> p
    BeforePosition(5)
    >>> print(p)
    <5
    >>> int(p)
    5
    >>> p + 10
    BeforePosition(15)

    Note this potentially surprising behavior:

    >>> p == ExactPosition(5)
    True
    >>> p == AfterPosition(5)
    True

    Just remember that for equality and sorting the position objects act
    like integers.
    """

    def __new__(cls, position, extension=0):
        if False:
            print('Hello World!')
        'Create a new instance in BeforePosition object.'
        if extension != 0:
            raise AttributeError(f'Non-zero extension {extension} for exact position.')
        return int.__new__(cls, position)

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Represent the location as a string for debugging.'
        return '%s(%i)' % (self.__class__.__name__, int(self))

    def __str__(self):
        if False:
            print('Hello World!')
        'Return a representation of the BeforePosition object (with python counting).'
        return f'<{int(self)}'

    def __add__(self, offset):
        if False:
            while True:
                i = 10
        'Return a copy of the position object with its location shifted (PRIVATE).'
        return self.__class__(int(self) + offset)

    def _flip(self, length):
        if False:
            print('Hello World!')
        'Return a copy of the location after the parent is reversed (PRIVATE).'
        return AfterPosition(length - int(self))

class AfterPosition(int, Position):
    """Specify a position where the actual location is found after it.

    Arguments:
     - position - The lower boundary of where the location can occur.
     - extension - An optional argument which must be zero since we don't
       have an extension. The argument is provided so that the same number
       of arguments can be passed to all position types.

    This is used to specify positions like (>10..100) where the location
    occurs somewhere after position 10.

    >>> p = AfterPosition(7)
    >>> p
    AfterPosition(7)
    >>> print(p)
    >7
    >>> int(p)
    7
    >>> p + 10
    AfterPosition(17)

    >>> isinstance(p, AfterPosition)
    True
    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    Note this potentially surprising behavior:

    >>> p == ExactPosition(7)
    True
    >>> p == BeforePosition(7)
    True

    Just remember that for equality and sorting the position objects act
    like integers.
    """

    def __new__(cls, position, extension=0):
        if False:
            while True:
                i = 10
        'Create a new instance of the AfterPosition object.'
        if extension != 0:
            raise AttributeError(f'Non-zero extension {extension} for exact position.')
        return int.__new__(cls, position)

    def __repr__(self):
        if False:
            return 10
        'Represent the location as a string for debugging.'
        return '%s(%i)' % (self.__class__.__name__, int(self))

    def __str__(self):
        if False:
            while True:
                i = 10
        'Return a representation of the AfterPosition object (with python counting).'
        return f'>{int(self)}'

    def __add__(self, offset):
        if False:
            while True:
                i = 10
        'Return a copy of the position object with its location shifted (PRIVATE).'
        return self.__class__(int(self) + offset)

    def _flip(self, length):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the location after the parent is reversed (PRIVATE).'
        return BeforePosition(length - int(self))

class OneOfPosition(int, Position):
    """Specify a position where the location can be multiple positions.

    This models the GenBank 'one-of(1888,1901)' function, and tries
    to make this fit within the Biopython Position models. If this was
    a start position it should act like 1888, but as an end position 1901.

    >>> p = OneOfPosition(1888, [ExactPosition(1888), ExactPosition(1901)])
    >>> p
    OneOfPosition(1888, choices=[ExactPosition(1888), ExactPosition(1901)])
    >>> int(p)
    1888

    Integer comparisons and operators act like using int(p),

    >>> p == 1888
    True
    >>> p <= 1888
    True
    >>> p > 1888
    False
    >>> p + 100
    OneOfPosition(1988, choices=[ExactPosition(1988), ExactPosition(2001)])

    >>> isinstance(p, OneOfPosition)
    True
    >>> isinstance(p, Position)
    True
    >>> isinstance(p, int)
    True

    """

    def __new__(cls, position, choices):
        if False:
            while True:
                i = 10
        'Initialize with a set of possible positions.\n\n        choices is a list of Position derived objects, specifying possible\n        locations.\n\n        position is an integer specifying the default behavior.\n        '
        if position not in choices:
            raise ValueError(f'OneOfPosition: {position!r} should match one of {choices!r}')
        obj = int.__new__(cls, position)
        obj.position_choices = choices
        return obj

    def __getnewargs__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the arguments accepted by __new__.\n\n        Necessary to allow pickling and unpickling of class instances.\n        '
        return (int(self), self.position_choices)

    @property
    def position(self):
        if False:
            print('Hello World!')
        'Legacy attribute to get (left) position as integer (DEPRECATED).'
        warnings.warn('Alias location.position is deprecated and will be removed in a future release. Use location directly, or int(location) which will return the preferred location defined for a OneOfPosition (which may not be the left-most position), or min(location.position_choices) instead.', BiopythonDeprecationWarning)
        return min((int(pos) for pos in self.position_choices))

    @property
    def extension(self):
        if False:
            i = 10
            return i + 15
        "Legacy attribute to get the one-of-position's 'width' as an integer (DEPRECATED)."
        warnings.warn('Alias location.extension is deprecated and will be removed in a future release. This is usually zero, but for a OneOfPosition you can use max(position.position_choices) - min(position.position_choices)', BiopythonDeprecationWarning)
        positions = [int(pos) for pos in self.position_choices]
        return max(positions) - min(positions)

    def __repr__(self):
        if False:
            return 10
        'Represent the OneOfPosition object as a string for debugging.'
        return '%s(%i, choices=%r)' % (self.__class__.__name__, int(self), self.position_choices)

    def __str__(self):
        if False:
            print('Hello World!')
        'Return a representation of the OneOfPosition object (with python counting).'
        out = 'one-of('
        for position in self.position_choices:
            out += f'{position},'
        return out[:-1] + ')'

    def __add__(self, offset):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the position object with its location shifted (PRIVATE).'
        return self.__class__(int(self) + offset, [p + offset for p in self.position_choices])

    def _flip(self, length):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy of the location after the parent is reversed (PRIVATE).'
        return self.__class__(length - int(self), [p._flip(length) for p in self.position_choices[::-1]])

class PositionGap:
    """Simple class to hold information about a gap between positions (DEPRECATED)."""

    def __init__(self, gap_size):
        if False:
            print('Hello World!')
        'Initialize with a position object containing the gap information.'
        self.gap_size = gap_size
        warnings.warn('The PositionGap class is deprecated and will be removed in a future release. It has not been used in Biopython for over ten years.', BiopythonDeprecationWarning)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Represent the position gap as a string for debugging.'
        return f'{self.__class__.__name__}({self.gap_size!r})'

    def __str__(self):
        if False:
            print('Hello World!')
        'Return a representation of the PositionGap object (with python counting).'
        return f'gap({self.gap_size})'
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()