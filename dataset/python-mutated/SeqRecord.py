"""Represent a Sequence Record, a sequence with annotation."""
from io import StringIO
import numbers
from typing import cast, overload, Any, Dict, Iterable, List, Mapping, NoReturn, Optional, Sequence, Union, TYPE_CHECKING
from Bio import StreamModeError
from Bio.Seq import UndefinedSequenceError
if TYPE_CHECKING:
    from Bio.Seq import Seq, MutableSeq
    from Bio.SeqFeature import SeqFeature
_NO_SEQRECORD_COMPARISON = 'SeqRecord comparison is deliberately not implemented. Explicitly compare the attributes of interest.'

class _RestrictedDict(Dict[str, Sequence[Any]]):
    """Dict which only allows sequences of given length as values (PRIVATE).

    This simple subclass of the Python dictionary is used in the SeqRecord
    object for holding per-letter-annotations.  This class is intended to
    prevent simple errors by only allowing python sequences (e.g. lists,
    strings and tuples) to be stored, and only if their length matches that
    expected (the length of the SeqRecord's seq object).  It cannot however
    prevent the entries being edited in situ (for example appending entries
    to a list).

    >>> x = _RestrictedDict(5)
    >>> x["test"] = "hello"
    >>> x
    {'test': 'hello'}

    Adding entries which don't have the expected length are blocked:

    >>> x["test"] = "hello world"
    Traceback (most recent call last):
    ...
    TypeError: Any per-letter annotation should be a Python sequence (list, tuple or string) of the same length as the biological sequence, here 5.

    The expected length is stored as a private attribute,

    >>> x._length
    5

    In order that the SeqRecord (and other objects using this class) can be
    pickled, for example for use in the multiprocessing library, we need to
    be able to pickle the restricted dictionary objects.

    Using the default protocol, which is 3 on Python 3,

    >>> import pickle
    >>> y = pickle.loads(pickle.dumps(x))
    >>> y
    {'test': 'hello'}
    >>> y._length
    5

    Using the highest protocol, which is 4 on Python 3,

    >>> import pickle
    >>> z = pickle.loads(pickle.dumps(x, pickle.HIGHEST_PROTOCOL))
    >>> z
    {'test': 'hello'}
    >>> z._length
    5
    """

    def __init__(self, length: int) -> None:
        if False:
            print('Hello World!')
        'Create an EMPTY restricted dictionary.'
        dict.__init__(self)
        self._length = int(length)

    def __setitem__(self, key: str, value: Sequence[Any]) -> None:
        if False:
            while True:
                i = 10
        if not hasattr(value, '__len__') or not hasattr(value, '__getitem__') or (hasattr(self, '_length') and len(value) != self._length):
            raise TypeError(f'Any per-letter annotation should be a Python sequence (list, tuple or string) of the same length as the biological sequence, here {self._length}.')
        dict.__setitem__(self, key, value)

    def update(self, new_dict):
        if False:
            for i in range(10):
                print('nop')
        for (key, value) in new_dict.items():
            self[key] = value

class SeqRecord:
    """A SeqRecord object holds a sequence and information about it.

    Main attributes:
     - id          - Identifier such as a locus tag (string)
     - seq         - The sequence itself (Seq object or similar)

    Additional attributes:
     - name        - Sequence name, e.g. gene name (string)
     - description - Additional text (string)
     - dbxrefs     - List of database cross references (list of strings)
     - features    - Any (sub)features defined (list of SeqFeature objects)
     - annotations - Further information about the whole sequence (dictionary).
       Most entries are strings, or lists of strings.
     - letter_annotations - Per letter/symbol annotation (restricted
       dictionary). This holds Python sequences (lists, strings
       or tuples) whose length matches that of the sequence.
       A typical use would be to hold a list of integers
       representing sequencing quality scores, or a string
       representing the secondary structure.

    You will typically use Bio.SeqIO to read in sequences from files as
    SeqRecord objects.  However, you may want to create your own SeqRecord
    objects directly (see the __init__ method for further details):

    >>> from Bio.Seq import Seq
    >>> from Bio.SeqRecord import SeqRecord
    >>> record = SeqRecord(Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"),
    ...                    id="YP_025292.1", name="HokC",
    ...                    description="toxic membrane protein")
    >>> print(record)
    ID: YP_025292.1
    Name: HokC
    Description: toxic membrane protein
    Number of features: 0
    Seq('MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF')

    If you want to save SeqRecord objects to a sequence file, use Bio.SeqIO
    for this.  For the special case where you want the SeqRecord turned into
    a string in a particular file format there is a format method which uses
    Bio.SeqIO internally:

    >>> print(record.format("fasta"))
    >YP_025292.1 toxic membrane protein
    MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF
    <BLANKLINE>

    You can also do things like slicing a SeqRecord, checking its length, etc

    >>> len(record)
    44
    >>> edited = record[:10] + record[11:]
    >>> print(edited.seq)
    MKQHKAMIVAIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF
    >>> print(record.seq)
    MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF

    """
    _AnnotationsDictValue = Union[str, int]
    _AnnotationsDict = Dict[str, _AnnotationsDictValue]
    annotations: _AnnotationsDict
    dbxrefs: List[str]

    def __init__(self, seq: Optional[Union['Seq', 'MutableSeq', str]], id: Optional[str]='<unknown id>', name: str='<unknown name>', description: str='<unknown description>', dbxrefs: Optional[List[str]]=None, features: Optional[List['SeqFeature']]=None, annotations: Optional[_AnnotationsDict]=None, letter_annotations: Optional[Dict[str, Sequence[Any]]]=None) -> None:
        if False:
            i = 10
            return i + 15
        "Create a SeqRecord.\n\n        Arguments:\n         - seq         - Sequence, required (Seq or MutableSeq)\n         - id          - Sequence identifier, recommended (string)\n         - name        - Sequence name, optional (string)\n         - description - Sequence description, optional (string)\n         - dbxrefs     - Database cross references, optional (list of strings)\n         - features    - Any (sub)features, optional (list of SeqFeature objects)\n         - annotations - Dictionary of annotations for the whole sequence\n         - letter_annotations - Dictionary of per-letter-annotations, values\n           should be strings, list or tuples of the same length as the full\n           sequence.\n\n        You will typically use Bio.SeqIO to read in sequences from files as\n        SeqRecord objects.  However, you may want to create your own SeqRecord\n        objects directly.\n\n        Note that while an id is optional, we strongly recommend you supply a\n        unique id string for each record.  This is especially important\n        if you wish to write your sequences to a file.\n\n        You can create a 'blank' SeqRecord object, and then populate the\n        attributes later.\n        "
        if id is not None and (not isinstance(id, str)):
            raise TypeError('id argument should be a string')
        if not isinstance(name, str):
            raise TypeError('name argument should be a string')
        if not isinstance(description, str):
            raise TypeError('description argument should be a string')
        self._seq = seq
        self.id = id
        self.name = name
        self.description = description
        if dbxrefs is None:
            dbxrefs = []
        elif not isinstance(dbxrefs, list):
            raise TypeError('dbxrefs argument should be a list (of strings)')
        self.dbxrefs = dbxrefs
        if annotations is None:
            annotations = {}
        elif not isinstance(annotations, dict):
            raise TypeError('annotations argument must be a dict or None')
        self.annotations = annotations
        if letter_annotations is None:
            if seq is None:
                self._per_letter_annotations: _RestrictedDict = _RestrictedDict(length=0)
            else:
                try:
                    self._per_letter_annotations = _RestrictedDict(length=len(seq))
                except TypeError:
                    raise TypeError('seq argument should be a Seq object or similar') from None
        else:
            self.letter_annotations = letter_annotations
        if features is None:
            features = []
        elif not isinstance(features, list):
            raise TypeError('features argument should be a list (of SeqFeature objects)')
        self.features = features

    def _set_per_letter_annotations(self, value: Mapping[str, str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(value, dict):
            raise TypeError('The per-letter-annotations should be a (restricted) dictionary.')
        try:
            self._per_letter_annotations = _RestrictedDict(length=len(self.seq))
        except AttributeError:
            self._per_letter_annotations = _RestrictedDict(length=0)
        self._per_letter_annotations.update(value)
    letter_annotations = property(fget=lambda self: self._per_letter_annotations, fset=_set_per_letter_annotations, doc='Dictionary of per-letter-annotation for the sequence.\n\n        For example, this can hold quality scores used in FASTQ or QUAL files.\n        Consider this example using Bio.SeqIO to read in an example Solexa\n        variant FASTQ file as a SeqRecord:\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Quality/solexa_faked.fastq", "fastq-solexa")\n        >>> print("%s %s" % (record.id, record.seq))\n        slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n        >>> print(list(record.letter_annotations))\n        [\'solexa_quality\']\n        >>> print(record.letter_annotations["solexa_quality"])\n        [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]\n\n        The letter_annotations get sliced automatically if you slice the\n        parent SeqRecord, for example taking the last ten bases:\n\n        >>> sub_record = record[-10:]\n        >>> print("%s %s" % (sub_record.id, sub_record.seq))\n        slxa_0001_1_0001_01 ACGTNNNNNN\n        >>> print(sub_record.letter_annotations["solexa_quality"])\n        [4, 3, 2, 1, 0, -1, -2, -3, -4, -5]\n\n        Any python sequence (i.e. list, tuple or string) can be recorded in\n        the SeqRecord\'s letter_annotations dictionary as long as the length\n        matches that of the SeqRecord\'s sequence.  e.g.\n\n        >>> len(sub_record.letter_annotations)\n        1\n        >>> sub_record.letter_annotations["dummy"] = "abcdefghij"\n        >>> len(sub_record.letter_annotations)\n        2\n\n        You can delete entries from the letter_annotations dictionary as usual:\n\n        >>> del sub_record.letter_annotations["solexa_quality"]\n        >>> sub_record.letter_annotations\n        {\'dummy\': \'abcdefghij\'}\n\n        You can completely clear the dictionary easily as follows:\n\n        >>> sub_record.letter_annotations = {}\n        >>> sub_record.letter_annotations\n        {}\n\n        Note that if replacing the record\'s sequence with a sequence of a\n        different length you must first clear the letter_annotations dict.\n        ')

    def _set_seq(self, value: Union['Seq', 'MutableSeq']) -> None:
        if False:
            return 10
        if self._per_letter_annotations:
            if len(self) != len(value):
                raise ValueError('You must empty the letter annotations first!')
            else:
                self._seq = value
        else:
            self._seq = value
            try:
                self._per_letter_annotations = _RestrictedDict(length=len(self.seq))
            except AttributeError:
                self._per_letter_annotations = _RestrictedDict(length=0)
    seq = property(fget=lambda self: self._seq, fset=_set_seq, doc='The sequence itself, as a Seq or MutableSeq object.')

    @overload
    def __getitem__(self, index: int) -> str:
        if False:
            return 10
        ...

    @overload
    def __getitem__(self, index: slice) -> 'SeqRecord':
        if False:
            for i in range(10):
                print('nop')
        ...

    def __getitem__(self, index):
        if False:
            print('Hello World!')
        'Return a sub-sequence or an individual letter.\n\n        Slicing, e.g. my_record[5:10], returns a new SeqRecord for\n        that sub-sequence with some annotation preserved as follows:\n\n        * The name, id and description are kept as-is.\n        * Any per-letter-annotations are sliced to match the requested\n          sub-sequence.\n        * Unless a stride is used, all those features which fall fully\n          within the subsequence are included (with their locations\n          adjusted accordingly). If you want to preserve any truncated\n          features (e.g. GenBank/EMBL source features), you must\n          explicitly add them to the new SeqRecord yourself.\n        * With the exception of any molecule type, the annotations\n          dictionary and the dbxrefs list are not used for the new\n          SeqRecord, as in general they may not apply to the\n          subsequence. If you want to preserve them, you must explicitly\n          copy them to the new SeqRecord yourself.\n\n        Using an integer index, e.g. my_record[5] is shorthand for\n        extracting that letter from the sequence, my_record.seq[5].\n\n        For example, consider this short protein and its secondary\n        structure as encoded by the PDB (e.g. H for alpha helices),\n        plus a simple feature for its histidine self phosphorylation\n        site:\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> from Bio.SeqFeature import SeqFeature, SimpleLocation\n        >>> rec = SeqRecord(Seq("MAAGVKQLADDRTLLMAGVSHDLRTPLTRIRLAT"\n        ...                     "EMMSEQDGYLAESINKDIEECNAIIEQFIDYLR"),\n        ...                 id="1JOY", name="EnvZ",\n        ...                 description="Homodimeric domain of EnvZ from E. coli")\n        >>> rec.letter_annotations["secondary_structure"] = "  S  SSSSSSHHHHHTTTHHHHHHHHHHHHHHHHHHHHHHTHHHHHHHHHHHHHHHHHHHHHTT  "\n        >>> rec.features.append(SeqFeature(SimpleLocation(20, 21),\n        ...                     type = "Site"))\n\n        Now let\'s have a quick look at the full record,\n\n        >>> print(rec)\n        ID: 1JOY\n        Name: EnvZ\n        Description: Homodimeric domain of EnvZ from E. coli\n        Number of features: 1\n        Per letter annotation for: secondary_structure\n        Seq(\'MAAGVKQLADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEE...YLR\')\n        >>> rec.letter_annotations["secondary_structure"]\n        \'  S  SSSSSSHHHHHTTTHHHHHHHHHHHHHHHHHHHHHHTHHHHHHHHHHHHHHHHHHHHHTT  \'\n        >>> print(rec.features[0].location)\n        [20:21]\n\n        Now let\'s take a sub sequence, here chosen as the first (fractured)\n        alpha helix which includes the histidine phosphorylation site:\n\n        >>> sub = rec[11:41]\n        >>> print(sub)\n        ID: 1JOY\n        Name: EnvZ\n        Description: Homodimeric domain of EnvZ from E. coli\n        Number of features: 1\n        Per letter annotation for: secondary_structure\n        Seq(\'RTLLMAGVSHDLRTPLTRIRLATEMMSEQD\')\n        >>> sub.letter_annotations["secondary_structure"]\n        \'HHHHHTTTHHHHHHHHHHHHHHHHHHHHHH\'\n        >>> print(sub.features[0].location)\n        [9:10]\n\n        You can also of course omit the start or end values, for\n        example to get the first ten letters only:\n\n        >>> print(rec[:10])\n        ID: 1JOY\n        Name: EnvZ\n        Description: Homodimeric domain of EnvZ from E. coli\n        Number of features: 0\n        Per letter annotation for: secondary_structure\n        Seq(\'MAAGVKQLAD\')\n\n        Or for the last ten letters:\n\n        >>> print(rec[-10:])\n        ID: 1JOY\n        Name: EnvZ\n        Description: Homodimeric domain of EnvZ from E. coli\n        Number of features: 0\n        Per letter annotation for: secondary_structure\n        Seq(\'IIEQFIDYLR\')\n\n        If you omit both, then you get a copy of the original record (although\n        lacking the annotations and dbxrefs):\n\n        >>> print(rec[:])\n        ID: 1JOY\n        Name: EnvZ\n        Description: Homodimeric domain of EnvZ from E. coli\n        Number of features: 1\n        Per letter annotation for: secondary_structure\n        Seq(\'MAAGVKQLADDRTLLMAGVSHDLRTPLTRIRLATEMMSEQDGYLAESINKDIEE...YLR\')\n\n        Finally, indexing with a simple integer is shorthand for pulling out\n        that letter from the sequence directly:\n\n        >>> rec[5]\n        \'K\'\n        >>> rec.seq[5]\n        \'K\'\n        '
        if isinstance(index, numbers.Integral):
            return self.seq[index]
        elif isinstance(index, slice):
            if self.seq is None:
                raise ValueError('If the sequence is None, we cannot slice it.')
            parent_length = len(self)
            try:
                from BioSQL.BioSeq import DBSeqRecord
                biosql_available = True
            except ImportError:
                biosql_available = False
            if biosql_available and isinstance(self, DBSeqRecord):
                answer = SeqRecord(self.seq[index], id=self.id, name=self.name, description=self.description)
            else:
                answer = self.__class__(self.seq[index], id=self.id, name=self.name, description=self.description)
            if 'molecule_type' in self.annotations:
                answer.annotations['molecule_type'] = self.annotations['molecule_type']
            (start, stop, step) = index.indices(parent_length)
            if step == 1:
                for f in self.features:
                    if f.ref or f.ref_db:
                        import warnings
                        warnings.warn('When slicing SeqRecord objects, any SeqFeature referencing other sequences (e.g. from segmented GenBank records) are ignored.')
                        continue
                    try:
                        if start <= f.location.start and f.location.end <= stop:
                            answer.features.append(f._shift(-start))
                    except TypeError:
                        pass
            for (key, value) in self.letter_annotations.items():
                answer._per_letter_annotations[key] = value[index]
            return answer
        raise ValueError('Invalid index')

    def __iter__(self) -> Iterable[Union['Seq', 'MutableSeq']]:
        if False:
            while True:
                i = 10
        'Iterate over the letters in the sequence.\n\n        For example, using Bio.SeqIO to read in a protein FASTA file:\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Fasta/loveliesbleeding.pro", "fasta")\n        >>> for amino in record:\n        ...     print(amino)\n        ...     if amino == "L": break\n        X\n        A\n        G\n        L\n        >>> print(record.seq[3])\n        L\n\n        This is just a shortcut for iterating over the sequence directly:\n\n        >>> for amino in record.seq:\n        ...     print(amino)\n        ...     if amino == "L": break\n        X\n        A\n        G\n        L\n        >>> print(record.seq[3])\n        L\n\n        Note that this does not facilitate iteration together with any\n        per-letter-annotation.  However, you can achieve that using the\n        python zip function on the record (or its sequence) and the relevant\n        per-letter-annotation:\n\n        >>> from Bio import SeqIO\n        >>> rec = SeqIO.read("Quality/solexa_faked.fastq", "fastq-solexa")\n        >>> print("%s %s" % (rec.id, rec.seq))\n        slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n        >>> print(list(rec.letter_annotations))\n        [\'solexa_quality\']\n        >>> for nuc, qual in zip(rec, rec.letter_annotations["solexa_quality"]):\n        ...     if qual > 35:\n        ...         print("%s %i" % (nuc, qual))\n        A 40\n        C 39\n        G 38\n        T 37\n        A 36\n\n        You may agree that using zip(rec.seq, ...) is more explicit than using\n        zip(rec, ...) as shown above.\n        '
        return iter(self.seq)

    def __contains__(self, char: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Implement the \'in\' keyword, searches the sequence.\n\n        e.g.\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Fasta/sweetpea.nu", "fasta")\n        >>> "GAATTC" in record\n        False\n        >>> "AAA" in record\n        True\n\n        This essentially acts as a proxy for using "in" on the sequence:\n\n        >>> "GAATTC" in record.seq\n        False\n        >>> "AAA" in record.seq\n        True\n\n        Note that you can also use Seq objects as the query,\n\n        >>> from Bio.Seq import Seq\n        >>> Seq("AAA") in record\n        True\n\n        See also the Seq object\'s __contains__ method.\n        '
        return char in self.seq

    def __bytes__(self) -> bytes:
        if False:
            i = 10
            return i + 15
        return bytes(self.seq)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        'Return a human readable summary of the record and its annotation (string).\n\n        The python built in function str works by calling the object\'s __str__\n        method.  e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> record = SeqRecord(Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"),\n        ...                    id="YP_025292.1", name="HokC",\n        ...                    description="toxic membrane protein, small")\n        >>> print(str(record))\n        ID: YP_025292.1\n        Name: HokC\n        Description: toxic membrane protein, small\n        Number of features: 0\n        Seq(\'MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\')\n\n        In this example you don\'t actually need to call str explicitly, as the\n        print command does this automatically:\n\n        >>> print(record)\n        ID: YP_025292.1\n        Name: HokC\n        Description: toxic membrane protein, small\n        Number of features: 0\n        Seq(\'MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\')\n\n        Note that long sequences are shown truncated.\n        '
        lines: List[str] = []
        if self.id:
            lines.append(f'ID: {self.id}')
        if self.name:
            lines.append(f'Name: {self.name}')
        if self.description:
            lines.append(f'Description: {self.description}')
        if self.dbxrefs:
            lines.append('Database cross-references: ' + ', '.join(self.dbxrefs))
        lines.append(f'Number of features: {len(self.features)}')
        for a in self.annotations:
            lines.append(f'/{a}={self.annotations[a]!s}')
        if self.letter_annotations:
            lines.append('Per letter annotation for: ' + ', '.join(self.letter_annotations))
        try:
            bytes(self.seq)
        except UndefinedSequenceError:
            lines.append(f'Undefined sequence of length {len(self.seq)}')
        else:
            seq = repr(self.seq)
            lines.append(seq)
        return '\n'.join(lines)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return a concise summary of the record for debugging (string).\n\n        The python built in function repr works by calling the object\'s __repr__\n        method.  e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> rec = SeqRecord(Seq("MASRGVNKVILVGNLGQDPEVRYMPNGGAVANITLATSESWRDKAT"\n        ...                     "GEMKEQTEWHRVVLFGKLAEVASEYLRKGSQVYIEGQLRTRKWTDQ"\n        ...                     "SGQDRYTTEVVVNVGGTMQMLGGRQGGGAPAGGNIGGGQPQGGWGQ"\n        ...                     "PQQPQGGNQFSGGAQSRPQQSAPAAPSNEPPMDFDDDIPF"),\n        ...                 id="NP_418483.1", name="b4059",\n        ...                 description="ssDNA-binding protein",\n        ...                 dbxrefs=["ASAP:13298", "GI:16131885", "GeneID:948570"])\n        >>> print(repr(rec))\n        SeqRecord(seq=Seq(\'MASRGVNKVILVGNLGQDPEVRYMPNGGAVANITLATSESWRDKATGEMKEQTE...IPF\'), id=\'NP_418483.1\', name=\'b4059\', description=\'ssDNA-binding protein\', dbxrefs=[\'ASAP:13298\', \'GI:16131885\', \'GeneID:948570\'])\n\n        At the python prompt you can also use this shorthand:\n\n        >>> rec\n        SeqRecord(seq=Seq(\'MASRGVNKVILVGNLGQDPEVRYMPNGGAVANITLATSESWRDKATGEMKEQTE...IPF\'), id=\'NP_418483.1\', name=\'b4059\', description=\'ssDNA-binding protein\', dbxrefs=[\'ASAP:13298\', \'GI:16131885\', \'GeneID:948570\'])\n\n        Note that long sequences are shown truncated. Also note that any\n        annotations, letter_annotations and features are not shown (as they\n        would lead to a very long string).\n        '
        return f'{self.__class__.__name__}(seq={self.seq!r}, id={self.id!r}, name={self.name!r}, description={self.description!r}, dbxrefs={self.dbxrefs!r})'

    def format(self, format: str) -> str:
        if False:
            return 10
        'Return the record as a string in the specified file format.\n\n        The format should be a lower case string supported as an output\n        format by Bio.SeqIO, which is used to turn the SeqRecord into a\n        string.  e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> record = SeqRecord(Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"),\n        ...                    id="YP_025292.1", name="HokC",\n        ...                    description="toxic membrane protein")\n        >>> record.format("fasta")\n        \'>YP_025292.1 toxic membrane protein\\nMKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\\n\'\n        >>> print(record.format("fasta"))\n        >YP_025292.1 toxic membrane protein\n        MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\n        <BLANKLINE>\n\n        The Python print function automatically appends a new line, meaning\n        in this example a blank line is shown.  If you look at the string\n        representation you can see there is a trailing new line (shown as\n        slash n) which is important when writing to a file or if\n        concatenating multiple sequence strings together.\n\n        Note that this method will NOT work on every possible file format\n        supported by Bio.SeqIO (e.g. some are for multiple sequences only,\n        and binary formats are not supported).\n        '
        return self.__format__(format)

    def __format__(self, format_spec: str) -> str:
        if False:
            i = 10
            return i + 15
        'Return the record as a string in the specified file format.\n\n        This method supports the Python format() function and f-strings.\n        The format_spec should be a lower case string supported by\n        Bio.SeqIO as a text output file format. Requesting a binary file\n        format raises a ValueError. e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> record = SeqRecord(Seq("MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF"),\n        ...                    id="YP_025292.1", name="HokC",\n        ...                    description="toxic membrane protein")\n        ...\n        >>> format(record, "fasta")\n        \'>YP_025292.1 toxic membrane protein\\nMKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\\n\'\n        >>> print(f"Here is {record.id} in FASTA format:\\n{record:fasta}")\n        Here is YP_025292.1 in FASTA format:\n        >YP_025292.1 toxic membrane protein\n        MKQHKAMIVALIVICITAVVAALVTRKDLCEVHIRTGQTEVAVF\n        <BLANKLINE>\n\n        See also the SeqRecord\'s format() method.\n        '
        if not format_spec:
            return str(self)
        from Bio import SeqIO
        if format_spec in SeqIO._FormatToString:
            return SeqIO._FormatToString[format_spec](self)
        handle = StringIO()
        try:
            SeqIO.write(self, handle, format_spec)
        except StreamModeError:
            raise ValueError('Binary format %s cannot be used with SeqRecord format method' % format_spec) from None
        return handle.getvalue()

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the length of the sequence.\n\n        For example, using Bio.SeqIO to read in a FASTA nucleotide file:\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Fasta/sweetpea.nu", "fasta")\n        >>> len(record)\n        309\n        >>> len(record.seq)\n        309\n        '
        return len(self.seq)

    def __lt__(self, other: Any) -> NoReturn:
        if False:
            print('Hello World!')
        'Define the less-than operand (not implemented).'
        raise NotImplementedError(_NO_SEQRECORD_COMPARISON)

    def __le__(self, other: Any) -> NoReturn:
        if False:
            while True:
                i = 10
        'Define the less-than-or-equal-to operand (not implemented).'
        raise NotImplementedError(_NO_SEQRECORD_COMPARISON)

    def __eq__(self, other: object) -> NoReturn:
        if False:
            i = 10
            return i + 15
        'Define the equal-to operand (not implemented).'
        raise NotImplementedError(_NO_SEQRECORD_COMPARISON)

    def __ne__(self, other: object) -> NoReturn:
        if False:
            for i in range(10):
                print('nop')
        'Define the not-equal-to operand (not implemented).'
        raise NotImplementedError(_NO_SEQRECORD_COMPARISON)

    def __gt__(self, other: Any) -> NoReturn:
        if False:
            print('Hello World!')
        'Define the greater-than operand (not implemented).'
        raise NotImplementedError(_NO_SEQRECORD_COMPARISON)

    def __ge__(self, other: Any) -> NoReturn:
        if False:
            return 10
        'Define the greater-than-or-equal-to operand (not implemented).'
        raise NotImplementedError(_NO_SEQRECORD_COMPARISON)

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        'Boolean value of an instance of this class (True).\n\n        This behaviour is for backwards compatibility, since until the\n        __len__ method was added, a SeqRecord always evaluated as True.\n\n        Note that in comparison, a Seq object will evaluate to False if it\n        has a zero length sequence.\n\n        WARNING: The SeqRecord may in future evaluate to False when its\n        sequence is of zero length (in order to better match the Seq\n        object behaviour)!\n        '
        return True

    def __add__(self, other: Union['SeqRecord', 'Seq', 'MutableSeq', str]) -> 'SeqRecord':
        if False:
            while True:
                i = 10
        'Add another sequence or string to this sequence.\n\n        The other sequence can be a SeqRecord object, a Seq object (or\n        similar, e.g. a MutableSeq) or a plain Python string. If you add\n        a plain string or a Seq (like) object, the new SeqRecord will simply\n        have this appended to the existing data. However, any per letter\n        annotation will be lost:\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Quality/solexa_faked.fastq", "fastq-solexa")\n        >>> print("%s %s" % (record.id, record.seq))\n        slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n        >>> print(list(record.letter_annotations))\n        [\'solexa_quality\']\n\n        >>> new = record + "ACT"\n        >>> print("%s %s" % (new.id, new.seq))\n        slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNNACT\n        >>> print(list(new.letter_annotations))\n        []\n\n        The new record will attempt to combine the annotation, but for any\n        ambiguities (e.g. different names) it defaults to omitting that\n        annotation.\n\n        >>> from Bio import SeqIO\n        >>> with open("GenBank/pBAD30.gb") as handle:\n        ...     plasmid = SeqIO.read(handle, "gb")\n        >>> print("%s %i" % (plasmid.id, len(plasmid)))\n        pBAD30 4923\n\n        Now let\'s cut the plasmid into two pieces, and join them back up the\n        other way round (i.e. shift the starting point on this plasmid, have\n        a look at the annotated features in the original file to see why this\n        particular split point might make sense):\n\n        >>> left = plasmid[:3765]\n        >>> right = plasmid[3765:]\n        >>> new = right + left\n        >>> print("%s %i" % (new.id, len(new)))\n        pBAD30 4923\n        >>> str(new.seq) == str(right.seq + left.seq)\n        True\n        >>> len(new.features) == len(left.features) + len(right.features)\n        True\n\n        When we add the left and right SeqRecord objects, their annotation\n        is all consistent, so it is all conserved in the new SeqRecord:\n\n        >>> new.id == left.id == right.id == plasmid.id\n        True\n        >>> new.name == left.name == right.name == plasmid.name\n        True\n        >>> new.description == plasmid.description\n        True\n        >>> new.annotations == left.annotations == right.annotations\n        True\n        >>> new.letter_annotations == plasmid.letter_annotations\n        True\n        >>> new.dbxrefs == left.dbxrefs == right.dbxrefs\n        True\n\n        However, we should point out that when we sliced the SeqRecord,\n        any annotations dictionary or dbxrefs list entries were lost.\n        You can explicitly copy them like this:\n\n        >>> new.annotations = plasmid.annotations.copy()\n        >>> new.dbxrefs = plasmid.dbxrefs[:]\n        '
        if not isinstance(other, SeqRecord):
            return type(self)(self.seq + other, id=self.id, name=self.name, description=self.description, features=self.features[:], annotations=self.annotations.copy(), dbxrefs=self.dbxrefs[:])
        answer = type(self)(self.seq + other.seq, features=self.features[:], dbxrefs=self.dbxrefs[:])
        length = len(self)
        for f in other.features:
            answer.features.append(f._shift(length))
        del length
        for ref in other.dbxrefs:
            if ref not in answer.dbxrefs:
                answer.dbxrefs.append(ref)
        if self.id == other.id:
            answer.id = self.id
        if self.name == other.name:
            answer.name = self.name
        if self.description == other.description:
            answer.description = self.description
        for (k, v) in self.annotations.items():
            if k in other.annotations and other.annotations[k] == v:
                answer.annotations[k] = v
        for (k, v) in self.letter_annotations.items():
            if k in other.letter_annotations:
                answer.letter_annotations[k] = v + other.letter_annotations[k]
        return answer

    def __radd__(self, other: Union['Seq', 'MutableSeq', str]) -> 'SeqRecord':
        if False:
            for i in range(10):
                print('nop')
        'Add another sequence or string to this sequence (from the left).\n\n        This method handles adding a Seq object (or similar, e.g. MutableSeq)\n        or a plain Python string (on the left) to a SeqRecord (on the right).\n        See the __add__ method for more details, but for example:\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Quality/solexa_faked.fastq", "fastq-solexa")\n        >>> print("%s %s" % (record.id, record.seq))\n        slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n        >>> print(list(record.letter_annotations))\n        [\'solexa_quality\']\n\n        >>> new = "ACT" + record\n        >>> print("%s %s" % (new.id, new.seq))\n        slxa_0001_1_0001_01 ACTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n        >>> print(list(new.letter_annotations))\n        []\n        '
        if isinstance(other, SeqRecord):
            raise RuntimeError('This should have happened via the __add__ of the other SeqRecord being added!')
        offset = len(other)
        return type(self)(other + self.seq, id=self.id, name=self.name, description=self.description, features=[f._shift(offset) for f in self.features], annotations=self.annotations.copy(), dbxrefs=self.dbxrefs[:])

    def count(self, sub, start=None, end=None):
        if False:
            print('Hello World!')
        'Return the number of non-overlapping occurrences of sub in seq[start:end].\n\n        Optional arguments start and end are interpreted as in slice notation.\n        This method behaves as the count method of Python strings.\n        '
        return self.seq.count(sub, start, end)

    def upper(self) -> 'SeqRecord':
        if False:
            while True:
                i = 10
        'Return a copy of the record with an upper case sequence.\n\n        All the annotation is preserved unchanged. e.g.\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> record = SeqRecord(Seq("acgtACGT"), id="Test",\n        ...                    description = "Made up for this example")\n        >>> record.letter_annotations["phred_quality"] = [1, 2, 3, 4, 5, 6, 7, 8]\n        >>> print(record.upper().format("fastq"))\n        @Test Made up for this example\n        ACGTACGT\n        +\n        "#$%&\'()\n        <BLANKLINE>\n\n        Naturally, there is a matching lower method:\n\n        >>> print(record.lower().format("fastq"))\n        @Test Made up for this example\n        acgtacgt\n        +\n        "#$%&\'()\n        <BLANKLINE>\n        '
        return type(self)(self.seq.upper(), id=self.id, name=self.name, description=self.description, dbxrefs=self.dbxrefs[:], features=self.features[:], annotations=self.annotations.copy(), letter_annotations=self.letter_annotations.copy())

    def lower(self) -> 'SeqRecord':
        if False:
            print('Hello World!')
        'Return a copy of the record with a lower case sequence.\n\n        All the annotation is preserved unchanged. e.g.\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Fasta/aster.pro", "fasta")\n        >>> print(record.format("fasta"))\n        >gi|3298468|dbj|BAA31520.1| SAMIPF\n        GGHVNPAVTFGAFVGGNITLLRGIVYIIAQLLGSTVACLLLKFVTNDMAVGVFSLSAGVG\n        VTNALVFEIVMTFGLVYTVYATAIDPKKGSLGTIAPIAIGFIVGANI\n        <BLANKLINE>\n        >>> print(record.lower().format("fasta"))\n        >gi|3298468|dbj|BAA31520.1| SAMIPF\n        gghvnpavtfgafvggnitllrgivyiiaqllgstvaclllkfvtndmavgvfslsagvg\n        vtnalvfeivmtfglvytvyataidpkkgslgtiapiaigfivgani\n        <BLANKLINE>\n\n        To take a more annotation rich example,\n\n        >>> from Bio import SeqIO\n        >>> old = SeqIO.read("EMBL/TRBG361.embl", "embl")\n        >>> len(old.features)\n        3\n        >>> new = old.lower()\n        >>> len(old.features) == len(new.features)\n        True\n        >>> old.annotations["organism"] == new.annotations["organism"]\n        True\n        >>> old.dbxrefs == new.dbxrefs\n        True\n        '
        return type(self)(self.seq.lower(), id=self.id, name=self.name, description=self.description, dbxrefs=self.dbxrefs[:], features=self.features[:], annotations=self.annotations.copy(), letter_annotations=self.letter_annotations.copy())

    def isupper(self):
        if False:
            while True:
                i = 10
        "Return True if all ASCII characters in the record's sequence are uppercase.\n\n        If there are no cased characters, the method returns False.\n        "
        return self.seq.isupper()

    def islower(self):
        if False:
            while True:
                i = 10
        "Return True if all ASCII characters in the record's sequence are lowercase.\n\n        If there are no cased characters, the method returns False.\n        "
        return self.seq.islower()

    def reverse_complement(self, id: bool=False, name: bool=False, description: bool=False, features: bool=True, annotations: bool=False, letter_annotations: bool=True, dbxrefs: bool=False) -> 'SeqRecord':
        if False:
            print('Hello World!')
        'Return new SeqRecord with reverse complement sequence.\n\n        By default the new record does NOT preserve the sequence identifier,\n        name, description, general annotation or database cross-references -\n        these are unlikely to apply to the reversed sequence.\n\n        You can specify the returned record\'s id, name and description as\n        strings, or True to keep that of the parent, or False for a default.\n\n        You can specify the returned record\'s features with a list of\n        SeqFeature objects, or True to keep that of the parent, or False to\n        omit them. The default is to keep the original features (with the\n        strand and locations adjusted).\n\n        You can also specify both the returned record\'s annotations and\n        letter_annotations as dictionaries, True to keep that of the parent,\n        or False to omit them. The default is to keep the original\n        annotations (with the letter annotations reversed).\n\n        To show what happens to the pre-letter annotations, consider an\n        example Solexa variant FASTQ file with a single entry, which we\'ll\n        read in as a SeqRecord:\n\n        >>> from Bio import SeqIO\n        >>> record = SeqIO.read("Quality/solexa_faked.fastq", "fastq-solexa")\n        >>> print("%s %s" % (record.id, record.seq))\n        slxa_0001_1_0001_01 ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTNNNNNN\n        >>> print(list(record.letter_annotations))\n        [\'solexa_quality\']\n        >>> print(record.letter_annotations["solexa_quality"])\n        [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]\n\n        Now take the reverse complement, here we explicitly give a new\n        identifier (the old identifier with a suffix):\n\n        >>> rc_record = record.reverse_complement(id=record.id + "_rc")\n        >>> print("%s %s" % (rc_record.id, rc_record.seq))\n        slxa_0001_1_0001_01_rc NNNNNNACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT\n\n        Notice that the per-letter-annotations have also been reversed,\n        although this may not be appropriate for all cases.\n\n        >>> print(rc_record.letter_annotations["solexa_quality"])\n        [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]\n\n        Now for the features, we need a different example. Parsing a GenBank\n        file is probably the easiest way to get an nice example with features\n        in it...\n\n        >>> from Bio import SeqIO\n        >>> with open("GenBank/pBAD30.gb") as handle:\n        ...     plasmid = SeqIO.read(handle, "gb")\n        >>> print("%s %i" % (plasmid.id, len(plasmid)))\n        pBAD30 4923\n        >>> plasmid.seq\n        Seq(\'GCTAGCGGAGTGTATACTGGCTTACTATGTTGGCACTGATGAGGGTGTCAGTGA...ATG\')\n        >>> len(plasmid.features)\n        13\n\n        Now, let\'s take the reverse complement of this whole plasmid:\n\n        >>> rc_plasmid = plasmid.reverse_complement(id=plasmid.id+"_rc")\n        >>> print("%s %i" % (rc_plasmid.id, len(rc_plasmid)))\n        pBAD30_rc 4923\n        >>> rc_plasmid.seq\n        Seq(\'CATGGGCAAATATTATACGCAAGGCGACAAGGTGCTGATGCCGCTGGCGATTCA...AGC\')\n        >>> len(rc_plasmid.features)\n        13\n\n        Let\'s compare the first CDS feature - it has gone from being the\n        second feature (index 1) to the second last feature (index -2), its\n        strand has changed, and the location switched round.\n\n        >>> print(plasmid.features[1])\n        type: CDS\n        location: [1081:1960](-)\n        qualifiers:\n            Key: label, Value: [\'araC\']\n            Key: note, Value: [\'araC regulator of the arabinose BAD promoter\']\n            Key: vntifkey, Value: [\'4\']\n        <BLANKLINE>\n        >>> print(rc_plasmid.features[-2])\n        type: CDS\n        location: [2963:3842](+)\n        qualifiers:\n            Key: label, Value: [\'araC\']\n            Key: note, Value: [\'araC regulator of the arabinose BAD promoter\']\n            Key: vntifkey, Value: [\'4\']\n        <BLANKLINE>\n\n        You can check this new location, based on the length of the plasmid:\n\n        >>> len(plasmid) - 1081\n        3842\n        >>> len(plasmid) - 1960\n        2963\n\n        Note that if the SeqFeature annotation includes any strand specific\n        information (e.g. base changes for a SNP), this information is not\n        amended, and would need correction after the reverse complement.\n\n        Note trying to reverse complement a protein SeqRecord raises an\n        exception:\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> protein_rec = SeqRecord(Seq("MAIVMGR"), id="Test",\n        ...                         annotations={"molecule_type": "protein"})\n        >>> protein_rec.reverse_complement()\n        Traceback (most recent call last):\n           ...\n        ValueError: Proteins do not have complements!\n\n        If you have RNA without any U bases, it must be annotated as RNA\n        otherwise it will be treated as DNA by default with A mapped to T:\n\n        >>> from Bio.Seq import Seq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> rna1 = SeqRecord(Seq("ACG"), id="Test")\n        >>> rna2 = SeqRecord(Seq("ACG"), id="Test", annotations={"molecule_type": "RNA"})\n        >>> print(rna1.reverse_complement(id="RC", description="unk").format("fasta"))\n        >RC unk\n        CGT\n        <BLANKLINE>\n        >>> print(rna2.reverse_complement(id="RC", description="RNA").format("fasta"))\n        >RC RNA\n        CGU\n        <BLANKLINE>\n\n        Also note you can reverse complement a SeqRecord using a MutableSeq:\n\n        >>> from Bio.Seq import MutableSeq\n        >>> from Bio.SeqRecord import SeqRecord\n        >>> rec = SeqRecord(MutableSeq("ACGT"), id="Test")\n        >>> rec.seq[0] = "T"\n        >>> print("%s %s" % (rec.id, rec.seq))\n        Test TCGT\n        >>> rc = rec.reverse_complement(id=True)\n        >>> print("%s %s" % (rc.id, rc.seq))\n        Test ACGA\n        '
        from Bio.Seq import Seq, MutableSeq
        if 'protein' in cast(str, self.annotations.get('molecule_type', '')):
            raise ValueError('Proteins do not have complements!')
        if 'RNA' in cast(str, self.annotations.get('molecule_type', '')):
            seq = self.seq.reverse_complement_rna(inplace=False)
        else:
            seq = self.seq.reverse_complement(inplace=False)
        if isinstance(self.seq, MutableSeq):
            seq = Seq(seq)
        answer = type(self)(seq)
        if isinstance(id, str):
            answer.id = id
        elif id:
            answer.id = self.id
        if isinstance(name, str):
            answer.name = name
        elif name:
            answer.name = self.name
        if isinstance(description, str):
            answer.description = description
        elif description:
            answer.description = self.description
        if isinstance(dbxrefs, list):
            answer.dbxrefs = dbxrefs
        elif dbxrefs:
            answer.dbxrefs = self.dbxrefs[:]
        if isinstance(features, list):
            answer.features = features
        elif features:
            length = len(answer)
            answer.features = [f._flip(length) for f in self.features]

            def key_fun(f):
                if False:
                    for i in range(10):
                        print('nop')
                'Sort on start position.'
                try:
                    return int(f.location.start)
                except TypeError:
                    return None
            answer.features.sort(key=key_fun)
        if isinstance(annotations, dict):
            answer.annotations = annotations
        elif annotations:
            answer.annotations = self.annotations.copy()
        if isinstance(letter_annotations, dict):
            answer.letter_annotations = letter_annotations
        elif letter_annotations:
            for (key, value) in self.letter_annotations.items():
                answer._per_letter_annotations[key] = value[::-1]
        return answer

    def translate(self, table: str='Standard', stop_symbol: str='*', to_stop: bool=False, cds: bool=False, gap: Optional[str]=None, id: bool=False, name: bool=False, description: bool=False, features: bool=False, annotations: bool=False, letter_annotations: bool=False, dbxrefs: bool=False) -> 'SeqRecord':
        if False:
            while True:
                i = 10
        'Return new SeqRecord with translated sequence.\n\n        This calls the record\'s .seq.translate() method (which describes\n        the translation related arguments, like table for the genetic code),\n\n        By default the new record does NOT preserve the sequence identifier,\n        name, description, general annotation or database cross-references -\n        these are unlikely to apply to the translated sequence.\n\n        You can specify the returned record\'s id, name and description as\n        strings, or True to keep that of the parent, or False for a default.\n\n        You can specify the returned record\'s features with a list of\n        SeqFeature objects, or False (default) to omit them.\n\n        You can also specify both the returned record\'s annotations and\n        letter_annotations as dictionaries, True to keep that of the parent\n        (annotations only), or False (default) to omit them.\n\n        e.g. Loading a FASTA gene and translating it,\n\n        >>> from Bio import SeqIO\n        >>> gene_record = SeqIO.read("Fasta/sweetpea.nu", "fasta")\n        >>> print(gene_record.format("fasta"))\n        >gi|3176602|gb|U78617.1|LOU78617 Lathyrus odoratus phytochrome A (PHYA) gene, partial cds\n        CAGGCTGCGCGGTTTCTATTTATGAAGAACAAGGTCCGTATGATAGTTGATTGTCATGCA\n        AAACATGTGAAGGTTCTTCAAGACGAAAAACTCCCATTTGATTTGACTCTGTGCGGTTCG\n        ACCTTAAGAGCTCCACATAGTTGCCATTTGCAGTACATGGCTAACATGGATTCAATTGCT\n        TCATTGGTTATGGCAGTGGTCGTCAATGACAGCGATGAAGATGGAGATAGCCGTGACGCA\n        GTTCTACCACAAAAGAAAAAGAGACTTTGGGGTTTGGTAGTTTGTCATAACACTACTCCG\n        AGGTTTGTT\n        <BLANKLINE>\n\n        And now translating the record, specifying the new ID and description:\n\n        >>> protein_record = gene_record.translate(table=11,\n        ...                                        id="phya",\n        ...                                        description="translation")\n        >>> print(protein_record.format("fasta"))\n        >phya translation\n        QAARFLFMKNKVRMIVDCHAKHVKVLQDEKLPFDLTLCGSTLRAPHSCHLQYMANMDSIA\n        SLVMAVVVNDSDEDGDSRDAVLPQKKKRLWGLVVCHNTTPRFV\n        <BLANKLINE>\n\n        '
        if 'protein' == self.annotations.get('molecule_type', ''):
            raise ValueError('Proteins cannot be translated!')
        answer = SeqRecord(self.seq.translate(table=table, stop_symbol=stop_symbol, to_stop=to_stop, cds=cds, gap=gap))
        if isinstance(id, str):
            answer.id = id
        elif id:
            answer.id = self.id
        if isinstance(name, str):
            answer.name = name
        elif name:
            answer.name = self.name
        if isinstance(description, str):
            answer.description = description
        elif description:
            answer.description = self.description
        if isinstance(dbxrefs, list):
            answer.dbxrefs = dbxrefs
        elif dbxrefs:
            answer.dbxrefs = self.dbxrefs[:]
        if isinstance(features, list):
            answer.features = features
        elif features:
            raise TypeError(f'Unexpected features argument {features!r}')
        if isinstance(annotations, dict):
            answer.annotations = annotations
        elif annotations:
            answer.annotations = self.annotations.copy()
        answer.annotations['molecule_type'] = 'protein'
        if isinstance(letter_annotations, dict):
            answer.letter_annotations = letter_annotations
        elif letter_annotations:
            raise TypeError(f'Unexpected letter_annotations argument {letter_annotations!r}')
        return answer
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()