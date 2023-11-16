"""Implementations of Biopython-like Seq objects on top of BioSQL.

This allows retrieval of items stored in a BioSQL database using
a biopython-like SeqRecord and Seq interface.

Note: Currently we do not support recording per-letter-annotations
(like quality scores) in BioSQL.
"""
from typing import List, Optional
from Bio.Seq import Seq, SequenceDataAbstractBaseClass
from Bio.SeqRecord import SeqRecord, _RestrictedDict
from Bio import SeqFeature

class _BioSQLSequenceData(SequenceDataAbstractBaseClass):
    """Retrieves sequence data from a BioSQL database (PRIVATE)."""
    __slots__ = ('primary_id', 'adaptor', '_length', 'start')

    def __init__(self, primary_id, adaptor, start=0, length=0):
        if False:
            i = 10
            return i + 15
        "Create a new _BioSQLSequenceData object referring to a BioSQL entry.\n\n        You wouldn't normally create a _BioSQLSequenceData object yourself,\n        this is done for you when retrieving a DBSeqRecord object from the\n        database, which creates a Seq object using a _BioSQLSequenceData\n        instance as the data provider.\n        "
        self.primary_id = primary_id
        self.adaptor = adaptor
        self._length = length
        self.start = start
        super().__init__()

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Return the length of the sequence.'
        return self._length

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        'Return a subsequence as a bytes or a _BioSQLSequenceData object.'
        if isinstance(key, slice):
            (start, end, step) = key.indices(self._length)
            size = len(range(start, end, step))
            if size == 0:
                return b''
        else:
            i = key
            if i < 0:
                i += self._length
                if i < 0:
                    raise IndexError(key)
            elif i >= self._length:
                raise IndexError(key)
            c = self.adaptor.get_subseq_as_string(self.primary_id, self.start + i, self.start + i + 1)
            return ord(c)
        if step == 1:
            if start == 0 and size == self._length:
                sequence = self.adaptor.get_subseq_as_string(self.primary_id, self.start, self.start + self._length)
                return sequence.encode('ASCII')
            else:
                return _BioSQLSequenceData(self.primary_id, self.adaptor, self.start + start, size)
        else:
            full = self.adaptor.get_subseq_as_string(self.primary_id, self.start + start, self.start + end)
            return full[::step].encode('ASCII')

def _retrieve_seq_len(adaptor, primary_id):
    if False:
        return 10
    seqs = adaptor.execute_and_fetchall('SELECT length FROM biosequence WHERE bioentry_id = %s', (primary_id,))
    if not seqs:
        return None
    if len(seqs) != 1:
        raise ValueError(f'Expected 1 response, got {len(seqs)}.')
    (given_length,) = seqs[0]
    return int(given_length)

def _retrieve_seq(adaptor, primary_id):
    if False:
        for i in range(10):
            print('nop')
    seqs = adaptor.execute_and_fetchall('SELECT alphabet, length, length(seq) FROM biosequence WHERE bioentry_id = %s', (primary_id,))
    if not seqs:
        return
    if len(seqs) != 1:
        raise ValueError(f'Expected 1 response, got {len(seqs)}.')
    (moltype, given_length, length) = seqs[0]
    try:
        length = int(length)
        given_length = int(given_length)
        if length != given_length:
            raise ValueError(f"'length' differs from sequence length, {given_length}, {length}")
        have_seq = True
    except TypeError:
        if length is not None:
            raise ValueError(f"Expected 'length' to be 'None', got {length}.")
        seqs = adaptor.execute_and_fetchall('SELECT alphabet, length, seq FROM biosequence WHERE bioentry_id = %s', (primary_id,))
        if len(seqs) != 1:
            raise ValueError(f'Expected 1 response, got {len(seqs)}.')
        (moltype, given_length, seq) = seqs[0]
        if seq:
            raise ValueError(f"Expected 'seq' to have a falsy value, got {seq}.")
        length = int(given_length)
        have_seq = False
        del seq
    del given_length
    if have_seq:
        data = _BioSQLSequenceData(primary_id, adaptor, start=0, length=length)
        return Seq(data)
    else:
        return Seq(None, length=length)

def _retrieve_dbxrefs(adaptor, primary_id):
    if False:
        i = 10
        return i + 15
    'Retrieve the database cross references for the sequence (PRIVATE).'
    _dbxrefs = []
    dbxrefs = adaptor.execute_and_fetchall('SELECT dbname, accession, version FROM bioentry_dbxref join dbxref using (dbxref_id) WHERE bioentry_id = %s ORDER BY "rank"', (primary_id,))
    for (dbname, accession, version) in dbxrefs:
        if version and version != '0':
            v = f'{accession}.{version}'
        else:
            v = accession
        _dbxrefs.append(f'{dbname}:{v}')
    return _dbxrefs

def _retrieve_features(adaptor, primary_id):
    if False:
        return 10
    sql = 'SELECT seqfeature_id, type.name, "rank" FROM seqfeature join term type on (type_term_id = type.term_id) WHERE bioentry_id = %s ORDER BY "rank"'
    results = adaptor.execute_and_fetchall(sql, (primary_id,))
    seq_feature_list = []
    for (seqfeature_id, seqfeature_type, seqfeature_rank) in results:
        qvs = adaptor.execute_and_fetchall('SELECT name, value FROM seqfeature_qualifier_value  join term using (term_id) WHERE seqfeature_id = %s ORDER BY "rank"', (seqfeature_id,))
        qualifiers = {}
        for (qv_name, qv_value) in qvs:
            qualifiers.setdefault(qv_name, []).append(qv_value)
        qvs = adaptor.execute_and_fetchall('SELECT dbxref.dbname, dbxref.accession FROM dbxref join seqfeature_dbxref using (dbxref_id) WHERE seqfeature_dbxref.seqfeature_id = %s ORDER BY "rank"', (seqfeature_id,))
        for (qv_name, qv_value) in qvs:
            value = f'{qv_name}:{qv_value}'
            qualifiers.setdefault('db_xref', []).append(value)
        results = adaptor.execute_and_fetchall('SELECT location_id, start_pos, end_pos, strand FROM location WHERE seqfeature_id = %s ORDER BY "rank"', (seqfeature_id,))
        locations = []
        for (location_id, start, end, strand) in results:
            if start:
                start -= 1
            if strand == 0:
                strand = None
            if strand not in (+1, -1, None):
                raise ValueError('Invalid strand %s found in database for seqfeature_id %s' % (strand, seqfeature_id))
            if start is not None and end is not None and (end < start):
                import warnings
                from Bio import BiopythonWarning
                warnings.warn('Inverted location start/end (%i and %i) for seqfeature_id %s' % (start, end, seqfeature_id), BiopythonWarning)
            if start is None:
                start = SeqFeature.UnknownPosition()
            if end is None:
                end = SeqFeature.UnknownPosition()
            locations.append((location_id, start, end, strand))
        remote_results = adaptor.execute_and_fetchall('SELECT location_id, dbname, accession, version FROM location join dbxref using (dbxref_id) WHERE seqfeature_id = %s', (seqfeature_id,))
        lookup = {}
        for (location_id, dbname, accession, version) in remote_results:
            if version and version != '0':
                v = f'{accession}.{version}'
            else:
                v = accession
            if dbname == '':
                dbname = None
            lookup[location_id] = (dbname, v)
        feature = SeqFeature.SeqFeature(type=seqfeature_type)
        feature._seqfeature_id = seqfeature_id
        feature.qualifiers = qualifiers
        if len(locations) == 0:
            pass
        elif len(locations) == 1:
            (location_id, start, end, strand) = locations[0]
            feature.location_operator = _retrieve_location_qualifier_value(adaptor, location_id)
            (dbname, version) = lookup.get(location_id, (None, None))
            feature.location = SeqFeature.SimpleLocation(start, end)
            feature.strand = strand
            feature.ref_db = dbname
            feature.ref = version
        else:
            locs = []
            for location in locations:
                (location_id, start, end, strand) = location
                (dbname, version) = lookup.get(location_id, (None, None))
                locs.append(SeqFeature.SimpleLocation(start, end, strand=strand, ref=version, ref_db=dbname))
            strands = {_.strand for _ in locs}
            if len(strands) == 1 and -1 in strands:
                locs = locs[::-1]
            feature.location = SeqFeature.CompoundLocation(locs, 'join')
        seq_feature_list.append(feature)
    return seq_feature_list

def _retrieve_location_qualifier_value(adaptor, location_id):
    if False:
        for i in range(10):
            print('nop')
    value = adaptor.execute_and_fetch_col0('SELECT value FROM location_qualifier_value WHERE location_id = %s', (location_id,))
    try:
        return value[0]
    except IndexError:
        return ''

def _retrieve_annotations(adaptor, primary_id, taxon_id):
    if False:
        for i in range(10):
            print('nop')
    annotations = {}
    annotations.update(_retrieve_alphabet(adaptor, primary_id))
    annotations.update(_retrieve_qualifier_value(adaptor, primary_id))
    annotations.update(_retrieve_reference(adaptor, primary_id))
    annotations.update(_retrieve_taxon(adaptor, primary_id, taxon_id))
    annotations.update(_retrieve_comment(adaptor, primary_id))
    return annotations

def _retrieve_alphabet(adaptor, primary_id):
    if False:
        return 10
    results = adaptor.execute_and_fetchall('SELECT alphabet FROM biosequence WHERE bioentry_id = %s', (primary_id,))
    if len(results) != 1:
        raise ValueError(f'Expected 1 response, got {len(results)}.')
    alphabets = results[0]
    if len(alphabets) != 1:
        raise ValueError(f'Expected 1 alphabet in response, got {len(alphabets)}.')
    alphabet = alphabets[0]
    if alphabet == 'dna':
        molecule_type = 'DNA'
    elif alphabet == 'rna':
        molecule_type = 'RNA'
    elif alphabet == 'protein':
        molecule_type = 'protein'
    else:
        molecule_type = None
    if molecule_type is not None:
        return {'molecule_type': molecule_type}
    else:
        return {}

def _retrieve_qualifier_value(adaptor, primary_id):
    if False:
        for i in range(10):
            print('nop')
    qvs = adaptor.execute_and_fetchall('SELECT name, value FROM bioentry_qualifier_value JOIN term USING (term_id) WHERE bioentry_id = %s ORDER BY "rank"', (primary_id,))
    qualifiers = {}
    for (name, value) in qvs:
        if name == 'keyword':
            name = 'keywords'
        elif name == 'date_changed':
            name = 'date'
        elif name == 'secondary_accession':
            name = 'accessions'
        qualifiers.setdefault(name, []).append(value)
    return qualifiers

def _retrieve_reference(adaptor, primary_id):
    if False:
        while True:
            i = 10
    refs = adaptor.execute_and_fetchall('SELECT start_pos, end_pos,  location, title, authors, dbname, accession FROM bioentry_reference JOIN reference USING (reference_id) LEFT JOIN dbxref USING (dbxref_id) WHERE bioentry_id = %s ORDER BY "rank"', (primary_id,))
    references = []
    for (start, end, location, title, authors, dbname, accession) in refs:
        reference = SeqFeature.Reference()
        if start is not None or end is not None:
            if start is not None:
                start -= 1
            reference.location = [SeqFeature.SimpleLocation(start, end)]
        if authors:
            reference.authors = authors
        if title:
            reference.title = title
        reference.journal = location
        if dbname == 'PUBMED':
            reference.pubmed_id = accession
        elif dbname == 'MEDLINE':
            reference.medline_id = accession
        references.append(reference)
    if references:
        return {'references': references}
    else:
        return {}

def _retrieve_taxon(adaptor, primary_id, taxon_id):
    if False:
        while True:
            i = 10
    a = {}
    common_names = adaptor.execute_and_fetch_col0("SELECT name FROM taxon_name WHERE taxon_id = %s AND name_class = 'genbank common name'", (taxon_id,))
    if common_names:
        a['source'] = common_names[0]
    scientific_names = adaptor.execute_and_fetch_col0("SELECT name FROM taxon_name WHERE taxon_id = %s AND name_class = 'scientific name'", (taxon_id,))
    if scientific_names:
        a['organism'] = scientific_names[0]
    ncbi_taxids = adaptor.execute_and_fetch_col0('SELECT ncbi_taxon_id FROM taxon WHERE taxon_id = %s', (taxon_id,))
    if ncbi_taxids and ncbi_taxids[0] and (ncbi_taxids[0] != '0'):
        a['ncbi_taxid'] = ncbi_taxids[0]
    taxonomy = []
    while taxon_id:
        (name, rank, parent_taxon_id) = adaptor.execute_one("SELECT taxon_name.name, taxon.node_rank, taxon.parent_taxon_id FROM taxon, taxon_name WHERE taxon.taxon_id=taxon_name.taxon_id AND taxon_name.name_class='scientific name' AND taxon.taxon_id = %s", (taxon_id,))
        if taxon_id == parent_taxon_id:
            break
        taxonomy.insert(0, name)
        taxon_id = parent_taxon_id
    if taxonomy:
        a['taxonomy'] = taxonomy
    return a

def _retrieve_comment(adaptor, primary_id):
    if False:
        while True:
            i = 10
    qvs = adaptor.execute_and_fetchall('SELECT comment_text FROM comment WHERE bioentry_id=%s ORDER BY "rank"', (primary_id,))
    comments = [comm[0] for comm in qvs]
    if comments:
        return {'comment': comments}
    else:
        return {}

class DBSeqRecord(SeqRecord):
    """BioSQL equivalent of the Biopython SeqRecord object."""

    def __init__(self, adaptor, primary_id):
        if False:
            i = 10
            return i + 15
        "Create a DBSeqRecord object.\n\n        Arguments:\n         - adaptor - A BioSQL.BioSeqDatabase.Adaptor object\n         - primary_id - An internal integer ID used by BioSQL\n\n        You wouldn't normally create a DBSeqRecord object yourself,\n        this is done for you when using a BioSeqDatabase object\n        "
        self._adaptor = adaptor
        self._primary_id = primary_id
        (self._biodatabase_id, self._taxon_id, self.name, accession, version, self._identifier, self._division, self.description) = self._adaptor.execute_one('SELECT biodatabase_id, taxon_id, name, accession, version, identifier, division, description FROM bioentry WHERE bioentry_id = %s', (self._primary_id,))
        if version and version != '0':
            self.id = f'{accession}.{version}'
        else:
            self.id = accession
        length = _retrieve_seq_len(adaptor, primary_id)
        self._per_letter_annotations = _RestrictedDict(length=length)

    def __get_seq(self):
        if False:
            print('Hello World!')
        if not hasattr(self, '_seq'):
            self._seq = _retrieve_seq(self._adaptor, self._primary_id)
        return self._seq

    def __set_seq(self, seq):
        if False:
            print('Hello World!')
        self._seq = seq

    def __del_seq(self):
        if False:
            print('Hello World!')
        del self._seq
    seq = property(__get_seq, __set_seq, __del_seq, 'Seq object')

    @property
    def dbxrefs(self) -> List[str]:
        if False:
            print('Hello World!')
        'Database cross references.'
        if not hasattr(self, '_dbxrefs'):
            self._dbxrefs = _retrieve_dbxrefs(self._adaptor, self._primary_id)
        return self._dbxrefs

    @dbxrefs.setter
    def dbxrefs(self, value: List[str]) -> None:
        if False:
            i = 10
            return i + 15
        self._dbxrefs = value

    @dbxrefs.deleter
    def dbxrefs(self) -> None:
        if False:
            print('Hello World!')
        del self._dbxrefs

    def __get_features(self):
        if False:
            while True:
                i = 10
        if not hasattr(self, '_features'):
            self._features = _retrieve_features(self._adaptor, self._primary_id)
        return self._features

    def __set_features(self, features):
        if False:
            for i in range(10):
                print('nop')
        self._features = features

    def __del_features(self):
        if False:
            while True:
                i = 10
        del self._features
    features = property(__get_features, __set_features, __del_features, 'Features')

    @property
    def annotations(self) -> SeqRecord._AnnotationsDict:
        if False:
            while True:
                i = 10
        'Annotations.'
        if not hasattr(self, '_annotations'):
            self._annotations = _retrieve_annotations(self._adaptor, self._primary_id, self._taxon_id)
            if self._identifier:
                self._annotations['gi'] = self._identifier
            if self._division:
                self._annotations['data_file_division'] = self._division
        return self._annotations

    @annotations.setter
    def annotations(self, value: Optional[SeqRecord._AnnotationsDict]) -> None:
        if False:
            print('Hello World!')
        if value:
            self._annotations = value
        else:
            self._annotations = {}

    @annotations.deleter
    def annotations(self) -> None:
        if False:
            print('Hello World!')
        del self._annotations