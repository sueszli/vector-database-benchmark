"""Bio.SeqIO support for accessing sequences in PDB and mmCIF files."""
import collections
import warnings
from Bio import BiopythonParserWarning
from Bio.Data.PDBData import protein_letters_3to1
from Bio.Data.PDBData import protein_letters_3to1_extended
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import SequenceIterator
_aa3to1_dict = {}
_aa3to1_dict.update(protein_letters_3to1)
_aa3to1_dict.update(protein_letters_3to1_extended)

def _res2aacode(residue, undef_code='X'):
    if False:
        while True:
            i = 10
    'Return the one-letter amino acid code from the residue name.\n\n    Non-amino acid are returned as "X".\n    '
    if isinstance(residue, str):
        return _aa3to1_dict.get(residue, undef_code)
    return _aa3to1_dict.get(residue.resname, undef_code)

def AtomIterator(pdb_id, structure):
    if False:
        i = 10
        return i + 15
    'Return SeqRecords from Structure objects.\n\n    Base function for sequence parsers that read structures Bio.PDB parsers.\n\n    Once a parser from Bio.PDB has been used to load a structure into a\n    Bio.PDB.Structure.Structure object, there is no difference in how the\n    sequence parser interprets the residue sequence. The functions in this\n    module may be used by SeqIO modules wishing to parse sequences from lists\n    of residues.\n\n    Calling functions must pass a Bio.PDB.Structure.Structure object.\n\n\n    See Bio.SeqIO.PdbIO.PdbAtomIterator and Bio.SeqIO.PdbIO.CifAtomIterator for\n    details.\n    '
    model = structure[0]
    for (chn_id, chain) in sorted(model.child_dict.items()):
        residues = [res for res in chain.get_unpacked_list() if _res2aacode(res.get_resname().upper()) != 'X']
        if not residues:
            continue
        gaps = []
        rnumbers = [r.id[1] for r in residues]
        for (i, rnum) in enumerate(rnumbers[:-1]):
            if rnumbers[i + 1] != rnum + 1 and rnumbers[i + 1] != rnum:
                gaps.append((i + 1, rnum, rnumbers[i + 1]))
        if gaps:
            res_out = []
            prev_idx = 0
            for (i, pregap, postgap) in gaps:
                if postgap > pregap:
                    gapsize = postgap - pregap - 1
                    res_out.extend((_res2aacode(x) for x in residues[prev_idx:i]))
                    prev_idx = i
                    res_out.append('X' * gapsize)
                else:
                    warnings.warn('Ignoring out-of-order residues after a gap', BiopythonParserWarning)
                    res_out.extend((_res2aacode(x) for x in residues[prev_idx:i]))
                    break
            else:
                res_out.extend((_res2aacode(x) for x in residues[prev_idx:]))
        else:
            res_out = [_res2aacode(x) for x in residues]
        record_id = f'{pdb_id}:{chn_id}'
        record = SeqRecord(Seq(''.join(res_out)), id=record_id, description=record_id)
        record.annotations['molecule_type'] = 'protein'
        record.annotations['model'] = model.id
        record.annotations['chain'] = chain.id
        record.annotations['start'] = int(rnumbers[0])
        record.annotations['end'] = int(rnumbers[-1])
        yield record

class PdbSeqresIterator(SequenceIterator):
    """Parser for PDB files."""

    def __init__(self, source):
        if False:
            print('Hello World!')
        'Return SeqRecord objects for each chain in a PDB file.\n\n        Arguments:\n         - source - input stream opened in text mode, or a path to a file\n\n        The sequences are derived from the SEQRES lines in the\n        PDB file header, not the atoms of the 3D structure.\n\n        Specifically, these PDB records are handled: DBREF, DBREF1, DBREF2, SEQADV, SEQRES, MODRES\n\n        See: http://www.wwpdb.org/documentation/format23/sect3.html\n\n        This gets called internally via Bio.SeqIO for the SEQRES based interpretation\n        of the PDB file format:\n\n        >>> from Bio import SeqIO\n        >>> for record in SeqIO.parse("PDB/1A8O.pdb", "pdb-seqres"):\n        ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n        ...     print(record.dbxrefs)\n        ...\n        Record id 1A8O:A, chain A\n        [\'UNP:P12497\', \'UNP:POL_HV1N5\']\n\n        Equivalently,\n\n        >>> with open("PDB/1A8O.pdb") as handle:\n        ...     for record in PdbSeqresIterator(handle):\n        ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n        ...         print(record.dbxrefs)\n        ...\n        Record id 1A8O:A, chain A\n        [\'UNP:P12497\', \'UNP:POL_HV1N5\']\n\n        Note the chain is recorded in the annotations dictionary, and any PDB DBREF\n        lines are recorded in the database cross-references list.\n        '
        super().__init__(source, mode='t', fmt='PDB')

    def parse(self, handle):
        if False:
            return 10
        'Start parsing the file, and return a SeqRecord generator.'
        records = self.iterate(handle)
        return records

    def iterate(self, handle):
        if False:
            i = 10
            return i + 15
        'Iterate over the records in the PDB file.'
        chains = collections.defaultdict(list)
        metadata = collections.defaultdict(list)
        rec_name = None
        for line in handle:
            rec_name = line[0:6].strip()
            if rec_name == 'SEQRES':
                chn_id = line[11]
                residues = [_res2aacode(res) for res in line[19:].split()]
                chains[chn_id].extend(residues)
            elif rec_name == 'DBREF':
                pdb_id = line[7:11]
                chn_id = line[12]
                database = line[26:32].strip()
                db_acc = line[33:41].strip()
                db_id_code = line[42:54].strip()
                metadata[chn_id].append({'pdb_id': pdb_id, 'database': database, 'db_acc': db_acc, 'db_id_code': db_id_code})
            elif rec_name == 'DBREF1':
                pdb_id = line[7:11]
                chn_id = line[12]
                database = line[26:32].strip()
                db_id_code = line[47:67].strip()
            elif rec_name == 'DBREF2':
                if pdb_id != line[7:11] or chn_id != line[12]:
                    raise ValueError('DBREF2 identifiers do not match')
                db_acc = line[18:40].strip()
                metadata[chn_id].append({'pdb_id': pdb_id, 'database': database, 'db_acc': db_acc, 'db_id_code': db_id_code})
        if rec_name is None:
            raise ValueError('Empty file.')
        for (chn_id, residues) in sorted(chains.items()):
            record = SeqRecord(Seq(''.join(residues)))
            record.annotations = {'chain': chn_id}
            record.annotations['molecule_type'] = 'protein'
            if chn_id in metadata:
                m = metadata[chn_id][0]
                record.id = record.name = f"{m['pdb_id']}:{chn_id}"
                record.description = f"{m['database']}:{m['db_acc']} {m['db_id_code']}"
                for melem in metadata[chn_id]:
                    record.dbxrefs.extend([f"{melem['database']}:{melem['db_acc']}", f"{melem['database']}:{melem['db_id_code']}"])
            else:
                record.id = chn_id
            yield record

def PdbAtomIterator(source):
    if False:
        i = 10
        return i + 15
    'Return SeqRecord objects for each chain in a PDB file.\n\n    Argument source is a file-like object or a path to a file.\n\n    The sequences are derived from the 3D structure (ATOM records), not the\n    SEQRES lines in the PDB file header.\n\n    Unrecognised three letter amino acid codes (e.g. "CSD") from HETATM entries\n    are converted to "X" in the sequence.\n\n    In addition to information from the PDB header (which is the same for all\n    records), the following chain specific information is placed in the\n    annotation:\n\n    record.annotations["residues"] = List of residue ID strings\n    record.annotations["chain"] = Chain ID (typically A, B ,...)\n    record.annotations["model"] = Model ID (typically zero)\n\n    Where amino acids are missing from the structure, as indicated by residue\n    numbering, the sequence is filled in with \'X\' characters to match the size\n    of the missing region, and  None is included as the corresponding entry in\n    the list record.annotations["residues"].\n\n    This function uses the Bio.PDB module to do most of the hard work. The\n    annotation information could be improved but this extra parsing should be\n    done in parse_pdb_header, not this module.\n\n    This gets called internally via Bio.SeqIO for the atom based interpretation\n    of the PDB file format:\n\n    >>> from Bio import SeqIO\n    >>> for record in SeqIO.parse("PDB/1A8O.pdb", "pdb-atom"):\n    ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n    ...\n    Record id 1A8O:A, chain A\n\n    Equivalently,\n\n    >>> with open("PDB/1A8O.pdb") as handle:\n    ...     for record in PdbAtomIterator(handle):\n    ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n    ...\n    Record id 1A8O:A, chain A\n\n    '
    from Bio.PDB import PDBParser
    structure = PDBParser().get_structure(None, source)
    pdb_id = structure.header['idcode']
    if not pdb_id:
        warnings.warn("'HEADER' line not found; can't determine PDB ID.", BiopythonParserWarning)
        pdb_id = '????'
    for record in AtomIterator(pdb_id, structure):
        record.annotations.update(structure.header)
        yield record
PDBX_POLY_SEQ_SCHEME_FIELDS = ('_pdbx_poly_seq_scheme.asym_id', '_pdbx_poly_seq_scheme.mon_id')
STRUCT_REF_FIELDS = ('_struct_ref.id', '_struct_ref.db_name', '_struct_ref.db_code', '_struct_ref.pdbx_db_accession')
STRUCT_REF_SEQ_FIELDS = ('_struct_ref_seq.ref_id', '_struct_ref_seq.pdbx_PDB_id_code', '_struct_ref_seq.pdbx_strand_id')

def CifSeqresIterator(source):
    if False:
        for i in range(10):
            print('nop')
    'Return SeqRecord objects for each chain in an mmCIF file.\n\n    Argument source is a file-like object or a path to a file.\n\n    The sequences are derived from the _entity_poly_seq entries in the mmCIF\n    file, not the atoms of the 3D structure.\n\n    Specifically, these mmCIF records are handled: _pdbx_poly_seq_scheme and\n    _struct_ref_seq. The _pdbx_poly_seq records contain sequence information,\n    and the _struct_ref_seq records contain database cross-references.\n\n    See:\n    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Categories/pdbx_poly_seq_scheme.html\n    and\n    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Categories/struct_ref_seq.html\n\n    This gets called internally via Bio.SeqIO for the sequence-based\n    interpretation of the mmCIF file format:\n\n    >>> from Bio import SeqIO\n    >>> for record in SeqIO.parse("PDB/1A8O.cif", "cif-seqres"):\n    ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n    ...     print(record.dbxrefs)\n    ...\n    Record id 1A8O:A, chain A\n    [\'UNP:P12497\', \'UNP:POL_HV1N5\']\n\n    Equivalently,\n\n    >>> with open("PDB/1A8O.cif") as handle:\n    ...     for record in CifSeqresIterator(handle):\n    ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n    ...         print(record.dbxrefs)\n    ...\n    Record id 1A8O:A, chain A\n    [\'UNP:P12497\', \'UNP:POL_HV1N5\']\n\n    Note the chain is recorded in the annotations dictionary, and any mmCIF\n    _struct_ref_seq entries are recorded in the database cross-references list.\n    '
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    chains = collections.defaultdict(list)
    metadata = collections.defaultdict(list)
    records = MMCIF2Dict(source)
    for field in PDBX_POLY_SEQ_SCHEME_FIELDS + STRUCT_REF_SEQ_FIELDS + STRUCT_REF_FIELDS:
        if field not in records:
            records[field] = []
        elif not isinstance(records[field], list):
            records[field] = [records[field]]
    for (asym_id, mon_id) in zip(records['_pdbx_poly_seq_scheme.asym_id'], records['_pdbx_poly_seq_scheme.mon_id']):
        mon_id_1l = _res2aacode(mon_id)
        chains[asym_id].append(mon_id_1l)
    struct_refs = {}
    for (ref_id, db_name, db_code, db_acc) in zip(records['_struct_ref.id'], records['_struct_ref.db_name'], records['_struct_ref.db_code'], records['_struct_ref.pdbx_db_accession']):
        struct_refs[ref_id] = {'database': db_name, 'db_id_code': db_code, 'db_acc': db_acc}
    for (ref_id, pdb_id, chain_id) in zip(records['_struct_ref_seq.ref_id'], records['_struct_ref_seq.pdbx_PDB_id_code'], records['_struct_ref_seq.pdbx_strand_id']):
        struct_ref = struct_refs[ref_id]
        metadata[chain_id].append({'pdb_id': pdb_id})
        metadata[chain_id][-1].update(struct_ref)
    for (chn_id, residues) in sorted(chains.items()):
        record = SeqRecord(Seq(''.join(residues)))
        record.annotations = {'chain': chn_id}
        record.annotations['molecule_type'] = 'protein'
        if chn_id in metadata:
            m = metadata[chn_id][0]
            record.id = record.name = f"{m['pdb_id']}:{chn_id}"
            record.description = f"{m['database']}:{m['db_acc']} {m['db_id_code']}"
            for melem in metadata[chn_id]:
                record.dbxrefs.extend([f"{melem['database']}:{melem['db_acc']}", f"{melem['database']}:{melem['db_id_code']}"])
        else:
            record.id = chn_id
        yield record

def CifAtomIterator(source):
    if False:
        while True:
            i = 10
    'Return SeqRecord objects for each chain in an mmCIF file.\n\n    Argument source is a file-like object or a path to a file.\n\n    The sequences are derived from the 3D structure (_atom_site.* fields)\n    in the mmCIF file.\n\n    Unrecognised three letter amino acid codes (e.g. "CSD") from HETATM entries\n    are converted to "X" in the sequence.\n\n    In addition to information from the PDB header (which is the same for all\n    records), the following chain specific information is placed in the\n    annotation:\n\n    record.annotations["residues"] = List of residue ID strings\n    record.annotations["chain"] = Chain ID (typically A, B ,...)\n    record.annotations["model"] = Model ID (typically zero)\n\n    Where amino acids are missing from the structure, as indicated by residue\n    numbering, the sequence is filled in with \'X\' characters to match the size\n    of the missing region, and  None is included as the corresponding entry in\n    the list record.annotations["residues"].\n\n    This function uses the Bio.PDB module to do most of the hard work. The\n    annotation information could be improved but this extra parsing should be\n    done in parse_pdb_header, not this module.\n\n    This gets called internally via Bio.SeqIO for the atom based interpretation\n    of the PDB file format:\n\n    >>> from Bio import SeqIO\n    >>> for record in SeqIO.parse("PDB/1A8O.cif", "cif-atom"):\n    ...     print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n    ...\n    Record id 1A8O:A, chain A\n\n    Equivalently,\n\n    >>> with open("PDB/1A8O.cif") as handle:\n    ...     for record in CifAtomIterator(handle):\n    ...         print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))\n    ...\n    Record id 1A8O:A, chain A\n\n    '
    from Bio.PDB.MMCIFParser import MMCIFParser
    structure = MMCIFParser().get_structure(None, source)
    pdb_id = structure.header['idcode']
    if not pdb_id:
        warnings.warn('Could not determine the PDB ID.', BiopythonParserWarning)
        pdb_id = '????'
    yield from AtomIterator(pdb_id, structure)
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)