"""Parses the mmCIF file format."""
import collections
import dataclasses
import functools
import io
from typing import Any, Mapping, Optional, Sequence, Tuple
from absl import logging
from Bio import PDB
from Bio.Data import SCOPData
from Bio.PDB.MMCIFParser import MMCIFParser
ChainId = str
PdbHeader = Mapping[str, Any]
PdbStructure = PDB.Structure.Structure
SeqRes = str
MmCIFDict = Mapping[str, Sequence[str]]

@dataclasses.dataclass(frozen=True)
class Monomer:
    id: str
    num: int

@dataclasses.dataclass(frozen=True)
class AtomSite:
    residue_name: str
    author_chain_id: str
    mmcif_chain_id: str
    author_seq_num: str
    mmcif_seq_num: int
    insertion_code: str
    hetatm_atom: str
    model_num: int

@dataclasses.dataclass(frozen=True)
class ResiduePosition:
    chain_id: str
    residue_number: int
    insertion_code: str

@dataclasses.dataclass(frozen=True)
class ResidueAtPosition:
    position: Optional[ResiduePosition]
    name: str
    is_missing: bool
    hetflag: str

@dataclasses.dataclass(frozen=True)
class MmcifObject:
    """Representation of a parsed mmCIF file.

    Contains:
        file_id: A meaningful name, e.g. a pdb_id. Should be unique amongst all
            files being processed.
        header: Biopython header.
        structure: Biopython structure.
        chain_to_seqres: Dict mapping chain_id to 1 letter amino acid sequence. E.g.
            {'A': 'ABCDEFG'}
        seqres_to_structure: Dict; for each chain_id contains a mapping between
            SEQRES index and a ResidueAtPosition. e.g. {'A': {0: ResidueAtPosition,  1: ResidueAtPosition, ...}}
        raw_string: The raw string used to construct the MmcifObject.
    """
    file_id: str
    header: PdbHeader
    structure: PdbStructure
    chain_to_seqres: Mapping[ChainId, SeqRes]
    seqres_to_structure: Mapping[ChainId, Mapping[int, ResidueAtPosition]]
    raw_string: Any
    mmcif_to_author_chain_id: Mapping[ChainId, ChainId]
    valid_chains: Mapping[ChainId, str]

@dataclasses.dataclass(frozen=True)
class ParsingResult:
    """Returned by the parse function.

    Contains:
        mmcif_object: A MmcifObject, may be None if no chain could be successfully
            parsed.
        errors: A dict mapping (file_id, chain_id) to any exception generated.
    """
    mmcif_object: Optional[MmcifObject]
    errors: Mapping[Tuple[str, str], Any]

class ParseError(Exception):
    """An error indicating that an mmCIF file could not be parsed."""

def mmcif_loop_to_list(prefix: str, parsed_info: MmCIFDict) -> Sequence[Mapping[str, str]]:
    if False:
        i = 10
        return i + 15
    "Extracts loop associated with a prefix from mmCIF data as a list.\n\n    Reference for loop_ in mmCIF:\n        http://mmcif.wwpdb.org/docs/tutorials/mechanics/pdbx-mmcif-syntax.html\n\n    Args:\n        prefix: Prefix shared by each of the data items in the loop.\n            e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,\n            _entity_poly_seq.mon_id. Should include the trailing period.\n        parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython\n            parser.\n\n    Returns:\n        Returns a list of dicts; each dict represents 1 entry from an mmCIF loop.\n    "
    cols = []
    data = []
    for (key, value) in parsed_info.items():
        if key.startswith(prefix):
            cols.append(key)
            data.append(value)
    assert all([len(xs) == len(data[0]) for xs in data]), 'mmCIF error: Not all loops are the same length: %s' % cols
    return [dict(zip(cols, xs)) for xs in zip(*data)]

def mmcif_loop_to_dict(prefix: str, index: str, parsed_info: MmCIFDict) -> Mapping[str, Mapping[str, str]]:
    if False:
        return 10
    "Extracts loop associated with a prefix from mmCIF data as a dictionary.\n\n    Args:\n        prefix: Prefix shared by each of the data items in the loop.\n            e.g. '_entity_poly_seq.', where the data items are _entity_poly_seq.num,\n            _entity_poly_seq.mon_id. Should include the trailing period.\n        index: Which item of loop data should serve as the key.\n        parsed_info: A dict of parsed mmCIF data, e.g. _mmcif_dict from a Biopython\n            parser.\n\n    Returns:\n        Returns a dict of dicts; each dict represents 1 entry from an mmCIF loop,\n        indexed by the index column.\n    "
    entries = mmcif_loop_to_list(prefix, parsed_info)
    return {entry[index]: entry for entry in entries}

@functools.lru_cache(16, typed=False)
def fast_parse(*, file_id: str, mmcif_string: str, catch_all_errors: bool=True) -> ParsingResult:
    if False:
        return 10
    'Entry point, parses an mmcif_string.\n\n    Args:\n        file_id: A string identifier for this file. Should be unique within the\n            collection of files being processed.\n        mmcif_string: Contents of an mmCIF file.\n        catch_all_errors: If True, all exceptions are caught and error messages are\n            returned as part of the ParsingResult. If False exceptions will be allowed\n            to propagate.\n\n    Returns:\n        A ParsingResult.\n    '
    errors = {}
    try:
        parser = MMCIFParser(QUIET=True)
        parsed_info = parser._mmcif_dict
        for (key, value) in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]
        header = _get_header(parsed_info)
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(None, {(file_id, ''): 'No protein chains found in this file.'})
        mmcif_to_author_chain_id = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != '1':
                continue
            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id
        mmcif_object = MmcifObject(file_id=file_id, header=header, structure=None, chain_to_seqres=None, seqres_to_structure=None, raw_string=parsed_info, mmcif_to_author_chain_id=mmcif_to_author_chain_id, valid_chains=valid_chains)
        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:
        errors[file_id, ''] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)

@functools.lru_cache(16, typed=False)
def parse(*, file_id: str, mmcif_string: str, catch_all_errors: bool=True) -> ParsingResult:
    if False:
        i = 10
        return i + 15
    'Entry point, parses an mmcif_string.\n\n    Args:\n        file_id: A string identifier for this file. Should be unique within the\n            collection of files being processed.\n        mmcif_string: Contents of an mmCIF file.\n        catch_all_errors: If True, all exceptions are caught and error messages are\n            returned as part of the ParsingResult. If False exceptions will be allowed\n            to propagate.\n\n    Returns:\n        A ParsingResult.\n    '
    errors = {}
    try:
        parser = PDB.MMCIFParser(QUIET=True)
        handle = io.StringIO(mmcif_string)
        full_structure = parser.get_structure('', handle)
        first_model_structure = _get_first_model(full_structure)
        parsed_info = parser._mmcif_dict
        for (key, value) in parsed_info.items():
            if not isinstance(value, list):
                parsed_info[key] = [value]
        header = _get_header(parsed_info)
        valid_chains = _get_protein_chains(parsed_info=parsed_info)
        if not valid_chains:
            return ParsingResult(None, {(file_id, ''): 'No protein chains found in this file.'})
        seq_start_num = {chain_id: min([monomer.num for monomer in seq]) for (chain_id, seq) in valid_chains.items()}
        mmcif_to_author_chain_id = {}
        seq_to_structure_mappings = {}
        for atom in _get_atom_site_list(parsed_info):
            if atom.model_num != '1':
                continue
            mmcif_to_author_chain_id[atom.mmcif_chain_id] = atom.author_chain_id
            if atom.mmcif_chain_id in valid_chains:
                hetflag = ' '
                if atom.hetatm_atom == 'HETATM':
                    if atom.residue_name in ('HOH', 'WAT'):
                        hetflag = 'W'
                    else:
                        hetflag = 'H_' + atom.residue_name
                insertion_code = atom.insertion_code
                if not _is_set(atom.insertion_code):
                    insertion_code = ' '
                position = ResiduePosition(chain_id=atom.author_chain_id, residue_number=int(atom.author_seq_num), insertion_code=insertion_code)
                seq_idx = int(atom.mmcif_seq_num) - seq_start_num[atom.mmcif_chain_id]
                current = seq_to_structure_mappings.get(atom.author_chain_id, {})
                current[seq_idx] = ResidueAtPosition(position=position, name=atom.residue_name, is_missing=False, hetflag=hetflag)
                seq_to_structure_mappings[atom.author_chain_id] = current
        for (chain_id, seq_info) in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            current_mapping = seq_to_structure_mappings[author_chain]
            for (idx, monomer) in enumerate(seq_info):
                if idx not in current_mapping:
                    current_mapping[idx] = ResidueAtPosition(position=None, name=monomer.id, is_missing=True, hetflag=' ')
        author_chain_to_sequence = {}
        for (chain_id, seq_info) in valid_chains.items():
            author_chain = mmcif_to_author_chain_id[chain_id]
            seq = []
            for monomer in seq_info:
                code = SCOPData.protein_letters_3to1.get(monomer.id, 'X')
                seq.append(code if len(code) == 1 else 'X')
            seq = ''.join(seq)
            author_chain_to_sequence[author_chain] = seq
        mmcif_object = MmcifObject(file_id=file_id, header=header, structure=first_model_structure, chain_to_seqres=author_chain_to_sequence, seqres_to_structure=seq_to_structure_mappings, raw_string=parsed_info, mmcif_to_author_chain_id=mmcif_to_author_chain_id, valid_chains=valid_chains)
        return ParsingResult(mmcif_object=mmcif_object, errors=errors)
    except Exception as e:
        errors[file_id, ''] = e
        if not catch_all_errors:
            raise
        return ParsingResult(mmcif_object=None, errors=errors)

def _get_first_model(structure: PdbStructure) -> PdbStructure:
    if False:
        for i in range(10):
            print('nop')
    'Returns the first model in a Biopython structure.'
    return next(structure.get_models())
_MIN_LENGTH_OF_CHAIN_TO_BE_COUNTED_AS_PEPTIDE = 21

def get_release_date(parsed_info: MmCIFDict) -> str:
    if False:
        while True:
            i = 10
    'Returns the oldest revision date.'
    revision_dates = parsed_info['_pdbx_audit_revision_history.revision_date']
    return min(revision_dates)

def _get_header(parsed_info: MmCIFDict) -> PdbHeader:
    if False:
        print('Hello World!')
    'Returns a basic header containing method, release date and resolution.'
    header = {}
    experiments = mmcif_loop_to_list('_exptl.', parsed_info)
    header['structure_method'] = ','.join([experiment['_exptl.method'].lower() for experiment in experiments])
    if '_pdbx_audit_revision_history.revision_date' in parsed_info:
        header['release_date'] = get_release_date(parsed_info)
    else:
        logging.warning('Could not determine release_date: %s', parsed_info['_entry.id'])
    header['resolution'] = 0.0
    for res_key in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution', '_reflns.d_resolution_high'):
        if res_key in parsed_info:
            try:
                raw_resolution = parsed_info[res_key][0]
                header['resolution'] = float(raw_resolution)
            except ValueError:
                logging.debug('Invalid resolution format: %s', parsed_info[res_key])
    return header

def _get_atom_site_list(parsed_info: MmCIFDict) -> Sequence[AtomSite]:
    if False:
        i = 10
        return i + 15
    'Returns list of atom sites; contains data not present in the structure.'
    return [AtomSite(*site) for site in zip(parsed_info['_atom_site.label_comp_id'], parsed_info['_atom_site.auth_asym_id'], parsed_info['_atom_site.label_asym_id'], parsed_info['_atom_site.auth_seq_id'], parsed_info['_atom_site.label_seq_id'], parsed_info['_atom_site.pdbx_PDB_ins_code'], parsed_info['_atom_site.group_PDB'], parsed_info['_atom_site.pdbx_PDB_model_num'])]

def _get_protein_chains(*, parsed_info: Mapping[str, Any]) -> Mapping[ChainId, Sequence[Monomer]]:
    if False:
        while True:
            i = 10
    'Extracts polymer information for protein chains only.\n\n    Args:\n        parsed_info: _mmcif_dict produced by the Biopython parser.\n\n    Returns:\n        A dict mapping mmcif chain id to a list of Monomers.\n    '
    entity_poly_seqs = mmcif_loop_to_list('_entity_poly_seq.', parsed_info)
    polymers = collections.defaultdict(list)
    for entity_poly_seq in entity_poly_seqs:
        polymers[entity_poly_seq['_entity_poly_seq.entity_id']].append(Monomer(id=entity_poly_seq['_entity_poly_seq.mon_id'], num=int(entity_poly_seq['_entity_poly_seq.num'])))
    chem_comps = mmcif_loop_to_dict('_chem_comp.', '_chem_comp.id', parsed_info)
    struct_asyms = mmcif_loop_to_list('_struct_asym.', parsed_info)
    entity_to_mmcif_chains = collections.defaultdict(list)
    for struct_asym in struct_asyms:
        chain_id = struct_asym['_struct_asym.id']
        entity_id = struct_asym['_struct_asym.entity_id']
        entity_to_mmcif_chains[entity_id].append(chain_id)
    valid_chains = {}
    for (entity_id, seq_info) in polymers.items():
        chain_ids = entity_to_mmcif_chains[entity_id]
        if any(['peptide' in chem_comps[monomer.id]['_chem_comp.type'] for monomer in seq_info]):
            for chain_id in chain_ids:
                valid_chains[chain_id] = seq_info
    return valid_chains

def _is_set(data: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Returns False if data is a special mmCIF character indicating 'unset'."
    return data not in ('.', '?')