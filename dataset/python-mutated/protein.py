"""Protein data type."""
import dataclasses
import io
from typing import Any, Mapping, Optional
import numpy as np
from Bio.PDB import PDBParser
from modelscope.models.science.unifold.data import residue_constants
FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]
PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)

@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""
    atom_positions: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    chain_index: np.ndarray
    b_factors: np.ndarray

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(f'Cannot build an instance with more than {PDB_MAX_CHAINS} chains because these cannot be written to PDB format.')

def from_pdb_string(pdb_str: str, chain_id: Optional[str]=None) -> Protein:
    if False:
        i = 10
        return i + 15
    'Takes a PDB string and constructs a Protein object.\n\n    WARNING: All non-standard residue types will be converted into UNK. All\n      non-standard atoms will be ignored.\n\n    Args:\n      pdb_str: The contents of the pdb file\n      chain_id: If chain_id is specified (e.g. A), then only that chain\n        is parsed. Otherwise all chains are parsed.\n\n    Returns:\n      A new `Protein` parsed from the pdb contents.\n    '
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('none', pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f'Only single model PDBs are supported. Found {len(models)} models.')
    model = models[0]
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != ' ':
                raise ValueError(f'PDB contains an insertion code at chain {chain.id} and residue index {res.id[1]}. These are not supported.')
            res_shortname = residue_constants.restype_3to1.get(res.resname, 'X')
            restype_idx = residue_constants.restype_order.get(res_shortname, residue_constants.restype_num)
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for (n, cid) in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])
    return Protein(atom_positions=np.array(atom_positions), atom_mask=np.array(atom_mask), aatype=np.array(aatype), residue_index=np.array(residue_index), chain_index=chain_index, b_factors=np.array(b_factors))

def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    if False:
        print('Hello World!')
    chain_end = 'TER'
    return f'{chain_end:<6}{atom_index:>5}      {end_resname:>3} {chain_name:>1}{residue_index:>4}'

def to_pdb(prot: Protein) -> str:
    if False:
        i = 10
        return i + 15
    'Converts a `Protein` instance to a PDB string.\n\n    Args:\n      prot: The protein to convert to PDB.\n\n    Returns:\n      PDB string.\n    '
    restypes = residue_constants.restypes + ['X']

    def res_1to3(r):
        if False:
            return 10
        return residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types
    pdb_lines = []
    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors
    if np.any(aatype > residue_constants.restype_num):
        raise ValueError('Invalid aatypes.')
    chain_ids = {}
    for i in np.unique(chain_index):
        if i >= PDB_MAX_CHAINS:
            raise ValueError(f'The PDB format supports at most {PDB_MAX_CHAINS} chains.')
        chain_ids[i] = PDB_CHAIN_IDS[i]
    pdb_lines.append('MODEL     1')
    atom_index = 1
    last_chain_index = chain_index[0]
    for i in range(aatype.shape[0]):
        if last_chain_index != chain_index[i]:
            pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[i - 1]), chain_ids[chain_index[i - 1]], residue_index[i - 1]))
            last_chain_index = chain_index[i]
            atom_index += 1
        res_name_3 = res_1to3(aatype[i])
        for (atom_name, pos, mask, b_factor) in zip(atom_types, atom_positions[i], atom_mask[i], b_factors[i]):
            if mask < 0.5:
                continue
            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.0
            element = atom_name[0]
            charge = ''
            atom_line = f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}{res_name_3:>3} {chain_ids[chain_index[i]]:>1}{residue_index[i]:>4}{insertion_code:>1}   {pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}{occupancy:>6.2f}{b_factor:>6.2f}          {element:>2}{charge:>2}'
            pdb_lines.append(atom_line)
            atom_index += 1
    pdb_lines.append(_chain_end(atom_index, res_1to3(aatype[-1]), chain_ids[chain_index[-1]], residue_index[-1]))
    pdb_lines.append('ENDMDL')
    pdb_lines.append('END')
    pdb_lines = [line.ljust(80) for line in pdb_lines]
    return '\n'.join(pdb_lines) + '\n'

def ideal_atom_mask(prot: Protein) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    'Computes an ideal atom mask.\n\n    `Protein.atom_mask` typically is defined according to the atoms that are\n    reported in the PDB. This function computes a mask according to heavy atoms\n    that should be present in the given sequence of amino acids.\n\n    Args:\n      prot: `Protein` whose fields are `numpy.ndarray` objects.\n\n    Returns:\n      An ideal atom mask.\n    '
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]

def from_prediction(features: FeatureDict, result: ModelOutput, b_factors: Optional[np.ndarray]=None) -> Protein:
    if False:
        print('Hello World!')
    'Assembles a protein from a prediction.\n\n    Args:\n      features: Dictionary holding model inputs.\n      fold_output: Dictionary holding model outputs.\n      b_factors: (Optional) B-factors to use for the protein.\n\n    Returns:\n      A protein instance.\n    '
    if 'asym_id' in features:
        chain_index = features['asym_id'] - 1
    else:
        chain_index = np.zeros_like(features['aatype'])
    if b_factors is None:
        b_factors = np.zeros_like(result['final_atom_mask'])
    return Protein(aatype=features['aatype'], atom_positions=result['final_atom_positions'], atom_mask=result['final_atom_mask'], residue_index=features['residue_index'] + 1, chain_index=chain_index, b_factors=b_factors)

def from_feature(features: FeatureDict, b_factors: Optional[np.ndarray]=None) -> Protein:
    if False:
        return 10
    'Assembles a standard pdb from input atom positions & mask.\n\n    Args:\n      features: Dictionary holding model inputs.\n      b_factors: (Optional) B-factors to use for the protein.\n\n    Returns:\n      A protein instance.\n    '
    if 'asym_id' in features:
        chain_index = features['asym_id'] - 1
    else:
        chain_index = np.zeros_like(features['aatype'])
    if b_factors is None:
        b_factors = np.zeros_like(features['all_atom_mask'])
    return Protein(aatype=features['aatype'], atom_positions=features['all_atom_positions'], atom_mask=features['all_atom_mask'], residue_index=features['residue_index'] + 1, chain_index=chain_index, b_factors=b_factors)