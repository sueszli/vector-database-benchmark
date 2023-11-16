from typing import Dict
import numpy as np
import torch
from . import residue_constants as rc
from .tensor_utils import tensor_tree_map, tree_map

def make_atom14_masks(protein: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if False:
        while True:
            i = 10
    'Construct denser atom positions (14 dimensions instead of 37).'
    restype_atom14_to_atom37_list = []
    restype_atom37_to_atom14_list = []
    restype_atom14_mask_list = []
    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37_list.append([rc.atom_order[name] if name else 0 for name in atom_names])
        atom_name_to_idx14 = {name: i for (i, name) in enumerate(atom_names)}
        restype_atom37_to_atom14_list.append([atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0 for name in rc.atom_types])
        restype_atom14_mask_list.append([1.0 if name else 0.0 for name in atom_names])
    restype_atom14_to_atom37_list.append([0] * 14)
    restype_atom37_to_atom14_list.append([0] * 37)
    restype_atom14_mask_list.append([0.0] * 14)
    restype_atom14_to_atom37 = torch.tensor(restype_atom14_to_atom37_list, dtype=torch.int32, device=protein['aatype'].device)
    restype_atom37_to_atom14 = torch.tensor(restype_atom37_to_atom14_list, dtype=torch.int32, device=protein['aatype'].device)
    restype_atom14_mask = torch.tensor(restype_atom14_mask_list, dtype=torch.float32, device=protein['aatype'].device)
    protein_aatype = protein['aatype'].to(torch.long)
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]
    protein['atom14_atom_exists'] = residx_atom14_mask
    protein['residx_atom14_to_atom37'] = residx_atom14_to_atom37.long()
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein['residx_atom37_to_atom14'] = residx_atom37_to_atom14.long()
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=protein['aatype'].device)
    for (restype, restype_letter) in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1
    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein['atom37_atom_exists'] = residx_atom37_mask
    return protein

def make_atom14_masks_np(batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    batch = tree_map(lambda n: torch.tensor(n, device=batch['aatype'].device), batch, np.ndarray)
    out = tensor_tree_map(lambda t: np.array(t), make_atom14_masks(batch))
    return out