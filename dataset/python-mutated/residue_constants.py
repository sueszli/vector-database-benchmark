"""Constants used in AlphaFold."""
import collections
import functools
import os
from typing import List, Mapping, Tuple
import numpy as np
from unicore.utils import tree_map
ca_ca = 3.80209737096
chi_angles_atoms = {'ALA': [], 'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']], 'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], 'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']], 'CYS': [['N', 'CA', 'CB', 'SG']], 'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']], 'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']], 'GLY': [], 'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']], 'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']], 'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']], 'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']], 'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']], 'SER': [['N', 'CA', 'CB', 'OG']], 'THR': [['N', 'CA', 'CB', 'OG1']], 'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']], 'VAL': [['N', 'CA', 'CB', 'CG1']]}
chi_angles_mask = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
chi_pi_periodic = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
rigid_group_atom_positions = {'ALA': [['N', 0, (-0.525, 1.363, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.526, -0.0, -0.0)], ['CB', 0, (-0.529, -0.774, -1.205)], ['O', 3, (0.627, 1.062, 0.0)]], 'ARG': [['N', 0, (-0.524, 1.362, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.525, -0.0, -0.0)], ['CB', 0, (-0.524, -0.778, -1.209)], ['O', 3, (0.626, 1.062, 0.0)], ['CG', 4, (0.616, 1.39, -0.0)], ['CD', 5, (0.564, 1.414, 0.0)], ['NE', 6, (0.539, 1.357, -0.0)], ['NH1', 7, (0.206, 2.301, 0.0)], ['NH2', 7, (2.078, 0.978, -0.0)], ['CZ', 7, (0.758, 1.093, -0.0)]], 'ASN': [['N', 0, (-0.536, 1.357, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.526, -0.0, -0.0)], ['CB', 0, (-0.531, -0.787, -1.2)], ['O', 3, (0.625, 1.062, 0.0)], ['CG', 4, (0.584, 1.399, 0.0)], ['ND2', 5, (0.593, -1.188, 0.001)], ['OD1', 5, (0.633, 1.059, 0.0)]], 'ASP': [['N', 0, (-0.525, 1.362, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.527, 0.0, -0.0)], ['CB', 0, (-0.526, -0.778, -1.208)], ['O', 3, (0.626, 1.062, -0.0)], ['CG', 4, (0.593, 1.398, -0.0)], ['OD1', 5, (0.61, 1.091, 0.0)], ['OD2', 5, (0.592, -1.101, -0.003)]], 'CYS': [['N', 0, (-0.522, 1.362, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.524, 0.0, 0.0)], ['CB', 0, (-0.519, -0.773, -1.212)], ['O', 3, (0.625, 1.062, -0.0)], ['SG', 4, (0.728, 1.653, 0.0)]], 'GLN': [['N', 0, (-0.526, 1.361, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.526, 0.0, 0.0)], ['CB', 0, (-0.525, -0.779, -1.207)], ['O', 3, (0.626, 1.062, -0.0)], ['CG', 4, (0.615, 1.393, 0.0)], ['CD', 5, (0.587, 1.399, -0.0)], ['NE2', 6, (0.593, -1.189, -0.001)], ['OE1', 6, (0.634, 1.06, 0.0)]], 'GLU': [['N', 0, (-0.528, 1.361, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.526, -0.0, -0.0)], ['CB', 0, (-0.526, -0.781, -1.207)], ['O', 3, (0.626, 1.062, 0.0)], ['CG', 4, (0.615, 1.392, 0.0)], ['CD', 5, (0.6, 1.397, 0.0)], ['OE1', 6, (0.607, 1.095, -0.0)], ['OE2', 6, (0.589, -1.104, -0.001)]], 'GLY': [['N', 0, (-0.572, 1.337, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.517, -0.0, -0.0)], ['O', 3, (0.626, 1.062, -0.0)]], 'HIS': [['N', 0, (-0.527, 1.36, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.525, 0.0, 0.0)], ['CB', 0, (-0.525, -0.778, -1.208)], ['O', 3, (0.625, 1.063, 0.0)], ['CG', 4, (0.6, 1.37, -0.0)], ['CD2', 5, (0.889, -1.021, 0.003)], ['ND1', 5, (0.744, 1.16, -0.0)], ['CE1', 5, (2.03, 0.851, 0.002)], ['NE2', 5, (2.145, -0.466, 0.004)]], 'ILE': [['N', 0, (-0.493, 1.373, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.527, -0.0, -0.0)], ['CB', 0, (-0.536, -0.793, -1.213)], ['O', 3, (0.627, 1.062, -0.0)], ['CG1', 4, (0.534, 1.437, -0.0)], ['CG2', 4, (0.54, -0.785, -1.199)], ['CD1', 5, (0.619, 1.391, 0.0)]], 'LEU': [['N', 0, (-0.52, 1.363, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.525, -0.0, -0.0)], ['CB', 0, (-0.522, -0.773, -1.214)], ['O', 3, (0.625, 1.063, -0.0)], ['CG', 4, (0.678, 1.371, 0.0)], ['CD1', 5, (0.53, 1.43, -0.0)], ['CD2', 5, (0.535, -0.774, 1.2)]], 'LYS': [['N', 0, (-0.526, 1.362, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.526, 0.0, 0.0)], ['CB', 0, (-0.524, -0.778, -1.208)], ['O', 3, (0.626, 1.062, -0.0)], ['CG', 4, (0.619, 1.39, 0.0)], ['CD', 5, (0.559, 1.417, 0.0)], ['CE', 6, (0.56, 1.416, 0.0)], ['NZ', 7, (0.554, 1.387, 0.0)]], 'MET': [['N', 0, (-0.521, 1.364, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.525, 0.0, 0.0)], ['CB', 0, (-0.523, -0.776, -1.21)], ['O', 3, (0.625, 1.062, -0.0)], ['CG', 4, (0.613, 1.391, -0.0)], ['SD', 5, (0.703, 1.695, 0.0)], ['CE', 6, (0.32, 1.786, -0.0)]], 'PHE': [['N', 0, (-0.518, 1.363, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.524, 0.0, -0.0)], ['CB', 0, (-0.525, -0.776, -1.212)], ['O', 3, (0.626, 1.062, -0.0)], ['CG', 4, (0.607, 1.377, 0.0)], ['CD1', 5, (0.709, 1.195, -0.0)], ['CD2', 5, (0.706, -1.196, 0.0)], ['CE1', 5, (2.102, 1.198, -0.0)], ['CE2', 5, (2.098, -1.201, -0.0)], ['CZ', 5, (2.794, -0.003, -0.001)]], 'PRO': [['N', 0, (-0.566, 1.351, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.527, -0.0, 0.0)], ['CB', 0, (-0.546, -0.611, -1.293)], ['O', 3, (0.621, 1.066, 0.0)], ['CG', 4, (0.382, 1.445, 0.0)], ['CD', 5, (0.477, 1.424, 0.0)]], 'SER': [['N', 0, (-0.529, 1.36, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.525, -0.0, -0.0)], ['CB', 0, (-0.518, -0.777, -1.211)], ['O', 3, (0.626, 1.062, -0.0)], ['OG', 4, (0.503, 1.325, 0.0)]], 'THR': [['N', 0, (-0.517, 1.364, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.526, 0.0, -0.0)], ['CB', 0, (-0.516, -0.793, -1.215)], ['O', 3, (0.626, 1.062, 0.0)], ['CG2', 4, (0.55, -0.718, -1.228)], ['OG1', 4, (0.472, 1.353, 0.0)]], 'TRP': [['N', 0, (-0.521, 1.363, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.525, -0.0, 0.0)], ['CB', 0, (-0.523, -0.776, -1.212)], ['O', 3, (0.627, 1.062, 0.0)], ['CG', 4, (0.609, 1.37, -0.0)], ['CD1', 5, (0.824, 1.091, 0.0)], ['CD2', 5, (0.854, -1.148, -0.005)], ['CE2', 5, (2.186, -0.678, -0.007)], ['CE3', 5, (0.622, -2.53, -0.007)], ['NE1', 5, (2.14, 0.69, -0.004)], ['CH2', 5, (3.028, -2.89, -0.013)], ['CZ2', 5, (3.283, -1.543, -0.011)], ['CZ3', 5, (1.715, -3.389, -0.011)]], 'TYR': [['N', 0, (-0.522, 1.362, 0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.524, -0.0, -0.0)], ['CB', 0, (-0.522, -0.776, -1.213)], ['O', 3, (0.627, 1.062, -0.0)], ['CG', 4, (0.607, 1.382, -0.0)], ['CD1', 5, (0.716, 1.195, -0.0)], ['CD2', 5, (0.713, -1.194, -0.001)], ['CE1', 5, (2.107, 1.2, -0.002)], ['CE2', 5, (2.104, -1.201, -0.003)], ['OH', 5, (4.168, -0.002, -0.005)], ['CZ', 5, (2.791, -0.001, -0.003)]], 'VAL': [['N', 0, (-0.494, 1.373, -0.0)], ['CA', 0, (0.0, 0.0, 0.0)], ['C', 0, (1.527, -0.0, -0.0)], ['CB', 0, (-0.533, -0.795, -1.213)], ['O', 3, (0.627, 1.062, -0.0)], ['CG1', 4, (0.54, 1.429, -0.0)], ['CG2', 4, (0.533, -0.776, 1.203)]]}
residue_atoms = {'ALA': ['C', 'CA', 'CB', 'N', 'O'], 'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'], 'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'], 'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'], 'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'], 'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'], 'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'], 'GLY': ['C', 'CA', 'N', 'O'], 'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'], 'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'], 'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'], 'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'], 'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'], 'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'], 'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'], 'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'], 'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'], 'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'N', 'NE1', 'O'], 'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O', 'OH'], 'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']}
residue_atom_renaming_swaps = {'ASP': {'OD1': 'OD2'}, 'GLU': {'OE1': 'OE2'}, 'PHE': {'CD1': 'CD2', 'CE1': 'CE2'}, 'TYR': {'CD1': 'CD2', 'CE1': 'CE2'}}
van_der_waals_radius = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
Bond = collections.namedtuple('Bond', ['atom1_name', 'atom2_name', 'length', 'stddev'])
BondAngle = collections.namedtuple('BondAngle', ['atom1_name', 'atom2_name', 'atom3name', 'angle_rad', 'stddev'])

@functools.lru_cache(maxsize=None)
def load_stereo_chemical_props():
    if False:
        i = 10
        return i + 15
    'Load stereo_chemical_props.txt into a nice structure.\n\n    Load literature values for bond lengths and bond angles and translate\n    bond angles into the length of the opposite edge of the triangle\n    ("residue_virtual_bonds").\n\n    Returns:\n        residue_bonds: Dict that maps resname -> list of Bond tuples.\n        residue_virtual_bonds: Dict that maps resname -> list of Bond tuples.\n        residue_bond_angles: Dict that maps resname -> list of BondAngle tuples.\n    '
    stereo_chemical_props_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stereo_chemical_props.txt')
    with open(stereo_chemical_props_path, 'rt', encoding='utf-8') as f:
        stereo_chemical_props = f.read()
    lines_iter = iter(stereo_chemical_props.splitlines())
    residue_bonds = {}
    next(lines_iter)
    for line in lines_iter:
        if line.strip() == '-':
            break
        (bond, resname, length, stddev) = line.split()
        (atom1, atom2) = bond.split('-')
        if resname not in residue_bonds:
            residue_bonds[resname] = []
        residue_bonds[resname].append(Bond(atom1, atom2, float(length), float(stddev)))
    residue_bonds['UNK'] = []
    residue_bond_angles = {}
    next(lines_iter)
    next(lines_iter)
    for line in lines_iter:
        if line.strip() == '-':
            break
        (bond, resname, angle_degree, stddev_degree) = line.split()
        (atom1, atom2, atom3) = bond.split('-')
        if resname not in residue_bond_angles:
            residue_bond_angles[resname] = []
        residue_bond_angles[resname].append(BondAngle(atom1, atom2, atom3, float(angle_degree) / 180.0 * np.pi, float(stddev_degree) / 180.0 * np.pi))
    residue_bond_angles['UNK'] = []

    def make_bond_key(atom1_name, atom2_name):
        if False:
            print('Hello World!')
        'Unique key to lookup bonds.'
        return '-'.join(sorted([atom1_name, atom2_name]))
    residue_virtual_bonds = {}
    for (resname, bond_angles) in residue_bond_angles.items():
        bond_cache = {}
        for b in residue_bonds[resname]:
            bond_cache[make_bond_key(b.atom1_name, b.atom2_name)] = b
        residue_virtual_bonds[resname] = []
        for ba in bond_angles:
            bond1 = bond_cache[make_bond_key(ba.atom1_name, ba.atom2_name)]
            bond2 = bond_cache[make_bond_key(ba.atom2_name, ba.atom3name)]
            gamma = ba.angle_rad
            length = np.sqrt(bond1.length ** 2 + bond2.length ** 2 - 2 * bond1.length * bond2.length * np.cos(gamma))
            dl_outer = 0.5 / length
            dl_dgamma = 2 * bond1.length * bond2.length * np.sin(gamma) * dl_outer
            dl_db1 = (2 * bond1.length - 2 * bond2.length * np.cos(gamma)) * dl_outer
            dl_db2 = (2 * bond2.length - 2 * bond1.length * np.cos(gamma)) * dl_outer
            stddev = np.sqrt((dl_dgamma * ba.stddev) ** 2 + (dl_db1 * bond1.stddev) ** 2 + (dl_db2 * bond2.stddev) ** 2)
            residue_virtual_bonds[resname].append(Bond(ba.atom1_name, ba.atom3name, length, stddev))
    return (residue_bonds, residue_virtual_bonds, residue_bond_angles)
between_res_bond_length_c_n = [1.329, 1.341]
between_res_bond_length_stddev_c_n = [0.014, 0.016]
between_res_cos_angles_c_n_ca = [-0.5203, 0.0353]
between_res_cos_angles_ca_c_n = [-0.4473, 0.0311]
atom_types = ['N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD', 'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3', 'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2', 'CZ3', 'NZ', 'OXT']
atom_order = {atom_type: i for (i, atom_type) in enumerate(atom_types)}
atom_type_num = len(atom_types)
restype_name_to_atom14_names = {'ALA': ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''], 'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''], 'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''], 'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''], 'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''], 'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''], 'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''], 'GLY': ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''], 'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''], 'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''], 'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''], 'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''], 'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''], 'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''], 'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''], 'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''], 'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''], 'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'], 'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''], 'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''], 'UNK': ['', '', '', '', '', '', '', '', '', '', '', '', '', '']}
restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
restype_order = {restype: i for (i, restype) in enumerate(restypes)}
restype_num = len(restypes)
unk_restype_index = restype_num
restypes_with_x = restypes + ['X']
restype_order_with_x = {restype: i for (i, restype) in enumerate(restypes_with_x)}

def sequence_to_onehot(sequence: str, mapping: Mapping[str, int], map_unknown_to_x: bool=False) -> np.ndarray:
    if False:
        print('Hello World!')
    "Maps the given sequence into a one-hot encoded matrix.\n\n    Args:\n        sequence: An amino acid sequence.\n        mapping: A dictionary mapping amino acids to integers.\n        map_unknown_to_x: If True, any amino acid that is not in the mapping will be\n            mapped to the unknown amino acid 'X'. If the mapping doesn't contain\n            amino acid 'X', an error will be thrown. If False, any amino acid not in\n            the mapping will throw an error.\n\n    Returns:\n        A numpy array of shape (seq_len, num_unique_aas) with one-hot encoding of\n        the sequence.\n\n    Raises:\n        ValueError: If the mapping doesn't contain values from 0 to\n            num_unique_aas - 1 without any gaps.\n    "
    num_entries = max(mapping.values()) + 1
    if sorted(set(mapping.values())) != list(range(num_entries)):
        raise ValueError('The mapping must have values from 0 to num_unique_aas-1 without any gaps. Got: %s' % sorted(mapping.values()))
    one_hot_arr = np.zeros((len(sequence), num_entries), dtype=np.int32)
    for (aa_index, aa_type) in enumerate(sequence):
        if map_unknown_to_x:
            if aa_type.isalpha() and aa_type.isupper():
                aa_id = mapping.get(aa_type, mapping['X'])
            else:
                raise ValueError(f'Invalid character in the sequence: {aa_type}')
        else:
            aa_id = mapping[aa_type]
        one_hot_arr[aa_index, aa_id] = 1
    return one_hot_arr
restype_1to3 = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
restype_3to1 = {v: k for (k, v) in restype_1to3.items()}
unk_restype = 'UNK'
resnames = [restype_1to3[r] for r in restypes] + [unk_restype]
resname_to_idx = {resname: i for (i, resname) in enumerate(resnames)}
HHBLITS_AA_TO_ID = {'A': 0, 'B': 2, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'J': 20, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'O': 20, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'U': 1, 'V': 17, 'W': 18, 'X': 20, 'Y': 19, 'Z': 3, '-': 21}
ID_TO_HHBLITS_AA = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X', 21: '-'}
restypes_with_x_and_gap = restypes + ['X', '-']
MAP_HHBLITS_AATYPE_TO_OUR_AATYPE = tuple((restypes_with_x_and_gap.index(ID_TO_HHBLITS_AA[i]) for i in range(len(restypes_with_x_and_gap))))

def _make_standard_atom_mask() -> np.ndarray:
    if False:
        return 10
    'Returns [num_res_types, num_atom_types] mask array.'
    mask = np.zeros([restype_num + 1, atom_type_num], dtype=np.int32)
    for (restype, restype_letter) in enumerate(restypes):
        restype_name = restype_1to3[restype_letter]
        atom_names = residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = atom_order[atom_name]
            mask[restype, atom_type] = 1
    return mask
STANDARD_ATOM_MASK = _make_standard_atom_mask()

def chi_angle_atom(atom_index: int) -> np.ndarray:
    if False:
        while True:
            i = 10
    'Define chi-angle rigid groups via one-hot representations.'
    chi_angles_index = {}
    one_hots = []
    for (k, v) in chi_angles_atoms.items():
        indices = [atom_types.index(s[atom_index]) for s in v]
        indices.extend([-1] * (4 - len(indices)))
        chi_angles_index[k] = indices
    for r in restypes:
        res3 = restype_1to3[r]
        one_hot = np.eye(atom_type_num)[chi_angles_index[res3]]
        one_hots.append(one_hot)
    one_hots.append(np.zeros([4, atom_type_num]))
    one_hot = np.stack(one_hots, axis=0)
    one_hot = np.transpose(one_hot, [0, 2, 1])
    return one_hot
chi_atom_1_one_hot = chi_angle_atom(1)
chi_atom_2_one_hot = chi_angle_atom(2)
chi_angles_atom_indices = [chi_angles_atoms[restype_1to3[r]] for r in restypes]
chi_angles_atom_indices = tree_map(lambda n: atom_order[n], chi_angles_atom_indices, leaf_type=str)
chi_angles_atom_indices = np.array([chi_atoms + [[0, 0, 0, 0]] * (4 - len(chi_atoms)) for chi_atoms in chi_angles_atom_indices])
chi_groups_for_atom = collections.defaultdict(list)
for (res_name, chi_angle_atoms_for_res) in chi_angles_atoms.items():
    for (chi_group_i, chi_group) in enumerate(chi_angle_atoms_for_res):
        for (atom_i, atom) in enumerate(chi_group):
            chi_groups_for_atom[res_name, atom].append((chi_group_i, atom_i))
chi_groups_for_atom = dict(chi_groups_for_atom)

def _make_rigid_transformation_4x4(ex, ey, translation):
    if False:
        print('Hello World!')
    'Create a rigid 4x4 transformation matrix from two axes and transl.'
    ex_normalized = ex / np.linalg.norm(ex)
    ey_normalized = ey - np.dot(ey, ex_normalized) * ex_normalized
    ey_normalized /= np.linalg.norm(ey_normalized)
    eznorm = np.cross(ex_normalized, ey_normalized)
    m = np.stack([ex_normalized, ey_normalized, eznorm, translation]).transpose()
    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)
    return m
restype_atom37_to_rigid_group = np.zeros([21, 37], dtype=np.int_)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)
restype_atom37_rigid_group_positions = np.zeros([21, 37, 3], dtype=np.float32)
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=np.int_)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)

def _make_rigid_group_constants():
    if False:
        for i in range(10):
            print('nop')
    'Fill the arrays above.'
    for (restype, restype_letter) in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        for (atomname, group_idx, atom_position) in rigid_group_atom_positions[resname]:
            atomtype = atom_order[atomname]
            restype_atom37_to_rigid_group[restype, atomtype] = group_idx
            restype_atom37_mask[restype, atomtype] = 1
            restype_atom37_rigid_group_positions[restype, atomtype, :] = atom_position
            atom14idx = restype_name_to_atom14_names[resname].index(atomname)
            restype_atom14_to_rigid_group[restype, atom14idx] = group_idx
            restype_atom14_mask[restype, atom14idx] = 1
            restype_atom14_rigid_group_positions[restype, atom14idx, :] = atom_position
    for (restype, restype_letter) in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_positions = {name: np.array(pos) for (name, _, pos) in rigid_group_atom_positions[resname]}
        restype_rigid_group_default_frame[restype, 0, :, :] = np.eye(4)
        restype_rigid_group_default_frame[restype, 1, :, :] = np.eye(4)
        mat = _make_rigid_transformation_4x4(ex=atom_positions['N'] - atom_positions['CA'], ey=np.array([1.0, 0.0, 0.0]), translation=atom_positions['N'])
        restype_rigid_group_default_frame[restype, 2, :, :] = mat
        mat = _make_rigid_transformation_4x4(ex=atom_positions['C'] - atom_positions['CA'], ey=atom_positions['CA'] - atom_positions['N'], translation=atom_positions['C'])
        restype_rigid_group_default_frame[restype, 3, :, :] = mat
        if chi_angles_mask[restype][0]:
            base_atom_names = chi_angles_atoms[resname][0]
            base_atom_positions = [atom_positions[name] for name in base_atom_names]
            mat = _make_rigid_transformation_4x4(ex=base_atom_positions[2] - base_atom_positions[1], ey=base_atom_positions[0] - base_atom_positions[1], translation=base_atom_positions[2])
            restype_rigid_group_default_frame[restype, 4, :, :] = mat
        for chi_idx in range(1, 4):
            if chi_angles_mask[restype][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[resname][chi_idx][2]
                axis_end_atom_position = atom_positions[axis_end_atom_name]
                mat = _make_rigid_transformation_4x4(ex=axis_end_atom_position, ey=np.array([-1.0, 0.0, 0.0]), translation=axis_end_atom_position)
                restype_rigid_group_default_frame[restype, 4 + chi_idx, :, :] = mat
_make_rigid_group_constants()

def make_atom14_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=15):
    if False:
        i = 10
        return i + 15
    'compute upper and lower bounds for bonds to assess violations.'
    restype_atom14_bond_lower_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_upper_bound = np.zeros([21, 14, 14], np.float32)
    restype_atom14_bond_stddev = np.zeros([21, 14, 14], np.float32)
    (residue_bonds, residue_virtual_bonds, _) = load_stereo_chemical_props()
    for (restype, restype_letter) in enumerate(restypes):
        resname = restype_1to3[restype_letter]
        atom_list = restype_name_to_atom14_names[resname]
        for (atom1_idx, atom1_name) in enumerate(atom_list):
            if not atom1_name:
                continue
            atom1_radius = van_der_waals_radius[atom1_name[0]]
            for (atom2_idx, atom2_name) in enumerate(atom_list):
                if not atom2_name or atom1_idx == atom2_idx:
                    continue
                atom2_radius = van_der_waals_radius[atom2_name[0]]
                lower = atom1_radius + atom2_radius - overlap_tolerance
                upper = 10000000000.0
                restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
                restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
                restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
                restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
        for b in residue_bonds[resname] + residue_virtual_bonds[resname]:
            atom1_idx = atom_list.index(b.atom1_name)
            atom2_idx = atom_list.index(b.atom2_name)
            lower = b.length - bond_length_tolerance_factor * b.stddev
            upper = b.length + bond_length_tolerance_factor * b.stddev
            restype_atom14_bond_lower_bound[restype, atom1_idx, atom2_idx] = lower
            restype_atom14_bond_lower_bound[restype, atom2_idx, atom1_idx] = lower
            restype_atom14_bond_upper_bound[restype, atom1_idx, atom2_idx] = upper
            restype_atom14_bond_upper_bound[restype, atom2_idx, atom1_idx] = upper
            restype_atom14_bond_stddev[restype, atom1_idx, atom2_idx] = b.stddev
            restype_atom14_bond_stddev[restype, atom2_idx, atom1_idx] = b.stddev
    return {'lower_bound': restype_atom14_bond_lower_bound, 'upper_bound': restype_atom14_bond_upper_bound, 'stddev': restype_atom14_bond_stddev}

def _make_atom14_and_atom37_constants():
    if False:
        return 10
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []
    for rt in restypes:
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]
        restype_atom14_to_atom37.append([atom_order[name] if name else 0 for name in atom_names])
        atom_name_to_idx14 = {name: i for (i, name) in enumerate(atom_names)}
        restype_atom37_to_atom14.append([atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0 for name in atom_types])
        restype_atom14_mask.append([1.0 if name else 0.0 for name in atom_names])
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)
    restype_atom14_to_atom37 = np.array(restype_atom14_to_atom37, dtype=np.int32)
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14, dtype=np.int32)
    restype_atom14_mask = np.array(restype_atom14_mask, dtype=np.float32)
    return (restype_atom14_to_atom37, restype_atom37_to_atom14, restype_atom14_mask)
(restype_atom14_to_atom37, restype_atom37_to_atom14, restype_atom14_mask) = _make_atom14_and_atom37_constants()

def _make_renaming_matrices():
    if False:
        while True:
            i = 10
    restype_3 = [restype_1to3[res] for res in restypes]
    restype_3 += ['UNK']
    all_matrices = {res: np.eye(14) for res in restype_3}
    for (resname, swap) in residue_atom_renaming_swaps.items():
        correspondences = np.arange(14)
        for (source_atom_swap, target_atom_swap) in swap.items():
            source_index = restype_name_to_atom14_names[resname].index(source_atom_swap)
            target_index = restype_name_to_atom14_names[resname].index(target_atom_swap)
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = np.zeros((14, 14))
            for (index, correspondence) in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix
    renaming_matrices = np.stack([all_matrices[restype] for restype in restype_3])
    return renaming_matrices
renaming_matrices = _make_renaming_matrices()

def _make_atom14_is_ambiguous():
    if False:
        while True:
            i = 10
    restype_atom14_is_ambiguous = np.zeros((21, 14))
    for (resname, swap) in residue_atom_renaming_swaps.items():
        for (atom_name1, atom_name2) in swap.items():
            restype = restype_order[restype_3to1[resname]]
            atom_idx1 = restype_name_to_atom14_names[resname].index(atom_name1)
            atom_idx2 = restype_name_to_atom14_names[resname].index(atom_name2)
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1
    return restype_atom14_is_ambiguous
restype_atom14_is_ambiguous = _make_atom14_is_ambiguous()

def get_chi_atom_indices():
    if False:
        return 10
    'Returns atom indices needed to compute chi angles for all residue types.\n\n    Returns:\n      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are\n      in the order specified in restypes + unknown residue type\n      at the end. For chi angles which are not defined on the residue, the\n      positions indices are by default set to 0.\n    '
    chi_atom_indices = []
    for residue_name in restypes:
        residue_name = restype_1to3[residue_name]
        residue_chi_angles = chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])
        chi_atom_indices.append(atom_indices)
    chi_atom_indices.append([[0, 0, 0, 0]] * 4)
    return chi_atom_indices
chi_atom_indices = get_chi_atom_indices()