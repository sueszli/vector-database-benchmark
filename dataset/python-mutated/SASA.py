"""Calculation of solvent accessible surface areas for Bio.PDB entities.

Uses the "rolling ball" algorithm developed by Shrake & Rupley algorithm,
which uses a sphere (of equal radius to a solvent molecule) to probe the
surface of the molecule.

Reference:
    Shrake, A; Rupley, JA. (1973). J Mol Biol
    "Environment and exposure to solvent of protein atoms. Lysozyme and insulin".
"""
import collections
import math
import numpy as np
from Bio.PDB.kdtrees import KDTree
__all__ = ['ShrakeRupley']
_ENTITY_HIERARCHY = {'A': 0, 'R': 1, 'C': 2, 'M': 3, 'S': 4}
ATOMIC_RADII = collections.defaultdict(lambda : 2.0)
ATOMIC_RADII.update({'H': 1.2, 'HE': 1.4, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'NA': 2.27, 'MG': 1.73, 'P': 1.8, 'S': 1.8, 'CL': 1.75, 'K': 2.75, 'CA': 2.31, 'NI': 1.63, 'CU': 1.4, 'ZN': 1.39, 'SE': 1.9, 'BR': 1.85, 'CD': 1.58, 'I': 1.98, 'HG': 1.55})

class ShrakeRupley:
    """Calculates SASAs using the Shrake-Rupley algorithm."""

    def __init__(self, probe_radius=1.4, n_points=100, radii_dict=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.\n\n        :param probe_radius: radius of the probe in A. Default is 1.40, roughly\n            the radius of a water molecule.\n        :type probe_radius: float\n\n        :param n_points: resolution of the surface of each atom. Default is 100.\n            A higher number of points results in more precise measurements, but\n            slows down the calculation.\n        :type n_points: int\n\n        :param radii_dict: user-provided dictionary of atomic radii to use in\n            the calculation. Values will replace/complement those in the\n            default ATOMIC_RADII dictionary.\n        :type radii_dict: dict\n\n        >>> sr = ShrakeRupley()\n        >>> sr = ShrakeRupley(n_points=960)\n        >>> sr = ShrakeRupley(radii_dict={"O": 3.1415})\n        '
        if probe_radius <= 0.0:
            raise ValueError(f'Probe radius must be a positive number: {probe_radius} <= 0')
        self.probe_radius = float(probe_radius)
        if n_points < 1:
            raise ValueError(f'Number of sphere points must be larger than 1: {n_points}')
        self.n_points = n_points
        self.radii_dict = ATOMIC_RADII.copy()
        if radii_dict is not None:
            self.radii_dict.update(radii_dict)
        self._sphere = self._compute_sphere()

    def _compute_sphere(self):
        if False:
            i = 10
            return i + 15
        "Return the 3D coordinates of n points on a sphere.\n\n        Uses the golden spiral algorithm to place points 'evenly' on the sphere\n        surface. We compute this once and then move the sphere to the centroid\n        of each atom as we compute the ASAs.\n        "
        n = self.n_points
        dl = np.pi * (3 - 5 ** 0.5)
        dz = 2.0 / n
        longitude = 0
        z = 1 - dz / 2
        coords = np.zeros((n, 3), dtype=np.float32)
        for k in range(n):
            r = (1 - z * z) ** 0.5
            coords[k, 0] = math.cos(longitude) * r
            coords[k, 1] = math.sin(longitude) * r
            coords[k, 2] = z
            z -= dz
            longitude += dl
        return coords

    def compute(self, entity, level='A'):
        if False:
            while True:
                i = 10
        'Calculate surface accessibility surface area for an entity.\n\n        The resulting atomic surface accessibility values are attached to the\n        .sasa attribute of each entity (or atom), depending on the level. For\n        example, if level="R", all residues will have a .sasa attribute. Atoms\n        will always be assigned a .sasa attribute with their individual values.\n\n        :param entity: input entity.\n        :type entity: Bio.PDB.Entity, e.g. Residue, Chain, ...\n\n        :param level: the level at which ASA values are assigned, which can be\n            one of "A" (Atom), "R" (Residue), "C" (Chain), "M" (Model), or\n            "S" (Structure). The ASA value of an entity is the sum of all ASA\n            values of its children. Defaults to "A".\n        :type entity: Bio.PDB.Entity\n\n        >>> from Bio.PDB import PDBParser\n        >>> from Bio.PDB.SASA import ShrakeRupley\n        >>> p = PDBParser(QUIET=1)\n        >>> # This assumes you have a local copy of 1LCD.pdb in a directory called "PDB"\n        >>> struct = p.get_structure("1LCD", "PDB/1LCD.pdb")\n        >>> sr = ShrakeRupley()\n        >>> sr.compute(struct, level="S")\n        >>> print(round(struct.sasa, 2))\n        7053.43\n        >>> print(round(struct[0]["A"][11]["OE1"].sasa, 2))\n        9.64\n        '
        is_valid = hasattr(entity, 'level') and entity.level in {'R', 'C', 'M', 'S'}
        if not is_valid:
            raise ValueError(f"Invalid entity type '{type(entity)}'. Must be Residue, Chain, Model, or Structure")
        if level not in _ENTITY_HIERARCHY:
            raise ValueError(f"Invalid level '{level}'. Must be A, R, C, M, or S.")
        elif _ENTITY_HIERARCHY[level] > _ENTITY_HIERARCHY[entity.level]:
            raise ValueError(f"Level '{level}' must be equal or smaller than input entity: {entity.level}")
        atoms = list(entity.get_atoms())
        n_atoms = len(atoms)
        if not n_atoms:
            raise ValueError('Entity has no child atoms.')
        coords = np.array([a.coord for a in atoms], dtype=np.float64)
        kdt = KDTree(coords, 10)
        radii_dict = self.radii_dict
        radii = np.array([radii_dict[a.element] for a in atoms], dtype=np.float64)
        radii += self.probe_radius
        twice_maxradii = np.max(radii) * 2
        asa_array = np.zeros((n_atoms, 1), dtype=np.int64)
        ptset = set(range(self.n_points))
        for i in range(n_atoms):
            r_i = radii[i]
            s_on_i = np.array(self._sphere, copy=True) * r_i + coords[i]
            available_set = ptset.copy()
            kdt_sphere = KDTree(s_on_i, 10)
            for jj in kdt.search(coords[i], twice_maxradii):
                j = jj.index
                if i == j:
                    continue
                if jj.radius < r_i + radii[j]:
                    available_set -= {pt.index for pt in kdt_sphere.search(coords[j], radii[j])}
            asa_array[i] = len(available_set)
        f = radii * radii * (4 * np.pi / self.n_points)
        asa_array = asa_array * f[:, np.newaxis]
        for (i, atom) in enumerate(atoms):
            atom.sasa = asa_array[i, 0]
        if level != 'A':
            entities = set(atoms)
            target = _ENTITY_HIERARCHY[level]
            for _ in range(target):
                entities = {e.parent for e in entities}
            atomdict = {a.full_id: idx for (idx, a) in enumerate(atoms)}
            for e in entities:
                e_atoms = [atomdict[a.full_id] for a in e.get_atoms()]
                e.sasa = asa_array[e_atoms].sum()