"""Half-sphere exposure and coordination number calculation."""
import warnings
from math import pi
from Bio.PDB.AbstractPropertyMap import AbstractPropertyMap
from Bio.PDB.Polypeptide import CaPPBuilder, is_aa
from Bio.PDB.vectors import rotaxis

class _AbstractHSExposure(AbstractPropertyMap):
    """Abstract class to calculate Half-Sphere Exposure (HSE).

    The HSE can be calculated based on the CA-CB vector, or the pseudo CB-CA
    vector based on three consecutive CA atoms. This is done by two separate
    subclasses.
    """

    def __init__(self, model, radius, offset, hse_up_key, hse_down_key, angle_key=None):
        if False:
            while True:
                i = 10
        'Initialize class.\n\n        :param model: model\n        :type model: L{Model}\n\n        :param radius: HSE radius\n        :type radius: float\n\n        :param offset: number of flanking residues that are ignored in the\n                       calculation of the number of neighbors\n        :type offset: int\n\n        :param hse_up_key: key used to store HSEup in the entity.xtra attribute\n        :type hse_up_key: string\n\n        :param hse_down_key: key used to store HSEdown in the entity.xtra attribute\n        :type hse_down_key: string\n\n        :param angle_key: key used to store the angle between CA-CB and CA-pCB in\n                          the entity.xtra attribute\n        :type angle_key: string\n        '
        assert offset >= 0
        self.ca_cb_list = []
        ppb = CaPPBuilder()
        ppl = ppb.build_peptides(model)
        hse_map = {}
        hse_list = []
        hse_keys = []
        for pp1 in ppl:
            for i in range(len(pp1)):
                if i == 0:
                    r1 = None
                else:
                    r1 = pp1[i - 1]
                r2 = pp1[i]
                if i == len(pp1) - 1:
                    r3 = None
                else:
                    r3 = pp1[i + 1]
                result = self._get_cb(r1, r2, r3)
                if result is None:
                    continue
                (pcb, angle) = result
                hse_u = 0
                hse_d = 0
                ca2 = r2['CA'].get_vector()
                for pp2 in ppl:
                    for j in range(len(pp2)):
                        if pp1 is pp2 and abs(i - j) <= offset:
                            continue
                        ro = pp2[j]
                        if not is_aa(ro) or not ro.has_id('CA'):
                            continue
                        cao = ro['CA'].get_vector()
                        d = cao - ca2
                        if d.norm() < radius:
                            if d.angle(pcb) < pi / 2:
                                hse_u += 1
                            else:
                                hse_d += 1
                res_id = r2.get_id()
                chain_id = r2.get_parent().get_id()
                hse_map[chain_id, res_id] = (hse_u, hse_d, angle)
                hse_list.append((r2, (hse_u, hse_d, angle)))
                hse_keys.append((chain_id, res_id))
                r2.xtra[hse_up_key] = hse_u
                r2.xtra[hse_down_key] = hse_d
                if angle_key:
                    r2.xtra[angle_key] = angle
        AbstractPropertyMap.__init__(self, hse_map, hse_keys, hse_list)

    def _get_cb(self, r1, r2, r3):
        if False:
            return 10
        return NotImplemented

    def _get_gly_cb_vector(self, residue):
        if False:
            for i in range(10):
                print('nop')
        'Return a pseudo CB vector for a Gly residue (PRIVATE).\n\n        The pseudoCB vector is centered at the origin.\n\n        CB coord=N coord rotated over -120 degrees\n        along the CA-C axis.\n        '
        try:
            n_v = residue['N'].get_vector()
            c_v = residue['C'].get_vector()
            ca_v = residue['CA'].get_vector()
        except Exception:
            return None
        n_v = n_v - ca_v
        c_v = c_v - ca_v
        rot = rotaxis(-pi * 120.0 / 180.0, c_v)
        cb_at_origin_v = n_v.left_multiply(rot)
        cb_v = cb_at_origin_v + ca_v
        self.ca_cb_list.append((ca_v, cb_v))
        return cb_at_origin_v

class HSExposureCA(_AbstractHSExposure):
    """Class to calculate HSE based on the approximate CA-CB vectors.

    Uses three consecutive CA positions.
    """

    def __init__(self, model, radius=12, offset=0):
        if False:
            return 10
        'Initialize class.\n\n        :param model: the model that contains the residues\n        :type model: L{Model}\n\n        :param radius: radius of the sphere (centred at the CA atom)\n        :type radius: float\n\n        :param offset: number of flanking residues that are ignored\n                       in the calculation of the number of neighbors\n        :type offset: int\n        '
        _AbstractHSExposure.__init__(self, model, radius, offset, 'EXP_HSE_A_U', 'EXP_HSE_A_D', 'EXP_CB_PCB_ANGLE')

    def _get_cb(self, r1, r2, r3):
        if False:
            print('Hello World!')
        'Calculate approx CA-CB direction (PRIVATE).\n\n        Calculate the approximate CA-CB direction for a central\n        CA atom based on the two flanking CA positions, and the angle\n        with the real CA-CB vector.\n\n        The CA-CB vector is centered at the origin.\n\n        :param r1, r2, r3: three consecutive residues\n        :type r1, r2, r3: L{Residue}\n        '
        if r1 is None or r3 is None:
            return None
        try:
            ca1 = r1['CA'].get_vector()
            ca2 = r2['CA'].get_vector()
            ca3 = r3['CA'].get_vector()
        except Exception:
            return None
        d1 = ca2 - ca1
        d3 = ca2 - ca3
        d1.normalize()
        d3.normalize()
        b = d1 + d3
        b.normalize()
        self.ca_cb_list.append((ca2, b + ca2))
        if r2.has_id('CB'):
            cb = r2['CB'].get_vector()
            cb_ca = cb - ca2
            cb_ca.normalize()
            angle = cb_ca.angle(b)
        elif r2.get_resname() == 'GLY':
            cb_ca = self._get_gly_cb_vector(r2)
            if cb_ca is None:
                angle = None
            else:
                angle = cb_ca.angle(b)
        else:
            angle = None
        return (b, angle)

    def pcb_vectors_pymol(self, filename='hs_exp.py'):
        if False:
            i = 10
            return i + 15
        'Write PyMol script for visualization.\n\n        Write a PyMol script that visualizes the pseudo CB-CA directions\n        at the CA coordinates.\n\n        :param filename: the name of the pymol script file\n        :type filename: string\n        '
        if not self.ca_cb_list:
            warnings.warn('Nothing to draw.', RuntimeWarning)
            return
        with open(filename, 'w') as fp:
            fp.write('from pymol.cgo import *\n')
            fp.write('from pymol import cmd\n')
            fp.write('obj=[\n')
            fp.write('BEGIN, LINES,\n')
            fp.write(f'COLOR, {1.0:.2f}, {1.0:.2f}, {1.0:.2f},\n')
            for (ca, cb) in self.ca_cb_list:
                (x, y, z) = ca.get_array()
                fp.write(f'VERTEX, {x:.2f}, {y:.2f}, {z:.2f},\n')
                (x, y, z) = cb.get_array()
                fp.write(f'VERTEX, {x:.2f}, {y:.2f}, {z:.2f},\n')
            fp.write('END]\n')
            fp.write("cmd.load_cgo(obj, 'HS')\n")

class HSExposureCB(_AbstractHSExposure):
    """Class to calculate HSE based on the real CA-CB vectors."""

    def __init__(self, model, radius=12, offset=0):
        if False:
            while True:
                i = 10
        'Initialize class.\n\n        :param model: the model that contains the residues\n        :type model: L{Model}\n\n        :param radius: radius of the sphere (centred at the CA atom)\n        :type radius: float\n\n        :param offset: number of flanking residues that are ignored\n                       in the calculation of the number of neighbors\n        :type offset: int\n        '
        _AbstractHSExposure.__init__(self, model, radius, offset, 'EXP_HSE_B_U', 'EXP_HSE_B_D')

    def _get_cb(self, r1, r2, r3):
        if False:
            return 10
        'Calculate CB-CA vector (PRIVATE).\n\n        :param r1, r2, r3: three consecutive residues (only r2 is used)\n        :type r1, r2, r3: L{Residue}\n        '
        if r2.get_resname() == 'GLY':
            return (self._get_gly_cb_vector(r2), 0.0)
        elif r2.has_id('CB') and r2.has_id('CA'):
            vcb = r2['CB'].get_vector()
            vca = r2['CA'].get_vector()
            return (vcb - vca, 0.0)
        return None

class ExposureCN(AbstractPropertyMap):
    """Residue exposure as number of CA atoms around its CA atom."""

    def __init__(self, model, radius=12.0, offset=0):
        if False:
            print('Hello World!')
        "Initialize class.\n\n        A residue's exposure is defined as the number of CA atoms around\n        that residue's CA atom. A dictionary is returned that uses a L{Residue}\n        object as key, and the residue exposure as corresponding value.\n\n        :param model: the model that contains the residues\n        :type model: L{Model}\n\n        :param radius: radius of the sphere (centred at the CA atom)\n        :type radius: float\n\n        :param offset: number of flanking residues that are ignored in\n                       the calculation of the number of neighbors\n        :type offset: int\n\n        "
        assert offset >= 0
        ppb = CaPPBuilder()
        ppl = ppb.build_peptides(model)
        fs_map = {}
        fs_list = []
        fs_keys = []
        for pp1 in ppl:
            for i in range(len(pp1)):
                fs = 0
                r1 = pp1[i]
                if not is_aa(r1) or not r1.has_id('CA'):
                    continue
                ca1 = r1['CA']
                for pp2 in ppl:
                    for j in range(len(pp2)):
                        if pp1 is pp2 and abs(i - j) <= offset:
                            continue
                        r2 = pp2[j]
                        if not is_aa(r2) or not r2.has_id('CA'):
                            continue
                        ca2 = r2['CA']
                        d = ca2 - ca1
                        if d < radius:
                            fs += 1
                res_id = r1.get_id()
                chain_id = r1.get_parent().get_id()
                fs_map[chain_id, res_id] = fs
                fs_list.append((r1, fs))
                fs_keys.append((chain_id, res_id))
                r1.xtra['EXP_CN'] = fs
        AbstractPropertyMap.__init__(self, fs_map, fs_keys, fs_list)