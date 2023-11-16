"""Use the DSSP program to calculate secondary structure and accessibility.

You need to have a working version of DSSP (and a license, free for academic
use) in order to use this. For DSSP, see https://swift.cmbi.umcn.nl/gv/dssp/.

The following Accessible surface area (ASA) values can be used, defaulting
to the Sander and Rost values:

    Ahmad
        Ahmad et al. 2003 https://doi.org/10.1002/prot.10328
    Miller
        Miller et al. 1987 https://doi.org/10.1016/0022-2836(87)90038-6
    Sander
        Sander and Rost 1994 https://doi.org/10.1002/prot.340200303
    Wilke
        Tien et al. 2013 https://doi.org/10.1371/journal.pone.0080635

The DSSP codes for secondary structure used here are:

    =====     ====
    Code      Structure
    =====     ====
     H        Alpha helix (4-12)
     B        Isolated beta-bridge residue
     E        Strand
     G        3-10 helix
     I        Pi helix
     T        Turn
     S        Bend
     \\-       None
    =====     ====

Usage
-----
The DSSP class can be used to run DSSP on a PDB or mmCIF file, and provides a
handle to the DSSP secondary structure and accessibility.

**Note** that DSSP can only handle one model, and will only run
calculations on the first model in the provided PDB file.

Examples
--------
Typical use::

    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    p = PDBParser()
    structure = p.get_structure("1MOT", "/local-pdb/1mot.pdb")
    model = structure[0]
    dssp = DSSP(model, "/local-pdb/1mot.pdb")

Note that the recent DSSP executable from the DSSP-2 package was
renamed from ``dssp`` to ``mkdssp``. If using a recent DSSP release,
you may need to provide the name of your DSSP executable::

    dssp = DSSP(model, '/local-pdb/1mot.pdb', dssp='mkdssp')

DSSP data is accessed by a tuple - (chain id, residue id)::

    a_key = list(dssp.keys())[2]
    dssp[a_key]

The dssp data returned for a single residue is a tuple in the form:

    ============ ===
    Tuple Index  Value
    ============ ===
    0            DSSP index
    1            Amino acid
    2            Secondary structure
    3            Relative ASA
    4            Phi
    5            Psi
    6            NH-->O_1_relidx
    7            NH-->O_1_energy
    8            O-->NH_1_relidx
    9            O-->NH_1_energy
    10           NH-->O_2_relidx
    11           NH-->O_2_energy
    12           O-->NH_2_relidx
    13           O-->NH_2_energy
    ============ ===

"""
import re
import os
from io import StringIO
import subprocess
import warnings
from Bio.PDB.AbstractPropertyMap import AbstractResiduePropertyMap
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.PDBParser import PDBParser
from Bio.Data.PDBData import protein_letters_3to1, residue_sasa_scales
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
_dssp_cys = re.compile('[a-z]')
residue_max_acc = residue_sasa_scales

def version(version_string):
    if False:
        return 10
    'Parse semantic version scheme for easy comparison.'
    return tuple(map(int, version_string.split('.')))

def ss_to_index(ss):
    if False:
        for i in range(10):
            print('nop')
    'Secondary structure symbol to index.\n\n    H=0\n    E=1\n    C=2\n    '
    if ss == 'H':
        return 0
    if ss == 'E':
        return 1
    if ss == 'C':
        return 2
    assert 0

def dssp_dict_from_pdb_file(in_file, DSSP='dssp', dssp_version='3.9.9'):
    if False:
        while True:
            i = 10
    'Create a DSSP dictionary from a PDB file.\n\n    Parameters\n    ----------\n    in_file : string\n        pdb file\n\n    DSSP : string\n        DSSP executable (argument to subprocess)\n\n    dssp_version : string\n        Version of DSSP excutable\n\n    Returns\n    -------\n    (out_dict, keys) : tuple\n        a dictionary that maps (chainid, resid) to\n        amino acid type, secondary structure code and\n        accessibility.\n\n    Examples\n    --------\n    How dssp_dict_from_pdb_file could be used::\n\n        from Bio.PDB.DSSP import dssp_dict_from_pdb_file\n        dssp_tuple = dssp_dict_from_pdb_file("/local-pdb/1fat.pdb")\n        dssp_dict = dssp_tuple[0]\n        print(dssp_dict[\'A\',(\' \', 1, \' \')])\n\n    '
    try:
        if version(dssp_version) < version('4.0.0'):
            DSSP_cmd = [DSSP, in_file]
        else:
            DSSP_cmd = [DSSP, '--output-format=dssp', in_file]
        p = subprocess.Popen(DSSP_cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        if DSSP == 'mkdssp':
            raise
        if version(dssp_version) < version('4.0.0'):
            DSSP_cmd = ['mkdssp', in_file]
        else:
            DSSP_cmd = ['mkdssp', '--output-format=dssp', in_file]
        p = subprocess.Popen(DSSP_cmd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = p.communicate()
    if err.strip():
        warnings.warn(err)
        if not out.strip():
            raise Exception('DSSP failed to produce an output')
    (out_dict, keys) = _make_dssp_dict(StringIO(out))
    return (out_dict, keys)

def make_dssp_dict(filename):
    if False:
        i = 10
        return i + 15
    'DSSP dictionary mapping identifiers to properties.\n\n    Return a DSSP dictionary that maps (chainid, resid) to\n    aa, ss and accessibility, from a DSSP file.\n\n    Parameters\n    ----------\n    filename : string\n        the DSSP output file\n\n    '
    with open(filename) as handle:
        return _make_dssp_dict(handle)

def _make_dssp_dict(handle):
    if False:
        print('Hello World!')
    'Return a DSSP dictionary, used by mask_dssp_dict (PRIVATE).\n\n    DSSP dictionary maps (chainid, resid) to an amino acid,\n    secondary structure symbol, solvent accessibility value, and hydrogen bond\n    information (relative dssp indices and hydrogen bond energies) from an open\n    DSSP file object.\n\n    Parameters\n    ----------\n    handle : file\n        the open DSSP output file handle\n\n    '
    dssp = {}
    start = 0
    keys = []
    for line in handle:
        sl = line.split()
        if len(sl) < 2:
            continue
        if sl[1] == 'RESIDUE':
            start = 1
            continue
        if not start:
            continue
        if line[9] == ' ':
            continue
        dssp_index = int(line[:5])
        resseq = int(line[5:10])
        icode = line[10]
        chainid = line[11]
        aa = line[13]
        ss = line[16]
        if ss == ' ':
            ss = '-'
        try:
            NH_O_1_relidx = int(line[38:45])
            NH_O_1_energy = float(line[46:50])
            O_NH_1_relidx = int(line[50:56])
            O_NH_1_energy = float(line[57:61])
            NH_O_2_relidx = int(line[61:67])
            NH_O_2_energy = float(line[68:72])
            O_NH_2_relidx = int(line[72:78])
            O_NH_2_energy = float(line[79:83])
            acc = int(line[34:38])
            phi = float(line[103:109])
            psi = float(line[109:115])
        except ValueError as exc:
            if line[34] != ' ':
                shift = line[34:].find(' ')
                NH_O_1_relidx = int(line[38 + shift:45 + shift])
                NH_O_1_energy = float(line[46 + shift:50 + shift])
                O_NH_1_relidx = int(line[50 + shift:56 + shift])
                O_NH_1_energy = float(line[57 + shift:61 + shift])
                NH_O_2_relidx = int(line[61 + shift:67 + shift])
                NH_O_2_energy = float(line[68 + shift:72 + shift])
                O_NH_2_relidx = int(line[72 + shift:78 + shift])
                O_NH_2_energy = float(line[79 + shift:83 + shift])
                acc = int(line[34 + shift:38 + shift])
                phi = float(line[103 + shift:109 + shift])
                psi = float(line[109 + shift:115 + shift])
            else:
                raise ValueError(exc) from None
        res_id = (' ', resseq, icode)
        dssp[chainid, res_id] = (aa, ss, acc, phi, psi, dssp_index, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
        keys.append((chainid, res_id))
    return (dssp, keys)

class DSSP(AbstractResiduePropertyMap):
    """Run DSSP and parse secondary structure and accessibility.

    Run DSSP on a PDB/mmCIF file, and provide a handle to the
    DSSP secondary structure and accessibility.

    **Note** that DSSP can only handle one model.

    Examples
    --------
    How DSSP could be used::

        from Bio.PDB import PDBParser
        from Bio.PDB.DSSP import DSSP
        p = PDBParser()
        structure = p.get_structure("1MOT", "/local-pdb/1mot.pdb")
        model = structure[0]
        dssp = DSSP(model, "/local-pdb/1mot.pdb")
        # DSSP data is accessed by a tuple (chain_id, res_id)
        a_key = list(dssp.keys())[2]
        # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
        # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
        # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
        dssp[a_key]

    """

    def __init__(self, model, in_file, dssp='dssp', acc_array='Sander', file_type=''):
        if False:
            print('Hello World!')
        'Create a DSSP object.\n\n        Parameters\n        ----------\n        model : Model\n            The first model of the structure\n        in_file : string\n            Either a PDB file or a DSSP file.\n        dssp : string\n            The dssp executable (ie. the argument to subprocess)\n        acc_array : string\n            Accessible surface area (ASA) from either Miller et al. (1987),\n            Sander & Rost (1994), Wilke: Tien et al. 2013, or Ahmad et al.\n            (2003) as string Sander/Wilke/Miller/Ahmad. Defaults to Sander.\n        file_type: string\n            File type switch: either PDB, MMCIF or DSSP. Inferred from the\n            file extension by default.\n\n        '
        self.residue_max_acc = residue_max_acc[acc_array]
        if file_type == '':
            file_type = os.path.splitext(in_file)[1][1:]
        file_type = file_type.upper()
        if file_type == 'CIF':
            file_type = 'MMCIF'
        assert file_type in ['PDB', 'MMCIF', 'DSSP'], 'File type must be PDB, mmCIF or DSSP'
        if file_type == 'PDB' or file_type == 'MMCIF':
            try:
                version_string = subprocess.check_output([dssp, '--version'], universal_newlines=True)
                dssp_version = re.search('\\s*([\\d.]+)', version_string).group(1)
                (dssp_dict, dssp_keys) = dssp_dict_from_pdb_file(in_file, dssp, dssp_version)
            except FileNotFoundError:
                if dssp == 'dssp':
                    dssp = 'mkdssp'
                elif dssp == 'mkdssp':
                    dssp = 'dssp'
                else:
                    raise
                version_string = subprocess.check_output([dssp, '--version'], universal_newlines=True)
                dssp_version = re.search('\\s*([\\d.]+)', version_string).group(1)
                (dssp_dict, dssp_keys) = dssp_dict_from_pdb_file(in_file, dssp, dssp_version)
        elif file_type == 'DSSP':
            (dssp_dict, dssp_keys) = make_dssp_dict(in_file)
        dssp_map = {}
        dssp_list = []

        def resid2code(res_id):
            if False:
                for i in range(10):
                    print('nop')
            "Serialize a residue's resseq and icode for easy comparison."
            return f'{res_id[1]}{res_id[2]}'
        if file_type == 'MMCIF' and version(dssp_version) < version('4.0.0'):
            mmcif_dict = MMCIF2Dict(in_file)
            mmcif_chain_dict = {}
            for (i, c) in enumerate(mmcif_dict['_atom_site.label_asym_id']):
                if c not in mmcif_chain_dict:
                    mmcif_chain_dict[c] = mmcif_dict['_atom_site.auth_asym_id'][i]
            dssp_mapped_keys = []
        for key in dssp_keys:
            (chain_id, res_id) = key
            if file_type == 'MMCIF' and version(dssp_version) < version('4.0.0'):
                chain_id = mmcif_chain_dict[chain_id]
                dssp_mapped_keys.append((chain_id, res_id))
            chain = model[chain_id]
            try:
                res = chain[res_id]
            except KeyError:
                res_seq_icode = resid2code(res_id)
                for r in chain:
                    if r.id[0] not in (' ', 'W'):
                        if resid2code(r.id) == res_seq_icode:
                            res = r
                            break
                else:
                    raise KeyError(res_id) from None
            if res.is_disordered() == 2:
                for rk in res.disordered_get_id_list():
                    altloc = res.child_dict[rk].get_list()[0].get_altloc()
                    if altloc in tuple('A1 '):
                        res.disordered_select(rk)
                        break
                else:
                    res.disordered_select(res.disordered_get_id_list()[0])
            elif res.is_disordered() == 1:
                altlocs = {a.get_altloc() for a in res.get_unpacked_list()}
                if altlocs.isdisjoint('A1 '):
                    res_seq_icode = resid2code(res_id)
                    for r in chain:
                        if r.id[0] not in (' ', 'W'):
                            if resid2code(r.id) == res_seq_icode and r.get_list()[0].get_altloc() in tuple('A1 '):
                                res = r
                                break
            (aa, ss, acc, phi, psi, dssp_index, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy) = dssp_dict[key]
            res.xtra['SS_DSSP'] = ss
            res.xtra['EXP_DSSP_ASA'] = acc
            res.xtra['PHI_DSSP'] = phi
            res.xtra['PSI_DSSP'] = psi
            res.xtra['DSSP_INDEX'] = dssp_index
            res.xtra['NH_O_1_RELIDX_DSSP'] = NH_O_1_relidx
            res.xtra['NH_O_1_ENERGY_DSSP'] = NH_O_1_energy
            res.xtra['O_NH_1_RELIDX_DSSP'] = O_NH_1_relidx
            res.xtra['O_NH_1_ENERGY_DSSP'] = O_NH_1_energy
            res.xtra['NH_O_2_RELIDX_DSSP'] = NH_O_2_relidx
            res.xtra['NH_O_2_ENERGY_DSSP'] = NH_O_2_energy
            res.xtra['O_NH_2_RELIDX_DSSP'] = O_NH_2_relidx
            res.xtra['O_NH_2_ENERGY_DSSP'] = O_NH_2_energy
            resname = res.get_resname()
            try:
                rel_acc = acc / self.residue_max_acc[resname]
            except KeyError:
                rel_acc = 'NA'
            else:
                if rel_acc > 1.0:
                    rel_acc = 1.0
            res.xtra['EXP_DSSP_RASA'] = rel_acc
            resname = protein_letters_3to1.get(resname, 'X')
            if resname == 'C':
                if _dssp_cys.match(aa):
                    aa = 'C'
            if resname != aa and (res.id[0] == ' ' or aa != 'X'):
                raise PDBException(f'Structure/DSSP mismatch at {res}')
            dssp_vals = (dssp_index, aa, ss, rel_acc, phi, psi, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy, NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
            dssp_map[chain_id, res_id] = dssp_vals
            dssp_list.append(dssp_vals)
        if file_type == 'MMCIF' and version(dssp_version) < version('4.0.0'):
            dssp_keys = dssp_mapped_keys
        AbstractResiduePropertyMap.__init__(self, dssp_map, dssp_keys, dssp_list)