"""Convert XYZ Structure to internal coordinates and back, test result."""
import re
from itertools import zip_longest
import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from io import StringIO
from Bio.File import as_handle
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.internal_coords import IC_Residue
from Bio.PDB.PICIO import write_PIC, read_PIC, enumerate_atoms, pdb_date
from typing import Dict, Union, Any, Tuple
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

def structure_rebuild_test(entity, verbose: bool=False, quick: bool=False) -> Dict:
    if False:
        i = 10
        return i + 15
    'Test rebuild PDB structure from internal coordinates.\n\n    Generates internal coordinates for entity and writes to a .pic file in\n    memory, then generates XYZ coordinates from the .pic file and compares the\n    resulting entity against the original.\n\n    See :data:`IC_Residue.pic_accuracy` to vary numeric accuracy of the\n    intermediate .pic file if the only issue is small differences in coordinates.\n\n    Note that with default settings, deuterated initial structures will fail\n    the comparison, as will structures loaded with alternate `IC_Residue.accept_atoms`\n    settings.  Use `quick=True` and/or variations on `AtomKey.d2h` and\n    `IC_Residue.accept_atoms` settings.\n\n    :param Entity entity: Biopython Structure, Model or Chain.\n        Structure to test\n    :param bool verbose: default False.\n        print extra messages\n    :param bool quick: default False.\n        only check the internal coords atomArrays are identical\n    :returns: dict\n        comparison dict from :func:`.compare_residues`\n    '
    sp = StringIO()
    entity.atom_to_internal_coordinates(verbose)
    write_PIC(entity, sp)
    sp.seek(0)
    pdb2 = read_PIC(sp, verbose=verbose, quick=quick)
    if isinstance(entity, Chain):
        pdb2 = next(pdb2.get_chains())
    if verbose:
        report_IC(pdb2, verbose=True)
    pdb2.internal_to_atom_coordinates(verbose)
    r = compare_residues(entity, pdb2, verbose=verbose, quick=quick)
    return r

def report_IC(entity: Union[Structure, Model, Chain, Residue], reportDict: Dict[str, Any]=None, verbose: bool=False) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    'Generate dict with counts of ic data elements for each entity level.\n\n    reportDict entries are:\n        - idcode : PDB ID\n        - hdr : PDB header lines\n        - mdl : models\n        - chn : chains\n        - res : residue objects\n        - res_e : residues with dihedra and/or hedra\n        - dih : dihedra\n        - hed : hedra\n\n    :param Entity entity: Biopython PDB Entity object: S, M, C or R\n    :raises PDBException: if entity level not S, M, C, or R\n    :raises Exception: if entity does not have .level attribute\n    :returns: dict with counts of IC data elements\n    '
    if reportDict is None:
        reportDict = {'idcode': None, 'hdr': 0, 'mdl': 0, 'chn': 0, 'chn_ids': [], 'res': 0, 'res_e': 0, 'dih': 0, 'hed': 0}
    try:
        if 'A' == entity.level:
            raise PDBException('No IC output at Atom level')
        elif isinstance(entity, (DisorderedResidue, Residue)):
            if entity.internal_coord:
                reportDict['res'] += 1
                dlen = len(entity.internal_coord.dihedra)
                hlen = len(entity.internal_coord.hedra)
                if 0 < dlen or 0 < hlen:
                    reportDict['res_e'] += 1
                    reportDict['dih'] += dlen
                    reportDict['hed'] += hlen
        elif isinstance(entity, Chain):
            reportDict['chn'] += 1
            reportDict['chn_ids'].append(entity.id)
            for res in entity:
                reportDict = report_IC(res, reportDict)
        elif isinstance(entity, Model):
            reportDict['mdl'] += 1
            for chn in entity:
                reportDict = report_IC(chn, reportDict)
        elif isinstance(entity, Structure):
            if hasattr(entity, 'header'):
                if reportDict['idcode'] is None:
                    reportDict['idcode'] = entity.header.get('idcode', None)
                hdr = entity.header.get('head', None)
                if hdr:
                    reportDict['hdr'] += 1
                nam = entity.header.get('name', None)
                if nam:
                    reportDict['hdr'] += 1
            for mdl in entity:
                reportDict = report_IC(mdl, reportDict)
        else:
            raise PDBException('Cannot identify level: ' + str(entity.level))
    except KeyError:
        raise Exception('write_PIC: argument is not a Biopython PDB Entity ' + str(entity))
    if verbose:
        print('{} : {} models {} chains {} {} residue objects {} residues with {} dihedra {} hedra'.format(reportDict['idcode'], reportDict['mdl'], reportDict['chn'], reportDict['chn_ids'], reportDict['res'], reportDict['res_e'], reportDict['dih'], reportDict['hed']))
    return reportDict

def IC_duplicate(entity) -> Structure:
    if False:
        while True:
            i = 10
    'Duplicate structure entity with IC data, no atom coordinates.\n\n    Employs :func:`.write_PIC`, :func:`.read_PIC` with StringIO buffer.\n    Calls :meth:`.Chain.atom_to_internal_coordinates` if needed.\n\n    :param Entity entity: Biopython PDB Entity (will fail for Atom)\n    :returns: Biopython PDBStructure, no Atom objects except initial coords\n    '
    sp = StringIO()
    hasInternalCoords = False
    for res in entity.get_residues():
        if res.internal_coord:
            if len(res.internal_coord.hedra) > 0:
                hasInternalCoords = True
                break
    if not hasInternalCoords:
        if isinstance(entity, Residue):
            res = entity
            if not res.internal_coord:
                res.internal_coord = IC_Residue(entity)
            res.internal_coord.atom_to_internal_coordinates()
        else:
            entity.atom_to_internal_coordinates()
    write_PIC(entity, sp)
    sp.seek(0)
    return read_PIC(sp)

def _atmfid_d2h(atm: Atom) -> Tuple:
    if False:
        return 10
    afid = list(atm.get_full_id())
    afid4 = list(afid[4])
    afid40 = re.sub('D', 'H', afid4[0], count=1)
    new_afid = (afid[0], afid[1], afid[2], afid[3], (afid40, afid4[1]))
    return tuple(new_afid)

def _cmp_atm(r0: Residue, r1: Residue, a0: Atom, a1: Atom, verbose: bool, cmpdict: Dict, rtol: float=None, atol: float=None) -> None:
    if False:
        while True:
            i = 10
    cmpdict['aCount'] += 1
    if a0 is None:
        if verbose:
            print(r1.get_full_id(), 'None !=', a1.get_full_id(), a1.parent.resname)
    elif a1 is None:
        if verbose:
            print(r0.get_full_id(), a0.get_full_id(), a0.parent.resname, '!= None')
    else:
        if a0.get_full_id() == a1.get_full_id() or _atmfid_d2h(a0) == a1.get_full_id():
            cmpdict['aFullIdMatchCount'] += 1
        elif verbose:
            print(r0.get_full_id(), a0.get_full_id(), a0.parent.resname, '!=', a1.get_full_id())
        ac_rslt = False
        if rtol is None and atol is None:
            a0c = np.round(a0.get_coord(), 3)
            a1c = np.round(a1.get_coord(), 3)
            ac_rslt = np.array_equal(a0c, a1c)
        else:
            a0c = a0.get_coord()
            a1c = a1.get_coord()
            ac_rslt = np.allclose(a0c, a1c, rtol=rtol, atol=atol)
        if ac_rslt:
            cmpdict['aCoordMatchCount'] += 1
        elif verbose:
            print('atom coords disagree:', r0.get_full_id(), a0.get_full_id(), a1.get_full_id(), a0c, '!=', a1c)

def _cmp_res(r0: Residue, r1: Residue, verbose: bool, cmpdict: Dict, rtol: float=None, atol: float=None) -> None:
    if False:
        return 10
    (r0id, r0fid, r1fid) = (r0.id, r0.full_id, r1.full_id)
    chn = r0.parent.id
    if chn not in cmpdict['chains']:
        cmpdict['chains'].append(chn)
    cmpdict['rCount'] += 1
    if r0fid == r1fid:
        cmpdict['rMatchCount'] += 1
    elif verbose:
        print(r0fid, '!=', r1fid)
    if hasattr(r0, 'internal_coord') and r0.internal_coord is not None:
        ric0 = r0.internal_coord
        ric1 = r1.internal_coord
        r0prev = sorted((ric.rbase for ric in ric0.rprev))
        r1prev = sorted((ric.rbase for ric in ric1.rprev))
        r0next = sorted((ric.rbase for ric in ric0.rnext))
        r1next = sorted((ric.rbase for ric in ric1.rnext))
        if r0prev != r1prev:
            if verbose:
                print(r0, 'rprev error:', r0prev, '!=', r1prev)
            cmpdict['rpnMismatchCount'] += 1
        if r0next != r1next:
            if verbose:
                print(r0, 'rnext error', r0next, '!=', r1next)
            cmpdict['rpnMismatchCount'] += 1
    if ' ' == r0id[0] and (not (' ' == r0.resname[0] or 2 == len(r0.resname))):
        cmpdict['residues'] += 1
        longer = r0 if len(r0.child_dict) >= len(r1.child_dict) else r1
        for ak in longer.child_dict:
            a0 = r0.child_dict.get(ak, None)
            if a0 is None:
                aknd = re.sub('D', 'H', ak, count=1)
                a0 = r0.child_dict.get(aknd, None)
            a1 = r1.child_dict.get(ak, None)
            if a1 is None:
                aknd = re.sub('D', 'H', ak, count=1)
                a1 = r1.child_dict.get(aknd, None)
            if a0 is None or a1 is None or 0 == a0.is_disordered() == a1.is_disordered():
                _cmp_atm(r0, r1, a0, a1, verbose, cmpdict, rtol=rtol, atol=atol)
            elif 2 == a0.is_disordered() == a1.is_disordered():
                cmpdict['disAtmCount'] += 1
                for da0k in a0.child_dict:
                    _cmp_atm(r0, r1, a0.child_dict.get(da0k, None), a1.child_dict.get(da0k, None), verbose, cmpdict, rtol=rtol, atol=atol)
            else:
                if verbose:
                    print('disorder disagreement:', r0.get_full_id(), ak)
                cmpdict['aCount'] += 1

def compare_residues(e0: Union[Structure, Model, Chain], e1: Union[Structure, Model, Chain], verbose: bool=False, quick: bool=False, rtol: float=None, atol: float=None) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    "Compare full IDs and atom coordinates for 2 Biopython PDB entities.\n\n    Skip DNA and HETATMs.\n\n    :param Entity e0,e1: Biopython PDB Entity objects (S, M or C).\n        Structures, Models or Chains to be compared\n    :param bool verbose:\n        Whether to print mismatch info, default False\n    :param bool quick: default False.\n        Only check atomArrays are identical, aCoordMatchCount=0 if different\n    :param float rtol, atol: default 1e-03, 1e-03 or round to 3 places.\n        NumPy allclose parameters; default is to round atom coordinates to 3\n        places and test equal.  For 'quick' will use defaults above for\n        comparing atomArrays\n    :returns dict:\n        Result counts for Residues, Full ID match Residues, Atoms,\n        Full ID match atoms, and Coordinate match atoms; report string;\n        error status (bool)\n    "
    cmpdict: Dict[str, Any] = {}
    cmpdict['chains'] = []
    cmpdict['residues'] = 0
    cmpdict['rCount'] = 0
    cmpdict['rMatchCount'] = 0
    cmpdict['rpnMismatchCount'] = 0
    cmpdict['aCount'] = 0
    cmpdict['disAtmCount'] = 0
    cmpdict['aCoordMatchCount'] = 0
    cmpdict['aFullIdMatchCount'] = 0
    cmpdict['id0'] = e0.get_full_id()
    cmpdict['id1'] = e1.get_full_id()
    cmpdict['pass'] = None
    cmpdict['report'] = None
    if quick:
        if isinstance(e0, Chain):
            if e0.internal_coord.atomArray is not None and np.shape(e0.internal_coord.atomArray) == np.shape(e1.internal_coord.atomArray) and np.allclose(e0.internal_coord.atomArray, e1.internal_coord.atomArray, rtol=0.001 if rtol is None else rtol, atol=0.001 if atol is None else atol):
                cmpdict['aCount'] = np.size(e0.internal_coord.atomArray, 0)
                cmpdict['aCoordMatchCount'] = np.size(e0.internal_coord.atomArray, 0)
                if cmpdict['aCoordMatchCount'] > 0:
                    cmpdict['pass'] = True
                else:
                    cmpdict['pass'] = False
            else:
                cmpdict['aCount'] = 0 if e0.internal_coord.atomArray is None else np.size(e0.internal_coord.atomArray, 0)
                cmpdict['pass'] = False
        else:
            cmpdict['pass'] = True
            for (c0, c1) in zip_longest(e0.get_chains(), e1.get_chains()):
                if c0.internal_coord.atomArray is not None:
                    if np.allclose(c0.internal_coord.atomArray, c1.internal_coord.atomArray, rtol=0.001 if rtol is None else rtol, atol=0.001 if atol is None else atol):
                        cmpdict['aCoordMatchCount'] += np.size(c0.internal_coord.atomArray, 0)
                    else:
                        cmpdict['pass'] = False
                    cmpdict['aCount'] += np.size(c0.internal_coord.atomArray, 0)
            if cmpdict['aCoordMatchCount'] < cmpdict['aCount']:
                cmpdict['pass'] = False
    else:
        for (r0, r1) in zip_longest(e0.get_residues(), e1.get_residues()):
            if 2 == r0.is_disordered() == r1.is_disordered():
                for (dr0, dr1) in zip_longest(r0.child_dict.values(), r1.child_dict.values()):
                    _cmp_res(dr0, dr1, verbose, cmpdict, rtol=rtol, atol=atol)
            else:
                _cmp_res(r0, r1, verbose, cmpdict, rtol=rtol, atol=atol)
        if cmpdict['rMatchCount'] == cmpdict['rCount'] and cmpdict['aCoordMatchCount'] == cmpdict['aCount'] and (cmpdict['aFullIdMatchCount'] == cmpdict['aCount']) and (cmpdict['rpnMismatchCount'] == 0):
            cmpdict['pass'] = True
        else:
            cmpdict['pass'] = False
    rstr = '{}:{} {} -- {} of {} residue IDs match; {} residues {} atom coords, {} full IDs of {} atoms ({} disordered) match : {}'.format(cmpdict['id0'], cmpdict['id1'], cmpdict['chains'], cmpdict['rMatchCount'], cmpdict['rCount'], cmpdict['residues'], cmpdict['aCoordMatchCount'], cmpdict['aFullIdMatchCount'], cmpdict['aCount'], cmpdict['disAtmCount'], 'ERROR' if not cmpdict['pass'] else 'ALL OK')
    if not cmpdict['pass']:
        if cmpdict['rMatchCount'] != cmpdict['rCount']:
            rstr += ' -RESIDUE IDS-'
        if cmpdict['aCoordMatchCount'] != cmpdict['aFullIdMatchCount']:
            rstr += ' -COORDINATES-'
        if cmpdict['aFullIdMatchCount'] != cmpdict['aCount']:
            rstr += ' -ATOM IDS-'
    cmpdict['report'] = rstr
    return cmpdict

def write_PDB(entity: Structure, file: str, pdbid: str=None, chainid: str=None) -> None:
    if False:
        i = 10
        return i + 15
    'Write PDB file with HEADER and TITLE if available.'
    enumerate_atoms(entity)
    with as_handle(file, 'w') as fp:
        try:
            if hasattr(entity, 'header'):
                if not pdbid:
                    pdbid = entity.header.get('idcode', None)
                hdr = entity.header.get('head', None)
                dd = pdb_date(entity.header.get('deposition_date', None))
                if hdr:
                    fp.write('HEADER    {:40}{:8}   {:4}\n'.format(hdr.upper(), dd or '', pdbid or ''))
                nam = entity.header.get('name', None)
                if nam:
                    fp.write('TITLE     ' + nam.upper() + '\n')
            io = PDBIO()
            io.set_structure(entity)
            io.save(fp, preserve_atom_numbering=True)
        except KeyError:
            raise Exception('write_PDB: argument is not a Biopython PDB Entity ' + str(entity))