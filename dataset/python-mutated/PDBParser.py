"""Parser for PDB files."""
import warnings
import numpy as np
from Bio.File import as_handle
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list

class PDBParser:
    """Parse a PDB file and return a Structure object."""

    def __init__(self, PERMISSIVE=True, get_header=False, structure_builder=None, QUIET=False, is_pqr=False):
        if False:
            i = 10
            return i + 15
        'Create a PDBParser object.\n\n        The PDB parser call a number of standard methods in an aggregated\n        StructureBuilder object. Normally this object is instantiated by the\n        PDBParser object itself, but if the user provides his/her own\n        StructureBuilder object, the latter is used instead.\n\n        Arguments:\n         - PERMISSIVE - Evaluated as a Boolean. If false, exceptions in\n           constructing the SMCRA data structure are fatal. If true (DEFAULT),\n           the exceptions are caught, but some residues or atoms will be missing.\n           THESE EXCEPTIONS ARE DUE TO PROBLEMS IN THE PDB FILE!.\n         - get_header - unused argument kept for historical compatibility.\n         - structure_builder - an optional user implemented StructureBuilder class.\n         - QUIET - Evaluated as a Boolean. If true, warnings issued in constructing\n           the SMCRA data will be suppressed. If false (DEFAULT), they will be shown.\n           These warnings might be indicative of problems in the PDB file!\n         - is_pqr - Evaluated as a Boolean. Specifies the type of file to be parsed.\n           If false (DEFAULT) a .pdb file format is assumed. Set it to true if you\n           want to parse a .pqr file instead.\n\n        '
        if structure_builder is not None:
            self.structure_builder = structure_builder
        else:
            self.structure_builder = StructureBuilder()
        self.header = None
        self.trailer = None
        self.line_counter = 0
        self.PERMISSIVE = bool(PERMISSIVE)
        self.QUIET = bool(QUIET)
        self.is_pqr = bool(is_pqr)

    def get_structure(self, id, file):
        if False:
            while True:
                i = 10
        'Return the structure.\n\n        Arguments:\n         - id - string, the id that will be used for the structure\n         - file - name of the PDB file OR an open filehandle\n\n        '
        with warnings.catch_warnings():
            if self.QUIET:
                warnings.filterwarnings('ignore', category=PDBConstructionWarning)
            self.header = None
            self.trailer = None
            self.structure_builder.init_structure(id)
            with as_handle(file) as handle:
                lines = handle.readlines()
                if not lines:
                    raise ValueError('Empty file.')
                self._parse(lines)
            self.structure_builder.set_header(self.header)
            structure = self.structure_builder.get_structure()
        return structure

    def get_header(self):
        if False:
            return 10
        'Return the header.'
        return self.header

    def get_trailer(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the trailer.'
        return self.trailer

    def _parse(self, header_coords_trailer):
        if False:
            while True:
                i = 10
        'Parse the PDB file (PRIVATE).'
        (self.header, coords_trailer) = self._get_header(header_coords_trailer)
        self.trailer = self._parse_coordinates(coords_trailer)

    def _get_header(self, header_coords_trailer):
        if False:
            while True:
                i = 10
        'Get the header of the PDB file, return the rest (PRIVATE).'
        structure_builder = self.structure_builder
        i = 0
        for i in range(len(header_coords_trailer)):
            structure_builder.set_line_counter(i + 1)
            line = header_coords_trailer[i]
            record_type = line[0:6]
            if record_type in ('ATOM  ', 'HETATM', 'MODEL '):
                break
        header = header_coords_trailer[0:i]
        self.line_counter = i
        coords_trailer = header_coords_trailer[i:]
        header_dict = _parse_pdb_header_list(header)
        return (header_dict, coords_trailer)

    def _parse_coordinates(self, coords_trailer):
        if False:
            i = 10
            return i + 15
        'Parse the atomic data in the PDB file (PRIVATE).'
        allowed_records = {'ATOM  ', 'HETATM', 'MODEL ', 'ENDMDL', 'TER   ', 'ANISOU', 'SIGATM', 'SIGUIJ', 'MASTER'}
        local_line_counter = 0
        structure_builder = self.structure_builder
        current_model_id = 0
        model_open = 0
        current_chain_id = None
        current_segid = None
        current_residue_id = None
        current_resname = None
        for i in range(len(coords_trailer)):
            line = coords_trailer[i].rstrip('\n')
            record_type = line[0:6]
            global_line_counter = self.line_counter + local_line_counter + 1
            structure_builder.set_line_counter(global_line_counter)
            if not line.strip():
                continue
            elif record_type == 'ATOM  ' or record_type == 'HETATM':
                if not model_open:
                    structure_builder.init_model(current_model_id)
                    current_model_id += 1
                    model_open = 1
                fullname = line[12:16]
                split_list = fullname.split()
                if len(split_list) != 1:
                    name = fullname
                else:
                    name = split_list[0]
                altloc = line[16]
                resname = line[17:20].strip()
                chainid = line[21]
                try:
                    serial_number = int(line[6:11])
                except Exception:
                    serial_number = 0
                resseq = int(line[22:26].split()[0])
                icode = line[26]
                if record_type == 'HETATM':
                    if resname == 'HOH' or resname == 'WAT':
                        hetero_flag = 'W'
                    else:
                        hetero_flag = 'H'
                else:
                    hetero_flag = ' '
                residue_id = (hetero_flag, resseq, icode)
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception:
                    raise PDBConstructionException('Invalid or missing coordinate(s) at line %i.' % global_line_counter) from None
                coord = np.array((x, y, z), 'f')
                if not self.is_pqr:
                    try:
                        occupancy = float(line[54:60])
                    except Exception:
                        self._handle_PDB_exception('Invalid or missing occupancy', global_line_counter)
                        occupancy = None
                    if occupancy is not None and occupancy < 0:
                        warnings.warn('Negative occupancy in one or more atoms', PDBConstructionWarning)
                    try:
                        bfactor = float(line[60:66])
                    except Exception:
                        self._handle_PDB_exception('Invalid or missing B factor', global_line_counter)
                        bfactor = 0.0
                elif self.is_pqr:
                    try:
                        pqr_charge = float(line[54:62])
                    except Exception:
                        self._handle_PDB_exception('Invalid or missing charge', global_line_counter)
                        pqr_charge = None
                    try:
                        radius = float(line[62:70])
                    except Exception:
                        self._handle_PDB_exception('Invalid or missing radius', global_line_counter)
                        radius = None
                    if radius is not None and radius < 0:
                        message = 'Negative atom radius'
                        self._handle_PDB_exception(message, global_line_counter)
                        radius = None
                segid = line[72:76]
                element = line[76:78].strip().upper()
                if current_segid != segid:
                    current_segid = segid
                    structure_builder.init_seg(current_segid)
                if current_chain_id != chainid:
                    current_chain_id = chainid
                    structure_builder.init_chain(current_chain_id)
                    current_residue_id = residue_id
                    current_resname = resname
                    try:
                        structure_builder.init_residue(resname, hetero_flag, resseq, icode)
                    except PDBConstructionException as message:
                        self._handle_PDB_exception(message, global_line_counter)
                elif current_residue_id != residue_id or current_resname != resname:
                    current_residue_id = residue_id
                    current_resname = resname
                    try:
                        structure_builder.init_residue(resname, hetero_flag, resseq, icode)
                    except PDBConstructionException as message:
                        self._handle_PDB_exception(message, global_line_counter)
                if not self.is_pqr:
                    try:
                        structure_builder.init_atom(name, coord, bfactor, occupancy, altloc, fullname, serial_number, element)
                    except PDBConstructionException as message:
                        self._handle_PDB_exception(message, global_line_counter)
                elif self.is_pqr:
                    try:
                        structure_builder.init_atom(name, coord, pqr_charge, radius, altloc, fullname, serial_number, element, pqr_charge, radius, self.is_pqr)
                    except PDBConstructionException as message:
                        self._handle_PDB_exception(message, global_line_counter)
            elif record_type == 'ANISOU':
                anisou = [float(x) for x in (line[28:35], line[35:42], line[43:49], line[49:56], line[56:63], line[63:70])]
                anisou_array = (np.array(anisou, 'f') / 10000.0).astype('f')
                structure_builder.set_anisou(anisou_array)
            elif record_type == 'MODEL ':
                try:
                    serial_num = int(line[10:14])
                except Exception:
                    self._handle_PDB_exception('Invalid or missing model serial number', global_line_counter)
                    serial_num = 0
                structure_builder.init_model(current_model_id, serial_num)
                current_model_id += 1
                model_open = 1
                current_chain_id = None
                current_residue_id = None
            elif record_type == 'END   ' or record_type == 'CONECT':
                self.line_counter += local_line_counter
                return coords_trailer[local_line_counter:]
            elif record_type == 'ENDMDL':
                model_open = 0
                current_chain_id = None
                current_residue_id = None
            elif record_type == 'SIGUIJ':
                siguij = [float(x) for x in (line[28:35], line[35:42], line[42:49], line[49:56], line[56:63], line[63:70])]
                siguij_array = (np.array(siguij, 'f') / 10000.0).astype('f')
                structure_builder.set_siguij(siguij_array)
            elif record_type == 'SIGATM':
                sigatm = [float(x) for x in (line[30:38], line[38:46], line[46:54], line[54:60], line[60:66])]
                sigatm_array = np.array(sigatm, 'f')
                structure_builder.set_sigatm(sigatm_array)
            elif record_type not in allowed_records:
                warnings.warn("Ignoring unrecognized record '{}' at line {}".format(record_type, global_line_counter), PDBConstructionWarning)
            local_line_counter += 1
        self.line_counter = self.line_counter + local_line_counter
        return []

    def _handle_PDB_exception(self, message, line_counter):
        if False:
            while True:
                i = 10
        'Handle exception (PRIVATE).\n\n        This method catches an exception that occurs in the StructureBuilder\n        object (if PERMISSIVE), or raises it again, this time adding the\n        PDB line number to the error message.\n        '
        message = '%s at line %i.' % (message, line_counter)
        if self.PERMISSIVE:
            warnings.warn('PDBConstructionException: %s\nException ignored.\nSome atoms or residues may be missing in the data structure.' % message, PDBConstructionWarning)
        else:
            raise PDBConstructionException(message) from None