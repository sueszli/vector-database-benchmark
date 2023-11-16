"""Parse header of PDB files into a python dictionary.

Emerged from the Columba database project www.columba-db.de, original author
Kristian Rother.
"""
import re
from Bio import File

def _get_journal(inl):
    if False:
        for i in range(10):
            print('nop')
    journal = ''
    for line in inl:
        if re.search('\\AJRNL', line):
            journal += line[19:72].lower()
    journal = re.sub('\\s\\s+', ' ', journal)
    return journal

def _get_references(inl):
    if False:
        return 10
    references = []
    actref = ''
    for line in inl:
        if re.search('\\AREMARK   1', line):
            if re.search('\\AREMARK   1 REFERENCE', line):
                if actref != '':
                    actref = re.sub('\\s\\s+', ' ', actref)
                    if actref != ' ':
                        references.append(actref)
                    actref = ''
            else:
                actref += line[19:72].lower()
    if actref != '':
        actref = re.sub('\\s\\s+', ' ', actref)
        if actref != ' ':
            references.append(actref)
    return references

def _format_date(pdb_date):
    if False:
        i = 10
        return i + 15
    'Convert dates from DD-Mon-YY to YYYY-MM-DD format (PRIVATE).'
    date = ''
    year = int(pdb_date[7:])
    if year < 50:
        century = 2000
    else:
        century = 1900
    date = str(century + year) + '-'
    all_months = ['xxx', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month = str(all_months.index(pdb_date[3:6]))
    if len(month) == 1:
        month = '0' + month
    date = date + month + '-' + pdb_date[:2]
    return date

def _chop_end_codes(line):
    if False:
        return 10
    "Chops lines ending with  '     1CSA  14' and the like (PRIVATE)."
    return re.sub('\\s\\s\\s\\s+[\\w]{4}.\\s+\\d*\\Z', '', line)

def _chop_end_misc(line):
    if False:
        print('Hello World!')
    "Chops lines ending with  '     14-JUL-97  1CSA' and the like (PRIVATE)."
    return re.sub('\\s+\\d\\d-\\w\\w\\w-\\d\\d\\s+[1-9][0-9A-Z]{3}\\s*\\Z', '', line)

def _nice_case(line):
    if False:
        while True:
            i = 10
    'Make A Lowercase String With Capitals (PRIVATE).'
    line_lower = line.lower()
    s = ''
    i = 0
    nextCap = 1
    while i < len(line_lower):
        c = line_lower[i]
        if c >= 'a' and c <= 'z' and nextCap:
            c = c.upper()
            nextCap = 0
        elif c in ' .,;:\t-_':
            nextCap = 1
        s += c
        i += 1
    return s

def parse_pdb_header(infile):
    if False:
        for i in range(10):
            print('nop')
    'Return the header lines of a pdb file as a dictionary.\n\n    Dictionary keys are: head, deposition_date, release_date, structure_method,\n    resolution, structure_reference, journal_reference, author and\n    compound.\n    '
    header = []
    with File.as_handle(infile) as f:
        for line in f:
            record_type = line[0:6]
            if record_type in ('ATOM  ', 'HETATM', 'MODEL '):
                break
            else:
                header.append(line)
    return _parse_pdb_header_list(header)

def _parse_remark_465(line):
    if False:
        while True:
            i = 10
    'Parse missing residue remarks.\n\n    Returns a dictionary describing the missing residue.\n    The specification for REMARK 465 at\n    http://www.wwpdb.org/documentation/file-format-content/format33/remarks2.html#REMARK%20465\n    only gives templates, but does not say they have to be followed.\n    So we assume that not all pdb-files with a REMARK 465 can be understood.\n\n    Returns a dictionary with the following keys:\n    "model", "res_name", "chain", "ssseq", "insertion"\n    '
    if line:
        assert line[0] != ' ' and line[-1] not in '\n ', 'line has to be stripped'
    pattern = re.compile('\n        (\\d+\\s[\\sA-Z][\\sA-Z][A-Z] |   # Either model number + residue name\n            [A-Z]{1,3})               # Or only residue name with 1 (RNA) to 3 letters\n        \\s ([A-Za-z0-9])              # A single character chain\n        \\s+(-?\\d+[A-Za-z]?)$          # Residue number: A digit followed by an optional\n                                      # insertion code (Hetero-flags make no sense in\n                                      # context with missing res)\n        ', re.VERBOSE)
    match = pattern.match(line)
    if match is None:
        return None
    residue = {}
    if ' ' in match.group(1):
        (model, residue['res_name']) = match.group(1).split()
        residue['model'] = int(model)
    else:
        residue['model'] = None
        residue['res_name'] = match.group(1)
    residue['chain'] = match.group(2)
    try:
        residue['ssseq'] = int(match.group(3))
    except ValueError:
        residue['insertion'] = match.group(3)[-1]
        residue['ssseq'] = int(match.group(3)[:-1])
    else:
        residue['insertion'] = None
    return residue

def _parse_pdb_header_list(header):
    if False:
        return 10
    pdbh_dict = {'name': '', 'head': '', 'idcode': '', 'deposition_date': '1909-01-08', 'release_date': '1909-01-08', 'structure_method': 'unknown', 'resolution': None, 'structure_reference': 'unknown', 'journal_reference': 'unknown', 'author': '', 'compound': {'1': {'misc': ''}}, 'source': {'1': {'misc': ''}}, 'has_missing_residues': False, 'missing_residues': []}
    pdbh_dict['structure_reference'] = _get_references(header)
    pdbh_dict['journal_reference'] = _get_journal(header)
    comp_molid = '1'
    last_comp_key = 'misc'
    last_src_key = 'misc'
    for hh in header:
        h = re.sub('[\\s\\n\\r]*\\Z', '', hh)
        key = h[:6].strip()
        tail = h[10:].strip()
        if key == 'TITLE':
            name = _chop_end_codes(tail).lower()
            pdbh_dict['name'] = ' '.join([pdbh_dict['name'], name]).strip()
        elif key == 'HEADER':
            rr = re.search('\\d\\d-\\w\\w\\w-\\d\\d', tail)
            if rr is not None:
                pdbh_dict['deposition_date'] = _format_date(_nice_case(rr.group()))
            rr = re.search('\\s+([1-9][0-9A-Z]{3})\\s*\\Z', tail)
            if rr is not None:
                pdbh_dict['idcode'] = rr.group(1)
            head = _chop_end_misc(tail).lower()
            pdbh_dict['head'] = head
        elif key == 'COMPND':
            tt = re.sub('\\;\\s*\\Z', '', _chop_end_codes(tail)).lower()
            rec = re.search('\\d+\\.\\d+\\.\\d+\\.\\d+', tt)
            if rec:
                pdbh_dict['compound'][comp_molid]['ec_number'] = rec.group()
                tt = re.sub('\\((e\\.c\\.)*\\d+\\.\\d+\\.\\d+\\.\\d+\\)', '', tt)
            tok = tt.split(':')
            if len(tok) >= 2:
                ckey = tok[0]
                cval = re.sub('\\A\\s*', '', tok[1])
                if ckey == 'mol_id':
                    pdbh_dict['compound'][cval] = {'misc': ''}
                    comp_molid = cval
                    last_comp_key = 'misc'
                else:
                    pdbh_dict['compound'][comp_molid][ckey] = cval
                    last_comp_key = ckey
            else:
                pdbh_dict['compound'][comp_molid][last_comp_key] += tok[0] + ' '
        elif key == 'SOURCE':
            tt = re.sub('\\;\\s*\\Z', '', _chop_end_codes(tail)).lower()
            tok = tt.split(':')
            if len(tok) >= 2:
                ckey = tok[0]
                cval = re.sub('\\A\\s*', '', tok[1])
                if ckey == 'mol_id':
                    pdbh_dict['source'][cval] = {'misc': ''}
                    comp_molid = cval
                    last_src_key = 'misc'
                else:
                    pdbh_dict['source'][comp_molid][ckey] = cval
                    last_src_key = ckey
            else:
                pdbh_dict['source'][comp_molid][last_src_key] += tok[0] + ' '
        elif key == 'KEYWDS':
            kwd = _chop_end_codes(tail).lower()
            if 'keywords' in pdbh_dict:
                pdbh_dict['keywords'] += ' ' + kwd
            else:
                pdbh_dict['keywords'] = kwd
        elif key == 'EXPDTA':
            expd = _chop_end_codes(tail)
            expd = re.sub('\\s\\s\\s\\s\\s\\s\\s.*\\Z', '', expd)
            pdbh_dict['structure_method'] = expd.lower()
        elif key == 'CAVEAT':
            pass
        elif key == 'REVDAT':
            rr = re.search('\\d\\d-\\w\\w\\w-\\d\\d', tail)
            if rr is not None:
                pdbh_dict['release_date'] = _format_date(_nice_case(rr.group()))
        elif key == 'JRNL':
            if 'journal' in pdbh_dict:
                pdbh_dict['journal'] += tail
            else:
                pdbh_dict['journal'] = tail
        elif key == 'AUTHOR':
            auth = _nice_case(_chop_end_codes(tail))
            if 'author' in pdbh_dict:
                pdbh_dict['author'] += auth
            else:
                pdbh_dict['author'] = auth
        elif key == 'REMARK':
            if re.search('REMARK   2 RESOLUTION.', hh):
                r = _chop_end_codes(re.sub('REMARK   2 RESOLUTION.', '', hh))
                r = re.sub('\\s+ANGSTROM.*', '', r)
                try:
                    pdbh_dict['resolution'] = float(r)
                except ValueError:
                    pdbh_dict['resolution'] = None
            elif hh.startswith('REMARK 465'):
                if tail:
                    pdbh_dict['has_missing_residues'] = True
                    missing_res_info = _parse_remark_465(tail)
                    if missing_res_info:
                        pdbh_dict['missing_residues'].append(missing_res_info)
            elif hh.startswith('REMARK  99 ASTRAL'):
                if tail:
                    remark_99_keyval = tail.replace('ASTRAL ', '').split(': ')
                    if isinstance(remark_99_keyval, list) and len(remark_99_keyval) == 2:
                        if 'astral' not in pdbh_dict:
                            pdbh_dict['astral'] = {remark_99_keyval[0]: remark_99_keyval[1]}
                        else:
                            pdbh_dict['astral'][remark_99_keyval[0]] = remark_99_keyval[1]
        else:
            pass
    if pdbh_dict['structure_method'] == 'unknown':
        res = pdbh_dict['resolution']
        if res is not None and res > 0.0:
            pdbh_dict['structure_method'] = 'x-ray diffraction'
    return pdbh_dict