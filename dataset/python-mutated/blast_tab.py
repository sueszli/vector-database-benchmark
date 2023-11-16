"""Bio.SearchIO parser for BLAST+ tab output format, with or without comments."""
import re
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
__all__ = ('BlastTabIndexer', 'BlastTabParser', 'BlastTabWriter')
_LONG_SHORT_MAP = {'query id': 'qseqid', 'query acc.': 'qacc', 'query acc.ver': 'qaccver', 'query length': 'qlen', 'subject id': 'sseqid', 'subject acc.': 'sacc', 'subject acc.ver': 'saccver', 'subject length': 'slen', 'alignment length': 'length', 'bit score': 'bitscore', 'score': 'score', 'evalue': 'evalue', 'identical': 'nident', '% identity': 'pident', 'positives': 'positive', '% positives': 'ppos', 'mismatches': 'mismatch', 'gaps': 'gaps', 'q. start': 'qstart', 'q. end': 'qend', 's. start': 'sstart', 's. end': 'send', 'query frame': 'qframe', 'sbjct frame': 'sframe', 'query/sbjct frames': 'frames', 'query seq': 'qseq', 'subject seq': 'sseq', 'gap opens': 'gapopen', 'query gi': 'qgi', 'subject ids': 'sallseqid', 'subject gi': 'sgi', 'subject gis': 'sallgi', 'BTOP': 'btop', 'subject accs.': 'sallacc', 'subject tax ids': 'staxids', 'subject sci names': 'sscinames', 'subject com names': 'scomnames', 'subject blast names': 'sblastnames', 'subject super kingdoms': 'sskingdoms', 'subject title': 'stitle', 'subject titles': 'salltitles', 'subject strand': 'sstrand', '% subject coverage': 'qcovs', '% hsp coverage': 'qcovhsp'}

def _list_semicol(s):
    if False:
        i = 10
        return i + 15
    return s.split(';')

def _list_diamond(s):
    if False:
        i = 10
        return i + 15
    return s.split('<>')
_COLUMN_QRESULT = {'qseqid': ('id', str), 'qacc': ('accession', str), 'qaccver': ('accession_version', str), 'qlen': ('seq_len', int), 'qgi': ('gi', str)}
_COLUMN_HIT = {'sseqid': ('id', str), 'sallseqid': ('id_all', _list_semicol), 'sacc': ('accession', str), 'saccver': ('accession_version', str), 'sallacc': ('accession_all', _list_semicol), 'sgi': ('gi', str), 'sallgi': ('gi_all', str), 'slen': ('seq_len', int), 'staxids': ('tax_ids', _list_semicol), 'sscinames': ('sci_names', _list_semicol), 'scomnames': ('com_names', _list_semicol), 'sblastnames': ('blast_names', _list_semicol), 'sskingdoms': ('super_kingdoms', _list_semicol), 'stitle': ('title', str), 'salltitles': ('title_all', _list_diamond), 'sstrand': ('strand', str), 'qcovs': ('query_coverage', float)}
_COLUMN_HSP = {'bitscore': ('bitscore', float), 'score': ('bitscore_raw', int), 'evalue': ('evalue', float), 'nident': ('ident_num', int), 'pident': ('ident_pct', float), 'positive': ('pos_num', int), 'ppos': ('pos_pct', float), 'mismatch': ('mismatch_num', int), 'gaps': ('gap_num', int), 'gapopen': ('gapopen_num', int), 'btop': ('btop', str), 'qcovhsp': ('query_coverage', float)}
_COLUMN_FRAG = {'length': ('aln_span', int), 'qstart': ('query_start', int), 'qend': ('query_end', int), 'sstart': ('hit_start', int), 'send': ('hit_end', int), 'qframe': ('query_frame', int), 'sframe': ('hit_frame', int), 'frames': ('frames', str), 'qseq': ('query', str), 'sseq': ('hit', str)}
_SUPPORTED_FIELDS = set(list(_COLUMN_QRESULT) + list(_COLUMN_HIT) + list(_COLUMN_HSP) + list(_COLUMN_FRAG))
_DEFAULT_FIELDS = ['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen', 'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore']
_MIN_QUERY_FIELDS = {'qseqid', 'qacc', 'qaccver'}
_MIN_HIT_FIELDS = {'sseqid', 'sacc', 'saccver', 'sallseqid'}
_RE_GAPOPEN = re.compile('\\w-')

def _compute_gapopen_num(hsp):
    if False:
        while True:
            i = 10
    'Return the number of gap openings in the given HSP (PRIVATE).'
    gapopen = 0
    for seq_type in ('query', 'hit'):
        seq = str(getattr(hsp, seq_type).seq)
        gapopen += len(re.findall(_RE_GAPOPEN, seq))
    return gapopen

def _augment_blast_hsp(hsp, attr):
    if False:
        while True:
            i = 10
    'Calculate the given HSP attribute, for writing (PRIVATE).'
    if not hasattr(hsp, attr) and (not attr.endswith('_pct')):
        if attr == 'aln_span':
            hsp.aln_span = hsp.ident_num + hsp.mismatch_num + hsp.gap_num
        elif attr.startswith('ident'):
            setattr(hsp, attr, hsp.aln_span - hsp.mismatch_num - hsp.gap_num)
        elif attr.startswith('gap'):
            setattr(hsp, attr, hsp.aln_span - hsp.ident_num - hsp.mismatch_num)
        elif attr == 'mismatch_num':
            setattr(hsp, attr, hsp.aln_span - hsp.ident_num - hsp.gap_num)
        elif attr == 'gapopen_num':
            if not hasattr(hsp, 'query') or not hasattr(hsp, 'hit'):
                raise AttributeError
            hsp.gapopen_num = _compute_gapopen_num(hsp)
    if attr == 'ident_pct':
        hsp.ident_pct = hsp.ident_num / hsp.aln_span * 100
    elif attr == 'pos_pct':
        hsp.pos_pct = hsp.pos_num / hsp.aln_span * 100
    elif attr == 'gap_pct':
        hsp.gap_pct = hsp.gap_num / hsp.aln_span * 100

class BlastTabParser:
    """Parser for the BLAST tabular format."""

    def __init__(self, handle, comments=False, fields=_DEFAULT_FIELDS):
        if False:
            return 10
        'Initialize the class.'
        self.handle = handle
        self.has_comments = comments
        self.fields = self._prep_fields(fields)
        self.line = self.handle.readline().strip()

    def __iter__(self):
        if False:
            return 10
        'Iterate over BlastTabParser, yields query results.'
        if not self.line:
            return
        elif self.has_comments:
            iterfunc = self._parse_commented_qresult
        else:
            if self.line.startswith('#'):
                raise ValueError("Encountered unexpected character '#' at the beginning of a line. Set comments=True if the file is a commented file.")
            iterfunc = self._parse_qresult
        yield from iterfunc()

    def _prep_fields(self, fields):
        if False:
            i = 10
            return i + 15
        'Validate and format the given fields for use by the parser (PRIVATE).'
        if isinstance(fields, str):
            fields = fields.strip().split(' ')
        if 'std' in fields:
            idx = fields.index('std')
            fields = fields[:idx] + _DEFAULT_FIELDS + fields[idx + 1:]
        if not set(fields).intersection(_MIN_QUERY_FIELDS) or not set(fields).intersection(_MIN_HIT_FIELDS):
            raise ValueError('Required query and/or hit ID field not found.')
        return fields

    def _parse_commented_qresult(self):
        if False:
            i = 10
            return i + 15
        'Yield ``QueryResult`` objects from a commented file (PRIVATE).'
        while True:
            comments = self._parse_comments()
            if comments:
                try:
                    self.fields = comments['fields']
                    qres_iter = self._parse_qresult()
                except KeyError:
                    assert 'fields' not in comments
                    qres_iter = iter([QueryResult()])
                for qresult in qres_iter:
                    for (key, value) in comments.items():
                        setattr(qresult, key, value)
                    yield qresult
            else:
                break

    def _parse_comments(self):
        if False:
            i = 10
            return i + 15
        'Return a dictionary containing tab file comments (PRIVATE).'
        comments = {}
        while True:
            if 'BLAST' in self.line and 'processed' not in self.line:
                program_line = self.line[len(' #'):].split(' ')
                comments['program'] = program_line[0].lower()
                comments['version'] = program_line[1]
            elif 'Query' in self.line:
                query_line = self.line[len('# Query: '):].split(' ', 1)
                comments['id'] = query_line[0]
                if len(query_line) == 2:
                    comments['description'] = query_line[1]
            elif 'Database' in self.line:
                comments['target'] = self.line[len('# Database: '):]
            elif 'RID' in self.line:
                comments['rid'] = self.line[len('# RID: '):]
            elif 'Fields' in self.line:
                comments['fields'] = self._parse_fields_line()
            elif ' hits found' in self.line or 'processed' in self.line:
                self.line = self.handle.readline().strip()
                return comments
            self.line = self.handle.readline()
            if not self.line:
                return comments
            else:
                self.line = self.line.strip()

    def _parse_fields_line(self):
        if False:
            while True:
                i = 10
        "Return column short names line from 'Fields' comment line (PRIVATE)."
        raw_field_str = self.line[len('# Fields: '):]
        long_fields = raw_field_str.split(', ')
        fields = [_LONG_SHORT_MAP[long_name] for long_name in long_fields]
        return self._prep_fields(fields)

    def _parse_result_row(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary of parsed row values (PRIVATE).'
        fields = self.fields
        columns = self.line.strip().split('\t')
        if len(fields) != len(columns):
            raise ValueError('Expected %i columns, found: %i' % (len(fields), len(columns)))
        (qresult, hit, hsp, frag) = ({}, {}, {}, {})
        for (idx, value) in enumerate(columns):
            sname = fields[idx]
            in_mapping = False
            for (parsed_dict, mapping) in ((qresult, _COLUMN_QRESULT), (hit, _COLUMN_HIT), (hsp, _COLUMN_HSP), (frag, _COLUMN_FRAG)):
                if sname in mapping:
                    (attr_name, caster) = mapping[sname]
                    if caster is not str:
                        value = caster(value)
                    parsed_dict[attr_name] = value
                    in_mapping = True
            if not in_mapping:
                assert sname not in _SUPPORTED_FIELDS
        return {'qresult': qresult, 'hit': hit, 'hsp': hsp, 'frag': frag}

    def _get_id(self, parsed):
        if False:
            return 10
        'Return the value used for a QueryResult or Hit ID from a parsed row (PRIVATE).'
        id_cache = parsed.get('id')
        if id_cache is None and 'id_all' in parsed:
            id_cache = parsed.get('id_all')[0]
        if id_cache is None:
            id_cache = parsed.get('accession')
        if id_cache is None:
            id_cache = parsed.get('accession_version')
        return id_cache

    def _parse_qresult(self):
        if False:
            i = 10
            return i + 15
        'Yield QueryResult objects (PRIVATE).'
        state_EOF = 0
        state_QRES_NEW = 1
        state_QRES_SAME = 3
        state_HIT_NEW = 2
        state_HIT_SAME = 4
        qres_state = None
        hit_state = None
        file_state = None
        cur_qid = None
        cur_hid = None
        prev_qid = None
        prev_hid = None
        (cur, prev) = (None, None)
        (hit_list, hsp_list) = ([], [])
        while True:
            if cur is not None:
                prev = cur
                prev_qid = cur_qid
                prev_hid = cur_hid
            if self.line and (not self.line.startswith('#')):
                cur = self._parse_result_row()
                cur_qid = self._get_id(cur['qresult'])
                cur_hid = self._get_id(cur['hit'])
            else:
                file_state = state_EOF
                (cur_qid, cur_hid) = (None, None)
            if prev_qid != cur_qid:
                qres_state = state_QRES_NEW
            else:
                qres_state = state_QRES_SAME
            if prev_hid != cur_hid or qres_state == state_QRES_NEW:
                hit_state = state_HIT_NEW
            else:
                hit_state = state_HIT_SAME
            if prev is not None:
                frag = HSPFragment(prev_hid, prev_qid)
                for (attr, value) in prev['frag'].items():
                    for seq_type in ('query', 'hit'):
                        if attr == seq_type + '_start':
                            value = min(value, prev['frag'][seq_type + '_end']) - 1
                        elif attr == seq_type + '_end':
                            value = max(value, prev['frag'][seq_type + '_start'])
                    setattr(frag, attr, value)
                for seq_type in ('hit', 'query'):
                    frame = self._get_frag_frame(frag, seq_type, prev['frag'])
                    setattr(frag, '%s_frame' % seq_type, frame)
                    strand = self._get_frag_strand(frag, seq_type, prev['frag'])
                    setattr(frag, '%s_strand' % seq_type, strand)
                hsp = HSP([frag])
                for (attr, value) in prev['hsp'].items():
                    setattr(hsp, attr, value)
                hsp_list.append(hsp)
                if hit_state == state_HIT_NEW:
                    hit = Hit(hsp_list)
                    for (attr, value) in prev['hit'].items():
                        if attr != 'id_all':
                            setattr(hit, attr, value)
                        else:
                            setattr(hit, '_id_alt', value[1:])
                    hit_list.append(hit)
                    hsp_list = []
                if qres_state == state_QRES_NEW or file_state == state_EOF:
                    qresult = QueryResult(hit_list, prev_qid)
                    for (attr, value) in prev['qresult'].items():
                        setattr(qresult, attr, value)
                    yield qresult
                    if file_state == state_EOF:
                        break
                    hit_list = []
            self.line = self.handle.readline().strip()

    def _get_frag_frame(self, frag, seq_type, parsedict):
        if False:
            for i in range(10):
                print('nop')
        'Return fragment frame for given object (PRIVATE).\n\n        Returns ``HSPFragment`` frame given the object, its sequence type,\n        and its parsed dictionary values.\n        '
        assert seq_type in ('query', 'hit')
        frame = getattr(frag, '%s_frame' % seq_type, None)
        if frame is not None:
            return frame
        elif 'frames' in parsedict:
            idx = 0 if seq_type == 'query' else 1
            return int(parsedict['frames'].split('/')[idx])

    def _get_frag_strand(self, frag, seq_type, parsedict):
        if False:
            for i in range(10):
                print('nop')
        'Return fragment strand for given object (PRIVATE).\n\n        Returns ``HSPFragment`` strand given the object, its sequence type,\n        and its parsed dictionary values.\n        '
        assert seq_type in ('query', 'hit')
        strand = getattr(frag, '%s_strand' % seq_type, None)
        if strand is not None:
            return strand
        else:
            start = parsedict.get('%s_start' % seq_type)
            end = parsedict.get('%s_end' % seq_type)
            if start is not None and end is not None:
                return 1 if start <= end else -1

class BlastTabIndexer(SearchIndexer):
    """Indexer class for BLAST+ tab output."""
    _parser = BlastTabParser

    def __init__(self, filename, comments=False, fields=_DEFAULT_FIELDS):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        SearchIndexer.__init__(self, filename, comments=comments, fields=fields)
        if not self._kwargs['comments']:
            if 'qseqid' in fields:
                self._key_idx = fields.index('qseqid')
            elif 'qacc' in fields:
                self._key_idx = fields.index('qacc')
            elif 'qaccver' in fields:
                self._key_idx = fields.index('qaccver')
            else:
                raise ValueError("Custom fields is missing an ID column. One of these must be present: 'qseqid', 'qacc', or 'qaccver'.")

    def __iter__(self):
        if False:
            return 10
        'Iterate over the file handle; yields key, start offset, and length.'
        handle = self._handle
        handle.seek(0)
        if not self._kwargs['comments']:
            iterfunc = self._qresult_index
        else:
            iterfunc = self._qresult_index_commented
        for (key, offset, length) in iterfunc():
            yield (key.decode(), offset, length)

    def _qresult_index_commented(self):
        if False:
            i = 10
            return i + 15
        'Indexer for commented BLAST tabular files (PRIVATE).'
        handle = self._handle
        handle.seek(0)
        start_offset = 0
        query_mark = None
        qid_mark = b'# Query: '
        end_mark = b'# BLAST processed'
        while True:
            end_offset = handle.tell()
            line = handle.readline()
            if query_mark is None:
                query_mark = line
                start_offset = end_offset
            elif line.startswith(qid_mark):
                qresult_key = line[len(qid_mark):].split()[0]
            elif line == query_mark or line.startswith(end_mark):
                yield (qresult_key, start_offset, end_offset - start_offset)
                start_offset = end_offset
            elif not line:
                break

    def _qresult_index(self):
        if False:
            for i in range(10):
                print('nop')
        'Indexer for noncommented BLAST tabular files (PRIVATE).'
        handle = self._handle
        handle.seek(0)
        start_offset = 0
        qresult_key = None
        key_idx = self._key_idx
        while True:
            end_offset = handle.tell()
            line = handle.readline()
            if qresult_key is None:
                qresult_key = line.split(b'\t')[key_idx]
            else:
                try:
                    curr_key = line.split(b'\t')[key_idx]
                except IndexError:
                    curr_key = b''
                if curr_key != qresult_key:
                    yield (qresult_key, start_offset, end_offset - start_offset)
                    qresult_key = curr_key
                    start_offset = end_offset
            if not line:
                break

    def get_raw(self, offset):
        if False:
            i = 10
            return i + 15
        'Return the raw bytes string of a QueryResult object from the given offset.'
        if self._kwargs['comments']:
            getfunc = self._get_raw_qresult_commented
        else:
            getfunc = self._get_raw_qresult
        return getfunc(offset)

    def _get_raw_qresult(self, offset):
        if False:
            print('Hello World!')
        'Return the raw bytes string of a single QueryResult from a noncommented file (PRIVATE).'
        handle = self._handle
        handle.seek(offset)
        qresult_raw = b''
        key_idx = self._key_idx
        qresult_key = None
        while True:
            line = handle.readline()
            if qresult_key is None:
                qresult_key = line.split(b'\t')[key_idx]
            else:
                try:
                    curr_key = line.split(b'\t')[key_idx]
                except IndexError:
                    curr_key = b''
                if curr_key != qresult_key:
                    break
            qresult_raw += line
        return qresult_raw

    def _get_raw_qresult_commented(self, offset):
        if False:
            while True:
                i = 10
        'Return the bytes raw string of a single QueryResult from a commented file (PRIVATE).'
        handle = self._handle
        handle.seek(offset)
        qresult_raw = b''
        end_mark = b'# BLAST processed'
        query_mark = None
        line = handle.readline()
        while line:
            if query_mark is None:
                query_mark = line
            elif line == query_mark or line.startswith(end_mark):
                break
            qresult_raw += line
            line = handle.readline()
        return qresult_raw

class BlastTabWriter:
    """Writer for blast-tab output format."""

    def __init__(self, handle, comments=False, fields=_DEFAULT_FIELDS):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.handle = handle
        self.has_comments = comments
        self.fields = fields

    def write_file(self, qresults):
        if False:
            print('Hello World!')
        'Write to the handle, return how many QueryResult objects were written.'
        handle = self.handle
        (qresult_counter, hit_counter, hsp_counter, frag_counter) = (0, 0, 0, 0)
        for qresult in qresults:
            if self.has_comments:
                handle.write(self._build_comments(qresult))
            if qresult:
                handle.write(self._build_rows(qresult))
                if not self.has_comments:
                    qresult_counter += 1
                hit_counter += len(qresult)
                hsp_counter += sum((len(hit) for hit in qresult))
                frag_counter += sum((len(hit.fragments) for hit in qresult))
            if self.has_comments:
                qresult_counter += 1
        if self.has_comments:
            handle.write('# BLAST processed %i queries' % qresult_counter)
        return (qresult_counter, hit_counter, hsp_counter, frag_counter)

    def _build_rows(self, qresult):
        if False:
            for i in range(10):
                print('nop')
        'Return a string containing tabular rows of the QueryResult object (PRIVATE).'
        coordinates = {'qstart', 'qend', 'sstart', 'send'}
        qresult_lines = ''
        for hit in qresult:
            for hsp in hit:
                line = []
                for field in self.fields:
                    if field in _COLUMN_QRESULT:
                        value = getattr(qresult, _COLUMN_QRESULT[field][0])
                    elif field in _COLUMN_HIT:
                        if field == 'sallseqid':
                            value = getattr(hit, 'id_all')
                        else:
                            value = getattr(hit, _COLUMN_HIT[field][0])
                    elif field == 'frames':
                        value = '%i/%i' % (hsp.query_frame, hsp.hit_frame)
                    elif field in _COLUMN_HSP:
                        try:
                            value = getattr(hsp, _COLUMN_HSP[field][0])
                        except AttributeError:
                            attr = _COLUMN_HSP[field][0]
                            _augment_blast_hsp(hsp, attr)
                            value = getattr(hsp, attr)
                    elif field in _COLUMN_FRAG:
                        value = getattr(hsp, _COLUMN_FRAG[field][0])
                    else:
                        assert field not in _SUPPORTED_FIELDS
                        continue
                    if field in coordinates:
                        value = self._adjust_coords(field, value, hsp)
                    value = self._adjust_output(field, value)
                    line.append(value)
                hsp_line = '\t'.join(line)
                qresult_lines += hsp_line + '\n'
        return qresult_lines

    def _adjust_coords(self, field, value, hsp):
        if False:
            while True:
                i = 10
        'Adjust start and end coordinates according to strand (PRIVATE).'
        assert field in ('qstart', 'qend', 'sstart', 'send')
        seq_type = 'query' if field.startswith('q') else 'hit'
        strand = getattr(hsp, '%s_strand' % seq_type, None)
        if strand is None:
            raise ValueError('Required attribute %r not found.' % ('%s_strand' % seq_type))
        if strand < 0:
            if field.endswith('start'):
                value = getattr(hsp, '%s_end' % seq_type)
            elif field.endswith('end'):
                value = getattr(hsp, '%s_start' % seq_type) + 1
        elif field.endswith('start'):
            value += 1
        return value

    def _adjust_output(self, field, value):
        if False:
            i = 10
            return i + 15
        'Adjust formatting of given field and value to mimic native tab output (PRIVATE).'
        if field in ('qseq', 'sseq'):
            value = str(value.seq)
        elif field == 'evalue':
            if value < 1e-180:
                value = '0.0'
            elif value < 1e-99:
                value = '%2.0e' % value
            elif value < 0.0009:
                value = '%3.0e' % value
            elif value < 0.1:
                value = '%4.3f' % value
            elif value < 1.0:
                value = '%3.2f' % value
            elif value < 10.0:
                value = '%2.1f' % value
            else:
                value = '%5.0f' % value
        elif field in ('pident', 'ppos'):
            value = '%.2f' % value
        elif field == 'bitscore':
            if value > 9999:
                value = '%4.3e' % value
            elif value > 99.9:
                value = '%4.0d' % value
            else:
                value = '%4.1f' % value
        elif field in ('qcovhsp', 'qcovs'):
            value = '%.0f' % value
        elif field == 'salltitles':
            value = '<>'.join(value)
        elif field in ('sallseqid', 'sallacc', 'staxids', 'sscinames', 'scomnames', 'sblastnames', 'sskingdoms'):
            value = ';'.join(value)
        else:
            value = str(value)
        return value

    def _build_comments(self, qres):
        if False:
            i = 10
            return i + 15
        'Return QueryResult tabular comment as a string (PRIVATE).'
        comments = []
        inv_field_map = {v: k for (k, v) in _LONG_SHORT_MAP.items()}
        program = qres.program.upper()
        try:
            version = qres.version
        except AttributeError:
            program_line = '# %s' % program
        else:
            program_line = f'# {program} {version}'
        comments.append(program_line)
        if qres.description is None:
            comments.append('# Query: %s' % qres.id)
        else:
            comments.append(f'# Query: {qres.id} {qres.description}')
        try:
            comments.append('# RID: %s' % qres.rid)
        except AttributeError:
            pass
        comments.append('# Database: %s' % qres.target)
        if qres:
            comments.append('# Fields: %s' % ', '.join((inv_field_map[field] for field in self.fields)))
        comments.append('# %i hits found' % len(qres))
        return '\n'.join(comments) + '\n'
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()