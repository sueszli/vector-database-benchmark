"""Bio.SearchIO parser for HMMER plain text output format."""
import re
from Bio.SearchIO._utils import read_forward
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
from ._base import _BaseHmmerTextIndexer
__all__ = ('Hmmer3TextParser', 'Hmmer3TextIndexer')
_RE_PROGRAM = re.compile('^# .*?(\\w?hmm\\w+) :: .*$')
_RE_VERSION = re.compile('# \\w+ ([\\w+\\.]+) .*; http.*$')
_RE_OPT = re.compile('^# (.+):\\s+(.+)$')
_QRE_ID_LEN_PTN = '^Query:\\s*(.*)\\s+\\[\\w=(\\d+)\\]'
_QRE_ID_LEN = re.compile(_QRE_ID_LEN_PTN)
_HRE_VALIDATE = re.compile('score:\\s(-?\\d+\\.?\\d+)\\sbits.*value:\\s(.*)')
_HRE_ANNOT_LINE = re.compile('^(\\s+)(.+)\\s(\\w+)')
_HRE_ID_LINE = re.compile('^(\\s+\\S+\\s+[0-9-]+ )(.+?)(\\s+[0-9-]+)')

class Hmmer3TextParser:
    """Parser for the HMMER 3.0 text output."""

    def __init__(self, handle):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.handle = handle
        self.line = read_forward(self.handle)
        self._meta = self._parse_preamble()

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Iterate over query results.'
        yield from self._parse_qresult()

    def _read_until(self, bool_func):
        if False:
            return 10
        'Read the file handle until the given function returns True (PRIVATE).'
        while True:
            if not self.line or bool_func(self.line):
                return
            else:
                self.line = read_forward(self.handle)

    def _parse_preamble(self):
        if False:
            print('Hello World!')
        "Parse HMMER preamble (lines beginning with '#') (PRIVATE)."
        meta = {}
        has_opts = False
        while True:
            if not self.line.startswith('#'):
                break
            elif '- - -' in self.line:
                if not has_opts:
                    has_opts = True
                else:
                    break
            elif not has_opts:
                regx = re.search(_RE_PROGRAM, self.line)
                if regx:
                    meta['program'] = regx.group(1)
                regx = re.search(_RE_VERSION, self.line)
                if regx:
                    meta['version'] = regx.group(1)
            elif has_opts:
                regx = re.search(_RE_OPT, self.line)
                if 'target' in regx.group(1):
                    meta['target'] = regx.group(2).strip()
                else:
                    meta[regx.group(1)] = regx.group(2)
            self.line = read_forward(self.handle)
        return meta

    def _parse_qresult(self):
        if False:
            print('Hello World!')
        'Parse a HMMER3 query block (PRIVATE).'
        self._read_until(lambda line: line.startswith('Query:'))
        while self.line:
            regx = re.search(_QRE_ID_LEN, self.line)
            while not regx:
                self.line = read_forward(self.handle)
                regx = re.search(_QRE_ID_LEN, self.line)
            qid = regx.group(1).strip()
            qresult_attrs = {'seq_len': int(regx.group(2)), 'program': self._meta.get('program'), 'version': self._meta.get('version'), 'target': self._meta.get('target')}
            qdesc = '<unknown description>'
            while not self.line.startswith('Scores for '):
                self.line = read_forward(self.handle)
                if self.line.startswith('Accession:'):
                    acc = self.line.strip().split(' ', 1)[1]
                    qresult_attrs['accession'] = acc.strip()
                elif self.line.startswith('Description:'):
                    qdesc = self.line.strip().split(' ', 1)[1].strip()
                    qresult_attrs['description'] = qdesc
            while self.line and '//' not in self.line:
                hit_list = self._parse_hit(qid, qdesc)
                if self.line.startswith('Internal pipeline'):
                    while self.line and '//' not in self.line:
                        self.line = read_forward(self.handle)
            qresult = QueryResult(id=qid, hits=hit_list)
            for (attr, value) in qresult_attrs.items():
                setattr(qresult, attr, value)
            yield qresult
            self.line = read_forward(self.handle)
            if self.line.startswith('#'):
                self.line = self.handle.readline()
            if '[ok]' in self.line:
                break

    def _parse_hit(self, qid, qdesc):
        if False:
            for i in range(10):
                print('nop')
        'Parse a HMMER3 hit block, beginning with the hit table (PRIVATE).'
        self._read_until(lambda line: line.startswith('    ------- ------ -----'))
        self.line = read_forward(self.handle)
        is_included = True
        hit_attr_list = []
        while True:
            if not self.line:
                return []
            elif self.line.startswith('  ------ inclusion'):
                is_included = False
                self.line = read_forward(self.handle)
            elif self.line.startswith('   [No hits detected that satisfy reporting'):
                while True:
                    self.line = read_forward(self.handle)
                    if self.line.startswith('Internal pipeline'):
                        assert len(hit_attr_list) == 0
                        return []
            elif self.line.startswith('Domain annotation for each '):
                hit_list = self._create_hits(hit_attr_list, qid, qdesc)
                return hit_list
            row = [x for x in self.line.strip().split(' ') if x]
            if len(row) > 10:
                row[9] = ' '.join(row[9:])
            elif len(row) < 10:
                row.append('')
                assert len(row) == 10
            hit_attrs = {'id': row[8], 'query_id': qid, 'evalue': float(row[0]), 'bitscore': float(row[1]), 'bias': float(row[2]), 'domain_exp_num': float(row[6]), 'domain_obs_num': int(row[7]), 'description': row[9], 'is_included': is_included}
            hit_attr_list.append(hit_attrs)
            self.line = read_forward(self.handle)

    def _create_hits(self, hit_attrs, qid, qdesc):
        if False:
            for i in range(10):
                print('nop')
        'Parse a HMMER3 hsp block, beginning with the hsp table (PRIVATE).'
        self._read_until(lambda line: line.startswith(('Internal pipeline', '>>')))
        hit_list = []
        while True:
            if self.line.startswith('Internal pipeline'):
                assert len(hit_attrs) == 0
                return hit_list
            assert self.line.startswith('>>')
            (hid, hdesc) = self.line[len('>> '):].split('  ', 1)
            hdesc = hdesc.strip()
            self._read_until(lambda line: line.startswith((' ---   ------ ----- --------', '   [No individual domains')))
            self.line = read_forward(self.handle)
            hsp_list = []
            while True:
                if self.line.startswith('   [No targets detected that satisfy') or self.line.startswith('   [No individual domains') or self.line.startswith('Internal pipeline statistics summary:') or self.line.startswith('  Alignments for each domain:') or self.line.startswith('>>'):
                    hit_attr = hit_attrs.pop(0)
                    hit = Hit(hsp_list)
                    for (attr, value) in hit_attr.items():
                        if attr == 'description':
                            cur_val = getattr(hit, attr)
                            if cur_val and value and cur_val.startswith(value):
                                continue
                        setattr(hit, attr, value)
                    if not hit:
                        hit.query_description = qdesc
                    hit_list.append(hit)
                    break
                parsed = [x for x in self.line.strip().split(' ') if x]
                assert len(parsed) == 16
                frag = HSPFragment(hid, qid)
                if qdesc:
                    frag.query_description = qdesc
                if hdesc:
                    frag.hit_description = hdesc
                frag.molecule_type = 'protein'
                if self._meta.get('program') == 'hmmscan':
                    frag.hit_start = int(parsed[6]) - 1
                    frag.hit_end = int(parsed[7])
                    frag.query_start = int(parsed[9]) - 1
                    frag.query_end = int(parsed[10])
                elif self._meta.get('program') in ['hmmsearch', 'phmmer']:
                    frag.hit_start = int(parsed[9]) - 1
                    frag.hit_end = int(parsed[10])
                    frag.query_start = int(parsed[6]) - 1
                    frag.query_end = int(parsed[7])
                frag.hit_strand = frag.query_strand = 0
                hsp = HSP([frag])
                hsp.domain_index = int(parsed[0])
                hsp.is_included = parsed[1] == '!'
                hsp.bitscore = float(parsed[2])
                hsp.bias = float(parsed[3])
                hsp.evalue_cond = float(parsed[4])
                hsp.evalue = float(parsed[5])
                if self._meta.get('program') == 'hmmscan':
                    hsp.hit_endtype = parsed[8]
                    hsp.query_endtype = parsed[11]
                elif self._meta.get('program') in ['hmmsearch', 'phmmer']:
                    hsp.hit_endtype = parsed[11]
                    hsp.query_endtype = parsed[8]
                hsp.env_start = int(parsed[12]) - 1
                hsp.env_end = int(parsed[13])
                hsp.env_endtype = parsed[14]
                hsp.acc_avg = float(parsed[15])
                hsp_list.append(hsp)
                self.line = read_forward(self.handle)
            if self.line.startswith('  Alignments for each domain:'):
                self._parse_aln_block(hid, hit.hsps)

    def _parse_aln_block(self, hid, hsp_list):
        if False:
            i = 10
            return i + 15
        'Parse a HMMER3 HSP alignment block (PRIVATE).'
        self.line = read_forward(self.handle)
        dom_counter = 0
        while True:
            if self.line.startswith('>>') or self.line.startswith('Internal pipeline'):
                return hsp_list
            assert self.line.startswith('  == domain %i' % (dom_counter + 1))
            frag = hsp_list[dom_counter][0]
            hmmseq = ''
            aliseq = ''
            annot = {}
            self.line = self.handle.readline()
            while True:
                regx = None
                regx = re.search(_HRE_ID_LINE, self.line)
                if regx:
                    if len(hmmseq) == len(aliseq):
                        hmmseq += regx.group(2)
                    elif len(hmmseq) > len(aliseq):
                        aliseq += regx.group(2)
                    assert len(hmmseq) >= len(aliseq)
                elif self.line.startswith('  == domain') or self.line.startswith('>>') or self.line.startswith('Internal pipeline'):
                    frag.aln_annotation = annot
                    if self._meta.get('program') == 'hmmscan':
                        frag.hit = hmmseq
                        frag.query = aliseq
                    elif self._meta.get('program') in ['hmmsearch', 'phmmer']:
                        frag.hit = aliseq
                        frag.query = hmmseq
                    dom_counter += 1
                    hmmseq = ''
                    aliseq = ''
                    annot = {}
                    break
                elif len(hmmseq) == len(aliseq):
                    regx = re.search(_HRE_ANNOT_LINE, self.line)
                    if regx:
                        annot_name = regx.group(3)
                        if annot_name in annot:
                            annot[annot_name] += regx.group(2)
                        else:
                            annot[annot_name] = regx.group(2)
                self.line = self.handle.readline()

class Hmmer3TextIndexer(_BaseHmmerTextIndexer):
    """Indexer class for HMMER plain text output."""
    _parser = Hmmer3TextParser
    qresult_start = b'Query: '
    qresult_end = b'//'

    def __iter__(self):
        if False:
            while True:
                i = 10
        "Iterate over Hmmer3TextIndexer; yields query results' key, offsets, 0."
        handle = self._handle
        handle.seek(0)
        start_offset = handle.tell()
        regex_id = re.compile(_QRE_ID_LEN_PTN.encode())
        while True:
            line = read_forward(handle)
            end_offset = handle.tell()
            if line.startswith(self.qresult_start):
                regx = re.search(regex_id, line)
                qresult_key = regx.group(1).strip()
                start_offset = end_offset - len(line)
            elif line.startswith(self.qresult_end):
                yield (qresult_key.decode(), start_offset, 0)
                start_offset = end_offset
            elif not line:
                break
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()