"""Bio.SearchIO parser for HMMER table output format."""
from itertools import chain
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
__all__ = ('Hmmer3TabParser', 'Hmmer3TabIndexer', 'Hmmer3TabWriter')

class Hmmer3TabParser:
    """Parser for the HMMER table format."""

    def __init__(self, handle):
        if False:
            return 10
        'Initialize the class.'
        self.handle = handle
        self.line = self.handle.readline()

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Iterate over Hmmer3TabParser, yields query results.'
        header_mark = '#'
        while self.line.startswith(header_mark):
            self.line = self.handle.readline()
        if self.line:
            yield from self._parse_qresult()

    def _parse_row(self):
        if False:
            while True:
                i = 10
        'Return a dictionary of parsed row values (PRIVATE).'
        cols = [x for x in self.line.strip().split(' ') if x]
        if len(cols) < 18:
            raise ValueError('Less columns than expected, only %i' % len(cols))
        cols[18] = ' '.join(cols[18:])
        qresult = {}
        qresult['id'] = cols[2]
        qresult['accession'] = cols[3]
        hit = {}
        hit['id'] = cols[0]
        hit['accession'] = cols[1]
        hit['evalue'] = float(cols[4])
        hit['bitscore'] = float(cols[5])
        hit['bias'] = float(cols[6])
        hit['domain_exp_num'] = float(cols[10])
        hit['region_num'] = int(cols[11])
        hit['cluster_num'] = int(cols[12])
        hit['overlap_num'] = int(cols[13])
        hit['env_num'] = int(cols[14])
        hit['domain_obs_num'] = int(cols[15])
        hit['domain_reported_num'] = int(cols[16])
        hit['domain_included_num'] = int(cols[17])
        hit['description'] = cols[18]
        hsp = {}
        hsp['evalue'] = float(cols[7])
        hsp['bitscore'] = float(cols[8])
        hsp['bias'] = float(cols[9])
        frag = {}
        frag['hit_strand'] = frag['query_strand'] = 0
        frag['molecule_type'] = 'protein'
        return {'qresult': qresult, 'hit': hit, 'hsp': hsp, 'frag': frag}

    def _parse_qresult(self):
        if False:
            while True:
                i = 10
        'Return QueryResult objects (PRIVATE).'
        state_EOF = 0
        state_QRES_NEW = 1
        state_QRES_SAME = 3
        qres_state = None
        file_state = None
        prev_qid = None
        (cur, prev) = (None, None)
        hit_list = []
        cur_qid = None
        while True:
            if cur is not None:
                prev = cur
                prev_qid = cur_qid
            if self.line and (not self.line.startswith('#')):
                cur = self._parse_row()
                cur_qid = cur['qresult']['id']
            else:
                file_state = state_EOF
                cur_qid = None
            if prev_qid != cur_qid:
                qres_state = state_QRES_NEW
            else:
                qres_state = state_QRES_SAME
            if prev is not None:
                prev_hid = prev['hit']['id']
                frag = HSPFragment(prev_hid, prev_qid)
                for (attr, value) in prev['frag'].items():
                    setattr(frag, attr, value)
                hsp = HSP([frag])
                for (attr, value) in prev['hsp'].items():
                    setattr(hsp, attr, value)
                hit = Hit([hsp])
                for (attr, value) in prev['hit'].items():
                    setattr(hit, attr, value)
                hit_list.append(hit)
                if qres_state == state_QRES_NEW or file_state == state_EOF:
                    qresult = QueryResult(hit_list, prev_qid)
                    for (attr, value) in prev['qresult'].items():
                        setattr(qresult, attr, value)
                    yield qresult
                    if file_state == state_EOF:
                        break
                    hit_list = []
            self.line = self.handle.readline()

class Hmmer3TabIndexer(SearchIndexer):
    """Indexer class for HMMER table output."""
    _parser = Hmmer3TabParser
    _query_id_idx = 2

    def __iter__(self):
        if False:
            print('Hello World!')
        'Iterate over the file handle; yields key, start offset, and length.'
        handle = self._handle
        handle.seek(0)
        query_id_idx = self._query_id_idx
        qresult_key = None
        header_mark = b'#'
        split_mark = b' '
        line = header_mark
        while line.startswith(header_mark):
            start_offset = handle.tell()
            line = handle.readline()
        while True:
            end_offset = handle.tell()
            if not line:
                break
            cols = [x for x in line.strip().split(split_mark) if x]
            if qresult_key is None:
                qresult_key = cols[query_id_idx]
            else:
                curr_key = cols[query_id_idx]
                if curr_key != qresult_key:
                    adj_end = end_offset - len(line)
                    yield (qresult_key.decode(), start_offset, adj_end - start_offset)
                    qresult_key = curr_key
                    start_offset = adj_end
            line = handle.readline()
            if not line:
                yield (qresult_key.decode(), start_offset, end_offset - start_offset)
                break

    def get_raw(self, offset):
        if False:
            while True:
                i = 10
        'Return the raw bytes string of a QueryResult object from the given offset.'
        handle = self._handle
        handle.seek(offset)
        query_id_idx = self._query_id_idx
        qresult_key = None
        qresult_raw = b''
        split_mark = b' '
        while True:
            line = handle.readline()
            if not line:
                break
            cols = [x for x in line.strip().split(split_mark) if x]
            if qresult_key is None:
                qresult_key = cols[query_id_idx]
            else:
                curr_key = cols[query_id_idx]
                if curr_key != qresult_key:
                    break
            qresult_raw += line
        return qresult_raw

class Hmmer3TabWriter:
    """Writer for hmmer3-tab output format."""

    def __init__(self, handle):
        if False:
            i = 10
            return i + 15
        'Initialize the class.'
        self.handle = handle

    def write_file(self, qresults):
        if False:
            for i in range(10):
                print('nop')
        'Write to the handle.\n\n        Returns a tuple of how many QueryResult, Hit, and HSP objects were written.\n\n        '
        handle = self.handle
        (qresult_counter, hit_counter, hsp_counter, frag_counter) = (0, 0, 0, 0)
        try:
            first_qresult = next(qresults)
        except StopIteration:
            handle.write(self._build_header())
        else:
            handle.write(self._build_header(first_qresult))
            for qresult in chain([first_qresult], qresults):
                if qresult:
                    handle.write(self._build_row(qresult))
                    qresult_counter += 1
                    hit_counter += len(qresult)
                    hsp_counter += sum((len(hit) for hit in qresult))
                    frag_counter += sum((len(hit.fragments) for hit in qresult))
        return (qresult_counter, hit_counter, hsp_counter, frag_counter)

    def _build_header(self, first_qresult=None):
        if False:
            i = 10
            return i + 15
        'Return the header string of a HMMER table output (PRIVATE).'
        if first_qresult is not None:
            qnamew = 20
            tnamew = max(20, len(first_qresult[0].id))
            qaccw = max(10, len(first_qresult.accession))
            taccw = max(10, len(first_qresult[0].accession))
        else:
            (qnamew, tnamew, qaccw, taccw) = (20, 20, 10, 10)
        header = '#%*s %22s %22s %33s\n' % (tnamew + qnamew + taccw + qaccw + 2, '', '--- full sequence ----', '--- best 1 domain ----', '--- domain number estimation ----')
        header += '#%-*s %-*s %-*s %-*s %9s %6s %5s %9s %6s %5s %5s %3s %3s %3s %3s %3s %3s %3s %s\n' % (tnamew - 1, ' target name', taccw, 'accession', qnamew, 'query name', qaccw, 'accession', '  E-value', ' score', ' bias', '  E-value', ' score', ' bias', 'exp', 'reg', 'clu', ' ov', 'env', 'dom', 'rep', 'inc', 'description of target')
        header += '#%*s %*s %*s %*s %9s %6s %5s %9s %6s %5s %5s %3s %3s %3s %3s %3s %3s %3s %s\n' % (tnamew - 1, '-------------------', taccw, '----------', qnamew, '--------------------', qaccw, '----------', '---------', '------', '-----', '---------', '------', '-----', '---', '---', '---', '---', '---', '---', '---', '---', '---------------------')
        return header

    def _build_row(self, qresult):
        if False:
            for i in range(10):
                print('nop')
        'Return a string or one row or more of the QueryResult object (PRIVATE).'
        rows = ''
        qnamew = max(20, len(qresult.id))
        tnamew = max(20, len(qresult[0].id))
        qaccw = max(10, len(qresult.accession))
        taccw = max(10, len(qresult[0].accession))
        for hit in qresult:
            rows += '%-*s %-*s %-*s %-*s %9.2g %6.1f %5.1f %9.2g %6.1f %5.1f %5.1f %3d %3d %3d %3d %3d %3d %3d %s\n' % (tnamew, hit.id, taccw, hit.accession, qnamew, qresult.id, qaccw, qresult.accession, hit.evalue, hit.bitscore, hit.bias, hit.hsps[0].evalue, hit.hsps[0].bitscore, hit.hsps[0].bias, hit.domain_exp_num, hit.region_num, hit.cluster_num, hit.overlap_num, hit.env_num, hit.domain_obs_num, hit.domain_reported_num, hit.domain_included_num, hit.description)
        return rows
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()