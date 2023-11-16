"""Bio.SearchIO parser for BLAST+ XML output formats."""
import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
__all__ = ('BlastXmlParser', 'BlastXmlIndexer', 'BlastXmlWriter')
_ELEM_QRESULT_OPT = {'Statistics_db-num': ('stat_db_num', int), 'Statistics_db-len': ('stat_db_len', int), 'Statistics_eff-space': ('stat_eff_space', float), 'Statistics_hsp-len': ('stat_hsp_len', int), 'Statistics_kappa': ('stat_kappa', float), 'Statistics_lambda': ('stat_lambda', float), 'Statistics_entropy': ('stat_entropy', float)}
_ELEM_HIT = {'Hit_accession': ('accession', str), 'Hit_len': ('seq_len', int)}
_ELEM_HSP = {'Hsp_bit-score': ('bitscore', float), 'Hsp_score': ('bitscore_raw', int), 'Hsp_evalue': ('evalue', float), 'Hsp_identity': ('ident_num', int), 'Hsp_positive': ('pos_num', int), 'Hsp_gaps': ('gap_num', int), 'Hsp_density': ('density', float)}
_ELEM_FRAG = {'Hsp_query-from': ('query_start', int), 'Hsp_query-to': ('query_end', int), 'Hsp_hit-from': ('hit_start', int), 'Hsp_hit-to': ('hit_end', int), 'Hsp_query-frame': ('query_frame', int), 'Hsp_hit-frame': ('hit_frame', int), 'Hsp_align-len': ('aln_span', int), 'Hsp_pattern-from': ('pattern_start', int), 'Hsp_pattern-to': ('pattern_end', int), 'Hsp_hseq': ('hit', str), 'Hsp_qseq': ('query', str)}
_ELEM_META = {'BlastOutput_db': ('target', str), 'BlastOutput_program': ('program', str), 'BlastOutput_version': ('version', str), 'BlastOutput_reference': ('reference', str), 'Parameters_expect': ('param_evalue_threshold', float), 'Parameters_entrez-query': ('param_entrez_query', str), 'Parameters_filter': ('param_filter', str), 'Parameters_gap-extend': ('param_gap_extend', int), 'Parameters_gap-open': ('param_gap_open', int), 'Parameters_include': ('param_include', str), 'Parameters_matrix': ('param_matrix', str), 'Parameters_pattern': ('param_pattern', str), 'Parameters_sc-match': ('param_score_match', int), 'Parameters_sc-mismatch': ('param_score_mismatch', int)}
_ELEM_QRESULT_FALLBACK = {'BlastOutput_query-ID': ('id', str), 'BlastOutput_query-def': ('description', str), 'BlastOutput_query-len': ('len', str)}
_WRITE_MAPS = {'preamble': (('program', 'program'), ('version', 'version'), ('reference', 'reference'), ('db', 'target'), ('query-ID', 'id'), ('query-def', 'description'), ('query-len', 'seq_len'), ('param', None)), 'param': (('matrix', 'param_matrix'), ('expect', 'param_evalue_threshold'), ('sc-match', 'param_score_match'), ('sc-mismatch', 'param_score_mismatch'), ('gap-open', 'param_gap_open'), ('gap-extend', 'param_gap_extend'), ('filter', 'param_filter'), ('pattern', 'param_pattern'), ('entrez-query', 'param_entrez_query')), 'qresult': (('query-ID', 'id'), ('query-def', 'description'), ('query-len', 'seq_len')), 'stat': (('db-num', 'stat_db_num'), ('db-len', 'stat_db_len'), ('hsp-len', 'stat_hsp_len'), ('eff-space', 'stat_eff_space'), ('kappa', 'stat_kappa'), ('lambda', 'stat_lambda'), ('entropy', 'stat_entropy')), 'hit': (('id', 'id'), ('def', 'description'), ('accession', 'accession'), ('len', 'seq_len')), 'hsp': (('bit-score', 'bitscore'), ('score', 'bitscore_raw'), ('evalue', 'evalue'), ('query-from', 'query_start'), ('query-to', 'query_end'), ('hit-from', 'hit_start'), ('hit-to', 'hit_end'), ('pattern-from', 'pattern_start'), ('pattern-to', 'pattern_end'), ('query-frame', 'query_frame'), ('hit-frame', 'hit_frame'), ('identity', 'ident_num'), ('positive', 'pos_num'), ('gaps', 'gap_num'), ('align-len', 'aln_span'), ('density', 'density'), ('qseq', 'query'), ('hseq', 'hit'), ('midline', None))}
_DTD_OPT = ('BlastOutput_query-seq', 'BlastOutput_mbstat', 'Iteration_query-def', 'Iteration_query-len', 'Iteration-hits', 'Iteration_stat', 'Iteration_message', 'Parameters_matrix', 'Parameters_include', 'Parameters_sc-match', 'Parameters_sc-mismatch', 'Parameters_filter', 'Parameters_pattern', 'Parameters_entrez-query', 'Hit_hsps', 'Hsp_pattern-from', 'Hsp_pattern-to', 'Hsp_query-frame', 'Hsp_hit-frame', 'Hsp_identity', 'Hsp_positive', 'Hsp_gaps', 'Hsp_align-len', 'Hsp_density', 'Hsp_midline')
_RE_VERSION = re.compile('\\d+\\.\\d+\\.\\d+\\+?')
_RE_ID_DESC_PAIRS_PATTERN = re.compile(' +>')
_RE_ID_DESC_PATTERN = re.compile(' +')

def _extract_ids_and_descs(raw_id, raw_desc):
    if False:
        i = 10
        return i + 15
    'Extract IDs, descriptions, and raw ID from raw values (PRIVATE).\n\n    Given values of the ``Hit_id`` and ``Hit_def`` elements, this function\n    returns a tuple of three elements: all IDs, all descriptions, and the\n    BLAST-generated ID. The BLAST-generated ID is set to ``None`` if no\n    BLAST-generated IDs are present.\n\n    '
    ids = []
    descs = []
    blast_gen_id = raw_id
    if raw_id.startswith('gnl|BL_ORD_ID|'):
        id_desc_line = raw_desc
    else:
        id_desc_line = raw_id + ' ' + raw_desc
    id_desc_pairs = [re.split(_RE_ID_DESC_PATTERN, x, maxsplit=1) for x in re.split(_RE_ID_DESC_PAIRS_PATTERN, id_desc_line)]
    for pair in id_desc_pairs:
        if len(pair) != 2:
            pair.append('')
        ids.append(pair[0])
        descs.append(pair[1])
    return (ids, descs, blast_gen_id)

class BlastXmlParser:
    """Parser for the BLAST XML format."""

    def __init__(self, handle, use_raw_query_ids=False, use_raw_hit_ids=False):
        if False:
            return 10
        'Initialize the class.'
        self.xml_iter = iter(ElementTree.iterparse(handle, events=('start', 'end')))
        self._use_raw_query_ids = use_raw_query_ids
        self._use_raw_hit_ids = use_raw_hit_ids
        (self._meta, self._fallback) = self._parse_preamble()

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        'Iterate over BlastXmlParser object yields query results.'
        yield from self._parse_qresult()

    def _parse_preamble(self):
        if False:
            return 10
        'Parse all tag data prior to the first query result (PRIVATE).'
        meta = {}
        fallback = {}
        for (event, elem) in self.xml_iter:
            if event == 'end' and elem.tag in _ELEM_META:
                (attr_name, caster) = _ELEM_META[elem.tag]
                if caster is not str:
                    meta[attr_name] = caster(elem.text)
                else:
                    meta[attr_name] = elem.text
                elem.clear()
                continue
            elif event == 'end' and elem.tag in _ELEM_QRESULT_FALLBACK:
                (attr_name, caster) = _ELEM_QRESULT_FALLBACK[elem.tag]
                if caster is not str:
                    fallback[attr_name] = caster(elem.text)
                else:
                    fallback[attr_name] = elem.text
                elem.clear()
                continue
            if event == 'start' and elem.tag == 'Iteration':
                break
        if meta.get('version') is not None:
            meta['version'] = re.search(_RE_VERSION, meta['version']).group(0)
        return (meta, fallback)

    def _parse_qresult(self):
        if False:
            for i in range(10):
                print('nop')
        'Parse query results (PRIVATE).'
        for (event, qresult_elem) in self.xml_iter:
            if event == 'end' and qresult_elem.tag == 'Iteration':
                query_id = qresult_elem.findtext('Iteration_query-ID')
                if query_id is None:
                    query_id = self._fallback['id']
                query_desc = qresult_elem.findtext('Iteration_query-def')
                if query_desc is None:
                    query_desc = self._fallback['description']
                query_len = qresult_elem.findtext('Iteration_query-len')
                if query_len is None:
                    query_len = self._fallback['len']
                blast_query_id = query_id
                if not self._use_raw_query_ids and query_id.startswith(('Query_', 'lcl|')):
                    id_desc = query_desc.split(' ', 1)
                    query_id = id_desc[0]
                    try:
                        query_desc = id_desc[1]
                    except IndexError:
                        query_desc = ''
                (hit_list, key_list) = ([], [])
                for hit in self._parse_hit(qresult_elem.find('Iteration_hits'), query_id):
                    if hit:
                        if hit.id in key_list:
                            warnings.warn('Renaming hit ID %r to a BLAST-generated ID %r since the ID was already matched by your query %r. Your BLAST database may contain duplicate entries.' % (hit.id, hit.blast_id, query_id), BiopythonParserWarning)
                            hit.description = f'{hit.id} {hit.description}'
                            hit.id = hit.blast_id
                            for hsp in hit:
                                hsp.hit_id = hit.blast_id
                        else:
                            key_list.append(hit.id)
                        hit_list.append(hit)
                qresult = QueryResult(hit_list, query_id)
                qresult.description = query_desc
                qresult.seq_len = int(query_len)
                qresult.blast_id = blast_query_id
                for (key, value) in self._meta.items():
                    setattr(qresult, key, value)
                stat_iter_elem = qresult_elem.find('Iteration_stat')
                if stat_iter_elem is not None:
                    stat_elem = stat_iter_elem.find('Statistics')
                    for (key, val_info) in _ELEM_QRESULT_OPT.items():
                        value = stat_elem.findtext(key)
                        if value is not None:
                            caster = val_info[1]
                            if value is not None and caster is not str:
                                value = caster(value)
                            setattr(qresult, val_info[0], value)
                qresult_elem.clear()
                yield qresult

    def _parse_hit(self, root_hit_elem, query_id):
        if False:
            i = 10
            return i + 15
        'Yield a generator object that transforms Iteration_hits XML elements into Hit objects (PRIVATE).\n\n        :param root_hit_elem: root element of the Iteration_hits tag.\n        :type root_hit_elem: XML element tag\n        :param query_id: QueryResult ID of this Hit\n        :type query_id: string\n\n        '
        if root_hit_elem is None:
            root_hit_elem = []
        for hit_elem in root_hit_elem:
            raw_hit_id = hit_elem.findtext('Hit_id')
            raw_hit_desc = hit_elem.findtext('Hit_def')
            if not self._use_raw_hit_ids:
                (ids, descs, blast_hit_id) = _extract_ids_and_descs(raw_hit_id, raw_hit_desc)
            else:
                (ids, descs, blast_hit_id) = ([raw_hit_id], [raw_hit_desc], raw_hit_id)
            (hit_id, alt_hit_ids) = (ids[0], ids[1:])
            (hit_desc, alt_hit_descs) = (descs[0], descs[1:])
            hsps = list(self._parse_hsp(hit_elem.find('Hit_hsps'), query_id, hit_id))
            hit = Hit(hsps)
            hit.description = hit_desc
            hit._id_alt = alt_hit_ids
            hit._description_alt = alt_hit_descs
            hit.blast_id = blast_hit_id
            for (key, val_info) in _ELEM_HIT.items():
                value = hit_elem.findtext(key)
                if value is not None:
                    caster = val_info[1]
                    if value is not None and caster is not str:
                        value = caster(value)
                    setattr(hit, val_info[0], value)
            hit_elem.clear()
            yield hit

    def _parse_hsp(self, root_hsp_frag_elem, query_id, hit_id):
        if False:
            print('Hello World!')
        'Yield a generator object that transforms Hit_hsps XML elements into HSP objects (PRIVATE).\n\n        :param root_hsp_frag_elem: the ``Hit_hsps`` tag\n        :type root_hsp_frag_elem: XML element tag\n        :param query_id: query ID\n        :type query_id: string\n        :param hit_id: hit ID\n        :type hit_id: string\n\n        '
        if root_hsp_frag_elem is None:
            root_hsp_frag_elem = []
        for hsp_frag_elem in root_hsp_frag_elem:
            coords = {}
            frag = HSPFragment(hit_id, query_id)
            for (key, val_info) in _ELEM_FRAG.items():
                value = hsp_frag_elem.findtext(key)
                caster = val_info[1]
                if value is not None:
                    if key.endswith(('-from', '-to')):
                        coords[val_info[0]] = caster(value)
                        continue
                    elif caster is not str:
                        value = caster(value)
                    setattr(frag, val_info[0], value)
            frag.aln_annotation['similarity'] = hsp_frag_elem.findtext('Hsp_midline')
            for coord_type in ('query', 'hit', 'pattern'):
                start_type = coord_type + '_start'
                end_type = coord_type + '_end'
                try:
                    start = coords[start_type]
                    end = coords[end_type]
                except KeyError:
                    continue
                else:
                    setattr(frag, start_type, min(start, end) - 1)
                    setattr(frag, end_type, max(start, end))
            prog = self._meta.get('program')
            if prog == 'blastn':
                frag.molecule_type = 'DNA'
            elif prog in ['blastp', 'blastx', 'tblastn', 'tblastx']:
                frag.molecule_type = 'protein'
            hsp = HSP([frag])
            for (key, val_info) in _ELEM_HSP.items():
                value = hsp_frag_elem.findtext(key)
                caster = val_info[1]
                if value is not None:
                    if caster is not str:
                        value = caster(value)
                    setattr(hsp, val_info[0], value)
            hsp_frag_elem.clear()
            yield hsp

class BlastXmlIndexer(SearchIndexer):
    """Indexer class for BLAST XML output."""
    _parser = BlastXmlParser
    qstart_mark = b'<Iteration>'
    qend_mark = b'</Iteration>'
    block_size = 16384

    def __init__(self, filename, **kwargs):
        if False:
            print('Hello World!')
        'Initialize the class.'
        SearchIndexer.__init__(self, filename)
        iter_obj = self._parser(self._handle, **kwargs)
        (self._meta, self._fallback) = (iter_obj._meta, iter_obj._fallback)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        "Iterate over BlastXmlIndexer yields qstart_id, start_offset, block's length."
        qstart_mark = self.qstart_mark
        qend_mark = self.qend_mark
        blast_id_mark = b'Query_'
        block_size = self.block_size
        handle = self._handle
        handle.seek(0)
        re_desc = re.compile(b'<Iteration_query-ID>(.*?)</Iteration_query-ID>\\s+?<Iteration_query-def>(.*?)</Iteration_query-def>')
        re_desc_end = re.compile(b'</Iteration_query-def>')
        counter = 0
        while True:
            start_offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if qstart_mark not in line:
                continue
            assert line.count(qstart_mark) == 1, 'XML without line breaks?'
            assert line.lstrip().startswith(qstart_mark), line
            if qend_mark in line:
                block = line
            else:
                block = [line]
                while line and qend_mark not in line:
                    line = handle.readline()
                    assert qstart_mark not in line, line
                    block.append(line)
                assert line.rstrip().endswith(qend_mark), line
                block = b''.join(block)
            assert block.count(qstart_mark) == 1, 'XML without line breaks? %r' % block
            assert block.count(qend_mark) == 1, 'XML without line breaks? %r' % block
            regx = re.search(re_desc, block)
            try:
                qstart_desc = regx.group(2)
                qstart_id = regx.group(1)
            except AttributeError:
                assert re.search(re_desc_end, block)
                qstart_desc = self._fallback['description'].encode()
                qstart_id = self._fallback['id'].encode()
            if qstart_id.startswith(blast_id_mark):
                qstart_id = qstart_desc.split(b' ', 1)[0]
            yield (qstart_id.decode(), start_offset, len(block))
            counter += 1

    def _parse(self, handle):
        if False:
            for i in range(10):
                print('nop')
        'Overwrite SearchIndexer parse (PRIVATE).\n\n        As we need to set the meta and fallback dictionaries to the parser.\n        '
        generator = self._parser(handle, **self._kwargs)
        generator._meta = self._meta
        generator._fallback = self._fallback
        return next(iter(generator))

    def get_raw(self, offset):
        if False:
            while True:
                i = 10
        'Return the raw record from the file as a bytes string.'
        qend_mark = self.qend_mark
        handle = self._handle
        handle.seek(offset)
        qresult_raw = handle.readline()
        assert qresult_raw.lstrip().startswith(self.qstart_mark)
        while qend_mark not in qresult_raw:
            qresult_raw += handle.readline()
        assert qresult_raw.rstrip().endswith(qend_mark)
        assert qresult_raw.count(qend_mark) == 1
        return qresult_raw

class _BlastXmlGenerator(XMLGenerator):
    """Event-based XML Generator."""

    def __init__(self, out, encoding='utf-8', indent=' ', increment=2):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        XMLGenerator.__init__(self, out, encoding)
        self._indent = indent
        self._level = 0
        self._increment = increment
        self._parent_stack = []

    def startDocument(self):
        if False:
            i = 10
            return i + 15
        'Start the XML document.'
        self._write('<?xml version="1.0"?>\n<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" "http://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">\n')

    def startElement(self, name, attrs=None, children=False):
        if False:
            return 10
        'Start an XML element.\n\n        :param name: element name\n        :type name: string\n        :param attrs: element attributes\n        :type attrs: dictionary {string: object}\n        :param children: whether the element has children or not\n        :type children: bool\n\n        '
        if attrs is None:
            attrs = {}
        self.ignorableWhitespace(self._indent * self._level)
        XMLGenerator.startElement(self, name, attrs)

    def endElement(self, name):
        if False:
            while True:
                i = 10
        'End and XML element of the given name.'
        XMLGenerator.endElement(self, name)
        self._write('\n')

    def startParent(self, name, attrs=None):
        if False:
            for i in range(10):
                print('nop')
        'Start an XML element which has children.\n\n        :param name: element name\n        :type name: string\n        :param attrs: element attributes\n        :type attrs: dictionary {string: object}\n\n        '
        if attrs is None:
            attrs = {}
        self.startElement(name, attrs, children=True)
        self._level += self._increment
        self._write('\n')
        self._parent_stack.append(name)

    def endParent(self):
        if False:
            while True:
                i = 10
        'End an XML element with children.'
        name = self._parent_stack.pop()
        self._level -= self._increment
        self.ignorableWhitespace(self._indent * self._level)
        self.endElement(name)

    def startParents(self, *names):
        if False:
            for i in range(10):
                print('nop')
        'Start XML elements without children.'
        for name in names:
            self.startParent(name)

    def endParents(self, num):
        if False:
            i = 10
            return i + 15
        'End XML elements, according to the given number.'
        for i in range(num):
            self.endParent()

    def simpleElement(self, name, content=None):
        if False:
            return 10
        'Create an XML element without children with the given content.'
        self.startElement(name, attrs={})
        if content:
            self.characters(content)
        self.endElement(name)

    def characters(self, content):
        if False:
            for i in range(10):
                print('nop')
        'Replace quotes and apostrophe.'
        content = escape(str(content))
        for (a, b) in (('"', '&quot;'), ("'", '&apos;')):
            content = content.replace(a, b)
        self._write(content)

class BlastXmlWriter:
    """Stream-based BLAST+ XML Writer."""

    def __init__(self, handle, use_raw_query_ids=True, use_raw_hit_ids=True):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.xml = _BlastXmlGenerator(handle, 'utf-8')
        self._use_raw_query_ids = use_raw_query_ids
        self._use_raw_hit_ids = use_raw_hit_ids

    def write_file(self, qresults):
        if False:
            for i in range(10):
                print('nop')
        'Write the XML contents to the output handle.'
        xml = self.xml
        (self.qresult_counter, self.hit_counter, self.hsp_counter, self.frag_counter) = (0, 0, 0, 0)
        first_qresult = next(qresults)
        xml.startDocument()
        xml.startParent('BlastOutput')
        self._write_preamble(first_qresult)
        xml.startParent('BlastOutput_iterations')
        self._write_qresults(chain([first_qresult], qresults))
        xml.endParents(2)
        xml.endDocument()
        return (self.qresult_counter, self.hit_counter, self.hsp_counter, self.frag_counter)

    def _write_elem_block(self, block_name, map_name, obj, opt_dict=None):
        if False:
            i = 10
            return i + 15
        'Write sibling XML elements (PRIVATE).\n\n        :param block_name: common element name prefix\n        :type block_name: string\n        :param map_name: name of mapping between element and attribute names\n        :type map_name: string\n        :param obj: object whose attribute value will be used\n        :type obj: object\n        :param opt_dict: custom element-attribute mapping\n        :type opt_dict: dictionary {string: string}\n\n        '
        if opt_dict is None:
            opt_dict = {}
        for (elem, attr) in _WRITE_MAPS[map_name]:
            elem = block_name + elem
            try:
                content = str(getattr(obj, attr))
            except AttributeError:
                if elem not in _DTD_OPT:
                    raise ValueError(f'Element {elem!r} (attribute {attr!r}) not found')
            else:
                if elem in opt_dict:
                    content = opt_dict[elem]
                self.xml.simpleElement(elem, content)

    def _write_preamble(self, qresult):
        if False:
            while True:
                i = 10
        'Write the XML file preamble (PRIVATE).'
        xml = self.xml
        for (elem, attr) in _WRITE_MAPS['preamble']:
            elem = 'BlastOutput_' + elem
            if elem == 'BlastOutput_param':
                xml.startParent(elem)
                self._write_param(qresult)
                xml.endParent()
                continue
            try:
                content = str(getattr(qresult, attr))
            except AttributeError:
                if elem not in _DTD_OPT:
                    raise ValueError(f'Element {elem} (attribute {attr}) not found')
            else:
                if elem == 'BlastOutput_version':
                    content = f'{qresult.program.upper()} {qresult.version}'
                elif qresult.blast_id:
                    if elem == 'BlastOutput_query-ID':
                        content = qresult.blast_id
                    elif elem == 'BlastOutput_query-def':
                        content = ' '.join([qresult.id, qresult.description]).strip()
                xml.simpleElement(elem, content)

    def _write_param(self, qresult):
        if False:
            i = 10
            return i + 15
        'Write the parameter block of the preamble (PRIVATE).'
        xml = self.xml
        xml.startParent('Parameters')
        self._write_elem_block('Parameters_', 'param', qresult)
        xml.endParent()

    def _write_qresults(self, qresults):
        if False:
            for i in range(10):
                print('nop')
        'Write QueryResult objects into iteration elements (PRIVATE).'
        xml = self.xml
        for (num, qresult) in enumerate(qresults):
            xml.startParent('Iteration')
            xml.simpleElement('Iteration_iter-num', str(num + 1))
            opt_dict = {}
            if self._use_raw_query_ids:
                query_id = qresult.blast_id
                query_desc = qresult.id + ' ' + qresult.description
            else:
                query_id = qresult.id
                query_desc = qresult.description
            opt_dict = {'Iteration_query-ID': query_id, 'Iteration_query-def': query_desc}
            self._write_elem_block('Iteration_', 'qresult', qresult, opt_dict)
            if qresult:
                xml.startParent('Iteration_hits')
                self._write_hits(qresult.hits)
                xml.endParent()
            else:
                xml.simpleElement('Iteration_hits', '')
            xml.startParents('Iteration_stat', 'Statistics')
            self._write_elem_block('Statistics_', 'stat', qresult)
            xml.endParents(2)
            if not qresult:
                xml.simpleElement('Iteration_message', 'No hits found')
            self.qresult_counter += 1
            xml.endParent()

    def _write_hits(self, hits):
        if False:
            for i in range(10):
                print('nop')
        'Write Hit objects (PRIVATE).'
        xml = self.xml
        for (num, hit) in enumerate(hits):
            xml.startParent('Hit')
            xml.simpleElement('Hit_num', str(num + 1))
            opt_dict = {}
            if self._use_raw_hit_ids:
                hit_id = hit.blast_id
                hit_desc = ' >'.join([f'{x} {y}' for (x, y) in zip(hit.id_all, hit.description_all)])
            else:
                hit_id = hit.id
                hit_desc = hit.description + ' >'.join([f'{x} {y}' for (x, y) in zip(hit.id_all[1:], hit.description_all[1:])])
            opt_dict = {'Hit_id': hit_id, 'Hit_def': hit_desc}
            self._write_elem_block('Hit_', 'hit', hit, opt_dict)
            xml.startParent('Hit_hsps')
            self._write_hsps(hit.hsps)
            self.hit_counter += 1
            xml.endParents(2)

    def _write_hsps(self, hsps):
        if False:
            i = 10
            return i + 15
        'Write HSP objects (PRIVATE).'
        xml = self.xml
        for (num, hsp) in enumerate(hsps):
            xml.startParent('Hsp')
            xml.simpleElement('Hsp_num', str(num + 1))
            for (elem, attr) in _WRITE_MAPS['hsp']:
                elem = 'Hsp_' + elem
                try:
                    content = self._adjust_output(hsp, elem, attr)
                except AttributeError:
                    if elem not in _DTD_OPT:
                        raise ValueError(f'Element {elem} (attribute {attr}) not found')
                else:
                    xml.simpleElement(elem, str(content))
            self.hsp_counter += 1
            self.frag_counter += len(hsp.fragments)
            xml.endParent()

    def _adjust_output(self, hsp, elem, attr):
        if False:
            while True:
                i = 10
        'Adjust output to mimic native BLAST+ XML as much as possible (PRIVATE).'
        if attr in ('query_start', 'query_end', 'hit_start', 'hit_end', 'pattern_start', 'pattern_end'):
            content = getattr(hsp, attr) + 1
            if '_start' in attr:
                content = getattr(hsp, attr) + 1
            else:
                content = getattr(hsp, attr)
            if hsp.query_frame != 0 and hsp.hit_frame < 0:
                if attr == 'hit_start':
                    content = getattr(hsp, 'hit_end')
                elif attr == 'hit_end':
                    content = getattr(hsp, 'hit_start') + 1
        elif elem in ('Hsp_hseq', 'Hsp_qseq'):
            content = str(getattr(hsp, attr).seq)
        elif elem == 'Hsp_midline':
            content = hsp.aln_annotation['similarity']
        elif elem in ('Hsp_evalue', 'Hsp_bit-score'):
            content = '%.*g' % (6, getattr(hsp, attr))
        else:
            content = getattr(hsp, attr)
        return content
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()