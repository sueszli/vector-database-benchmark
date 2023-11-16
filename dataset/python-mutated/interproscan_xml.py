"""Bio.SearchIO parser for InterProScan XML output formats."""
import re
from xml.etree import ElementTree
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
_ELEM_HIT = {'name': ('accession', str), 'ac': ('id', str), 'desc': ('description', str)}
_ELEM_HSP = {'score': ('bitscore', float), 'evalue': ('evalue', float)}
_ELEM_FRAG = {'start': ('query_start', int), 'end': ('query_end', int), 'hmm-start': ('hit_start', int), 'hmm-end': ('hit_end', int)}

class InterproscanXmlParser:
    """Parser for the InterProScan XML format."""

    def __init__(self, handle):
        if False:
            return 10
        'Initialize the class.'
        self.xml_iter = iter(ElementTree.iterparse(handle, events=('start', 'end')))
        self._meta = self._parse_header()

    def __iter__(self):
        if False:
            while True:
                i = 10
        'Iterate qresults.'
        yield from self._parse_qresult()

    def _parse_header(self):
        if False:
            for i in range(10):
                print('nop')
        'Parse the header for the InterProScan version (PRIVATE).'
        (event, elem) = next(self.xml_iter)
        meta = {}
        meta['target'] = 'InterPro'
        meta['program'] = 'InterProScan'
        meta['version'] = elem.attrib['interproscan-version']
        self.NS = re.sub('protein-matches', '', elem.tag)
        return meta

    def _parse_qresult(self):
        if False:
            while True:
                i = 10
        'Parse query results (PRIVATE).'
        for (event, elem) in self.xml_iter:
            if event == 'end' and elem.tag == self.NS + 'protein':
                seq = elem.find(self.NS + 'sequence')
                query_seq = seq.text
                xref = elem.find(self.NS + 'xref')
                query_id = xref.attrib['id']
                query_desc = xref.attrib['name']
                hit_list = []
                for hit_new in self._parse_hit(elem.find(self.NS + 'matches'), query_id, query_seq):
                    for hit in hit_list:
                        if hit.id == hit_new.id:
                            for hsp in hit_new.hsps:
                                hit.hsps.append(hsp)
                            break
                    else:
                        hit_list.append(hit_new)
                qresult = QueryResult(hit_list, query_id)
                setattr(qresult, 'description', query_desc)
                for (key, value) in self._meta.items():
                    setattr(qresult, key, value)
                yield qresult

    def _parse_hit(self, root_hit_elem, query_id, query_seq=None):
        if False:
            while True:
                i = 10
        'Parse hit (PRIVATE).'
        if root_hit_elem is None:
            root_hit_elem = []
        for hit_elem in root_hit_elem:
            hit_type = re.sub('%s(\\w+)-match' % self.NS, '\\1', hit_elem.find('.').tag)
            signature = hit_elem.find(self.NS + 'signature')
            hit_id = signature.attrib['ac']
            xrefs = self._parse_xrefs(signature.find(self.NS + 'entry'))
            hsps = list(self._parse_hsp(hit_elem.find(self.NS + 'locations'), query_id, hit_id, query_seq))
            hit = Hit(hsps, hit_id)
            setattr(hit, 'dbxrefs', xrefs)
            for (key, (attr, caster)) in _ELEM_HIT.items():
                value = signature.attrib.get(key)
                if value is not None:
                    setattr(hit, attr, caster(value))
            hit.attributes['Hit type'] = hit_type
            signature_lib = signature.find(self.NS + 'signature-library-release')
            hit.attributes['Target'] = str(signature_lib.attrib.get('library'))
            hit.attributes['Target version'] = str(signature_lib.attrib.get('version'))
            yield hit

    def _parse_hsp(self, root_hsp_elem, query_id, hit_id, query_seq=None):
        if False:
            return 10
        'Parse hsp (PRIVATE).'
        if root_hsp_elem is None:
            root_hsp_elem = []
        for hsp_elem in root_hsp_elem:
            frag = HSPFragment(hit_id, query_id)
            setattr(frag, 'molecule_type', 'protein')
            if query_seq is not None:
                setattr(frag, 'query', query_seq)
            for (key, (attr, caster)) in _ELEM_FRAG.items():
                value = hsp_elem.attrib.get(key)
                if value is not None:
                    if attr.endswith('start'):
                        value = caster(value) - 1
                    if attr == 'query_start':
                        start = int(value)
                    if attr == 'query_end':
                        end = int(value)
                    setattr(frag, attr, caster(value))
            setattr(frag, 'aln_span', end - start)
            hsp = HSP([frag])
            setattr(hsp, 'query_id', query_id)
            setattr(hsp, 'hit_id', hit_id)
            for (key, (attr, caster)) in _ELEM_HSP.items():
                value = hsp_elem.attrib.get(key)
                if value is not None:
                    setattr(hsp, attr, caster(value))
            yield hsp

    def _parse_xrefs(self, root_entry_elem):
        if False:
            for i in range(10):
                print('nop')
        'Parse xrefs (PRIVATE).'
        xrefs = []
        if root_entry_elem is not None:
            xrefs.append('IPR:' + root_entry_elem.attrib['ac'])
        if root_entry_elem is not None:
            xref_elems = []
            xref_elems = xref_elems + root_entry_elem.findall(self.NS + 'go-xref')
            xref_elems = xref_elems + root_entry_elem.findall(self.NS + 'pathway-xref')
            for entry in xref_elems:
                xref = entry.attrib['id']
                if ':' not in xref:
                    xref = entry.attrib['db'] + ':' + xref
                xrefs.append(xref)
        return xrefs
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()