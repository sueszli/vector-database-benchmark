"""Bio.SeqIO support for the "uniprot-xml" file format.

See Also:
http://www.uniprot.org

The UniProt XML format essentially replaces the old plain text file format
originally introduced by SwissProt ("swiss" format in Bio.SeqIO).

"""
from xml.etree import ElementTree
from xml.parsers.expat import errors
from Bio import SeqFeature
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
NS = '{http://uniprot.org/uniprot}'
REFERENCE_JOURNAL = '%(name)s %(volume)s:%(first)s-%(last)s(%(pub_date)s)'

def UniprotIterator(source, alphabet=None, return_raw_comments=False):
    if False:
        i = 10
        return i + 15
    'Iterate over UniProt XML as SeqRecord objects.\n\n    parses an XML entry at a time from any UniProt XML file\n    returns a SeqRecord for each iteration\n\n    This generator can be used in Bio.SeqIO\n\n    Argument source is a file-like object or a path to a file.\n\n    Optional argument alphabet should not be used anymore.\n\n    return_raw_comments = True --> comment fields are returned as complete XML to allow further processing\n    skip_parsing_errors = True --> if parsing errors are found, skip to next entry\n    '
    if alphabet is not None:
        raise ValueError('The alphabet argument is no longer supported')
    try:
        for (event, elem) in ElementTree.iterparse(source, events=('start', 'start-ns', 'end')):
            if event == 'start-ns' and (not (elem[1].startswith('http://www.w3.org/') or NS == f'{{{elem[1]}}}')):
                raise ValueError(f"SeqIO format 'uniprot-xml' only parses xml with namespace: {NS} but xml has namespace: {{{elem[1]}}}")
            if event == 'end' and elem.tag == NS + 'entry':
                yield Parser(elem, return_raw_comments=return_raw_comments).parse()
                elem.clear()
    except ElementTree.ParseError as exception:
        if errors.messages[exception.code] == errors.XML_ERROR_NO_ELEMENTS:
            assert exception.position == (1, 0)
            raise ValueError('Empty file.') from None
        else:
            raise

class Parser:
    """Parse a UniProt XML entry to a SeqRecord.

    Optional argument alphabet is no longer used.

    return_raw_comments=True to get back the complete comment field in XML format
    """

    def __init__(self, elem, alphabet=None, return_raw_comments=False):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        self.entry = elem
        self.return_raw_comments = return_raw_comments

    def parse(self):
        if False:
            return 10
        'Parse the input.'
        assert self.entry.tag == NS + 'entry'

        def append_to_annotations(key, value):
            if False:
                i = 10
                return i + 15
            if key not in self.ParsedSeqRecord.annotations:
                self.ParsedSeqRecord.annotations[key] = []
            if value not in self.ParsedSeqRecord.annotations[key]:
                self.ParsedSeqRecord.annotations[key].append(value)

        def _parse_name(element):
            if False:
                while True:
                    i = 10
            self.ParsedSeqRecord.name = element.text
            self.ParsedSeqRecord.dbxrefs.append(self.dbname + ':' + element.text)

        def _parse_accession(element):
            if False:
                i = 10
                return i + 15
            append_to_annotations('accessions', element.text)
            self.ParsedSeqRecord.dbxrefs.append(self.dbname + ':' + element.text)

        def _parse_protein(element):
            if False:
                while True:
                    i = 10
            'Parse protein names (PRIVATE).'
            descr_set = False
            for protein_element in element:
                if protein_element.tag in [NS + 'recommendedName', NS + 'submittedName', NS + 'alternativeName']:
                    for rec_name in protein_element:
                        ann_key = '%s_%s' % (protein_element.tag.replace(NS, ''), rec_name.tag.replace(NS, ''))
                        append_to_annotations(ann_key, rec_name.text)
                        if rec_name.tag == NS + 'fullName' and (not descr_set):
                            self.ParsedSeqRecord.description = rec_name.text
                            descr_set = True
                elif protein_element.tag == NS + 'component':
                    pass
                elif protein_element.tag == NS + 'domain':
                    pass

        def _parse_gene(element):
            if False:
                for i in range(10):
                    print('nop')
            for genename_element in element:
                if 'type' in genename_element.attrib:
                    ann_key = 'gene_%s_%s' % (genename_element.tag.replace(NS, ''), genename_element.attrib['type'])
                    if genename_element.attrib['type'] == 'primary':
                        self.ParsedSeqRecord.annotations[ann_key] = genename_element.text
                    else:
                        append_to_annotations(ann_key, genename_element.text)

        def _parse_geneLocation(element):
            if False:
                while True:
                    i = 10
            append_to_annotations('geneLocation', element.attrib['type'])

        def _parse_organism(element):
            if False:
                for i in range(10):
                    print('nop')
            organism_name = com_name = sci_name = ''
            for organism_element in element:
                if organism_element.tag == NS + 'name':
                    if organism_element.text:
                        if organism_element.attrib['type'] == 'scientific':
                            sci_name = organism_element.text
                        elif organism_element.attrib['type'] == 'common':
                            com_name = organism_element.text
                        else:
                            append_to_annotations('organism_name', organism_element.text)
                elif organism_element.tag == NS + 'dbReference':
                    self.ParsedSeqRecord.dbxrefs.append(organism_element.attrib['type'] + ':' + organism_element.attrib['id'])
                elif organism_element.tag == NS + 'lineage':
                    for taxon_element in organism_element:
                        if taxon_element.tag == NS + 'taxon':
                            append_to_annotations('taxonomy', taxon_element.text)
            if sci_name and com_name:
                organism_name = f'{sci_name} ({com_name})'
            elif sci_name:
                organism_name = sci_name
            elif com_name:
                organism_name = com_name
            self.ParsedSeqRecord.annotations['organism'] = organism_name

        def _parse_organismHost(element):
            if False:
                return 10
            for organism_element in element:
                if organism_element.tag == NS + 'name':
                    append_to_annotations('organism_host', organism_element.text)

        def _parse_keyword(element):
            if False:
                print('Hello World!')
            append_to_annotations('keywords', element.text)

        def _parse_comment(element):
            if False:
                for i in range(10):
                    print('nop')
            'Parse comments (PRIVATE).\n\n            Comment fields are very heterogeneus. each type has his own (frequently mutated) schema.\n            To store all the contained data, more complex data structures are needed, such as\n            annotated dictionaries. This is left to end user, by optionally setting:\n\n            return_raw_comments=True\n\n            The original XML is returned in the annotation fields.\n\n            Available comment types at december 2009:\n             - "allergen"\n             - "alternative products"\n             - "biotechnology"\n             - "biophysicochemical properties"\n             - "catalytic activity"\n             - "caution"\n             - "cofactor"\n             - "developmental stage"\n             - "disease"\n             - "domain"\n             - "disruption phenotype"\n             - "enzyme regulation"\n             - "function"\n             - "induction"\n             - "miscellaneous"\n             - "pathway"\n             - "pharmaceutical"\n             - "polymorphism"\n             - "PTM"\n             - "RNA editing"\n             - "similarity"\n             - "subcellular location"\n             - "sequence caution"\n             - "subunit"\n             - "tissue specificity"\n             - "toxic dose"\n             - "online information"\n             - "mass spectrometry"\n             - "interaction"\n\n            '
            simple_comments = ['allergen', 'biotechnology', 'biophysicochemical properties', 'catalytic activity', 'caution', 'cofactor', 'developmental stage', 'disease', 'domain', 'disruption phenotype', 'enzyme regulation', 'function', 'induction', 'miscellaneous', 'pathway', 'pharmaceutical', 'polymorphism', 'PTM', 'RNA editing', 'similarity', 'subunit', 'tissue specificity', 'toxic dose']
            if element.attrib['type'] in simple_comments:
                ann_key = f"comment_{element.attrib['type'].replace(' ', '')}"
                for text_element in element.iter(NS + 'text'):
                    if text_element.text:
                        append_to_annotations(ann_key, text_element.text)
            elif element.attrib['type'] == 'subcellular location':
                for subloc_element in element.iter(NS + 'subcellularLocation'):
                    for el in subloc_element:
                        if el.text:
                            ann_key = 'comment_%s_%s' % (element.attrib['type'].replace(' ', ''), el.tag.replace(NS, ''))
                            append_to_annotations(ann_key, el.text)
            elif element.attrib['type'] == 'interaction':
                for interact_element in element.iter(NS + 'interactant'):
                    ann_key = f"comment_{element.attrib['type']}_intactId"
                    append_to_annotations(ann_key, interact_element.attrib['intactId'])
            elif element.attrib['type'] == 'alternative products':
                for alt_element in element.iter(NS + 'isoform'):
                    ann_key = 'comment_%s_isoform' % element.attrib['type'].replace(' ', '')
                    for id_element in alt_element.iter(NS + 'id'):
                        append_to_annotations(ann_key, id_element.text)
            elif element.attrib['type'] == 'mass spectrometry':
                ann_key = f"comment_{element.attrib['type'].replace(' ', '')}"
                start = end = 0
                for el in element.iter(NS + 'location'):
                    pos_els = list(el.iter(NS + 'position'))
                    try:
                        if pos_els:
                            end = int(pos_els[0].attrib['position'])
                            start = end - 1
                        else:
                            start = int(next(el.iter(NS + 'begin')).attrib['position'])
                            start -= 1
                            end = int(next(el.iter(NS + 'end')).attrib['position'])
                    except (ValueError, KeyError):
                        pass
                mass = element.attrib['mass']
                method = element.attrib['method']
                if start == end == 0:
                    append_to_annotations(ann_key, f'undefined:{mass}|{method}')
                else:
                    append_to_annotations(ann_key, f'{start}..{end}:{mass}|{method}')
            elif element.attrib['type'] == 'sequence caution':
                pass
            elif element.attrib['type'] == 'online information':
                for link_element in element.iter(NS + 'link'):
                    ann_key = f"comment_{element.attrib['type'].replace(' ', '')}"
                    for id_element in link_element.iter(NS + 'link'):
                        append_to_annotations(ann_key, f"{element.attrib['name']}@{link_element.attrib['uri']}")
            if self.return_raw_comments:
                ann_key = f"comment_{element.attrib['type'].replace(' ', '')}_xml"
                append_to_annotations(ann_key, ElementTree.tostring(element))

        def _parse_dbReference(element):
            if False:
                return 10
            self.ParsedSeqRecord.dbxrefs.append(element.attrib['type'] + ':' + element.attrib['id'])
            if 'type' in element.attrib:
                if element.attrib['type'] == 'PDB':
                    method = ''
                    resolution = ''
                    for ref_element in element:
                        if ref_element.tag == NS + 'property':
                            dat_type = ref_element.attrib['type']
                            if dat_type == 'method':
                                method = ref_element.attrib['value']
                            if dat_type == 'resolution':
                                resolution = ref_element.attrib['value']
                            if dat_type == 'chains':
                                pairs = ref_element.attrib['value'].split(',')
                                for elem in pairs:
                                    pair = elem.strip().split('=')
                                    if pair[1] != '-':
                                        feature = SeqFeature.SeqFeature()
                                        feature.type = element.attrib['type']
                                        feature.qualifiers['name'] = element.attrib['id']
                                        feature.qualifiers['method'] = method
                                        feature.qualifiers['resolution'] = resolution
                                        feature.qualifiers['chains'] = pair[0].split('/')
                                        start = int(pair[1].split('-')[0]) - 1
                                        end = int(pair[1].split('-')[1])
                                        feature.location = SeqFeature.SimpleLocation(start, end)
            for ref_element in element:
                if ref_element.tag == NS + 'property':
                    pass

        def _parse_reference(element):
            if False:
                print('Hello World!')
            reference = SeqFeature.Reference()
            authors = []
            scopes = []
            tissues = []
            journal_name = ''
            pub_type = ''
            pub_date = ''
            for ref_element in element:
                if ref_element.tag == NS + 'citation':
                    pub_type = ref_element.attrib['type']
                    if pub_type == 'submission':
                        pub_type += ' to the ' + ref_element.attrib['db']
                    if 'name' in ref_element.attrib:
                        journal_name = ref_element.attrib['name']
                    pub_date = ref_element.attrib.get('date', '')
                    j_volume = ref_element.attrib.get('volume', '')
                    j_first = ref_element.attrib.get('first', '')
                    j_last = ref_element.attrib.get('last', '')
                    for cit_element in ref_element:
                        if cit_element.tag == NS + 'title':
                            reference.title = cit_element.text
                        elif cit_element.tag == NS + 'authorList':
                            for person_element in cit_element:
                                authors.append(person_element.attrib['name'])
                        elif cit_element.tag == NS + 'dbReference':
                            self.ParsedSeqRecord.dbxrefs.append(cit_element.attrib['type'] + ':' + cit_element.attrib['id'])
                            if cit_element.attrib['type'] == 'PubMed':
                                reference.pubmed_id = cit_element.attrib['id']
                            elif ref_element.attrib['type'] == 'MEDLINE':
                                reference.medline_id = cit_element.attrib['id']
                elif ref_element.tag == NS + 'scope':
                    scopes.append(ref_element.text)
                elif ref_element.tag == NS + 'source':
                    for source_element in ref_element:
                        if source_element.tag == NS + 'tissue':
                            tissues.append(source_element.text)
            if scopes:
                scopes_str = 'Scope: ' + ', '.join(scopes)
            else:
                scopes_str = ''
            if tissues:
                tissues_str = 'Tissue: ' + ', '.join(tissues)
            else:
                tissues_str = ''
            reference.location = []
            reference.authors = ', '.join(authors)
            if journal_name:
                if pub_date and j_volume and j_first and j_last:
                    reference.journal = REFERENCE_JOURNAL % {'name': journal_name, 'volume': j_volume, 'first': j_first, 'last': j_last, 'pub_date': pub_date}
                else:
                    reference.journal = journal_name
            reference.comment = ' | '.join((pub_type, pub_date, scopes_str, tissues_str))
            append_to_annotations('references', reference)

        def _parse_position(element, offset=0):
            if False:
                while True:
                    i = 10
            try:
                position = int(element.attrib['position']) + offset
            except KeyError:
                position = None
            status = element.attrib.get('status', '')
            if status == 'unknown':
                assert position is None
                return SeqFeature.UnknownPosition()
            elif not status:
                return SeqFeature.ExactPosition(position)
            elif status == 'greater than':
                return SeqFeature.AfterPosition(position)
            elif status == 'less than':
                return SeqFeature.BeforePosition(position)
            elif status == 'uncertain':
                return SeqFeature.UncertainPosition(position)
            else:
                raise NotImplementedError(f'Position status {status!r}')

        def _parse_feature(element):
            if False:
                i = 10
                return i + 15
            feature = SeqFeature.SeqFeature()
            for (k, v) in element.attrib.items():
                feature.qualifiers[k] = v
            feature.type = element.attrib.get('type', '')
            if 'id' in element.attrib:
                feature.id = element.attrib['id']
            for feature_element in element:
                if feature_element.tag == NS + 'location':
                    position_elements = feature_element.findall(NS + 'position')
                    if position_elements:
                        element = position_elements[0]
                        start_position = _parse_position(element, -1)
                        end_position = _parse_position(element)
                    else:
                        element = feature_element.findall(NS + 'begin')[0]
                        start_position = _parse_position(element, -1)
                        element = feature_element.findall(NS + 'end')[0]
                        end_position = _parse_position(element)
                    feature.location = SeqFeature.SimpleLocation(start_position, end_position)
                else:
                    try:
                        feature.qualifiers[feature_element.tag.replace(NS, '')] = feature_element.text
                    except Exception:
                        pass
            self.ParsedSeqRecord.features.append(feature)

        def _parse_proteinExistence(element):
            if False:
                i = 10
                return i + 15
            append_to_annotations('proteinExistence', element.attrib['type'])

        def _parse_evidence(element):
            if False:
                return 10
            for (k, v) in element.attrib.items():
                ann_key = k
                append_to_annotations(ann_key, v)

        def _parse_sequence(element):
            if False:
                i = 10
                return i + 15
            for (k, v) in element.attrib.items():
                if k in ('length', 'mass', 'version'):
                    self.ParsedSeqRecord.annotations[f'sequence_{k}'] = int(v)
                else:
                    self.ParsedSeqRecord.annotations[f'sequence_{k}'] = v
            self.ParsedSeqRecord.seq = Seq(''.join(element.text.split()))
            self.ParsedSeqRecord.annotations['molecule_type'] = 'protein'
        self.ParsedSeqRecord = SeqRecord('', id='')
        self.dbname = self.entry.attrib.get('dataset', 'UnknownDataset')
        for (k, v) in self.entry.attrib.items():
            if k in 'version':
                self.ParsedSeqRecord.annotations[k] = int(v)
            else:
                self.ParsedSeqRecord.annotations[k] = v
        for element in self.entry:
            if element.tag == NS + 'name':
                _parse_name(element)
            elif element.tag == NS + 'accession':
                _parse_accession(element)
            elif element.tag == NS + 'protein':
                _parse_protein(element)
            elif element.tag == NS + 'gene':
                _parse_gene(element)
            elif element.tag == NS + 'geneLocation':
                _parse_geneLocation(element)
            elif element.tag == NS + 'organism':
                _parse_organism(element)
            elif element.tag == NS + 'organismHost':
                _parse_organismHost(element)
            elif element.tag == NS + 'keyword':
                _parse_keyword(element)
            elif element.tag == NS + 'comment':
                _parse_comment(element)
            elif element.tag == NS + 'dbReference':
                _parse_dbReference(element)
            elif element.tag == NS + 'reference':
                _parse_reference(element)
            elif element.tag == NS + 'feature':
                _parse_feature(element)
            elif element.tag == NS + 'proteinExistence':
                _parse_proteinExistence(element)
            elif element.tag == NS + 'evidence':
                _parse_evidence(element)
            elif element.tag == NS + 'sequence':
                _parse_sequence(element)
            else:
                pass
        self.ParsedSeqRecord.dbxrefs = sorted(set(self.ParsedSeqRecord.dbxrefs))
        if not self.ParsedSeqRecord.id:
            self.ParsedSeqRecord.id = self.ParsedSeqRecord.annotations['accessions'][0]
        return self.ParsedSeqRecord