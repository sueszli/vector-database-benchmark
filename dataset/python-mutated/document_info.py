"""
This class represents the meta-information belonging to a PDF document
"""
import typing
from decimal import Decimal
from borb.io.read.types import Name
from borb.io.write.conformance_level import ConformanceLevel

class DocumentInfo:
    """
    This class represents the meta-information belonging to a PDF document
    """

    def __init__(self, document: 'Document'):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._document: 'Document' = document

    def check_signatures(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        This method verifies the signatures in the Document,\n        it returns True if the signatures match the digest of the Document\n        (or if the Document has no signatures), False otherwise\n        '
        return True

    def get_author(self) -> typing.Optional[str]:
        if False:
            print('Hello World!')
        '\n        (Optional; PDF 1.1) The name of the person who created the document.\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['Author']
        except:
            return None

    def get_conformance_level_upon_create(self) -> typing.Optional[ConformanceLevel]:
        if False:
            i = 10
            return i + 15
        '\n        This function returns the ConformanceLevel that was\n        set for writing operations upon creating the Document instance.\n        This allows the user to specify whether they want to enable things like tagging.\n        A document that was already tagged, and read by borb will of course remain tagged.\n        A document that was not tagged, will similarly not magically be provided with tags.\n        This ConformanceLevel only applies to Document instances that were created by borb.\n        :return:    the ConformanceLevel to be used when writing the PDF\n        '
        return self._document._conformance_level_upon_create

    def get_creation_date(self) -> typing.Optional[str]:
        if False:
            return 10
        '\n        (Optional) The date and time the document was created, in human-\n        readable form (see 7.9.4, “Dates”).\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['CreationDate']
        except:
            return None

    def get_creator(self) -> typing.Optional[str]:
        if False:
            while True:
                i = 10
        '\n        (Optional) If the document was converted to PDF from another format,\n        the name of the conforming product that created the original document\n        from which it was converted.\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['Creator']
        except:
            return None

    def get_file_size(self) -> typing.Optional[Decimal]:
        if False:
            return 10
        '\n        This function returns the filesize (in bytes) of this Document\n        '
        return self._document.get('FileSize', None)

    def get_ids(self) -> typing.Optional[typing.List[str]]:
        if False:
            print('Hello World!')
        '\n        File identifiers shall be defined by the optional ID entry in a PDF file’s trailer dictionary (see 7.5.5, “File Trailer”).\n        The ID entry is optional but should be used. The value of this entry shall be an array of two byte strings. The\n        first byte string shall be a permanent identifier based on the contents of the file at the time it was originally\n        created and shall not change when the file is incrementally updated. The second byte string shall be a\n        changing identifier based on the file’s contents at the time it was last updated. When a file is first written, both\n        identifiers shall be set to the same value. If both identifiers match when a file reference is resolved, it is very\n        likely that the correct and unchanged file has been found. If only the first identifier matches, a different version\n        of the correct file has been found.\n        '
        if 'XRef' in self._document and 'Trailer' in self._document['XRef'] and ('ID' in self._document['XRef']['Trailer']):
            return self._document['XRef']['Trailer']['ID']
        return None

    def get_keywords(self) -> typing.Optional[str]:
        if False:
            print('Hello World!')
        '\n        (Optional; PDF 1.1) Keywords associated with the document.\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['Keywords']
        except:
            return None

    def get_language(self) -> typing.Optional[str]:
        if False:
            return 10
        '\n        (Optional; PDF 1.4) A language identifier that shall specify the\n        natural language for all text in the document except where\n        overridden by language specifications for structure elements or\n        marked content (see 14.9.2, "Natural Language Specification"). If\n        this entry is absent, the language shall be considered unknown.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Lang']
        except:
            return None

    def get_modification_date(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Required if PieceInfo is present in the document catalogue;\n        otherwise optional; PDF 1.1) The date and time the document was\n        most recently modified, in human-readable form (see 7.9.4, “Dates”).\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['ModDate']
        except:
            return None

    def get_number_of_pages(self) -> typing.Optional[Decimal]:
        if False:
            return 10
        '\n        This function returns the number of pages in the Document\n        '
        return self._document['XRef']['Trailer']['Root']['Pages']['Count']

    def get_optional_content_group_names(self) -> typing.List[str]:
        if False:
            while True:
                i = 10
        '\n        This function returns the name(s) of the optional content group(s),\n        suitable for presentation in a reader’s user interface\n        '
        if not self.has_optional_content():
            return []
        return [str(x['Name']) for x in self._document['XRef']['Trailer']['OCProperties'] if 'Name' in x]

    def get_producer(self) -> typing.Optional[str]:
        if False:
            while True:
                i = 10
        '\n        (Optional) If the document was converted to PDF from another format,\n        the name of the conforming product that converted it to PDF.\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['Producer']
        except:
            return None

    def get_subject(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        (Optional; PDF 1.1) The subject of the document.\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['Subject']
        except:
            return None

    def get_title(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        (Optional; PDF 1.1) The document’s title.\n        '
        try:
            return self._document['XRef']['Trailer']['Info']['Title']
        except:
            return None

    def has_optional_content(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Optional content (PDF 1.5) refers to sub-clauses of content in a PDF document that can be selectively viewed\n        or hidden by document authors or consumers. This capability is useful in items such as CAD drawings, layered\n        artwork, maps, and multi-language documents.\n        '
        return 'OCProperties' in self._document['XRef']['Trailer']

    def has_signatures(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        This function returns True if this Document has signatures, False otherwise\n        '
        catalog_dict = self._document['XRef']['Trailer']['Root']
        has_approval_signatures: bool = any([d.get(Name('FT'), None) == Name('Sig') for d in catalog_dict.get(Name('AcroForm'), {}).get(Name('Fields'), []) if isinstance(d, dict)])
        has_certification_signature: bool = any([d.get(Name('FT'), None) == Name('Sig') and Name('DocMDP') in d for d in catalog_dict.get(Name('AcroForm'), {}).get(Name('Fields'), []) if isinstance(d, dict)])
        has_usage_rights_signatures: bool = catalog_dict.get(Name('Perm'), {}).get(Name('UR3'), None) is not None
        return has_approval_signatures or has_certification_signature or has_usage_rights_signatures

class XMPDocumentInfo(DocumentInfo):
    """
    This class represents the (XMP) meta-information belonging to a PDF document
    """

    def __init__(self, document: 'Document'):
        if False:
            print('Hello World!')
        super(XMPDocumentInfo, self).__init__(document)

    def get_author(self) -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        (Optional; PDF 1.1) The name of the person who created the document.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}creator')[0][0][0].text
        except:
            return None

    def get_creation_date(self) -> typing.Optional[str]:
        if False:
            while True:
                i = 10
        '\n        (Optional) The date and time the document was created, in human-\n        readable form (see 7.9.4, “Dates”).\n        '
        try:
            return next(iter([v for (k, v) in self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}Description')[0].attrib.items() if k.endswith('CreateDate')]), None)
        except:
            return None

    def get_creator(self) -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        (Optional) If the document was converted to PDF from another format,\n        the name of the conforming product that created the original document\n        from which it was converted.\n        '
        try:
            return next(iter([v for (k, v) in self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}Description')[0].attrib.items() if k.endswith('CreatorTool')]), None)
        except:
            return None

    def get_document_id(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        The common identifier for all versions and renditions of a document.\n        It should be based on a UUID; see Document and Instance IDs.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}DocumentID')[0].text
        except:
            return None

    def get_instance_id(self) -> typing.Optional[str]:
        if False:
            return 10
        '\n        An identifier for a specific incarnation of a document, updated each time a file is saved.\n        It should be based on a UUID; see Document and Instance IDs.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}InstanceID')[0].text
        except:
            return None

    def get_keywords(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        (Optional; PDF 1.1) Keywords associated with the document.\n        '
        try:
            return next(iter([v for (k, v) in self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}Description')[0].attrib.items() if k.endswith('Keywords')]), None)
        except:
            return None

    def get_metadata_date(self) -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        (Optional) The date and time the metadata for this document was created, in human-\n        readable form (see 7.9.4, “Dates”).\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}MetadataDate')[0].text
        except:
            return None

    def get_modification_date(self) -> typing.Optional[str]:
        if False:
            print('Hello World!')
        '\n        Required if PieceInfo is present in the document catalogue;\n        otherwise optional; PDF 1.1) The date and time the document was\n        most recently modified, in human-readable form (see 7.9.4, “Dates”).\n        '
        try:
            return next(iter([v for (k, v) in self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}Description')[0].attrib.items() if k.endswith('ModifyDate')]), None)
        except:
            return None

    def get_original_document_id(self) -> typing.Optional[str]:
        if False:
            return 10
        '\n        Refer to Part 1, Data Model, Serialization, and Core Properties, for definition.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}OriginalDocumentID')[0].text
        except:
            return None

    def get_producer(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        (Optional) If the document was converted to PDF from another format,\n        the name of the conforming product that converted it to PDF.\n        '
        try:
            return next(iter([v for (k, v) in self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}Description')[0].attrib.items() if k.endswith('Producer')]), None)
        except:
            return None

    def get_publisher(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        (Optional; PDF 1.1) The name of the person/software who/which published the document.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}publisher')[0].text
        except:
            return None

    def get_subject(self) -> typing.Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        (Optional; PDF 1.1) The subject of the document.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}description')[0][0][0].text
        except:
            return None

    def get_title(self) -> typing.Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        (Optional; PDF 1.1) The document’s title.\n        '
        try:
            return self._document['XRef']['Trailer']['Root']['Metadata'].findall('.//{*}title')[0][0][0].text
        except:
            return None