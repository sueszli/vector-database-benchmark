from . import xmlwriter
schema_root = 'http://schemas.openxmlformats.org'
package_schema = schema_root + '/package/2006/relationships'
document_schema = schema_root + '/officeDocument/2006/relationships'

class Relationships(xmlwriter.XMLwriter):
    """
    A class for writing the Excel XLSX Relationships file.


    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Constructor.\n\n        '
        super(Relationships, self).__init__()
        self.relationships = []
        self.id = 1

    def _assemble_xml_file(self):
        if False:
            i = 10
            return i + 15
        self._xml_declaration()
        self._write_relationships()
        self._xml_close()

    def _add_document_relationship(self, rel_type, target, target_mode=None):
        if False:
            while True:
                i = 10
        rel_type = document_schema + rel_type
        self.relationships.append((rel_type, target, target_mode))

    def _add_package_relationship(self, rel_type, target):
        if False:
            print('Hello World!')
        rel_type = package_schema + rel_type
        self.relationships.append((rel_type, target, None))

    def _add_ms_package_relationship(self, rel_type, target):
        if False:
            for i in range(10):
                print('nop')
        schema = 'http://schemas.microsoft.com/office/2006/relationships'
        rel_type = schema + rel_type
        self.relationships.append((rel_type, target, None))

    def _write_relationships(self):
        if False:
            for i in range(10):
                print('nop')
        attributes = [('xmlns', package_schema)]
        self._xml_start_tag('Relationships', attributes)
        for relationship in self.relationships:
            self._write_relationship(relationship)
        self._xml_end_tag('Relationships')

    def _write_relationship(self, relationship):
        if False:
            i = 10
            return i + 15
        (rel_type, target, target_mode) = relationship
        attributes = [('Id', 'rId' + str(self.id)), ('Type', rel_type), ('Target', target)]
        self.id += 1
        if target_mode:
            attributes.append(('TargetMode', target_mode))
        self._xml_empty_tag('Relationship', attributes)