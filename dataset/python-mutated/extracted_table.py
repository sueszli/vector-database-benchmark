from msrest.serialization import Model

class ExtractedTable(Model):
    """Extraction information about a table
    contained in a page.

    :param id: Table identifier.
    :type id: str
    :param columns: List of columns contained in the table.
    :type columns:
     list[~azure.cognitiveservices.formrecognizer.models.ExtractedTableColumn]
    """
    _attribute_map = {'id': {'key': 'id', 'type': 'str'}, 'columns': {'key': 'columns', 'type': '[ExtractedTableColumn]'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(ExtractedTable, self).__init__(**kwargs)
        self.id = kwargs.get('id', None)
        self.columns = kwargs.get('columns', None)