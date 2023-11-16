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

    def __init__(self, *, id: str=None, columns=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(ExtractedTable, self).__init__(**kwargs)
        self.id = id
        self.columns = columns