from msrest.serialization import Model

class ExtractedTableColumn(Model):
    """Extraction information of a column in
    a table.

    :param header: List of extracted tokens for the column header.
    :type header:
     list[~azure.cognitiveservices.formrecognizer.models.ExtractedToken]
    :param entries: Extracted text for each cell of a column. Each cell
     in the column can have a list of one or more tokens.
    :type entries:
     list[list[~azure.cognitiveservices.formrecognizer.models.ExtractedToken]]
    """
    _attribute_map = {'header': {'key': 'header', 'type': '[ExtractedToken]'}, 'entries': {'key': 'entries', 'type': '[[ExtractedToken]]'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ExtractedTableColumn, self).__init__(**kwargs)
        self.header = kwargs.get('header', None)
        self.entries = kwargs.get('entries', None)