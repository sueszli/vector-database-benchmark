from msrest.serialization import Model

class FormDocumentReport(Model):
    """FormDocumentReport.

    :param document_name: Reference to the data that the report is for.
    :type document_name: str
    :param pages: Total number of pages trained on.
    :type pages: int
    :param errors: List of errors per page.
    :type errors: list[str]
    :param status: Status of the training operation. Possible values include:
     'success', 'partialSuccess', 'failure'
    :type status: str or ~azure.cognitiveservices.formrecognizer.models.enum
    """
    _attribute_map = {'document_name': {'key': 'documentName', 'type': 'str'}, 'pages': {'key': 'pages', 'type': 'int'}, 'errors': {'key': 'errors', 'type': '[str]'}, 'status': {'key': 'status', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super(FormDocumentReport, self).__init__(**kwargs)
        self.document_name = kwargs.get('document_name', None)
        self.pages = kwargs.get('pages', None)
        self.errors = kwargs.get('errors', None)
        self.status = kwargs.get('status', None)