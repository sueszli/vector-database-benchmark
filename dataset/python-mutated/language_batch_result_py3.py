from msrest.serialization import Model

class LanguageBatchResult(Model):
    """LanguageBatchResult.

    Variables are only populated by the server, and will be ignored when
    sending a request.

    :ivar documents: Response by document
    :vartype documents:
     list[~azure.cognitiveservices.language.textanalytics.models.LanguageBatchResultItem]
    :ivar errors: Errors and Warnings by document
    :vartype errors:
     list[~azure.cognitiveservices.language.textanalytics.models.ErrorRecord]
    :ivar statistics: (Optional) if showStats=true was specified in the
     request this field will contain information about the request payload.
    :vartype statistics:
     ~azure.cognitiveservices.language.textanalytics.models.RequestStatistics
    """
    _validation = {'documents': {'readonly': True}, 'errors': {'readonly': True}, 'statistics': {'readonly': True}}
    _attribute_map = {'documents': {'key': 'documents', 'type': '[LanguageBatchResultItem]'}, 'errors': {'key': 'errors', 'type': '[ErrorRecord]'}, 'statistics': {'key': 'statistics', 'type': 'RequestStatistics'}}

    def __init__(self, **kwargs) -> None:
        if False:
            print('Hello World!')
        super(LanguageBatchResult, self).__init__(**kwargs)
        self.documents = None
        self.errors = None
        self.statistics = None