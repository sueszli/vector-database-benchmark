from msrest.serialization import Model

class TrainResult(Model):
    """Response of the Train API call.

    :param model_id: Identifier of the model.
    :type model_id: str
    :param training_documents: List of documents used to train the model and
     the
     train operation error reported by each.
    :type training_documents:
     list[~azure.cognitiveservices.formrecognizer.models.FormDocumentReport]
    :param errors: Errors returned during the training operation.
    :type errors:
     list[~azure.cognitiveservices.formrecognizer.models.FormOperationError]
    """
    _attribute_map = {'model_id': {'key': 'modelId', 'type': 'str'}, 'training_documents': {'key': 'trainingDocuments', 'type': '[FormDocumentReport]'}, 'errors': {'key': 'errors', 'type': '[FormOperationError]'}}

    def __init__(self, *, model_id: str=None, training_documents=None, errors=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(TrainResult, self).__init__(**kwargs)
        self.model_id = model_id
        self.training_documents = training_documents
        self.errors = errors