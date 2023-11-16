from msrest.serialization import Model

class ResponseBase(Model):
    """Response base.

    You probably want to use the sub-classes and not this class directly. Known
    sub-classes are: Identifiable

    All required parameters must be populated in order to send to Azure.

    :param _type: Required. Constant filled by server.
    :type _type: str
    """
    _validation = {'_type': {'required': True}}
    _attribute_map = {'_type': {'key': '_type', 'type': 'str'}}
    _subtype_map = {'_type': {'Identifiable': 'Identifiable'}}

    def __init__(self, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(ResponseBase, self).__init__(**kwargs)
        self._type = None