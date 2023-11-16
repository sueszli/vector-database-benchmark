from msrest.serialization import Model

class RewardRequest(Model):
    """Reward given to a rank response.

    All required parameters must be populated in order to send to Azure.

    :param value: Required. Reward to be assigned to an action. Value should
     be between -1 and 1 inclusive.
    :type value: float
    """
    _validation = {'value': {'required': True}}
    _attribute_map = {'value': {'key': 'value', 'type': 'float'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(RewardRequest, self).__init__(**kwargs)
        self.value = kwargs.get('value', None)