from msrest.serialization import Model

class EndpointKeysDTO(Model):
    """Schema for EndpointKeys generate/refresh operations.

    :param primary_endpoint_key: Primary Access Key.
    :type primary_endpoint_key: str
    :param secondary_endpoint_key: Secondary Access Key.
    :type secondary_endpoint_key: str
    :param installed_version: Current version of runtime.
    :type installed_version: str
    :param last_stable_version: Latest version of runtime.
    :type last_stable_version: str
    :param language: Language setting of runtime.
    :type language: str
    """
    _attribute_map = {'primary_endpoint_key': {'key': 'primaryEndpointKey', 'type': 'str'}, 'secondary_endpoint_key': {'key': 'secondaryEndpointKey', 'type': 'str'}, 'installed_version': {'key': 'installedVersion', 'type': 'str'}, 'last_stable_version': {'key': 'lastStableVersion', 'type': 'str'}, 'language': {'key': 'language', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(EndpointKeysDTO, self).__init__(**kwargs)
        self.primary_endpoint_key = kwargs.get('primary_endpoint_key', None)
        self.secondary_endpoint_key = kwargs.get('secondary_endpoint_key', None)
        self.installed_version = kwargs.get('installed_version', None)
        self.last_stable_version = kwargs.get('last_stable_version', None)
        self.language = kwargs.get('language', None)