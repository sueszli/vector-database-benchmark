from msrest.serialization import Model

class EventsAiInfo(Model):
    """AI related application info for an event result.

    :param i_key: iKey of the app
    :type i_key: str
    :param app_name: Name of the application
    :type app_name: str
    :param app_id: ID of the application
    :type app_id: str
    :param sdk_version: SDK version of the application
    :type sdk_version: str
    """
    _attribute_map = {'i_key': {'key': 'iKey', 'type': 'str'}, 'app_name': {'key': 'appName', 'type': 'str'}, 'app_id': {'key': 'appId', 'type': 'str'}, 'sdk_version': {'key': 'sdkVersion', 'type': 'str'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(EventsAiInfo, self).__init__(**kwargs)
        self.i_key = kwargs.get('i_key', None)
        self.app_name = kwargs.get('app_name', None)
        self.app_id = kwargs.get('app_id', None)
        self.sdk_version = kwargs.get('sdk_version', None)