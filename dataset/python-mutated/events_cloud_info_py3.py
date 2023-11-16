from msrest.serialization import Model

class EventsCloudInfo(Model):
    """Cloud info for an event result.

    :param role_name: Role name of the cloud
    :type role_name: str
    :param role_instance: Role instance of the cloud
    :type role_instance: str
    """
    _attribute_map = {'role_name': {'key': 'roleName', 'type': 'str'}, 'role_instance': {'key': 'roleInstance', 'type': 'str'}}

    def __init__(self, *, role_name: str=None, role_instance: str=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(EventsCloudInfo, self).__init__(**kwargs)
        self.role_name = role_name
        self.role_instance = role_instance