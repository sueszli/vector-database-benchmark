from msrest.serialization import Model

class EventsPageViewInfo(Model):
    """The page view information.

    :param name: The name of the page
    :type name: str
    :param url: The URL of the page
    :type url: str
    :param duration: The duration of the page view
    :type duration: str
    :param performance_bucket: The performance bucket of the page view
    :type performance_bucket: str
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}, 'url': {'key': 'url', 'type': 'str'}, 'duration': {'key': 'duration', 'type': 'str'}, 'performance_bucket': {'key': 'performanceBucket', 'type': 'str'}}

    def __init__(self, *, name: str=None, url: str=None, duration: str=None, performance_bucket: str=None, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super(EventsPageViewInfo, self).__init__(**kwargs)
        self.name = name
        self.url = url
        self.duration = duration
        self.performance_bucket = performance_bucket