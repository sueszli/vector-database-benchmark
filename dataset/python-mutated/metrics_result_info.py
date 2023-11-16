from msrest.serialization import Model

class MetricsResultInfo(Model):
    """A metric result data.

    :param additional_properties: Unmatched properties from the message are
     deserialized this collection
    :type additional_properties: dict[str, object]
    :param start: Start time of the metric.
    :type start: datetime
    :param end: Start time of the metric.
    :type end: datetime
    :param interval: The interval used to segment the metric data.
    :type interval: timedelta
    :param segments: Segmented metric data (if segmented).
    :type segments: list[~azure.applicationinsights.models.MetricsSegmentInfo]
    """
    _attribute_map = {'additional_properties': {'key': '', 'type': '{object}'}, 'start': {'key': 'start', 'type': 'iso-8601'}, 'end': {'key': 'end', 'type': 'iso-8601'}, 'interval': {'key': 'interval', 'type': 'duration'}, 'segments': {'key': 'segments', 'type': '[MetricsSegmentInfo]'}}

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MetricsResultInfo, self).__init__(**kwargs)
        self.additional_properties = kwargs.get('additional_properties', None)
        self.start = kwargs.get('start', None)
        self.end = kwargs.get('end', None)
        self.interval = kwargs.get('interval', None)
        self.segments = kwargs.get('segments', None)