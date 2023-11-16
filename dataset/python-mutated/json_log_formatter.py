import json
import logging
import datetime

class JSONFormatter(logging.Formatter):
    """
    Render logs as JSON.

    To add details to a log record, store them in a ``event_data``
    custom attribute. This dict is merged into the event.

    """

    def __init__(self):
        if False:
            return 10
        pass

    def format(self, record):
        if False:
            while True:
                i = 10
        event = {'timestamp': self.getTimestamp(record.created), 'message': record.getMessage(), 'level': record.levelname, 'logger': record.name}
        event_data = getattr(record, 'event_data', None)
        if event_data:
            event.update(event_data)
        if record.exc_info:
            event['exc_info'] = self.formatException(record.exc_info)
        if record.stack_info:
            event['stack_info'] = self.formatStack(record.stack_info)
        return json.dumps(event)

    def getTimestamp(self, created):
        if False:
            i = 10
            return i + 15
        return datetime.datetime.utcfromtimestamp(created).isoformat()