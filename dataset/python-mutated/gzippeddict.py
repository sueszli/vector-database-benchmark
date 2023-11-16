from __future__ import annotations
import logging
import pickle
from django.db.models import TextField
from sentry.db.models.utils import Creator
from sentry.utils import json
from sentry.utils.strings import decompress
__all__ = ('GzippedDictField',)
logger = logging.getLogger('sentry')

class GzippedDictField(TextField):
    """
    Slightly different from a JSONField in the sense that the default
    value is a dictionary.
    """

    def contribute_to_class(self, cls, name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a descriptor for backwards compatibility\n        with previous Django behavior.\n        '
        super().contribute_to_class(cls, name)
        setattr(cls, name, Creator(self))

    def to_python(self, value):
        if False:
            while True:
                i = 10
        try:
            if not value:
                return {}
            return json.loads(value, skip_trace=True)
        except (ValueError, TypeError):
            if isinstance(value, str) and value:
                try:
                    value = pickle.loads(decompress(value))
                except Exception as e:
                    logger.exception(e)
                    return {}
            elif not value:
                return {}
            return value

    def from_db_value(self, value, expression, connection):
        if False:
            for i in range(10):
                print('nop')
        return self.to_python(value)

    def get_prep_value(self, value):
        if False:
            print('Hello World!')
        if not value and self.null:
            return None
        elif isinstance(value, bytes):
            value = value.decode('utf-8')
        if value is None and self.null:
            return None
        return json.dumps(value)

    def value_to_string(self, obj):
        if False:
            print('Hello World!')
        return self.get_prep_value(self.value_from_object(obj))