import logging
from raven.handlers.logging import SentryHandler
DEFAULT_SENTRY_ENABLED = False

class SwitchedSentryHandler(SentryHandler):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            self.enabled = bool(kwargs.pop('enabled', DEFAULT_SENTRY_ENABLED))
        except Exception:
            self.enabled = DEFAULT_SENTRY_ENABLED
        self._user = {}
        super().__init__(*args, **kwargs)

    def emit(self, record):
        if False:
            print('Hello World!')
        if not self.enabled:
            return None
        self.client.context.merge({'user': self._user})
        return super().emit(record)

    def update_user(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._user.update(kwargs)

    def set_version(self, version=None, env=None):
        if False:
            print('Hello World!')
        if version is not None:
            self.client.release = version
        if env is not None:
            self.client.environment = env

    def set_enabled(self, value):
        if False:
            print('Hello World!')
        try:
            self.enabled = bool(value)
        except Exception:
            self.enabled = DEFAULT_SENTRY_ENABLED

class SentryMetricsFilter(logging.Filter):

    def filter(self, record):
        if False:
            i = 10
            return i + 15
        return record.getMessage().startswith('METRIC')