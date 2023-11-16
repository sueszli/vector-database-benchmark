import logging
import os
import sys
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum, auto
from hashlib import md5
from typing import Any, Dict, Optional
import sentry_sdk
from faker import Faker
from sentry_sdk.integrations.logging import LoggingIntegration, ignore_logger
from sentry_sdk.integrations.threading import ThreadingIntegration
from tribler.core import version
from tribler.core.sentry_reporter.sentry_tools import delete_item, get_first_item, get_last_item, get_value, parse_last_core_output
VALUE = 'value'
TYPE = 'type'
LAST_CORE_OUTPUT = 'last_core_output'
LAST_PROCESSES = 'last_processes'
PLATFORM = 'platform'
OS = 'os'
MACHINE = 'machine'
COMMENTS = 'comments'
TRIBLER = 'Tribler'
NAME = 'name'
VERSION = 'version'
BROWSER = 'browser'
STACKTRACE = '_stacktrace'
STACKTRACE_EXTRA = f'{STACKTRACE}_extra'
STACKTRACE_CONTEXT = f'{STACKTRACE}_context'
SYSINFO = 'sysinfo'
OS_ENVIRON = 'os.environ'
SYS_ARGV = 'sys.argv'
TAGS = 'tags'
CONTEXTS = 'contexts'
EXTRA = 'extra'
BREADCRUMBS = 'breadcrumbs'
LOGENTRY = 'logentry'
REPORTER = 'reporter'
VALUES = 'values'
RELEASE = 'release'
EXCEPTION = 'exception'
ADDITIONAL_INFORMATION = 'additional_information'

class SentryStrategy(Enum):
    """Class describes all available Sentry Strategies

    SentryReporter can work with 3 strategies:
    1. Send reports are allowed
    2. Send reports are allowed with a confirmation dialog
    3. Send reports are suppressed (but the last event will be stored)
    """
    SEND_ALLOWED = auto()
    SEND_ALLOWED_WITH_CONFIRMATION = auto()
    SEND_SUPPRESSED = auto()

@contextmanager
def this_sentry_strategy(reporter, strategy: SentryStrategy):
    if False:
        i = 10
        return i + 15
    saved_strategy = reporter.thread_strategy.get()
    try:
        reporter.thread_strategy.set(strategy)
        yield reporter
    finally:
        reporter.thread_strategy.set(saved_strategy)

class SentryReporter:
    """SentryReporter designed for sending reports to the Sentry server from
    a Tribler Client.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.scrubber = None
        self.last_event = None
        self.ignored_exceptions = [KeyboardInterrupt, SystemExit]
        self.global_strategy = SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION
        self.thread_strategy = ContextVar('context_strategy', default=None)
        self.collecting_breadcrumbs_allowed = True
        self.additional_information = defaultdict(dict)
        self._sentry_logger_name = 'SentryReporter'
        self._logger = logging.getLogger(self._sentry_logger_name)

    def init(self, sentry_url='', release_version='', scrubber=None, strategy=SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION):
        if False:
            while True:
                i = 10
        "Initialization.\n\n        This method should be called in each process that uses SentryReporter.\n\n        Args:\n            sentry_url: URL for Sentry server. If it is empty then Sentry's\n                sending mechanism will not be initialized.\n\n            scrubber: a class that will be used for scrubbing sending events.\n                Only a single method should be implemented in the class:\n                ```\n                    def scrub_event(self, event):\n                        pass\n                ```\n            release_version: string that represents a release version.\n                See Also: https://docs.sentry.io/platforms/python/configuration/releases/\n            strategy: a Sentry strategy for sending events (see class Strategy\n                for more information)\n        Returns:\n            Sentry Guard.\n        "
        self._logger.debug(f'Init: {sentry_url}')
        self.scrubber = scrubber
        self.global_strategy = strategy
        rv = sentry_sdk.init(sentry_url, release=release_version, integrations=[LoggingIntegration(level=logging.INFO, event_level=None), ThreadingIntegration(propagate_hub=True)], auto_enabling_integrations=False, before_send=self._before_send, before_breadcrumb=self._before_breadcrumb, ignore_errors=[KeyboardInterrupt, ConnectionResetError])
        ignore_logger(self._sentry_logger_name)
        return rv

    def ignore_logger(self, logger_name: str):
        if False:
            while True:
                i = 10
        self._logger.debug(f'Ignore logger: {logger_name}')
        ignore_logger(logger_name)

    def add_breadcrumb(self, message='', category='', level='info', **kwargs):
        if False:
            while True:
                i = 10
        'Adds a breadcrumb for current Sentry client.\n\n        It is necessary to specify a message, a category and a level to make this\n        breadcrumb visible in Sentry server.\n\n        Args:\n            **kwargs: named arguments that will be added to Sentry event as well\n        '
        crumb = {'message': message, 'category': category, 'level': level}
        self._logger.debug(f'Add the breadcrumb: {crumb}')
        return sentry_sdk.add_breadcrumb(crumb, **kwargs)

    def send_event(self, event: Dict, tags: Optional[Dict[str, Any]]=None, info: Optional[Dict[str, Any]]=None, last_core_output: Optional[str]=None, tribler_version='<not set>'):
        if False:
            return 10
        "Send the event to the Sentry server\n\n        This method\n            1. Enable Sentry's sending mechanism.\n            2. Extend sending event by the information from post_data.\n            3. Send the event.\n            4. Disables Sentry's sending mechanism.\n\n        Scrubbing the information will be performed in the `_before_send` method.\n\n        During the execution of this method, all unhandled exceptions that\n        will be raised, will be sent to Sentry automatically.\n\n        Args:\n            event: event to send. It should be taken from SentryReporter\n            tags: tags that will be added to the event\n            info: additional information that will be added to the event\n            last_core_output: string that represents last core output\n            tribler_version: Tribler version\n\n        Returns:\n            Event that was sent to Sentry server\n        "
        self._logger.info(f'Send: {tags}, {info}, {event}')
        tags = tags or {}
        info = info or {}
        if CONTEXTS not in event:
            event[CONTEXTS] = {}
        if TAGS not in event:
            event[TAGS] = {}
        event[TAGS].update(tags)
        if last_core_output:
            info[LAST_CORE_OUTPUT] = last_core_output.split('\n')
            if (last_core_exception := parse_last_core_output(last_core_output)):
                exceptions = event.get(EXCEPTION, {})
                gui_exception = get_last_item(exceptions.get(VALUES, []), {})
                core_exception = {TYPE: last_core_exception.type, VALUE: last_core_exception.message}
                delete_item(gui_exception, 'stacktrace')
                exceptions[VALUES] = [gui_exception, core_exception]
        event[CONTEXTS][REPORTER] = info
        event[CONTEXTS][BROWSER] = {VERSION: tribler_version, NAME: TRIBLER}
        with this_sentry_strategy(self, SentryStrategy.SEND_ALLOWED):
            sentry_sdk.capture_event(event)
        return event

    def get_confirmation(self, exception):
        if False:
            for i in range(10):
                print('nop')
        'Get confirmation on sending exception to the Team.\n\n        There are two message boxes, that will be triggered:\n        1. Message box with the error_text\n        2. Message box with confirmation about sending this report to the Tribler\n            team.\n\n        Args:\n            exception: exception to be sent.\n        '
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox
        except ImportError:
            self._logger.debug('PyQt5 is not available. User confirmation is not possible.')
            return False
        self._logger.debug(f'Get confirmation: {exception}')
        _ = QApplication(sys.argv)
        messagebox = QMessageBox(icon=QMessageBox.Critical, text=f'{exception}.')
        messagebox.setWindowTitle('Error')
        messagebox.exec()
        messagebox = QMessageBox(icon=QMessageBox.Question, text='Do you want to send this crash report to the Tribler team? We anonymize all your data, who you are and what you downloaded.')
        messagebox.setWindowTitle('Error')
        messagebox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        return messagebox.exec() == QMessageBox.Yes

    def capture_exception(self, exception):
        if False:
            i = 10
            return i + 15
        self._logger.info(f'Capture exception: {exception}')
        sentry_sdk.capture_exception(exception)

    def event_from_exception(self, exception) -> Dict:
        if False:
            return 10
        'This function format the exception by passing it through sentry\n        Args:\n            exception: an exception that will be passed to `sentry_sdk.capture_exception(exception)`\n\n        Returns:\n            the event that has been saved in `_before_send` method\n        '
        self._logger.debug(f'Event from exception: {exception}')
        if not exception:
            return {}
        with this_sentry_strategy(self, SentryStrategy.SEND_SUPPRESSED):
            sentry_sdk.capture_exception(exception)
            return self.last_event

    def set_user(self, user_id):
        if False:
            while True:
                i = 10
        'Set the user to identify the event on a Sentry server\n\n        The algorithm is the following:\n        1. Calculate hash from `user_id`.\n        2. Generate fake user, based on the hash.\n\n        No real `user_id` will be used in Sentry.\n\n        Args:\n            user_id: Real user id.\n\n        Returns:\n            Generated user (dictionary: {id, username}).\n        '
        user_id_hash = md5(user_id).hexdigest()
        self._logger.debug(f'Set user: {user_id_hash}')
        Faker.seed(user_id_hash)
        user_name = Faker().name()
        user = {'id': user_id_hash, 'username': user_name}
        sentry_sdk.set_user(user)
        return user

    def get_actual_strategy(self):
        if False:
            print('Hello World!')
        'This method is used to determine actual strategy.\n\n        Strategy can be global: self.strategy\n        and local: self._context_strategy.\n\n        Returns: the local strategy if it is defined, the global strategy otherwise\n        '
        strategy = self.thread_strategy.get()
        return strategy if strategy else self.global_strategy

    @staticmethod
    def get_sentry_url() -> Optional[str]:
        if False:
            return 10
        return version.sentry_url or os.environ.get('TRIBLER_SENTRY_URL', None)

    @staticmethod
    def get_test_sentry_url() -> Optional[str]:
        if False:
            print('Hello World!')
        return os.environ.get('TRIBLER_TEST_SENTRY_URL', None)

    @staticmethod
    def is_in_test_mode():
        if False:
            print('Hello World!')
        return bool(SentryReporter.get_test_sentry_url())

    def _before_send(self, event: Optional[Dict], hint: Optional[Dict]) -> Optional[Dict]:
        if False:
            i = 10
            return i + 15
        'The method that is called before each send. Both allowed and\n        disallowed.\n\n        The algorithm:\n        1. If sending is allowed, then scrub the event and send.\n        2. If sending is disallowed, then store the event in\n            `self.last_event`\n\n        Args:\n            event: event that generated by Sentry\n            hint: root exception (can be used in some cases)\n\n        Returns:\n            The event, prepared for sending, or `None`, if sending is suppressed.\n        '
        if not event:
            return event
        strategy = self.get_actual_strategy()
        self._logger.info(f'Before send strategy: {strategy}')
        exc_info = get_value(hint, 'exc_info')
        error_type = get_first_item(exc_info)
        if error_type in self.ignored_exceptions:
            self._logger.debug(f'Exception is in ignored: {hint}. Skipped.')
            return None
        if strategy == SentryStrategy.SEND_SUPPRESSED:
            self._logger.debug('Suppress sending. Storing the event.')
            self.last_event = event
            return None
        if strategy == SentryStrategy.SEND_ALLOWED_WITH_CONFIRMATION:
            self._logger.debug('Request confirmation.')
            if not self.get_confirmation(hint):
                return None
        self._logger.debug(f'Clean up the event with scrubber: {self.scrubber}')
        if self.scrubber:
            event = self.scrubber.scrub_event(event)
        return event

    def _before_breadcrumb(self, breadcrumb: Optional[Dict], hint: Optional[Dict]) -> Optional[Dict]:
        if False:
            return 10
        'This function is called with an SDK-specific breadcrumb object before the breadcrumb is added to the scope.\n         When nothing is returned from the function, the breadcrumb is dropped. To pass the breadcrumb through, return\n         the first argument, which contains the breadcrumb object'
        if not self.collecting_breadcrumbs_allowed:
            return None
        return breadcrumb