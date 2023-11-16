import base64
import copy
import datetime
import email
import email.policy
import logging
import os
import signal
import socket
import tempfile
import threading
import time
import urllib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from email.message import EmailMessage
from functools import wraps
from types import FrameType
from typing import Any, Callable, Dict, List, Mapping, MutableSequence, Optional, Sequence, Set, Tuple, Type, TypeVar
import orjson
import sentry_sdk
from django.conf import settings
from django.core.mail.backends.base import BaseEmailBackend
from django.db import connection, transaction
from django.db.models import F
from django.db.utils import IntegrityError
from django.utils.timezone import now as timezone_now
from django.utils.translation import gettext as _
from django.utils.translation import override as override_language
from returns.curry import partial
from sentry_sdk import add_breadcrumb, configure_scope
from typing_extensions import override
from zulip_bots.lib import extract_query_without_mention
from zerver.actions.invites import do_send_confirmation_email
from zerver.actions.message_edit import do_update_embedded_data
from zerver.actions.message_flags import do_mark_stream_messages_as_read
from zerver.actions.message_send import internal_send_private_message, render_incoming_message
from zerver.actions.presence import do_update_user_presence
from zerver.actions.realm_export import notify_realm_export
from zerver.actions.user_activity import do_update_user_activity, do_update_user_activity_interval
from zerver.context_processors import common_context
from zerver.lib.bot_lib import EmbeddedBotHandler, EmbeddedBotQuitError, get_bot_handler
from zerver.lib.context_managers import lockfile
from zerver.lib.db import reset_queries
from zerver.lib.digest import bulk_handle_digest_email
from zerver.lib.email_mirror import decode_stream_email_address, is_missed_message_address, rate_limit_mirror_by_realm
from zerver.lib.email_mirror import process_message as mirror_email
from zerver.lib.email_notifications import MissedMessageData, handle_missedmessage_emails
from zerver.lib.exceptions import RateLimitedError
from zerver.lib.export import export_realm_wrapper
from zerver.lib.outgoing_webhook import do_rest_call, get_outgoing_webhook_service_handler
from zerver.lib.per_request_cache import flush_per_request_caches
from zerver.lib.push_notifications import clear_push_device_tokens, handle_push_notification, handle_remove_push_notification, initialize_push_notifications
from zerver.lib.pysa import mark_sanitized
from zerver.lib.queue import SimpleQueueClient, retry_event
from zerver.lib.remote_server import PushNotificationBouncerRetryLaterError
from zerver.lib.send_email import EmailNotDeliveredError, FromAddress, handle_send_email_format_changes, initialize_connection, send_email, send_future_email
from zerver.lib.soft_deactivation import reactivate_user_if_soft_deactivated
from zerver.lib.timestamp import timestamp_to_datetime
from zerver.lib.upload import handle_reupload_emojis_event
from zerver.lib.url_preview import preview as url_preview
from zerver.lib.url_preview.types import UrlEmbedData
from zerver.models import Message, PreregistrationUser, Realm, RealmAuditLog, ScheduledMessageNotificationEmail, Stream, UserMessage, UserProfile, filter_to_valid_prereg_users, get_bot_services, get_client, get_system_bot, get_user_profile_by_id
logger = logging.getLogger(__name__)

class WorkerTimeoutError(Exception):

    def __init__(self, queue_name: str, limit: int, event_count: int) -> None:
        if False:
            while True:
                i = 10
        self.queue_name = queue_name
        self.limit = limit
        self.event_count = event_count

    @override
    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return f'Timed out in {self.queue_name} after {self.limit * self.event_count} seconds processing {self.event_count} events'

class InterruptConsumeError(Exception):
    """
    This exception is to be thrown inside event consume function
    if the intention is to simply interrupt the processing
    of the current event and normally continue the work of the queue.
    """

class WorkerDeclarationError(Exception):
    pass
ConcreteQueueWorker = TypeVar('ConcreteQueueWorker', bound='QueueProcessingWorker')

def assign_queue(queue_name: str, enabled: bool=True, is_test_queue: bool=False) -> Callable[[Type[ConcreteQueueWorker]], Type[ConcreteQueueWorker]]:
    if False:
        i = 10
        return i + 15

    def decorate(clazz: Type[ConcreteQueueWorker]) -> Type[ConcreteQueueWorker]:
        if False:
            i = 10
            return i + 15
        clazz.queue_name = queue_name
        if enabled:
            register_worker(queue_name, clazz, is_test_queue)
        return clazz
    return decorate
worker_classes: Dict[str, Type['QueueProcessingWorker']] = {}
test_queues: Set[str] = set()

def register_worker(queue_name: str, clazz: Type['QueueProcessingWorker'], is_test_queue: bool=False) -> None:
    if False:
        return 10
    worker_classes[queue_name] = clazz
    if is_test_queue:
        test_queues.add(queue_name)

def get_worker(queue_name: str, threaded: bool=False, disable_timeout: bool=False) -> 'QueueProcessingWorker':
    if False:
        while True:
            i = 10
    return worker_classes[queue_name](threaded=threaded, disable_timeout=disable_timeout)

def get_active_worker_queues(only_test_queues: bool=False) -> List[str]:
    if False:
        while True:
            i = 10
    'Returns all (either test, or real) worker queues.'
    return [queue_name for queue_name in worker_classes if bool(queue_name in test_queues) == only_test_queues]

def check_and_send_restart_signal() -> None:
    if False:
        print('Hello World!')
    try:
        if not connection.is_usable():
            logging.warning('*** Sending self SIGUSR1 to trigger a restart.')
            os.kill(os.getpid(), signal.SIGUSR1)
    except Exception:
        pass

def retry_send_email_failures(func: Callable[[ConcreteQueueWorker, Dict[str, Any]], None]) -> Callable[[ConcreteQueueWorker, Dict[str, Any]], None]:
    if False:
        print('Hello World!')

    @wraps(func)
    def wrapper(worker: ConcreteQueueWorker, data: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        try:
            func(worker, data)
        except (socket.gaierror, socket.timeout, EmailNotDeliveredError) as e:
            error_class_name = type(e).__name__

            def on_failure(event: Dict[str, Any]) -> None:
                if False:
                    i = 10
                    return i + 15
                logging.exception('Event %r failed due to exception %s', event, error_class_name, stack_info=True)
            retry_event(worker.queue_name, data, on_failure)
    return wrapper

class QueueProcessingWorker(ABC):
    queue_name: str
    MAX_CONSUME_SECONDS: Optional[int] = 30
    CONSUME_ITERATIONS_BEFORE_UPDATE_STATS_NUM = 50
    MAX_SECONDS_BEFORE_UPDATE_STATS = 30
    PREFETCH = 100

    def __init__(self, threaded: bool=False, disable_timeout: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.q: Optional[SimpleQueueClient] = None
        self.threaded = threaded
        self.disable_timeout = disable_timeout
        if not hasattr(self, 'queue_name'):
            raise WorkerDeclarationError('Queue worker declared without queue_name')
        self.initialize_statistics()

    def initialize_statistics(self) -> None:
        if False:
            i = 10
            return i + 15
        self.queue_last_emptied_timestamp = time.time()
        self.consumed_since_last_emptied = 0
        self.recent_consume_times: MutableSequence[Tuple[int, float]] = deque(maxlen=50)
        self.consume_iteration_counter = 0
        self.idle = True
        self.last_statistics_update_time = 0.0
        self.update_statistics()

    def update_statistics(self) -> None:
        if False:
            while True:
                i = 10
        total_seconds = sum((seconds for (_, seconds) in self.recent_consume_times))
        total_events = sum((events_number for (events_number, _) in self.recent_consume_times))
        if total_events == 0:
            recent_average_consume_time = None
        else:
            recent_average_consume_time = total_seconds / total_events
        stats_dict = dict(update_time=time.time(), recent_average_consume_time=recent_average_consume_time, queue_last_emptied_timestamp=self.queue_last_emptied_timestamp, consumed_since_last_emptied=self.consumed_since_last_emptied)
        os.makedirs(settings.QUEUE_STATS_DIR, exist_ok=True)
        fname = f'{self.queue_name}.stats'
        fn = os.path.join(settings.QUEUE_STATS_DIR, fname)
        with lockfile(fn + '.lock'):
            tmp_fn = fn + '.tmp'
            with open(tmp_fn, 'wb') as f:
                f.write(orjson.dumps(stats_dict, option=orjson.OPT_APPEND_NEWLINE | orjson.OPT_INDENT_2))
            os.rename(tmp_fn, fn)
        self.last_statistics_update_time = time.time()

    def get_remaining_local_queue_size(self) -> int:
        if False:
            while True:
                i = 10
        if self.q is not None:
            return self.q.local_queue_size()
        else:
            return 0

    @abstractmethod
    def consume(self, data: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def do_consume(self, consume_func: Callable[[List[Dict[str, Any]]], None], events: List[Dict[str, Any]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        consume_time_seconds: Optional[float] = None
        with configure_scope() as scope:
            scope.clear_breadcrumbs()
            add_breadcrumb(type='debug', category='queue_processor', message=f'Consuming {self.queue_name}', data={'events': events, 'local_queue_size': self.get_remaining_local_queue_size()})
        try:
            if self.idle:
                self.idle = False
                self.update_statistics()
            time_start = time.time()
            if self.MAX_CONSUME_SECONDS and (not self.threaded) and (not self.disable_timeout):
                try:
                    signal.signal(signal.SIGALRM, partial(self.timer_expired, self.MAX_CONSUME_SECONDS, events))
                    try:
                        signal.alarm(self.MAX_CONSUME_SECONDS * len(events))
                        consume_func(events)
                    finally:
                        signal.alarm(0)
                finally:
                    signal.signal(signal.SIGALRM, signal.SIG_DFL)
            else:
                consume_func(events)
            consume_time_seconds = time.time() - time_start
            self.consumed_since_last_emptied += len(events)
        except Exception as e:
            self._handle_consume_exception(events, e)
        finally:
            flush_per_request_caches()
            reset_queries()
            if consume_time_seconds is not None:
                self.recent_consume_times.append((len(events), consume_time_seconds))
            remaining_local_queue_size = self.get_remaining_local_queue_size()
            if remaining_local_queue_size == 0:
                self.queue_last_emptied_timestamp = time.time()
                self.consumed_since_last_emptied = 0
                self.update_statistics()
                self.idle = True
            else:
                self.consume_iteration_counter += 1
                if self.consume_iteration_counter >= self.CONSUME_ITERATIONS_BEFORE_UPDATE_STATS_NUM or time.time() - self.last_statistics_update_time >= self.MAX_SECONDS_BEFORE_UPDATE_STATS:
                    self.consume_iteration_counter = 0
                    self.update_statistics()

    def consume_single_event(self, event: Dict[str, Any]) -> None:
        if False:
            return 10
        consume_func = lambda events: self.consume(events[0])
        self.do_consume(consume_func, [event])

    def timer_expired(self, limit: int, events: List[Dict[str, Any]], signal: int, frame: Optional[FrameType]) -> None:
        if False:
            i = 10
            return i + 15
        raise WorkerTimeoutError(self.queue_name, limit, len(events))

    def _handle_consume_exception(self, events: List[Dict[str, Any]], exception: Exception) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(exception, InterruptConsumeError):
            return
        with configure_scope() as scope:
            scope.set_context('events', {'data': events, 'queue_name': self.queue_name})
            if isinstance(exception, WorkerTimeoutError):
                with sentry_sdk.push_scope() as scope:
                    scope.fingerprint = ['worker-timeout', self.queue_name]
                    logging.exception(exception, stack_info=True)
            else:
                logging.exception('Problem handling data on queue %s', self.queue_name, stack_info=True)
        if not os.path.exists(settings.QUEUE_ERROR_DIR):
            os.mkdir(settings.QUEUE_ERROR_DIR)
        fname = mark_sanitized(f'{self.queue_name}.errors')
        fn = os.path.join(settings.QUEUE_ERROR_DIR, fname)
        line = f'{time.asctime()}\t{orjson.dumps(events).decode()}\n'
        lock_fn = fn + '.lock'
        with lockfile(lock_fn):
            with open(fn, 'a') as f:
                f.write(line)
        check_and_send_restart_signal()

    def setup(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.q = SimpleQueueClient(prefetch=self.PREFETCH)

    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        assert self.q is not None
        self.initialize_statistics()
        self.q.start_json_consumer(self.queue_name, lambda events: self.consume_single_event(events[0]))

    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        assert self.q is not None
        self.q.stop_consuming()

class LoopQueueProcessingWorker(QueueProcessingWorker):
    sleep_delay = 1
    batch_size = 100

    @override
    def setup(self) -> None:
        if False:
            i = 10
            return i + 15
        self.q = SimpleQueueClient(prefetch=max(self.PREFETCH, self.batch_size))

    @override
    def start(self) -> None:
        if False:
            print('Hello World!')
        assert self.q is not None
        self.initialize_statistics()
        self.q.start_json_consumer(self.queue_name, lambda events: self.do_consume(self.consume_batch, events), batch_size=self.batch_size, timeout=self.sleep_delay)

    @abstractmethod
    def consume_batch(self, events: List[Dict[str, Any]]) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @override
    def consume(self, event: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        'In LoopQueueProcessingWorker, consume is used just for automated tests'
        self.consume_batch([event])

@assign_queue('invites')
class ConfirmationEmailWorker(QueueProcessingWorker):

    @override
    def consume(self, data: Mapping[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        if 'invite_expires_in_days' in data:
            invite_expires_in_minutes = data['invite_expires_in_days'] * 24 * 60
        elif 'invite_expires_in_minutes' in data:
            invite_expires_in_minutes = data['invite_expires_in_minutes']
        invitee = filter_to_valid_prereg_users(PreregistrationUser.objects.filter(id=data['prereg_id']), invite_expires_in_minutes).first()
        if invitee is None:
            return
        referrer = get_user_profile_by_id(data['referrer_id'])
        logger.info('Sending invitation for realm %s to %s', referrer.realm.string_id, invitee.email)
        if 'email_language' in data:
            email_language = data['email_language']
        else:
            email_language = referrer.realm.default_language
        activate_url = do_send_confirmation_email(invitee, referrer, email_language, invite_expires_in_minutes)
        if invite_expires_in_minutes is None:
            return
        if invite_expires_in_minutes >= 4 * 24 * 60:
            context = common_context(referrer)
            context.update(activate_url=activate_url, referrer_name=referrer.full_name, referrer_email=referrer.delivery_email, referrer_realm_name=referrer.realm.name)
            send_future_email('zerver/emails/invitation_reminder', referrer.realm, to_emails=[invitee.email], from_address=FromAddress.tokenized_no_reply_placeholder, language=email_language, context=context, delay=datetime.timedelta(minutes=invite_expires_in_minutes - 2 * 24 * 60))

@assign_queue('user_activity')
class UserActivityWorker(LoopQueueProcessingWorker):
    """The UserActivity queue is perhaps our highest-traffic queue, and
    requires some care to ensure it performs adequately.

    We use a LoopQueueProcessingWorker as a performance optimization
    for managing the queue.  The structure of UserActivity records is
    such that they are easily deduplicated before being sent to the
    database; we take advantage of that to make this queue highly
    effective at dealing with a backlog containing many similar
    events.  Such a backlog happen in a few ways:

    * In abuse/DoS situations, if a client is sending huge numbers of
      similar requests to the server.
    * If the queue ends up with several minutes of backlog e.g. due to
      downtime of the queue processor, many clients will have several
      common events from doing an action multiple times.

    """
    client_id_map: Dict[str, int] = {}

    @override
    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        self.client_id_map = {}
        super().start()

    @override
    def consume_batch(self, user_activity_events: List[Dict[str, Any]]) -> None:
        if False:
            print('Hello World!')
        uncommitted_events: Dict[Tuple[int, int, str], Tuple[int, float]] = {}
        for event in user_activity_events:
            user_profile_id = event['user_profile_id']
            client_id = event['client_id']
            key_tuple = (user_profile_id, client_id, event['query'])
            if key_tuple not in uncommitted_events:
                uncommitted_events[key_tuple] = (1, event['time'])
            else:
                (count, time) = uncommitted_events[key_tuple]
                uncommitted_events[key_tuple] = (count + 1, max(time, event['time']))
        for key_tuple in uncommitted_events:
            (user_profile_id, client_id, query) = key_tuple
            (count, time) = uncommitted_events[key_tuple]
            log_time = timestamp_to_datetime(time)
            do_update_user_activity(user_profile_id, client_id, query, count, log_time)

@assign_queue('user_activity_interval')
class UserActivityIntervalWorker(QueueProcessingWorker):

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        user_profile = get_user_profile_by_id(event['user_profile_id'])
        log_time = timestamp_to_datetime(event['time'])
        do_update_user_activity_interval(user_profile, log_time)

@assign_queue('user_presence')
class UserPresenceWorker(QueueProcessingWorker):

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        logging.debug('Received presence event: %s', event)
        user_profile = get_user_profile_by_id(event['user_profile_id'])
        client = get_client(event['client'])
        log_time = timestamp_to_datetime(event['time'])
        status = event['status']
        do_update_user_presence(user_profile, client, log_time, status)

@assign_queue('missedmessage_emails')
class MissedMessageWorker(QueueProcessingWorker):
    CHECK_FREQUENCY_SECONDS = 5
    worker_thread: Optional[threading.Thread] = None
    cv = threading.Condition()
    stopping = False
    has_timeout = False

    @override
    def consume(self, event: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        logging.debug('Processing missedmessage_emails event: %s', event)
        user_profile_id: int = event['user_profile_id']
        user_profile = get_user_profile_by_id(user_profile_id)
        batch_duration_seconds = user_profile.email_notifications_batching_period_seconds
        batch_duration = datetime.timedelta(seconds=batch_duration_seconds)
        try:
            pending_email = ScheduledMessageNotificationEmail.objects.filter(user_profile_id=user_profile_id)[0]
            scheduled_timestamp = pending_email.scheduled_timestamp
        except IndexError:
            scheduled_timestamp = timezone_now() + batch_duration
        with self.cv:
            try:
                ScheduledMessageNotificationEmail.objects.create(user_profile_id=user_profile_id, message_id=event['message_id'], trigger=event['trigger'], scheduled_timestamp=scheduled_timestamp, mentioned_user_group_id=event.get('mentioned_user_group_id'))
                if not self.has_timeout:
                    self.cv.notify()
            except IntegrityError:
                logging.debug('ScheduledMessageNotificationEmail row could not be created. The message may have been deleted. Skipping event.')

    @override
    def start(self) -> None:
        if False:
            print('Hello World!')
        with self.cv:
            self.stopping = False
        self.worker_thread = threading.Thread(target=lambda : self.work())
        self.worker_thread.start()
        super().start()

    def work(self) -> None:
        if False:
            i = 10
            return i + 15
        while True:
            try:
                finished = self.background_loop()
                if finished:
                    break
            except Exception:
                logging.exception('Exception in MissedMessage background worker; restarting the loop', stack_info=True)

    def background_loop(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        with self.cv:
            if self.stopping:
                return True
            timeout: Optional[int] = None
            if ScheduledMessageNotificationEmail.objects.exists():
                timeout = self.CHECK_FREQUENCY_SECONDS
            self.has_timeout = timeout is not None

            def wait_condition() -> bool:
                if False:
                    while True:
                        i = 10
                if self.stopping:
                    return True
                if timeout is None:
                    return ScheduledMessageNotificationEmail.objects.exists()
                return False
            was_notified = self.cv.wait_for(wait_condition, timeout=timeout)
        if not was_notified:
            self.maybe_send_batched_emails()
        return False

    def maybe_send_batched_emails(self) -> None:
        if False:
            while True:
                i = 10
        current_time = timezone_now()
        with transaction.atomic():
            events_to_process = ScheduledMessageNotificationEmail.objects.filter(scheduled_timestamp__lte=current_time).select_for_update()
            events_by_recipient: Dict[int, Dict[int, MissedMessageData]] = defaultdict(dict)
            for event in events_to_process:
                events_by_recipient[event.user_profile_id][event.message_id] = MissedMessageData(trigger=event.trigger, mentioned_user_group_id=event.mentioned_user_group_id)
            for user_profile_id in events_by_recipient:
                events = events_by_recipient[user_profile_id]
                logging.info('Batch-processing %s missedmessage_emails events for user %s', len(events), user_profile_id)
                try:
                    handle_missedmessage_emails(user_profile_id, events)
                except Exception:
                    logging.exception('Failed to process %d missedmessage_emails for user %s', len(events), user_profile_id, stack_info=True)
            events_to_process.delete()

    @override
    def stop(self) -> None:
        if False:
            while True:
                i = 10
        with self.cv:
            self.stopping = True
            self.cv.notify()
        if self.worker_thread is not None:
            self.worker_thread.join()
        super().stop()

@assign_queue('email_senders')
class EmailSendingWorker(LoopQueueProcessingWorker):

    def __init__(self, threaded: bool=False, disable_timeout: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(threaded, disable_timeout)
        self.connection: BaseEmailBackend = initialize_connection(None)

    @retry_send_email_failures
    def send_email(self, event: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        copied_event = copy.deepcopy(event)
        if 'failed_tries' in copied_event:
            del copied_event['failed_tries']
        handle_send_email_format_changes(copied_event)
        self.connection = initialize_connection(self.connection)
        send_email(**copied_event, connection=self.connection)

    @override
    def consume_batch(self, events: List[Dict[str, Any]]) -> None:
        if False:
            print('Hello World!')
        for event in events:
            self.send_email(event)

    @override
    def stop(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            self.connection.close()
        finally:
            super().stop()

@assign_queue('missedmessage_mobile_notifications')
class PushNotificationsWorker(QueueProcessingWorker):
    MAX_CONSUME_SECONDS = None

    @override
    def start(self) -> None:
        if False:
            i = 10
            return i + 15
        initialize_push_notifications()
        super().start()

    @override
    def consume(self, event: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        try:
            if event.get('type', 'add') == 'remove':
                message_ids = event['message_ids']
                handle_remove_push_notification(event['user_profile_id'], message_ids)
            else:
                handle_push_notification(event['user_profile_id'], event)
        except PushNotificationBouncerRetryLaterError:

            def failure_processor(event: Dict[str, Any]) -> None:
                if False:
                    while True:
                        i = 10
                logger.warning('Maximum retries exceeded for trigger:%s event:push_notification', event['user_profile_id'])
            retry_event(self.queue_name, event, failure_processor)

@assign_queue('digest_emails')
class DigestWorker(QueueProcessingWorker):

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            return 10
        if 'user_ids' in event:
            user_ids = event['user_ids']
        else:
            user_ids = [event['user_profile_id']]
        bulk_handle_digest_email(user_ids, event['cutoff'])

@assign_queue('email_mirror')
class MirrorWorker(QueueProcessingWorker):

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        rcpt_to = event['rcpt_to']
        msg = email.message_from_bytes(base64.b64decode(event['msg_base64']), policy=email.policy.default)
        assert isinstance(msg, EmailMessage)
        if not is_missed_message_address(rcpt_to):
            recipient_realm = decode_stream_email_address(rcpt_to)[0].realm
            try:
                rate_limit_mirror_by_realm(recipient_realm)
            except RateLimitedError:
                logger.warning('MirrorWorker: Rejecting an email from: %s to realm: %s - rate limited.', msg['From'], recipient_realm.subdomain)
                return
        mirror_email(msg, rcpt_to=rcpt_to)

@assign_queue('embed_links')
class FetchLinksEmbedData(QueueProcessingWorker):
    CONSUME_ITERATIONS_BEFORE_UPDATE_STATS_NUM = 1

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            print('Hello World!')
        url_embed_data: Dict[str, Optional[UrlEmbedData]] = {}
        for url in event['urls']:
            start_time = time.time()
            url_embed_data[url] = url_preview.get_link_embed_data(url)
            logging.info('Time spent on get_link_embed_data for %s: %s', url, time.time() - start_time)
        with transaction.atomic():
            try:
                message = Message.objects.select_for_update().get(id=event['message_id'])
            except Message.DoesNotExist:
                return
            if message.content != event['message_content']:
                return
            realm = Realm.objects.get(id=event['message_realm_id'])
            rendering_result = render_incoming_message(message, message.content, realm, url_embed_data=url_embed_data)
            do_update_embedded_data(message.sender, message, message.content, rendering_result)

    @override
    def timer_expired(self, limit: int, events: List[Dict[str, Any]], signal: int, frame: Optional[FrameType]) -> None:
        if False:
            print('Hello World!')
        assert len(events) == 1
        event = events[0]
        logging.warning('Timed out in %s after %s seconds while fetching URLs for message %s: %s', self.queue_name, limit, event['message_id'], event['urls'])
        raise InterruptConsumeError

@assign_queue('outgoing_webhooks')
class OutgoingWebhookWorker(QueueProcessingWorker):

    @override
    def consume(self, event: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        message = event['message']
        event['command'] = message['content']
        services = get_bot_services(event['user_profile_id'])
        for service in services:
            event['service_name'] = str(service.name)
            service_handler = get_outgoing_webhook_service_handler(service)
            do_rest_call(service.base_url, event, service_handler)

@assign_queue('embedded_bots')
class EmbeddedBotWorker(QueueProcessingWorker):

    def get_bot_api_client(self, user_profile: UserProfile) -> EmbeddedBotHandler:
        if False:
            while True:
                i = 10
        return EmbeddedBotHandler(user_profile)

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        user_profile_id = event['user_profile_id']
        user_profile = get_user_profile_by_id(user_profile_id)
        message: Dict[str, Any] = event['message']
        services = get_bot_services(user_profile_id)
        for service in services:
            bot_handler = get_bot_handler(str(service.name))
            if bot_handler is None:
                logging.error('Error: User %s has bot with invalid embedded bot service %s', user_profile_id, service.name)
                continue
            try:
                if hasattr(bot_handler, 'initialize'):
                    bot_handler.initialize(self.get_bot_api_client(user_profile))
                if event['trigger'] == 'mention':
                    message['content'] = extract_query_without_mention(message=message, client=self.get_bot_api_client(user_profile))
                    assert message['content'] is not None
                bot_handler.handle_message(message=message, bot_handler=self.get_bot_api_client(user_profile))
            except EmbeddedBotQuitError as e:
                logging.warning('%s', e)

@assign_queue('deferred_work')
class DeferredWorker(QueueProcessingWorker):
    """This queue processor is intended for cases where we want to trigger a
    potentially expensive, not urgent, job to be run on a separate
    thread from the Django worker that initiated it (E.g. so we that
    can provide a low-latency HTTP response or avoid risk of request
    timeouts for an operation that could in rare cases take minutes).
    """
    MAX_CONSUME_SECONDS = None

    @override
    def consume(self, event: Dict[str, Any]) -> None:
        if False:
            return 10
        start = time.time()
        if event['type'] == 'mark_stream_messages_as_read':
            user_profile = get_user_profile_by_id(event['user_profile_id'])
            logger.info('Marking messages as read for user %s, stream_recipient_ids %s', user_profile.id, event['stream_recipient_ids'])
            for recipient_id in event['stream_recipient_ids']:
                count = do_mark_stream_messages_as_read(user_profile, recipient_id)
                logger.info('Marked %s messages as read for user %s, stream_recipient_id %s', count, user_profile.id, recipient_id)
        elif event['type'] == 'mark_stream_messages_as_read_for_everyone':
            logger.info('Marking messages as read for all users, stream_recipient_id %s', event['stream_recipient_id'])
            stream = Stream.objects.get(recipient_id=event['stream_recipient_id'])
            batch_size = 100
            offset = 0
            while True:
                messages = Message.objects.filter(realm_id=stream.realm_id, recipient_id=event['stream_recipient_id']).order_by('id')[offset:offset + batch_size]
                with transaction.atomic(savepoint=False):
                    UserMessage.select_for_update_query().filter(message__in=messages).extra(where=[UserMessage.where_unread()]).update(flags=F('flags').bitor(UserMessage.flags.read))
                offset += len(messages)
                if len(messages) < batch_size:
                    break
            logger.info('Marked %s messages as read for all users, stream_recipient_id %s', offset, event['stream_recipient_id'])
        elif event['type'] == 'clear_push_device_tokens':
            logger.info('Clearing push device tokens for user_profile_id %s', event['user_profile_id'])
            try:
                clear_push_device_tokens(event['user_profile_id'])
            except PushNotificationBouncerRetryLaterError:

                def failure_processor(event: Dict[str, Any]) -> None:
                    if False:
                        while True:
                            i = 10
                    logger.warning('Maximum retries exceeded for trigger:%s event:clear_push_device_tokens', event['user_profile_id'])
                retry_event(self.queue_name, event, failure_processor)
        elif event['type'] == 'realm_export':
            realm = Realm.objects.get(id=event['realm_id'])
            output_dir = tempfile.mkdtemp(prefix='zulip-export-')
            export_event = RealmAuditLog.objects.get(id=event['id'])
            user_profile = get_user_profile_by_id(event['user_profile_id'])
            extra_data = export_event.extra_data
            if extra_data.get('started_timestamp') is not None:
                logger.error('Marking export for realm %s as failed due to retry -- possible OOM during export?', realm.string_id)
                extra_data['failed_timestamp'] = timezone_now().timestamp()
                export_event.extra_data = extra_data
                export_event.save(update_fields=['extra_data'])
                notify_realm_export(user_profile)
                return
            extra_data['started_timestamp'] = timezone_now().timestamp()
            export_event.extra_data = extra_data
            export_event.save(update_fields=['extra_data'])
            logger.info('Starting realm export for realm %s into %s, initiated by user_profile_id %s', realm.string_id, output_dir, event['user_profile_id'])
            try:
                public_url = export_realm_wrapper(realm=realm, output_dir=output_dir, threads=1 if self.threaded else 6, upload=True, public_only=True)
            except Exception:
                extra_data['failed_timestamp'] = timezone_now().timestamp()
                export_event.extra_data = extra_data
                export_event.save(update_fields=['extra_data'])
                logging.exception('Data export for %s failed after %s', user_profile.realm.string_id, time.time() - start, stack_info=True)
                notify_realm_export(user_profile)
                return
            assert public_url is not None
            extra_data['export_path'] = urllib.parse.urlparse(public_url).path
            export_event.extra_data = extra_data
            export_event.save(update_fields=['extra_data'])
            with override_language(user_profile.default_language):
                content = _('Your data export is complete. [View and download exports]({export_settings_link}).').format(export_settings_link='/#organization/data-exports-admin')
            internal_send_private_message(sender=get_system_bot(settings.NOTIFICATION_BOT, realm.id), recipient_user=user_profile, content=content)
            notify_realm_export(user_profile)
            logging.info('Completed data export for %s in %s', user_profile.realm.string_id, time.time() - start)
        elif event['type'] == 'reupload_realm_emoji':
            realm = Realm.objects.get(id=event['realm_id'])
            logger.info('Processing reupload_realm_emoji event for realm %s', realm.id)
            handle_reupload_emojis_event(realm, logger)
        elif event['type'] == 'soft_reactivate':
            logger.info('Starting soft reactivation for user_profile_id %s', event['user_profile_id'])
            user_profile = get_user_profile_by_id(event['user_profile_id'])
            reactivate_user_if_soft_deactivated(user_profile)
        end = time.time()
        logger.info('deferred_work processed %s event (%dms)', event['type'], (end - start) * 1000)

@assign_queue('test', is_test_queue=True)
class TestWorker(QueueProcessingWorker):

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        fn = settings.ZULIP_WORKER_TEST_FILE
        message = orjson.dumps(event)
        logging.info('TestWorker should append this message to %s: %s', fn, message.decode())
        with open(fn, 'ab') as f:
            f.write(message + b'\n')

@assign_queue('noop', is_test_queue=True)
class NoopWorker(QueueProcessingWorker):
    """Used to profile the queue processing framework, in zilencer's queue_rate."""

    def __init__(self, threaded: bool=False, disable_timeout: bool=False, max_consume: int=1000, slow_queries: Sequence[int]=[]) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(threaded, disable_timeout)
        self.consumed = 0
        self.max_consume = max_consume
        self.slow_queries: Set[int] = set(slow_queries)

    @override
    def consume(self, event: Mapping[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.consumed += 1
        if self.consumed in self.slow_queries:
            logging.info('Slow request...')
            time.sleep(60)
            logging.info('Done!')
        if self.consumed >= self.max_consume:
            self.stop()

@assign_queue('noop_batch', is_test_queue=True)
class BatchNoopWorker(LoopQueueProcessingWorker):
    """Used to profile the queue processing framework, in zilencer's queue_rate."""
    batch_size = 100

    def __init__(self, threaded: bool=False, disable_timeout: bool=False, max_consume: int=1000, slow_queries: Sequence[int]=[]) -> None:
        if False:
            return 10
        super().__init__(threaded, disable_timeout)
        self.consumed = 0
        self.max_consume = max_consume
        self.slow_queries: Set[int] = set(slow_queries)

    @override
    def consume_batch(self, events: List[Dict[str, Any]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        event_numbers = set(range(self.consumed + 1, self.consumed + 1 + len(events)))
        found_slow = self.slow_queries & event_numbers
        if found_slow:
            logging.info('%d slow requests...', len(found_slow))
            time.sleep(60 * len(found_slow))
            logging.info('Done!')
        self.consumed += len(events)
        if self.consumed >= self.max_consume:
            self.stop()