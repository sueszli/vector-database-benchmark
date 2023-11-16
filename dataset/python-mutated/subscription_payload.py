from typing import Any, Optional
from celery.utils.log import get_task_logger
from django.conf import settings
from django.utils import timezone
from django.utils.functional import SimpleLazyObject
from graphql import get_default_backend, parse
from graphql.error import GraphQLError
from promise import Promise
from ...app.models import App
from ...core.exceptions import PermissionDenied
from ...core.utils import get_domain
from ..core import SaleorContext
from ..utils import format_error
logger = get_task_logger(__name__)

def initialize_request(requestor=None, sync_event=False, allow_replica=False, event_type: Optional[str]=None) -> SaleorContext:
    if False:
        while True:
            i = 10
    'Prepare a request object for webhook subscription.\n\n    It creates a dummy request object.\n\n    return: HttpRequest\n    '
    request_time = timezone.now()
    request = SaleorContext()
    request.path = '/graphql/'
    request.path_info = '/graphql/'
    request.method = 'GET'
    request.META = {'SERVER_NAME': SimpleLazyObject(get_domain), 'SERVER_PORT': '80'}
    if settings.ENABLE_SSL:
        request.META['HTTP_X_FORWARDED_PROTO'] = 'https'
        request.META['SERVER_PORT'] = '443'
    setattr(request, 'sync_event', sync_event)
    setattr(request, 'event_type', event_type)
    request.requestor = requestor
    request.request_time = request_time
    request.allow_replica = allow_replica
    return request

def get_event_payload(event):
    if False:
        print('Hello World!')
    if isinstance(event, Promise):
        return event.get()
    return event

def generate_payload_from_subscription(event_type: str, subscribable_object, subscription_query: Optional[str], request: SaleorContext, app: Optional[App]=None) -> Optional[dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    "Generate webhook payload from subscription query.\n\n    It uses a graphql's engine to build payload by using the same logic as response.\n    As an input it expects given event type and object and the query which will be\n    used to resolve a payload.\n    event_type: is an event which will be triggered.\n    subscribable_object: is an object which have a dedicated own type in Subscription\n    definition.\n    subscription_query: query used to prepare a payload via graphql engine.\n    context: A dummy request used to share context between apps in order to use\n    dataloaders benefits.\n    app: the owner of the given payload. Required in case when webhook contains\n    protected fields.\n    return: A payload ready to send via webhook. None if the function was not able to\n    generate a payload\n    "
    from ..api import schema
    from ..context import get_context_value
    graphql_backend = get_default_backend()
    ast = parse(subscription_query)
    document = graphql_backend.document_from_string(schema, ast)
    app_id = app.pk if app else None
    request.app = app
    results = document.execute(allow_subscriptions=True, root=(event_type, subscribable_object), context=get_context_value(request))
    if hasattr(results, 'errors'):
        logger.warning('Unable to build a payload for subscription. \nerror: %s' % str(results.errors), extra={'query': subscription_query, 'app': app_id})
        return None
    payload: list[Any] = []
    results.subscribe(payload.append)
    if not payload:
        logger.warning('Subscription did not return a payload.', extra={'query': subscription_query, 'app': app_id})
        return None
    payload_instance = payload[0]
    event_payload = get_event_payload(payload_instance.data.get('event'))
    if payload_instance.errors:
        event_payload['errors'] = [format_error(error, (GraphQLError, PermissionDenied)) for error in payload_instance.errors]
    return event_payload