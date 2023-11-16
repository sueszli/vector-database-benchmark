import logging
from typing import Any
from django.conf import settings
from django.db import router, transaction
from rest_framework.request import Request
from rest_framework.response import Response
import sentry
from sentry import options
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import Endpoint, all_silo_endpoint
from sentry.api.permissions import SuperuserPermission
from sentry.utils.email import is_smtp_enabled
logger = logging.getLogger('sentry')
SYSTEM_OPTIONS_ALLOWLIST = 'system.admin-email'

@all_silo_endpoint
class SystemOptionsEndpoint(Endpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}
    permission_classes = (SuperuserPermission,)

    def get(self, request: Request) -> Response:
        if False:
            while True:
                i = 10
        query = request.GET.get('query')
        if query == 'is:required':
            option_list = options.filter(flag=options.FLAG_REQUIRED)
        elif query:
            return Response(f'{query} is not a supported search query', status=400)
        else:
            option_list = options.all()
        smtp_disabled = not is_smtp_enabled()
        results = {}
        for k in option_list:
            (disabled, disabled_reason) = (False, None)
            if smtp_disabled and k.name[:5] == 'mail.':
                (disabled_reason, disabled) = ('smtpDisabled', True)
            elif bool(k.flags & options.FLAG_PRIORITIZE_DISK and settings.SENTRY_OPTIONS.get(k.name)):
                (disabled_reason, disabled) = ('diskPriority', True)
            results[k.name] = {'value': options.get(k.name) if not self.__is_secret(k) else '[redacted]', 'field': {'default': k.default(), 'required': bool(k.flags & options.FLAG_REQUIRED), 'disabled': disabled, 'disabledReason': disabled_reason, 'isSet': options.isset(k.name), 'allowEmpty': bool(k.flags & options.FLAG_ALLOW_EMPTY)}}
        return Response(results)

    def __is_secret(self, k: Any) -> bool:
        if False:
            i = 10
            return i + 15
        keywords = ['secret', 'private', 'token']
        return k.flags & options.FLAG_CREDENTIAL or any([keyword in k.name for keyword in keywords])

    def has_permission(self, request: Request):
        if False:
            print('Hello World!')
        if not request.access.has_permission('options.admin'):
            if all([k in SYSTEM_OPTIONS_ALLOWLIST for k in request.data.keys()]):
                return True
            return False
        return True

    def put(self, request: Request):
        if False:
            return 10
        if not self.has_permission(request):
            return Response(status=403)
        for (k, v) in request.data.items():
            if v and isinstance(v, str):
                v = v.strip()
            try:
                option = options.lookup_key(k)
            except options.UnknownOption:
                return Response({'error': 'unknown_option', 'errorDetail': {'option': k}}, status=400)
            try:
                with transaction.atomic(router.db_for_write(options.default_store.model)):
                    if not option.flags & options.FLAG_ALLOW_EMPTY and (not v):
                        options.delete(k)
                    else:
                        options.set(k, v, channel=options.UpdateChannel.APPLICATION)
                    logger.info('options.update', extra={'ip_address': request.META['REMOTE_ADDR'], 'user_id': request.user.id, 'option_key': k, 'option_value': v})
            except (TypeError, AssertionError) as e:
                return Response({'error': 'invalid_type' if type(e) is TypeError else 'immutable_option', 'errorDetail': {'option': k, 'message': str(e)}}, status=400)
        options.set('sentry:version-configured', sentry.get_version(), channel=options.UpdateChannel.APPLICATION)
        return Response(status=200)