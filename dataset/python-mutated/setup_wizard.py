from __future__ import annotations
import logging
from django.utils.crypto import get_random_string
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import ratelimits
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import Endpoint, control_silo_endpoint
from sentry.api.serializers import serialize
from sentry.cache import default_cache
logger = logging.getLogger('sentry.api')
SETUP_WIZARD_CACHE_KEY = 'setup-wizard-keys:v1:'
SETUP_WIZARD_CACHE_TIMEOUT = 600

@control_silo_endpoint
class SetupWizard(Endpoint):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN}
    permission_classes = ()

    def delete(self, request: Request, wizard_hash=None) -> Response | None:
        if False:
            while True:
                i = 10
        '\n        This removes the cache content for a specific hash\n        '
        if wizard_hash is not None:
            key = f'{SETUP_WIZARD_CACHE_KEY}{wizard_hash}'
            default_cache.delete(key)
            return Response(status=200)
        return None

    def get(self, request: Request, wizard_hash=None) -> Response:
        if False:
            print('Hello World!')
        '\n        This tries to retrieve and return the cache content if possible\n        otherwise creates new cache\n        '
        if wizard_hash is not None:
            key = f'{SETUP_WIZARD_CACHE_KEY}{wizard_hash}'
            wizard_data = default_cache.get(key)
            if wizard_data is None:
                return Response(status=404)
            elif wizard_data == 'empty':
                return Response(status=400)
            return Response(serialize(wizard_data))
        else:
            rate_limited = ratelimits.is_limited(key='rl:setup-wizard:ip:%s' % request.META['REMOTE_ADDR'], limit=10)
            if rate_limited:
                logger.info('setup-wizard.rate-limit')
                return Response({'Too many wizard requests'}, status=403)
            wizard_hash = get_random_string(64, allowed_chars='abcdefghijklmnopqrstuvwxyz012345679')
            key = f'{SETUP_WIZARD_CACHE_KEY}{wizard_hash}'
            default_cache.set(key, 'empty', SETUP_WIZARD_CACHE_TIMEOUT)
            return Response(serialize({'hash': wizard_hash}))