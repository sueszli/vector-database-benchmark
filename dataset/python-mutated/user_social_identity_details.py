import logging
from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import control_silo_endpoint
from sentry.api.bases.user import UserEndpoint
from social_auth.backends import get_backend
from social_auth.models import UserSocialAuth
logger = logging.getLogger('sentry.accounts')

@control_silo_endpoint
class UserSocialIdentityDetailsEndpoint(UserEndpoint):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN}

    def delete(self, request: Request, user, identity_id) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Disconnect a Identity from Account\n        ```````````````````````````````````````````````````````\n\n        Disconnects a social auth identity from a sentry account\n\n        :pparam string identity_id: identity id\n        :auth: required\n        '
        try:
            auth = UserSocialAuth.objects.get(id=identity_id)
        except UserSocialAuth.DoesNotExist:
            return Response(status=404)
        backend = get_backend(auth.provider, request, '/')
        if backend is None:
            raise Exception(f'Backend was not found for request: {auth.provider}')
        backend.disconnect(user, identity_id)
        assert not UserSocialAuth.objects.filter(user=user, id=identity_id).exists()
        logger.info('user.identity.disconnect', extra={'user_id': user.id, 'ip_address': request.META['REMOTE_ADDR'], 'usersocialauth_id': identity_id})
        return Response(status=204)