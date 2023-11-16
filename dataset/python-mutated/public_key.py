import logging
from rest_framework.permissions import IsAuthenticated
from rest_framework.request import Request
from rest_framework.response import Response
from sentry import features
from sentry.api.api_owners import ApiOwner
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import Endpoint, region_silo_endpoint
from sentry.api.endpoints.relocations import ERR_FEATURE_DISABLED
from sentry.backup.helpers import DEFAULT_CRYPTO_KEY_VERSION, GCPKMSEncryptor
logger = logging.getLogger(__name__)

@region_silo_endpoint
class RelocationPublicKeyEndpoint(Endpoint):
    owner = ApiOwner.RELOCATION
    publish_status = {'GET': ApiPublishStatus.EXPERIMENTAL}
    permission_classes = (IsAuthenticated,)

    def get(self, request: Request) -> Response:
        if False:
            return 10
        '\n        Get your public key for relocation encryption.\n        ``````````````````````````````````````````````\n\n        Returns a public key which can be used to create an encrypted export tarball.\n\n        :auth: required\n        '
        logger.info('get.start', extra={'caller': request.user.id})
        if not features.has('relocation:enabled'):
            return Response({'detail': ERR_FEATURE_DISABLED}, status=400)
        public_key_pem = GCPKMSEncryptor.from_crypto_key_version(DEFAULT_CRYPTO_KEY_VERSION).get_public_key_pem()
        return Response({'public_key': public_key_pem.decode('utf-8')}, status=200)