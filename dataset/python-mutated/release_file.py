from base64 import urlsafe_b64decode, urlsafe_b64encode
from sentry.api.serializers import Serializer, register
from sentry.models.distribution import Distribution
from sentry.models.releasefile import ReleaseFile

def encode_release_file_id(obj):
    if False:
        print('Hello World!')
    'Generate ID for artifacts that only exist in a bundle\n\n    This ID is only unique per release.\n\n    We use the name of the release file because it is also the key for lookups\n    in ArtifactIndex. To prevent any urlencode confusion, we base64 encode it.\n\n    '
    if obj.id:
        return str(obj.id)
    if obj.name:
        dist_name = ''
        if obj.dist_id:
            dist_name = Distribution.objects.get(pk=obj.dist_id).name
        return urlsafe_b64encode(f'{dist_name}_{obj.name}'.encode())

def decode_release_file_id(id: str):
    if False:
        for i in range(10):
            print('nop')
    'May raise ValueError'
    try:
        return int(id)
    except ValueError:
        decoded = urlsafe_b64decode(id).decode()
        (dist, url) = decoded.split('_', 1)
        return (dist or None, url)

@register(ReleaseFile)
class ReleaseFileSerializer(Serializer):

    def serialize(self, obj, attrs, user):
        if False:
            return 10
        dist_name = None
        if obj.dist_id:
            dist_name = Distribution.objects.get(pk=obj.dist_id).name
        return {'id': encode_release_file_id(obj), 'name': obj.name, 'dist': dist_name, 'headers': obj.file.headers, 'size': obj.file.size, 'sha1': obj.file.checksum, 'dateCreated': obj.file.timestamp}