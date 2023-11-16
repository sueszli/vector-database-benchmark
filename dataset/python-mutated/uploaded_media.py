from io import BytesIO
from typing import Dict, Optional
import structlog
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from drf_spectacular.utils import extend_schema
from PIL import Image
from rest_framework import status, viewsets
from rest_framework.exceptions import APIException, UnsupportedMediaType, ValidationError
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from rest_framework.response import Response
from statshog.defaults.django import statsd
from posthog.api.routing import StructuredViewSetMixin
from posthog.models import UploadedMedia
from posthog.models.uploaded_media import ObjectStorageUnavailable
from posthog.permissions import ProjectMembershipNecessaryPermissions, TeamMemberAccessPermission
from posthog.storage import object_storage
FOUR_MEGABYTES = 4 * 1024 * 1024
logger = structlog.getLogger(__name__)

def validate_image_file(file: Optional[bytes], user: int) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Django validates file content type by reading "magic bytes" from the start of the file.\n    It doesn\'t then check that file really is the type it claims to be.\n\n    This could allow an attacker to attempt to upload HTML with magic bytes that pretend to be an image file.\n    We would store that and then serve it back to a dashboard. ☠️\n\n    Here we check that the file is actually a valid image file by opening and transposing it.\n    '
    if file is None:
        return False
    try:
        im = Image.open(BytesIO(file))
        im.transpose(Image.FLIP_LEFT_RIGHT)
        im.close()
        return True
    except Exception as e:
        logger.error('uploaded_media.image_verification_error', user=user, exception=e, exc_info=True)
        return False

@csrf_exempt
def download(request, *args, **kwargs) -> HttpResponse:
    if False:
        for i in range(10):
            print('nop')
    '\n    Images are immutable, so we can cache them forever\n    They are served unauthenticated as they might be presented on shared dashboards\n    '
    instance: Optional[UploadedMedia] = None
    try:
        instance = UploadedMedia.objects.get(pk=kwargs['image_uuid'])
    except UploadedMedia.DoesNotExist:
        return HttpResponse(status=404)
    file_bytes = object_storage.read_bytes(instance.media_location)
    statsd.incr('uploaded_media.served', tags={'team_id': instance.team_id, 'uuid': kwargs['image_uuid']})
    return HttpResponse(file_bytes, content_type=instance.content_type, headers={'Cache-Control': 'public, max-age=315360000, immutable'})

class MediaViewSet(StructuredViewSetMixin, viewsets.GenericViewSet):
    queryset = UploadedMedia.objects.all()
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [IsAuthenticatedOrReadOnly, ProjectMembershipNecessaryPermissions, TeamMemberAccessPermission]

    @extend_schema(description="\n    When object storage is available this API allows upload of media which can be used, for example, in text cards on dashboards.\n\n    Uploaded media must have a content type beginning with 'image/' and be less than 4MB.\n    ")
    def create(self, request, *args, **kwargs) -> Response:
        if False:
            while True:
                i = 10
        try:
            file = request.data['image']
            if file.size > FOUR_MEGABYTES:
                raise ValidationError(code='file_too_large', detail='Uploaded media must be less than 4MB')
            if file.content_type.startswith('image/'):
                uploaded_media = UploadedMedia.save_content(team=self.team, created_by=request.user, file_name=file.name, content_type=file.content_type, content=file.file)
                if uploaded_media is None:
                    raise APIException('Could not save media')
                bytes_to_verify = object_storage.read_bytes(uploaded_media.media_location)
                if not validate_image_file(bytes_to_verify, user=request.user.id):
                    statsd.incr('uploaded_media.image_failed_validation', tags={'file_name': file.name, 'team': self.team_id})
                    uploaded_media.delete()
                    raise ValidationError(code='invalid_image', detail='Uploaded media must be a valid image')
                headers = self.get_success_headers(uploaded_media.get_absolute_url())
                statsd.incr('uploaded_media.uploaded', tags={'team_id': self.team.pk, 'content_type': file.content_type})
                return Response({'id': uploaded_media.id, 'image_location': uploaded_media.get_absolute_url(), 'name': uploaded_media.file_name}, status=status.HTTP_201_CREATED, headers=headers)
            else:
                raise UnsupportedMediaType(file.content_type)
        except KeyError:
            raise ValidationError(code='no-image-provided', detail='An image file must be provided')
        except ObjectStorageUnavailable:
            raise ValidationError(code='object_storage_required', detail='Object storage must be available to allow media uploads.')

    def get_success_headers(self, location: str) -> Dict:
        if False:
            while True:
                i = 10
        try:
            return {'Location': location}
        except (TypeError, KeyError):
            return {}