from django.conf import settings
from sentry.api.serializers import Serializer, register
from social_auth.models import UserSocialAuth

def get_provider_label(obj: UserSocialAuth) -> str:
    if False:
        for i in range(10):
            print('nop')
    return settings.AUTH_PROVIDER_LABELS[obj.provider]

@register(UserSocialAuth)
class UserSocialAuthSerializer(Serializer):

    def serialize(self, obj, attrs, user):
        if False:
            i = 10
            return i + 15
        return {'id': str(obj.id), 'provider': obj.provider, 'providerLabel': get_provider_label(obj)}