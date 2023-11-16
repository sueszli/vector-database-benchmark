from django.core.exceptions import MultipleObjectsReturned, ObjectDoesNotExist
from social_auth.exceptions import AuthException
from social_auth.models import UserSocialAuth

def associate_by_email(details, user=None, *args, **kwargs):
    if False:
        print('Hello World!')
    'Return user entry with same email address as one returned on details.'
    if user:
        return None
    email = details.get('email')
    if email:
        try:
            return {'user': UserSocialAuth.get_user_by_email(email=email)}
        except MultipleObjectsReturned:
            raise AuthException(kwargs['backend'], 'Not unique email address.')
        except ObjectDoesNotExist:
            pass