import hashlib
from django.conf import settings
from zerver.models import UserProfile

def gravatar_hash(email: str) -> str:
    if False:
        return 10
    'Compute the Gravatar hash for an email address.'
    return hashlib.md5(email.lower().encode()).hexdigest()

def user_avatar_hash(uid: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    user_key = uid + settings.AVATAR_SALT
    return hashlib.sha1(user_key.encode()).hexdigest()

def user_avatar_path(user_profile: UserProfile) -> str:
    if False:
        return 10
    return user_avatar_path_from_ids(user_profile.id, user_profile.realm_id)

def user_avatar_path_from_ids(user_profile_id: int, realm_id: int) -> str:
    if False:
        i = 10
        return i + 15
    user_id_hash = user_avatar_hash(str(user_profile_id))
    return f'{realm_id}/{user_id_hash}'

def user_avatar_content_hash(ldap_avatar: bytes) -> str:
    if False:
        i = 10
        return i + 15
    return hashlib.sha256(ldap_avatar).hexdigest()