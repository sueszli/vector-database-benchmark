from typing import List, Optional, Tuple
from django.conf import settings
from django.db.models import F, Sum
from django.db.models.functions import Length
from zerver.models import BotStorageData, UserProfile

class StateError(Exception):
    pass

def get_bot_storage(bot_profile: UserProfile, key: str) -> str:
    if False:
        return 10
    try:
        return BotStorageData.objects.get(bot_profile=bot_profile, key=key).value
    except BotStorageData.DoesNotExist:
        raise StateError('Key does not exist.')

def get_bot_storage_size(bot_profile: UserProfile, key: Optional[str]=None) -> int:
    if False:
        return 10
    if key is None:
        return BotStorageData.objects.filter(bot_profile=bot_profile).annotate(key_size=Length('key'), value_size=Length('value')).aggregate(sum=Sum(F('key_size') + F('value_size')))['sum'] or 0
    else:
        try:
            return len(key) + len(BotStorageData.objects.get(bot_profile=bot_profile, key=key).value)
        except BotStorageData.DoesNotExist:
            return 0

def set_bot_storage(bot_profile: UserProfile, entries: List[Tuple[str, str]]) -> None:
    if False:
        i = 10
        return i + 15
    storage_size_limit = settings.USER_STATE_SIZE_LIMIT
    storage_size_difference = 0
    for (key, value) in entries:
        assert isinstance(key, str), 'Key type should be str.'
        assert isinstance(value, str), 'Value type should be str.'
        storage_size_difference += len(key) + len(value) - get_bot_storage_size(bot_profile, key)
    new_storage_size = get_bot_storage_size(bot_profile) + storage_size_difference
    if new_storage_size > storage_size_limit:
        raise StateError('Request exceeds storage limit by {} characters. The limit is {} characters.'.format(new_storage_size - storage_size_limit, storage_size_limit))
    else:
        for (key, value) in entries:
            BotStorageData.objects.update_or_create(bot_profile=bot_profile, key=key, defaults={'value': value})

def remove_bot_storage(bot_profile: UserProfile, keys: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    queryset = BotStorageData.objects.filter(bot_profile=bot_profile, key__in=keys)
    if len(queryset) < len(keys):
        raise StateError('Key does not exist.')
    queryset.delete()

def is_key_in_bot_storage(bot_profile: UserProfile, key: str) -> bool:
    if False:
        return 10
    return BotStorageData.objects.filter(bot_profile=bot_profile, key=key).exists()

def get_keys_in_bot_storage(bot_profile: UserProfile) -> List[str]:
    if False:
        return 10
    return list(BotStorageData.objects.filter(bot_profile=bot_profile).values_list('key', flat=True))