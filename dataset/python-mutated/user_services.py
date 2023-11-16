"""Services for user data."""
from __future__ import annotations
import datetime
import hashlib
import imghdr
import itertools
import logging
import re
import urllib
from core import feconf
from core import utils
from core.constants import constants
from core.domain import auth_domain
from core.domain import auth_services
from core.domain import exp_fetchers
from core.domain import fs_services
from core.domain import role_services
from core.domain import state_domain
from core.domain import user_domain
from core.platform import models
import requests
from typing import Dict, Final, List, Literal, Optional, Sequence, TypedDict, overload
MYPY = False
if MYPY:
    from mypy_imports import audit_models
    from mypy_imports import auth_models
    from mypy_imports import bulk_email_services
    from mypy_imports import suggestion_models
    from mypy_imports import transaction_services
    from mypy_imports import user_models
(auth_models, user_models, audit_models, suggestion_models) = models.Registry.import_models([models.Names.AUTH, models.Names.USER, models.Names.AUDIT, models.Names.SUGGESTION])
bulk_email_services = models.Registry.import_bulk_email_services()
transaction_services = models.Registry.import_transaction_services()
GRAVATAR_SIZE_PX: Final = 150
DEFAULT_IDENTICON_DATA_URL: Final = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEwAAABMCAYAAADHl1ErAAAAAXNSR0IArs4c6QAADhtJREFUeAHtXHlwVdUZ/859jyxmIQESyCaglC0iAgkJIntrIpvKphSwY2ttxbFOp9R/cGGqdhykLaMVO2OtoyRSCEKNEpYKyBIVQ1iNkBhNMCtb8shiQpJ3b7/fTW7m5uUlecu9L4nTM5Pce8895zvf93vnnPud833fEdQLKXb5jsC6%2BuZERZbHKaSMYRbGKERxgpQQUkSIIigEbAmFavlfrUKiVhCVcFa%2BIJEvJOlCcNCAnNKMFQ0o58vEfPgmhS5Mn0ot8n2KIs8lIZJJUfy8almIJqbxhRDSIbJKe2s%2BXvWlV/RcrGwqYGGp20bI1LyaeVmjKMrodp4EycGBAy6MjgsrSxozqG7O5GgxcVREeEigNDAwwBpmsUiRKGu3y1caGltstQ3yjbOFV6sPnypXTuRXBReU2GLqGprHkUKSRlMIUcD3WyUakGbbt7JYyzf6agpgYfe9O8kui/U8nB7UhJIkUTljwrBTTz449mZKUlyCEBTnjTCKQiX7T5ScfGP3Rf9j5ysny7IyTKXHPwYP690WSXnZtvcXp71pw1ldQwELm59%2BlyzbX%2BbeNL%2Btscb4EYOyNz2ZWD99wtAFnGdxxoQBefbs85f3rHsjJyivuGo60wsATe51WZJkWW/LWnXGgDZUEoYAFr58x0B7beOLPHGv5XnFIpGoS0mKOfze%2Bpmj/f2smNR9lm42teQ/8vLRgv0nyuZwVwtm1Ows5BZLSMBz1RkrbnjLiNeAhaWmPWgn%2BxYeejwkRMu9idH7tm%2BYE8/z0EhvmfOmPs9/RQ9tOJx3IKc8lUixkqBKC1nW2vat3u0NXY8Bi1%2B%2Bw6%2BktnETD7%2BnwEB4iP/pL/5xf03U4IBZ3jBkdN2K641Hkn/7YWh17c1JoM3D9PW4kIB1eRkrmjxpyyPAeK4aLttbPuAhOIU5aHpm1cTMZ1ffuRT8eMKED%2BooL6Wd%2B2Bj%2BtnFUGeYyVzJYl3Kc9sld9t2W8Dw%2BWkTWuz2fdxQ9ACr9P3Jfy7%2BZuSw0HnuNtwb5Ysqaw4mPJb5k%2BYW%2BVZuv9xqsaRWZ60%2B7w4vbgEWnrJ1hp3kTO5ZYUPCAnK%2B3bYiitWDWHca7O2yrI6U3r5yR8U1W2MiC2%2BzkLS4ev%2BaY67y1a749VQBYLUIZT/AGhUTduS7f68Y39/AgozgGbxDBsgCmSBbT/Jr710CDMMQPYvHf2DC2Mj9p95efA8TCNKI9MNrEGSALJAJskFGV%2BTocUhigrfbWz5jYtH4VdrAMksBdYVnI8vYJ/8q83hhmW0WEy23WKx39/Qh6LaHQXXA1xBgYc5isBL4/scCFoC3QCbIBhkhK2TGi65St4CpeharDvgaYoJnIv15GHaFQRBkg4w8p02BzF0VRH6XgEGDV5VS1rOgOvTHCb47wfXvIBtkhE4JmSG7/r3%2B3ilg6toQyx1OUEr7i56lF8zde8gIWVEPSz1g4IyGU8CwkMbaEMudNg3eWd0fXR5khcyQXcXAiYSdAMMWDY/ltVhIY23IdXr8kjqh21%2BzRKvMogUYAAtHQToBhv0sbNFg16GvLaQdmTfjGTJDdmCgYuHQSIfe07pTSqewn3V9z6qrvb1F48Crzx6xNTR4QXoE9tN4c2%2ByfufWqudC3VbmAYzNPwZrkf6dL%2B4LSm5Q9vkrVH79B6qs%2BoH8B1goatAtNCIqmOZOiabw4G5VJMNYREdhDD7ae6J0USsmtEwj3t7DYLCwK83f8WbbzauZP7/kq53SxiY7vfmfC5R24Fv6prTrDVEWgqbfEUlPLY2nlKkxGv%2BmXbFzG7H4/eE8g/tZyO92zbDSPoe1WncUgT14X4G189NimvjobnrhX6e6BQuo8DCho2crafnzB2n%2BMwe4PL5H5iVgACx4wEltli%2B1sXbA%2BGkNcmCwUN%2BY%2BI%2B3WOjZt3Lpl68cpQoefu6m4%2Bcqae7TWfTfk%2BXuVnWrvA4LFRtUVockjKxKc8sJmMJsWWsiON/U9eJvNmXTtk%2B%2BdYt5Z4WZX0p/bjYtmBbn7LURefaw%2BVuvwoQnBliTYCxu7WFskQb1WROjcvliKlibM/IMAQv8siD0643H6etiGx7NSBbYUlXCbRipgKnme859Ysl4jwwDrnKaV2SjDe%2B0tu9qnZ7KsQWch/YxVpt6KunZexieUVPDSIJjCC86k3lwyikJ0di%2BMS09/3au2iuMbuDr4mpKN2CIO%2BMLVnpgA4yAlVRX1ziV4fODrwOv2k2bDM4UVvEkXeaMJ0PyXn3/nCF0HIkAE2ADjICVpChiLArBMcSxsJHPmdmXjCTXiVZRRS19VVTdKd%2BIDA0bYCW1%2BWcRvGiMIN4Vjb1flHb1yrD8rM9LDKOlJ6RhA6ww6au%2BD3A50hcy%2Bt5sRRP8FpSYo8zqsBnDPax13oJ/ltEgafSqam5SU7NdezTtWsHrTzOShg2wYtWP3SQ5wZnNjMZA80Z9s1mkO9CtMakdDRtgJcGnFK3C869D6wY%2BRISp7loGUnROKtKkdtqxYawkzQGXdwNUN0nnrHiXGxxoJf40e0fEhdpRg29xoZT7RTRsgJV%2B8e0%2BJTdqJIwd4kZpz4pOGWN%2BG5Lq2s38wQHXMzZdq2XiAlllgP2%2BaH6yOX4xGjbAinejlVq0CG9l10T3rNT99wwnf96KMyvNuHMoDR0UaAr5dmwYK1YrhAoYXLtNaa2N6DAW5vFF6qLClGZeeHSyKXRBVMMGWLFaoUZYEPzgTWuxjfC6lROI/RgMb2bZ7JGUaOIcqWEDrDDp50MCBA0YLokDQRgx0p%2BdTezH4PDG88dxI8LotaeneU7AhZo6bPK5hwkVMERYuFDX6yLT2JDx99/fTVY2anibYiOCaPuGuayydDB%2BeUu2U30NG2AlCaFcRAmEo3QqaVLGynm30a6X5sHz2uMWksZH0pHXF9CIYeb/zho2CAqTgoMDvoTXCmJ3EI7isQRuVpw9KYqytyykhxk8qASuJoD84mNTKGvjveSLFQQwUeOaGCNE0Flqvs5o8b/9gZ8xwyMmj404NComZJyrzHtbLjTIjxZNv1X9C/S30pXqRrLVdd4lh7EjOX4oPfHAOHrzD9Np9l1RZMHnygeJ45kOZXxaPJ6byr6WueotdfAjhI73rGdu2ZXnn5oY7QM2OjZxx8hw%2BvPjCepf2bUfqJz/Llc1qHpb1OBAiosMpoFB5i%2BtOnLV%2BoTgL9ypYYZ8bZ0tOd6QmuUNbCiFMoN9GPM0TCbeXYoZcgvhr48kOyLlVF6AESf1UwV7G88jBbC/ISqsjzDb62wAC9UmydhoAaz6b/tWcIgQul7ntI8woMNCxQZstQOGSFYeqQriDeGI0Ud47jU2gIEae8kmtlZsWllpB6zNO2UXZwcg3rDXOO0jDbdhEIDoXs1zB6y1A4YHhP3iiuBMOJXh3tfJzuZ/qBbfX65nR5UGqmto8TUL2OoqAgZoWMNEY6KTMhOa%2Bt4ehCDfmxjz8c4X5y3UChp5hVk/j63Vpwuu0zdlNVTIrkuFfC1hkOobO%2B//Qw8LD/an26JDaFRsKI2KCWU76kCaOi6CoHYYnZY9d/DjAzllC/lDmFWz75EFevqdFmGIkbbL9hREsiI40yg/11wGhxex9PlXV%2BjEhatUU99ZQdUzpr%2BH08n1mkb1L%2BfiVf0rGs5Lo2nxkXT3HUPZ0S7WawAhsxrFy6HPwKJDY/zQqYehAPey1%2BDgDxfsSxkPwZPYaTmU7S7BPWDXkWLafayYLlWaaidW2cASK5nBWzJzOD3AG5YebCgqw5dvP4PoXab1Oveu3znK5xQIOPW31DZchL/6M6vv2sn%2B68scK3b1jDlo%2B6Hv6G878ij/e1M3cbtiQc3HML4vKZbWrbyTpowe3G1Z7SVH7e7cmHZmGXePSmtI4FhnQfVOAQMBNfhdse/CwvzsO/cf6ykapKlZpq0HCmlzxlc%2B6U2akK5c2XJNf3x4At3D29hdJUTrTnz0wxlwOrEIy5Kugum7BAyEtaGJwKVrH63mrSDn0besEdNTmz9XJ%2B6uGOoL%2BbAr/OXJJIoM77jryx%2Bh0iGL0mSENnc1FDX%2BO6gVWqZ2RfQ9I5oLQgj75fxO/q%2BvpJ9TnXTxlevr6cPjlyj5iUx2bb%2BsZ7UesqlgsayQWf/S8b7bHobC3QWYrv3rZ%2BwuXuhIs88/Y4v8vfWz4BvrdoBpj4BBejWE2W4/yupTGMJ%2BD21O/emf3j1t2bTNrYD8PgWkv7/FflvUwE8uFFelMAg2i8Uy05UTBlwCTAWtLUieJ8XA2MiQIxXX6xNYI%2B6XC3Wep%2Br5xz/Jsszij1qDVREprp4s4DJgGmjaMQzcUA5bgaNkRTbH3GxSf5SEVMoxRBUMlrnHMIB//ArounxbjgZZuWWtSzlokmyGkwWv4Bm8QwZ1GLpxZgUYcquHaRLgQ6A/SobJ4IiGpeyc7RE9ja55V/aKEOID5s/3R8loQjkeVsTzwmmeF2oYuFlamT5xFeII/4qh3LMmgR/oWT4/rEgPhONxWEKifUJW4mWikfpyvr5nBbNIkUQeD8BU7lm9fxyWHgDHA9fYQlzHg/0w/6qjuZzqdKwvb/J9PveiAl4Hz%2BE5q%2B8duKYXHjHSjkf6sXkqWyEZK4QFLIQ51iihWrr2CJKCeE6fzm2pax8Grm8e6acHDffth0YSLdF9CCoZvFye55okRU7gIetV1AkPuRJZSCfZUdefezJMYf3v0MhOwHVzLKlQxAWSRJlQlDr%2BzrPcUjjbGwbyBB2mCKH62/K7KwywjWM8b5CQq%2BH9x%2B%2BCSVZiFKH8eI4ldQQOz4jJ/P/Bt86QcSFPPVqZA50Qu4NwFK7i3tHK7HEEJ5reOFr5fwkK97jkk8ywAAAAAElFTkSuQmCC'
LABEL_FOR_USER_BEING_DELETED: Final = '[User being deleted]'
USERNAME_FOR_USER_BEING_DELETED: Final = 'UserBeingDeleted'
TIMEOUT_SECS = 60

class DashboardStatsDict(TypedDict):
    """Dictionary representing the dashborad stats dictionary."""
    num_ratings: int
    average_ratings: Optional[float]
    total_plays: int

def is_username_taken(username: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Returns whether the given username has already been taken.\n\n    Args:\n        username: str. Identifiable username to display in the UI.\n\n    Returns:\n        bool. Whether the given username is taken.\n    '
    return user_models.UserSettingsModel.is_normalized_username_taken(user_domain.UserSettings.normalize_username(username))

def get_email_from_user_id(user_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Gets the email from a given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        str. The user_email corresponding to the given user_id.\n\n    Raises:\n        Exception. The user is not found.\n    '
    user_settings = get_user_settings(user_id)
    return user_settings.email

@overload
def get_user_id_from_username(username: str, *, strict: Literal[True]) -> str:
    if False:
        while True:
            i = 10
    ...

@overload
def get_user_id_from_username(username: str) -> Optional[str]:
    if False:
        return 10
    ...

@overload
def get_user_id_from_username(username: str, *, strict: Literal[False]) -> Optional[str]:
    if False:
        while True:
            i = 10
    ...

def get_user_id_from_username(username: str, strict: bool=False) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Gets the user_id for a given username.\n\n    Args:\n        username: str. Identifiable username to display in the UI.\n        strict: bool. Whether to fail noisily if no UserSettingsModel with a\n            given username found in the datastore.\n\n    Returns:\n        str or None. If the user with given username does not exist, return\n        None. Otherwise return the user_id corresponding to given username.\n\n    Raises:\n        Exception. No user_id found for the given username.\n    '
    user_model = user_models.UserSettingsModel.get_by_normalized_username(user_domain.UserSettings.normalize_username(username))
    if user_model is None:
        if strict:
            raise Exception('No user_id found for the given username: %s' % username)
        return None
    else:
        return user_model.id

@overload
def get_multi_user_ids_from_usernames(usernames: List[str], *, strict: Literal[True]) -> List[str]:
    if False:
        print('Hello World!')
    ...

@overload
def get_multi_user_ids_from_usernames(usernames: List[str]) -> List[Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_multi_user_ids_from_usernames(usernames: List[str], *, strict: Literal[False]) -> List[Optional[str]]:
    if False:
        i = 10
        return i + 15
    ...

def get_multi_user_ids_from_usernames(usernames: List[str], strict: bool=False) -> Sequence[Optional[str]]:
    if False:
        i = 10
        return i + 15
    'Gets the user_ids for a given list of usernames.\n\n    Args:\n        usernames: list(str). Identifiable usernames to display in the UI.\n        strict: bool. Whether to fail noisily if no user_id with the given\n            useranme found.\n\n    Returns:\n        list(str|None). Return the list of user ids corresponding to given\n        usernames.\n\n    Raises:\n        Exception. No user_id found for the username.\n    '
    if len(usernames) == 0:
        return []
    normalized_usernames = [user_domain.UserSettings.normalize_username(username) for username in usernames]
    found_models: Sequence[user_models.UserSettingsModel] = user_models.UserSettingsModel.query(user_models.UserSettingsModel.normalized_username.IN(normalized_usernames)).fetch()
    username_to_user_id_map = {model.normalized_username: model.id for model in found_models}
    user_ids = []
    for username in normalized_usernames:
        user_id = username_to_user_id_map.get(username)
        if strict and user_id is None:
            raise Exception('No user_id found for the username: %s' % username)
        user_ids.append(user_id)
    return user_ids

def get_user_settings_from_username(username: str) -> Optional[user_domain.UserSettings]:
    if False:
        for i in range(10):
            print('nop')
    'Gets the user settings for a given username.\n\n    Args:\n        username: str. Identifiable username to display in the UI.\n\n    Returns:\n        UserSettingsModel or None. The UserSettingsModel instance corresponding\n        to the given username, or None if no such model was found.\n    '
    user_model = user_models.UserSettingsModel.get_by_normalized_username(user_domain.UserSettings.normalize_username(username))
    if user_model is None:
        return None
    else:
        return get_user_settings(user_model.id)

def get_user_settings_from_email(email: str) -> Optional[user_domain.UserSettings]:
    if False:
        return 10
    'Gets the user settings for a given email.\n\n    Args:\n        email: str. Email of the user.\n\n    Returns:\n        UserSettingsModel or None. The UserSettingsModel instance corresponding\n        to the given email, or None if no such model was found.\n    '
    user_model = user_models.UserSettingsModel.get_by_email(email)
    if user_model is None:
        return None
    else:
        return get_user_settings(user_model.id)

@overload
def get_users_settings(user_ids: Sequence[Optional[str]], *, strict: Literal[True], include_marked_deleted: bool=False) -> Sequence[user_domain.UserSettings]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_users_settings(user_ids: Sequence[Optional[str]], *, strict: Literal[False], include_marked_deleted: bool=False) -> Sequence[Optional[user_domain.UserSettings]]:
    if False:
        print('Hello World!')
    ...

@overload
def get_users_settings(user_ids: Sequence[Optional[str]], *, strict: bool=..., include_marked_deleted: bool=False) -> Sequence[Optional[user_domain.UserSettings]]:
    if False:
        while True:
            i = 10
    ...

def get_users_settings(user_ids: Sequence[Optional[str]], strict: bool=False, include_marked_deleted: bool=False) -> Sequence[Optional[user_domain.UserSettings]]:
    if False:
        print('Hello World!')
    "Gets domain objects representing the settings for the given user_ids.\n\n    Args:\n        user_ids: list(str). The list of user_ids to get UserSettings\n            domain objects for.\n        strict: bool. Whether to fail noisily if one or more user IDs don't\n            exist in the datastore. Defaults to False.\n        include_marked_deleted: bool. Whether to included users that are being\n            deleted. This should be used only for retrieving the usernames.\n\n    Returns:\n        list(UserSettings|None). The UserSettings domain objects corresponding\n        to the given user ids. If the given user_id does not exist, the\n        corresponding entry in the returned list is None.\n\n    Raises:\n        Exception. When strict mode is enabled and some user is not found.\n    "
    user_settings_models = user_models.UserSettingsModel.get_multi(user_ids, include_deleted=include_marked_deleted)
    if strict:
        for (user_id, user_settings_model) in zip(user_ids, user_settings_models):
            if user_settings_model is None:
                raise Exception("User with ID '%s' not found." % user_id)
    result: List[Optional[user_domain.UserSettings]] = []
    for (i, model) in enumerate(user_settings_models):
        if user_ids[i] == feconf.SYSTEM_COMMITTER_ID:
            result.append(user_domain.UserSettings(user_id=feconf.SYSTEM_COMMITTER_ID, email=feconf.SYSTEM_EMAIL_ADDRESS, roles=[feconf.ROLE_ID_FULL_USER, feconf.ROLE_ID_CURRICULUM_ADMIN, feconf.ROLE_ID_MODERATOR, feconf.ROLE_ID_VOICEOVER_ADMIN], banned=False, username='admin', has_viewed_lesson_info_modal_once=False, last_agreed_to_terms=datetime.datetime.utcnow()))
        else:
            if model is not None and model.deleted:
                model.username = USERNAME_FOR_USER_BEING_DELETED
            result.append(_get_user_settings_from_model(model) if model is not None else None)
    return result

def generate_initial_profile_picture(user_id: str) -> None:
    if False:
        while True:
            i = 10
    "Generates a profile picture for a new user and\n    updates the user's settings in the datastore.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    "
    user_email = get_email_from_user_id(user_id)
    user_gravatar = fetch_gravatar(user_email)
    update_profile_picture_data_url(user_id, user_gravatar)

def get_gravatar_url(email: str) -> str:
    if False:
        print('Hello World!')
    'Returns the gravatar url for the specified email.\n\n    Args:\n        email: str. The user email.\n\n    Returns:\n        str. The gravatar url for the specified email.\n    '
    return 'https://www.gravatar.com/avatar/%s?d=identicon&s=%s' % (hashlib.md5(email.encode('utf-8')).hexdigest(), GRAVATAR_SIZE_PX)

def fetch_gravatar(email: str) -> str:
    if False:
        print('Hello World!')
    "Returns the gravatar corresponding to the user's email, or an\n    identicon generated from the email if the gravatar doesn't exist.\n\n    Args:\n        email: str. The user email.\n\n    Returns:\n        str. The gravatar url corresponding to the given user email. If the call\n        to the gravatar service fails, this returns DEFAULT_IDENTICON_DATA_URL\n        and logs an error.\n    "
    gravatar_url = get_gravatar_url(email)
    try:
        response = requests.get(gravatar_url, headers={b'Content-Type': b'image/png'}, allow_redirects=False, timeout=TIMEOUT_SECS)
    except Exception:
        logging.exception('Failed to fetch Gravatar from %s' % gravatar_url)
    else:
        if response.ok:
            if imghdr.what(None, h=response.content) == 'png':
                return utils.convert_image_binary_to_data_url(response.content, 'png')
        else:
            logging.error('[Status %s] Failed to fetch Gravatar from %s' % (response.status_code, gravatar_url))
    return DEFAULT_IDENTICON_DATA_URL

@overload
def get_user_settings(user_id: str) -> user_domain.UserSettings:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_user_settings(user_id: str, *, strict: Literal[True]) -> user_domain.UserSettings:
    if False:
        return 10
    ...

@overload
def get_user_settings(user_id: str, *, strict: Literal[False]) -> Optional[user_domain.UserSettings]:
    if False:
        i = 10
        return i + 15
    ...

def get_user_settings(user_id: str, strict: bool=True) -> Optional[user_domain.UserSettings]:
    if False:
        while True:
            i = 10
    'Return the user settings for a single user.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        strict: bool. Whether to fail noisily if no user with the given\n            id exists in the datastore. Defaults to True.\n\n    Returns:\n        UserSettings or None. If the given user_id does not exist and strict\n        is False, returns None. Otherwise, returns the corresponding\n        UserSettings domain object.\n\n    Raises:\n        Exception. The value of strict is True and given user_id does not exist.\n    '
    user_settings = get_users_settings([user_id])[0]
    if strict and user_settings is None:
        logging.error('Could not find user with id %s' % user_id)
        raise Exception('User not found.')
    return user_settings

@overload
def get_user_settings_by_auth_id(auth_id: str, *, strict: Literal[True]) -> user_domain.UserSettings:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_user_settings_by_auth_id(auth_id: str) -> Optional[user_domain.UserSettings]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_user_settings_by_auth_id(auth_id: str, *, strict: Literal[False]) -> Optional[user_domain.UserSettings]:
    if False:
        while True:
            i = 10
    ...

def get_user_settings_by_auth_id(auth_id: str, strict: bool=False) -> Optional[user_domain.UserSettings]:
    if False:
        while True:
            i = 10
    'Return the user settings for a single user.\n\n    Args:\n        auth_id: str. The auth ID of the user.\n        strict: bool. Whether to fail noisily if no user with the given\n            id exists in the datastore. Defaults to False.\n\n    Returns:\n        UserSettings or None. If the given auth_id does not exist and strict is\n        False, returns None. Otherwise, returns the corresponding UserSettings\n        domain object.\n\n    Raises:\n        Exception. The value of strict is True and given auth_id does not exist.\n    '
    user_id = auth_services.get_user_id_from_auth_id(auth_id, include_deleted=True)
    user_settings_model = None if user_id is None else user_models.UserSettingsModel.get_by_id(user_id)
    if user_settings_model is not None:
        return _get_user_settings_from_model(user_settings_model)
    elif strict:
        logging.error('Could not find user with id %s' % auth_id)
        raise Exception('User not found.')
    else:
        return None

def get_user_roles_from_id(user_id: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns roles of the user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        list(str). Roles of the user with given id.\n    '
    user_settings = get_user_settings(user_id, strict=False)
    if user_settings is None:
        return [feconf.ROLE_ID_GUEST]
    return user_settings.roles

def _create_user_contribution_rights_from_model(user_contribution_rights_model: Optional[user_models.UserContributionRightsModel]) -> user_domain.UserContributionRights:
    if False:
        for i in range(10):
            print('nop')
    'Creates a UserContributionRights object from the given model. If the\n    model is None, an empty UserContributionRights object is returned.\n\n    Args:\n        user_contribution_rights_model: UserContributionRightsModel. The model\n            used to create the UserContributionRights domain object.\n\n    Returns:\n        UserContributionRights. The UserContributionRights domain object\n        associated with the model, or an empty UserContributionRights domain\n        object if the model is None.\n    '
    if user_contribution_rights_model is not None:
        return user_domain.UserContributionRights(user_contribution_rights_model.id, user_contribution_rights_model.can_review_translation_for_language_codes, user_contribution_rights_model.can_review_voiceover_for_language_codes, user_contribution_rights_model.can_review_questions, user_contribution_rights_model.can_submit_questions)
    else:
        return user_domain.UserContributionRights('', [], [], False, False)

def get_user_contribution_rights(user_id: str) -> user_domain.UserContributionRights:
    if False:
        while True:
            i = 10
    'Returns the UserContributionRights domain object for the given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        UserContributionRights. The UserContributionRights domain object for the\n        corresponding user.\n    '
    return get_users_contribution_rights([user_id])[0]

def get_users_contribution_rights(user_ids: List[str]) -> List[user_domain.UserContributionRights]:
    if False:
        i = 10
        return i + 15
    'Returns the UserContributionRights domain object for each user_id in\n    user_ids.\n\n    Args:\n        user_ids: list(str). A list of user ids.\n\n    Returns:\n        list(UserContributionRights). A list containing the\n        UserContributionRights domain object for each user.\n    '
    user_contribution_rights_models = user_models.UserContributionRightsModel.get_multi(user_ids)
    users_contribution_rights = []
    for (index, user_contribution_rights_model) in enumerate(user_contribution_rights_models):
        user_contribution_rights = _create_user_contribution_rights_from_model(user_contribution_rights_model)
        if user_contribution_rights_model is None:
            user_contribution_rights.id = user_ids[index]
        users_contribution_rights.append(user_contribution_rights)
    return users_contribution_rights

def get_reviewer_user_ids_to_notify() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Gets a list of the reviewer user_ids who want to be notified of\n    Contributor Dashboard reviewer updates.\n\n    Returns:\n        list(str). A list of reviewer user_ids who want to be notified of\n        Contributor Dashboard reviewer updates.\n    '
    users_contribution_rights = get_all_reviewers_contribution_rights()
    reviewer_ids = [user_contribution_rights.id for user_contribution_rights in users_contribution_rights]
    users_global_prefs = get_users_email_preferences(reviewer_ids)
    reviewer_ids_to_notify = []
    for (index, user_global_pref) in enumerate(users_global_prefs):
        if user_global_pref.can_receive_email_updates:
            reviewer_ids_to_notify.append(reviewer_ids[index])
    return reviewer_ids_to_notify

def get_all_reviewers_contribution_rights() -> List[user_domain.UserContributionRights]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of UserContributionRights objects corresponding to each\n    UserContributionRightsModel.\n\n    Returns:\n        list(UserContributionRights). A list of UserContributionRights objects.\n    '
    user_contribution_rights_models = user_models.UserContributionRightsModel.get_all()
    return [_create_user_contribution_rights_from_model(user_contribution_rights_model) for user_contribution_rights_model in user_contribution_rights_models]

def _save_user_contribution_rights(user_contribution_rights: user_domain.UserContributionRights) -> None:
    if False:
        i = 10
        return i + 15
    'Saves the UserContributionRights object into the datastore.\n\n    Args:\n        user_contribution_rights: UserContributionRights. The\n            UserContributionRights object of the user.\n    '
    user_contribution_rights.validate()
    _update_reviewer_counts_in_community_contribution_stats(user_contribution_rights)
    user_models.UserContributionRightsModel(id=user_contribution_rights.id, can_review_translation_for_language_codes=user_contribution_rights.can_review_translation_for_language_codes, can_review_voiceover_for_language_codes=user_contribution_rights.can_review_voiceover_for_language_codes, can_review_questions=user_contribution_rights.can_review_questions, can_submit_questions=user_contribution_rights.can_submit_questions).put()

def _update_user_contribution_rights(user_contribution_rights: user_domain.UserContributionRights) -> None:
    if False:
        i = 10
        return i + 15
    'Updates the users rights model if the updated object has review rights in\n    at least one item else delete the existing model.\n\n    Args:\n        user_contribution_rights: UserContributionRights. The updated\n            UserContributionRights object of the user.\n    '
    if user_contribution_rights.can_review_at_least_one_item():
        _save_user_contribution_rights(user_contribution_rights)
    else:
        remove_contribution_reviewer(user_contribution_rights.id)

@transaction_services.run_in_transaction_wrapper
def _update_reviewer_counts_in_community_contribution_stats_transactional(future_user_contribution_rights: user_domain.UserContributionRights) -> None:
    if False:
        return 10
    'Updates the reviewer counts in the community contribution stats based\n    on the given user contribution rights with the most up-to-date values.\n    This method is intended to be called right before the new updates to the\n    user contribution rights have been saved in the datastore. Note that this\n    method should only ever be called in a transaction.\n\n    Args:\n        future_user_contribution_rights: UserContributionRights. The most\n            up-to-date user contribution rights.\n    '
    past_user_contribution_rights = get_user_contribution_rights(future_user_contribution_rights.id)
    stats_model = suggestion_models.CommunityContributionStatsModel.get()
    future_languages_that_reviewer_can_review = set(future_user_contribution_rights.can_review_translation_for_language_codes)
    past_languages_that_reviewer_can_review = set(past_user_contribution_rights.can_review_translation_for_language_codes)
    languages_that_reviewer_can_no_longer_review = past_languages_that_reviewer_can_review.difference(future_languages_that_reviewer_can_review)
    new_languages_that_reviewer_can_review = future_languages_that_reviewer_can_review.difference(past_languages_that_reviewer_can_review)
    if past_user_contribution_rights.can_review_questions and (not future_user_contribution_rights.can_review_questions):
        stats_model.question_reviewer_count -= 1
    if not past_user_contribution_rights.can_review_questions and future_user_contribution_rights.can_review_questions:
        stats_model.question_reviewer_count += 1
    for language_code in languages_that_reviewer_can_no_longer_review:
        stats_model.translation_reviewer_counts_by_lang_code[language_code] -= 1
        if stats_model.translation_reviewer_counts_by_lang_code[language_code] == 0:
            del stats_model.translation_reviewer_counts_by_lang_code[language_code]
    for language_code in new_languages_that_reviewer_can_review:
        if language_code not in stats_model.translation_reviewer_counts_by_lang_code:
            stats_model.translation_reviewer_counts_by_lang_code[language_code] = 1
        else:
            stats_model.translation_reviewer_counts_by_lang_code[language_code] += 1
    stats_model.update_timestamps()
    stats_model.put()

def _update_reviewer_counts_in_community_contribution_stats(user_contribution_rights: user_domain.UserContributionRights) -> None:
    if False:
        i = 10
        return i + 15
    'Updates the reviewer counts in the community contribution stats based\n    on the updates to the given user contribution rights. The GET and PUT is\n    done in a transaction to avoid loss of updates that come in rapid\n    succession.\n\n    Args:\n        user_contribution_rights: UserContributionRights. The user contribution\n            rights.\n    '
    _update_reviewer_counts_in_community_contribution_stats_transactional(user_contribution_rights)

def get_usernames_by_role(role: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Get usernames of all the users with given role ID.\n\n    Args:\n        role: str. The role ID of users requested.\n\n    Returns:\n        list(str). List of usernames of users with given role ID.\n    '
    user_settings = user_models.UserSettingsModel.get_by_role(role)
    return [user.username for user in user_settings]

def get_user_ids_by_role(role: str) -> List[str]:
    if False:
        while True:
            i = 10
    'Get user ids of all the users with given role ID.\n\n    Args:\n        role: str. The role ID of users requested.\n\n    Returns:\n        list(str). List of user ids of users with given role ID.\n    '
    user_settings = user_models.UserSettingsModel.get_by_role(role)
    return [user.id for user in user_settings]

def get_user_actions_info(user_id: Optional[str]) -> user_domain.UserActionsInfo:
    if False:
        print('Hello World!')
    'Gets user actions info for a user.\n\n    Args:\n        user_id: str|None. The user ID of the user we want to get actions for,\n            or None if the user is not logged in.\n\n    Returns:\n        UserActionsInfo. User object with system committer user id.\n    '
    roles = get_user_roles_from_id(user_id) if user_id else [feconf.ROLE_ID_GUEST]
    actions = role_services.get_all_actions(roles)
    return user_domain.UserActionsInfo(user_id, roles, actions)

def get_system_user() -> user_domain.UserActionsInfo:
    if False:
        for i in range(10):
            print('nop')
    'Returns user object with system committer user id.\n\n    Returns:\n        UserActionsInfo. User object with system committer user id.\n    '
    return get_user_actions_info(feconf.SYSTEM_COMMITTER_ID)

def save_user_settings(user_settings: user_domain.UserSettings) -> None:
    if False:
        i = 10
        return i + 15
    'Commits a user settings object to the datastore.\n\n    Args:\n        user_settings: UserSettings. The user setting domain object to be saved.\n\n    Returns:\n        UserSettingsModel. The updated user settings model that was saved.\n    '
    user_model = convert_to_user_settings_model(user_settings)
    user_model.update_timestamps()
    user_model.put()

def convert_to_user_settings_model(user_settings: user_domain.UserSettings) -> user_models.UserSettingsModel:
    if False:
        for i in range(10):
            print('nop')
    'Converts a UserSettings domain object to a UserSettingsModel.\n\n    Args:\n        user_settings: UserSettings. The user setting domain object to be\n            converted.\n\n    Returns:\n        UserSettingsModel. The user settings model that was converted.\n    '
    user_settings.validate()
    user_settings_dict = user_settings.to_dict()
    user_model = user_models.UserSettingsModel.get_by_id(user_settings.user_id)
    if user_model is not None:
        user_model.populate(**user_settings_dict)
    else:
        user_settings_dict['id'] = user_settings.user_id
        user_model = user_models.UserSettingsModel(**user_settings_dict)
    return user_model

def _get_user_settings_from_model(user_settings_model: user_models.UserSettingsModel) -> user_domain.UserSettings:
    if False:
        return 10
    'Transform user settings storage model to domain object.\n\n    Args:\n        user_settings_model: UserSettingsModel. The model to be converted.\n\n    Returns:\n        UserSettings. Domain object for user settings.\n    '
    return user_domain.UserSettings(user_id=user_settings_model.id, email=user_settings_model.email, roles=user_settings_model.roles, banned=user_settings_model.banned, username=user_settings_model.username, last_agreed_to_terms=user_settings_model.last_agreed_to_terms, last_started_state_editor_tutorial=user_settings_model.last_started_state_editor_tutorial, last_started_state_translation_tutorial=user_settings_model.last_started_state_translation_tutorial, last_logged_in=user_settings_model.last_logged_in, last_edited_an_exploration=user_settings_model.last_edited_an_exploration, last_created_an_exploration=user_settings_model.last_created_an_exploration, default_dashboard=user_settings_model.default_dashboard, creator_dashboard_display_pref=user_settings_model.creator_dashboard_display_pref, user_bio=user_settings_model.user_bio, subject_interests=user_settings_model.subject_interests, first_contribution_msec=user_settings_model.first_contribution_msec, preferred_language_codes=user_settings_model.preferred_language_codes, preferred_site_language_code=user_settings_model.preferred_site_language_code, preferred_audio_language_code=user_settings_model.preferred_audio_language_code, preferred_translation_language_code=user_settings_model.preferred_translation_language_code, pin=user_settings_model.pin, display_alias=user_settings_model.display_alias, deleted=user_settings_model.deleted, created_on=user_settings_model.created_on, has_viewed_lesson_info_modal_once=user_settings_model.has_viewed_lesson_info_modal_once)

def is_user_registered(user_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if a user is registered with the given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. Whether a user with the given user_id is registered.\n    '
    user_settings = user_models.UserSettingsModel.get(user_id, strict=False)
    return bool(user_settings)

def has_ever_registered(user_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks if a user has ever been registered with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. Whether a user with the given user_id has ever been registered.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    return bool(user_settings.username and user_settings.last_agreed_to_terms)

def has_fully_registered_account(user_id: str) -> bool:
    if False:
        return 10
    'Checks if a user has fully registered.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. Whether a user with the given user_id has fully registered.\n    '
    user_settings = get_user_settings(user_id, strict=False)
    if user_settings is None:
        return False
    return bool(user_settings.username and user_settings.last_agreed_to_terms and (user_settings.last_agreed_to_terms >= feconf.TERMS_PAGE_LAST_UPDATED_UTC))

def get_all_profiles_auth_details_by_parent_user_id(parent_user_id: str) -> List[auth_domain.UserAuthDetails]:
    if False:
        return 10
    'Gets domain objects representing the auth details for all profiles\n    associated with the user having the given parent_user_id.\n\n    Args:\n        parent_user_id: str. User id of the parent_user whose associated\n            profiles we are querying for.\n\n    Returns:\n        list(UserAuthDetails). The UserAuthDetails domain objects corresponding\n        to the profiles linked to given parent_user_id. If that parent user does\n        not have any profiles linked to it, the returned list will be empty.\n\n    Raises:\n        Exception. Parent user with the given parent_user_id not found.\n    '
    if auth_models.UserAuthDetailsModel.has_reference_to_user_id(parent_user_id) is False:
        raise Exception('Parent user not found.')
    return [auth_services.get_user_auth_details_from_model(model) for model in auth_services.get_all_profiles_by_parent_user_id(parent_user_id) if not model.deleted]

def create_new_user(auth_id: str, email: str) -> user_domain.UserSettings:
    if False:
        return 10
    'Creates a new user and commits it to the datastore.\n\n    Args:\n        auth_id: str. The unique auth ID of the user.\n        email: str. The user email.\n\n    Returns:\n        UserSettings. The newly-created user settings domain object.\n\n    Raises:\n        Exception. A user with the given auth_id already exists.\n    '
    user_settings = get_user_settings_by_auth_id(auth_id, strict=False)
    if user_settings is not None:
        raise Exception('User %s already exists for auth_id %s.' % (user_settings.user_id, auth_id))
    user_id = user_models.UserSettingsModel.get_new_id('')
    user_settings = user_domain.UserSettings(user_id, email, [feconf.ROLE_ID_FULL_USER], False, False, preferred_language_codes=[constants.DEFAULT_LANGUAGE_CODE])
    _create_new_user_transactional(auth_id, user_settings)
    return user_settings

@transaction_services.run_in_transaction_wrapper
def _create_new_user_transactional(auth_id: str, user_settings: user_domain.UserSettings) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Save user models for new users as a transaction.\n\n    Args:\n        auth_id: str. The auth_id of the newly created user.\n        user_settings: UserSettings. The user settings domain object\n            corresponding to the newly created user.\n    '
    save_user_settings(user_settings)
    user_contributions = get_or_create_new_user_contributions(user_settings.user_id)
    save_user_contributions(user_contributions)
    auth_services.associate_auth_id_with_user_id(auth_domain.AuthIdUserIdPair(auth_id, user_settings.user_id))

def create_new_profiles(auth_id: str, email: str, modifiable_user_data_list: List[user_domain.ModifiableUserData]) -> List[user_domain.UserSettings]:
    if False:
        for i in range(10):
            print('nop')
    'Creates new profiles for the users specified in the\n    modifiable_user_data_list and commits them to the datastore.\n\n    Args:\n        auth_id: str. The auth ID of the full (parent) user trying to create new\n            profiles.\n        email: str. The email address of the full (parent) user trying to create\n            new profiles.\n        modifiable_user_data_list: list(ModifiableUserData). The list of\n            modifiable user data objects used for creation of new profiles.\n\n    Returns:\n        list(UserSettings). List of UserSettings objects created for the new\n        users.\n\n    Raises:\n        Exception. The pin for parent user trying to create a new profile\n            must be set.\n        Exception. The user_id is already set for any user in its corresponding\n            modifiable_user_data object.\n    '
    parent_user_settings = get_user_settings_by_auth_id(auth_id, strict=True)
    if parent_user_settings.pin is None:
        raise Exception('Pin must be set for a full user before creating a profile.')
    parent_user_id = parent_user_settings.user_id
    user_settings_list = []
    for modifiable_user_data in modifiable_user_data_list:
        if modifiable_user_data.user_id is not None:
            raise Exception('User id cannot already exist for a new user.')
        user_id = user_models.UserSettingsModel.get_new_id()
        user_settings = user_domain.UserSettings(user_id, email, [feconf.ROLE_ID_MOBILE_LEARNER], False, False, preferred_language_codes=[constants.DEFAULT_LANGUAGE_CODE], pin=modifiable_user_data.pin)
        user_settings.populate_from_modifiable_user_data(modifiable_user_data)
        user_auth_details = auth_services.create_profile_user_auth_details(user_id, parent_user_id)
        _create_new_profile_transactional(user_settings, user_auth_details)
        user_settings_list.append(user_settings)
    return user_settings_list

@transaction_services.run_in_transaction_wrapper
def _create_new_profile_transactional(user_settings: user_domain.UserSettings, user_auth_details: auth_domain.UserAuthDetails) -> None:
    if False:
        while True:
            i = 10
    'Save user models for new users as a transaction.\n\n    Args:\n        user_settings: UserSettings. The user settings domain object\n            corresponding to the newly created user.\n        user_auth_details: UserAuthDetails. The user auth details domain\n            object corresponding to the newly created list of users.\n    '
    save_user_settings(user_settings)
    _save_user_auth_details(user_auth_details)

def update_multiple_users_data(modifiable_user_data_list: List[user_domain.ModifiableUserData]) -> None:
    if False:
        return 10
    'Updates user settings and user auth model details for the users\n    specified in the modifiable_user_data_list.\n\n    Args:\n        modifiable_user_data_list: list(ModifiableUserData). The list of\n            modifiable_user_data entries corresponding to the users whose\n            data has to be updated.\n\n    Raises:\n        Exception. A user id is None.\n        Exception. UserSettings or UserAuthDetail for a given user_id is\n            not found.\n    '
    user_ids = [user.user_id for user in modifiable_user_data_list]
    user_settings_list_with_none = get_users_settings(user_ids, strict=False)
    user_settings_list = []
    user_auth_details_list = get_multiple_user_auth_details(user_ids)
    for (modifiable_user_data, user_settings) in zip(modifiable_user_data_list, user_settings_list_with_none):
        user_id = modifiable_user_data.user_id
        if user_id is None:
            raise Exception('Missing user ID.')
        if not user_settings:
            raise Exception('User not found.')
        user_settings.populate_from_modifiable_user_data(modifiable_user_data)
        user_settings_list.append(user_settings)
    _save_existing_users_settings(user_settings_list)
    _save_existing_users_auth_details(user_auth_details_list)

def _save_existing_users_settings(user_settings_list: List[user_domain.UserSettings]) -> None:
    if False:
        print('Hello World!')
    "Commits a list of existing users' UserSettings objects to the datastore.\n\n    Args:\n        user_settings_list: list(UserSettings). The list of UserSettings\n            objects to be saved.\n    "
    user_ids = [user.user_id for user in user_settings_list]
    user_settings_models_with_none = user_models.UserSettingsModel.get_multi(user_ids, include_deleted=True)
    user_settings_models = []
    for (user_model, user_settings) in zip(user_settings_models_with_none, user_settings_list):
        assert user_model is not None
        user_settings.validate()
        user_model.populate(**user_settings.to_dict())
        user_settings_models.append(user_model)
    user_models.UserSettingsModel.update_timestamps_multi(user_settings_models)
    user_models.UserSettingsModel.put_multi(user_settings_models)

def _save_existing_users_auth_details(user_auth_details_list: List[auth_domain.UserAuthDetails]) -> None:
    if False:
        print('Hello World!')
    "Commits a list of existing users' UserAuthDetails objects to the\n    datastore.\n\n    Args:\n        user_auth_details_list: list(UserAuthDetails). The list of\n            UserAuthDetails objects to be saved.\n    "
    user_ids = [user.user_id for user in user_auth_details_list]
    user_auth_models_with_none = auth_models.UserAuthDetailsModel.get_multi(user_ids, include_deleted=True)
    user_auth_models = []
    for (user_auth_details_model, user_auth_details) in zip(user_auth_models_with_none, user_auth_details_list):
        assert user_auth_details_model is not None
        user_auth_details.validate()
        user_auth_details_model.populate(**user_auth_details.to_dict())
        user_auth_models.append(user_auth_details_model)
    auth_models.UserAuthDetailsModel.update_timestamps_multi(user_auth_models)
    auth_models.UserAuthDetailsModel.put_multi(user_auth_models)

def _save_user_auth_details(user_auth_details: auth_domain.UserAuthDetails) -> None:
    if False:
        i = 10
        return i + 15
    'Commits a user auth details object to the datastore.\n\n    Args:\n        user_auth_details: UserAuthDetails. The user auth details domain object\n            to be saved.\n    '
    user_auth_details.validate()
    user_auth_details_model = auth_models.UserAuthDetailsModel.get_by_id(user_auth_details.user_id)
    user_auth_details_dict = user_auth_details.to_dict()
    if user_auth_details_model is not None:
        user_auth_details_model.populate(**user_auth_details_dict)
        user_auth_details_model.update_timestamps()
        user_auth_details_model.put()
    else:
        user_auth_details_dict['id'] = user_auth_details.user_id
        model = auth_models.UserAuthDetailsModel(**user_auth_details_dict)
        model.update_timestamps()
        model.put()

def get_multiple_user_auth_details(user_ids: List[Optional[str]]) -> List[auth_domain.UserAuthDetails]:
    if False:
        while True:
            i = 10
    'Gets domain objects representing the auth details\n    for the given user_ids.\n\n    Args:\n        user_ids: list(str). The list of user_ids for which we need to fetch\n            the user auth details.\n\n    Returns:\n        list(UserAuthDetails). The UserAuthDetails domain objects\n        corresponding to the given user ids.\n    '
    user_settings_models = auth_models.UserAuthDetailsModel.get_multi(user_ids)
    return [auth_services.get_user_auth_details_from_model(model) for model in user_settings_models if model is not None]

@overload
def get_auth_details_by_user_id(user_id: str, *, strict: Literal[True]) -> auth_domain.UserAuthDetails:
    if False:
        return 10
    ...

@overload
def get_auth_details_by_user_id(user_id: str) -> Optional[auth_domain.UserAuthDetails]:
    if False:
        print('Hello World!')
    ...

@overload
def get_auth_details_by_user_id(user_id: str, *, strict: Literal[False]) -> Optional[auth_domain.UserAuthDetails]:
    if False:
        for i in range(10):
            print('nop')
    ...

def get_auth_details_by_user_id(user_id: str, strict: bool=False) -> Optional[auth_domain.UserAuthDetails]:
    if False:
        for i in range(10):
            print('nop')
    'Return the user auth details for a single user.\n\n    Args:\n        user_id: str. The unique user ID of the user.\n        strict: bool. Whether to fail noisily if no user with the given\n            id exists in the datastore. Defaults to False.\n\n    Returns:\n        UserAuthDetails or None. If the given user_id does not exist and\n        strict is False, returns None. Otherwise, returns the corresponding\n        UserAuthDetails domain object.\n\n    Raises:\n        Exception. The value of strict is True and given user_id does not exist.\n    '
    user_auth_details_model = auth_models.UserAuthDetailsModel.get(user_id, strict=False)
    if user_auth_details_model is not None:
        return auth_services.get_user_auth_details_from_model(user_auth_details_model)
    elif strict:
        logging.error('Could not find user with id %s' % user_id)
        raise Exception('User not found.')
    else:
        return None

def get_pseudonymous_username(pseudonymous_id: str) -> str:
    if False:
        return 10
    "Get the username from pseudonymous ID.\n\n    Args:\n        pseudonymous_id: str. The pseudonymous ID from which to generate\n            the username.\n\n    Returns:\n        str. The pseudonymous username, starting with 'User' and ending with\n        the last eight letters from the pseudonymous_id.\n    "
    return 'User_%s%s' % (pseudonymous_id[-8].upper(), pseudonymous_id[-7:])

def get_username(user_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Gets username corresponding to the given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        str. Username corresponding to the given user_id.\n    '
    return get_usernames([user_id], strict=True)[0]

@overload
def get_usernames(user_ids: List[str], *, strict: Literal[True]) -> Sequence[str]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_usernames(user_ids: List[str]) -> Sequence[Optional[str]]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_usernames(user_ids: List[str], *, strict: Literal[False]) -> Sequence[Optional[str]]:
    if False:
        i = 10
        return i + 15
    ...

def get_usernames(user_ids: List[str], strict: bool=False) -> Sequence[Optional[str]]:
    if False:
        print('Hello World!')
    'Gets usernames corresponding to the given user_ids.\n\n    Args:\n        user_ids: list(str). The list of user_ids to get usernames for.\n        strict: bool. Whether to fail noisily if no user with the given ID\n            exists in the datastore. Defaults to False.\n\n    Returns:\n        list(str|None). Containing usernames based on given user_ids.\n        If a user_id does not exist, the corresponding entry in the\n        returned list is None. Can also return username of pseudonymized user\n        or a temporary username of user that is being deleted.\n    '
    usernames: List[Optional[str]] = [None] * len(user_ids)
    non_system_user_indices = []
    non_system_user_ids = []
    for (index, user_id) in enumerate(user_ids):
        if user_id in feconf.SYSTEM_USERS:
            usernames[index] = feconf.SYSTEM_USERS[user_id]
        elif utils.is_pseudonymous_id(user_id):
            usernames[index] = get_pseudonymous_username(user_id)
        else:
            non_system_user_indices.append(index)
            non_system_user_ids.append(user_id)
    non_system_users_settings = get_users_settings(non_system_user_ids, strict=strict, include_marked_deleted=True)
    for (index, user_settings) in enumerate(non_system_users_settings):
        if user_settings:
            usernames[non_system_user_indices[index]] = user_settings.username
    return usernames

def set_username(user_id: str, new_username: str) -> None:
    if False:
        return 10
    'Updates the username of the user with the given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        new_username: str. The new username to set.\n\n    Raises:\n        ValidationError. The new_username supplied is already taken.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_domain.UserSettings.require_valid_username(new_username)
    if is_username_taken(new_username):
        raise utils.ValidationError('Sorry, the username "%s" is already taken! Please pick a different one.' % new_username)
    user_settings.username = new_username
    save_user_settings(user_settings)

def record_agreement_to_terms(user_id: str) -> None:
    if False:
        return 10
    'Records that the user with given user_id has agreed to the license terms.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.last_agreed_to_terms = datetime.datetime.utcnow()
    save_user_settings(user_settings)

def update_profile_picture_data_url(user_id: str, profile_picture_data_url: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates profile_picture_data_url of user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        profile_picture_data_url: str. New profile picture url to be set.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    username = user_settings.username
    assert isinstance(username, str)
    fs = fs_services.GcsFileSystem(feconf.ENTITY_TYPE_USER, username)
    filename_png = 'profile_picture.png'
    png_binary = utils.convert_data_url_to_binary(profile_picture_data_url, 'png')
    fs.commit(filename_png, png_binary, mimetype='image/png')
    webp_binary = utils.convert_png_binary_to_webp_binary(png_binary)
    filename_webp = 'profile_picture.webp'
    fs.commit(filename_webp, webp_binary, mimetype='image/webp')

def update_user_bio(user_id: str, user_bio: str) -> None:
    if False:
        i = 10
        return i + 15
    'Updates user_bio of user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        user_bio: str. New user biography to be set.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.user_bio = user_bio
    save_user_settings(user_settings)

def update_user_default_dashboard(user_id: str, default_dashboard: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the default dashboard of user with given user id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        default_dashboard: str. The dashboard the user wants.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.default_dashboard = default_dashboard
    save_user_settings(user_settings)

def update_user_creator_dashboard_display(user_id: str, creator_dashboard_display_pref: str) -> None:
    if False:
        return 10
    'Updates the creator dashboard preference of user with given user id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        creator_dashboard_display_pref: str. The creator dashboard preference\n            the user wants.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.creator_dashboard_display_pref = creator_dashboard_display_pref
    save_user_settings(user_settings)

def update_subject_interests(user_id: str, subject_interests: List[str]) -> None:
    if False:
        print('Hello World!')
    'Updates subject_interests of user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        subject_interests: list(str). New subject interests to be set.\n    '
    if not isinstance(subject_interests, list):
        raise utils.ValidationError('Expected subject_interests to be a list.')
    for interest in subject_interests:
        if not isinstance(interest, str):
            raise utils.ValidationError('Expected each subject interest to be a string.')
        if not interest:
            raise utils.ValidationError('Expected each subject interest to be non-empty.')
        if not re.match(constants.TAG_REGEX, interest):
            raise utils.ValidationError('Expected each subject interest to consist only of lowercase alphabetic characters and spaces.')
    if len(set(subject_interests)) != len(subject_interests):
        raise utils.ValidationError('Expected each subject interest to be distinct.')
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.subject_interests = subject_interests
    save_user_settings(user_settings)

def update_preferred_language_codes(user_id: str, preferred_language_codes: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Updates preferred_language_codes of user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        preferred_language_codes: list(str). New exploration language\n            preferences to set.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.preferred_language_codes = preferred_language_codes
    save_user_settings(user_settings)

def update_preferred_site_language_code(user_id: str, preferred_site_language_code: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates preferred_site_language_code of user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        preferred_site_language_code: str. New system language preference\n            to set.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.preferred_site_language_code = preferred_site_language_code
    save_user_settings(user_settings)

def update_preferred_audio_language_code(user_id: str, preferred_audio_language_code: str) -> None:
    if False:
        print('Hello World!')
    'Updates preferred_audio_language_code of user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        preferred_audio_language_code: str. New audio language preference\n            to set.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.preferred_audio_language_code = preferred_audio_language_code
    save_user_settings(user_settings)

def update_preferred_translation_language_code(user_id: str, preferred_translation_language_code: str) -> None:
    if False:
        return 10
    'Updates preferred_translation_language_code of user with\n    given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        preferred_translation_language_code: str. New text translation\n            language preference to set.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.preferred_translation_language_code = preferred_translation_language_code
    save_user_settings(user_settings)

def add_user_role(user_id: str, role: str) -> None:
    if False:
        print('Hello World!')
    'Updates the roles of the user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user whose role is to be updated.\n        role: str. The role to be assigned to user with given id.\n\n    Raises:\n        Exception. The given role does not exist.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    if feconf.ROLE_ID_MOBILE_LEARNER in user_settings.roles:
        raise Exception('The role of a Mobile Learner cannot be changed.')
    if role in feconf.ALLOWED_DEFAULT_USER_ROLES_ON_REGISTRATION:
        raise Exception('Adding a %s role is not allowed.' % role)
    if role in user_settings.roles:
        raise Exception('The user already has this role.')
    user_settings.roles.append(role)
    role_services.log_role_query(user_id, feconf.ROLE_ACTION_ADD, role=role, username=user_settings.username)
    save_user_settings(user_settings)

def remove_user_role(user_id: str, role: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the roles of the user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user whose role is to be updated.\n        role: str. The role to be assigned to user with given id.\n\n    Raises:\n        Exception. The given role does not exist.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    if feconf.ROLE_ID_MOBILE_LEARNER in user_settings.roles:
        raise Exception('The role of a Mobile Learner cannot be changed.')
    if role in feconf.ALLOWED_DEFAULT_USER_ROLES_ON_REGISTRATION:
        raise Exception('Removing a default role is not allowed.')
    user_settings.roles.remove(role)
    role_services.log_role_query(user_id, feconf.ROLE_ACTION_REMOVE, role=role, username=user_settings.username)
    save_user_settings(user_settings)

def mark_user_for_deletion(user_id: str) -> None:
    if False:
        return 10
    "Set the 'deleted' property of the user with given user_id to True.\n\n    Args:\n        user_id: str. The unique ID of the user who should be deleted.\n    "
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.deleted = True
    save_user_settings(user_settings)
    user_auth_details = auth_services.get_user_auth_details_from_model(auth_models.UserAuthDetailsModel.get(user_id))
    user_auth_details.deleted = True
    _save_user_auth_details(user_auth_details)
    auth_services.mark_user_for_deletion(user_id)

def save_deleted_username(normalized_username: str) -> None:
    if False:
        i = 10
        return i + 15
    'Save the username of deleted user.\n\n    Args:\n        normalized_username: str. Normalized version of the username to be\n            saved.\n    '
    hashed_normalized_username = utils.convert_to_hash(normalized_username, user_models.DeletedUsernameModel.ID_LENGTH)
    deleted_user_model = user_models.DeletedUsernameModel(id=hashed_normalized_username)
    deleted_user_model.update_timestamps()
    deleted_user_model.put()

def get_human_readable_user_ids(user_ids: List[str], strict: bool=True) -> List[str]:
    if False:
        print('Hello World!')
    "Converts the given ids to usernames, or truncated email addresses.\n    Requires all users to be known.\n\n    Args:\n        user_ids: list(str). The list of user_ids to get UserSettings domain\n            objects for.\n        strict: bool. Whether to fail noisily if no user with the given\n            id exists in the datastore. Defaults to True.\n\n    Returns:\n        list(str). List of usernames corresponding to given user_ids. If\n        username does not exist, the corresponding entry in the returned\n        list is the user's truncated email address. If the user is scheduled to\n        be deleted USER_IDENTIFICATION_FOR_USER_BEING_DELETED is returned.\n\n    Raises:\n        Exception. At least one of the user_ids does not correspond to a valid\n            UserSettingsModel.\n    "
    users_settings = get_users_settings(user_ids, include_marked_deleted=True)
    usernames = []
    for (ind, user_settings) in enumerate(users_settings):
        if user_settings is None:
            if strict:
                logging.error('User id %s not known in list of user_ids %s' % (user_ids[ind], user_ids))
                raise Exception('User not found.')
        elif user_settings.deleted:
            usernames.append(LABEL_FOR_USER_BEING_DELETED)
        elif user_settings.username:
            usernames.append(user_settings.username)
        else:
            usernames.append('[Awaiting user registration: %s]' % user_settings.truncated_email)
    return usernames

def record_user_started_state_editor_tutorial(user_id: str) -> None:
    if False:
        return 10
    'Updates last_started_state_editor_tutorial to the current datetime\n    for the user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.last_started_state_editor_tutorial = datetime.datetime.utcnow()
    save_user_settings(user_settings)

def record_user_started_state_translation_tutorial(user_id: str) -> None:
    if False:
        while True:
            i = 10
    'Updates last_started_state_translation_tutorial to the current datetime\n    for the user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.last_started_state_translation_tutorial = datetime.datetime.utcnow()
    save_user_settings(user_settings)

def record_user_logged_in(user_id: str) -> None:
    if False:
        while True:
            i = 10
    'Updates last_logged_in to the current datetime for the user with\n    given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    user_settings.last_logged_in = datetime.datetime.utcnow()
    save_user_settings(user_settings)

def record_user_created_an_exploration(user_id: str) -> None:
    if False:
        print('Hello World!')
    'Updates last_created_an_exploration to the current datetime for\n    the user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    '
    user_settings = get_user_settings(user_id, strict=False)
    if user_settings is not None:
        user_settings.last_created_an_exploration = datetime.datetime.utcnow()
        save_user_settings(user_settings)

def add_user_to_mailing_list(email: str, name: str, tag: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Adds user to the bulk email provider with the relevant tag and required\n    merge fields.\n\n    Args:\n        email: str. Email of the user.\n        name: str. Name of the user.\n        tag: str. Tag for the mailing list.\n\n    Returns:\n        bool. Whether the operation was successful or not.\n    '
    merge_fields = {'NAME': name}
    return bulk_email_services.add_or_update_user_status(email, merge_fields, tag, can_receive_email_updates=True)

def update_email_preferences(user_id: str, can_receive_email_updates: bool, can_receive_editor_role_email: bool, can_receive_feedback_email: bool, can_receive_subscription_email: bool, bulk_email_db_already_updated: bool=False) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Updates whether the user has chosen to receive email updates.\n\n    If no UserEmailPreferencesModel exists for this user, a new one will\n    be created.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        can_receive_email_updates: bool. Whether the given user can receive\n            email updates.\n        can_receive_editor_role_email: bool. Whether the given user can receive\n            emails notifying them of role changes.\n        can_receive_feedback_email: bool. Whether the given user can receive\n            emails when users submit feedback to their explorations.\n        can_receive_subscription_email: bool. Whether the given user can receive\n            emails related to his/her creator subscriptions.\n        bulk_email_db_already_updated: bool. Whether the bulk email provider's\n            database is already updated. This is set to true only when calling\n            from the webhook controller since in that case, the external update\n            to the bulk email provider's database initiated the update here.\n\n    Returns:\n        bool. Whether to send a mail to the user to complete bulk email service\n        signup.\n    "
    email_preferences_model = user_models.UserEmailPreferencesModel.get(user_id, strict=False)
    if email_preferences_model is None:
        email_preferences_model = user_models.UserEmailPreferencesModel(id=user_id)
    email_preferences_model.editor_role_notifications = can_receive_editor_role_email
    email_preferences_model.feedback_message_notifications = can_receive_feedback_email
    email_preferences_model.subscription_notifications = can_receive_subscription_email
    email = get_email_from_user_id(user_id)
    if not bulk_email_db_already_updated and feconf.CAN_SEND_EMAILS:
        user_creation_successful = bulk_email_services.add_or_update_user_status(email, {}, 'Account', can_receive_email_updates=can_receive_email_updates)
        if not user_creation_successful:
            email_preferences_model.site_updates = False
            email_preferences_model.update_timestamps()
            email_preferences_model.put()
            return True
    email_preferences_model.site_updates = can_receive_email_updates
    email_preferences_model.update_timestamps()
    email_preferences_model.put()
    return False

def get_email_preferences(user_id: str) -> user_domain.UserGlobalPrefs:
    if False:
        return 10
    'Gives email preferences of user with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        UserGlobalPrefs. Representing whether the user has chosen to receive\n        email updates.\n    '
    email_preferences_model = user_models.UserEmailPreferencesModel.get(user_id, strict=False)
    if email_preferences_model is None:
        return user_domain.UserGlobalPrefs.create_default_prefs()
    else:
        return user_domain.UserGlobalPrefs(email_preferences_model.site_updates, email_preferences_model.editor_role_notifications, email_preferences_model.feedback_message_notifications, email_preferences_model.subscription_notifications)

def get_users_email_preferences(user_ids: List[str]) -> List[user_domain.UserGlobalPrefs]:
    if False:
        i = 10
        return i + 15
    'Get email preferences for the list of users.\n\n    Args:\n        user_ids: list(str). A list of user IDs for whom we want to get email\n            preferences.\n\n    Returns:\n        list(UserGlobalPrefs). Representing whether the users had chosen to\n        receive email updates.\n    '
    user_email_preferences_models = user_models.UserEmailPreferencesModel.get_multi(user_ids)
    result = []
    for email_preferences_model in user_email_preferences_models:
        if email_preferences_model is None:
            result.append(user_domain.UserGlobalPrefs.create_default_prefs())
        else:
            result.append(user_domain.UserGlobalPrefs(email_preferences_model.site_updates, email_preferences_model.editor_role_notifications, email_preferences_model.feedback_message_notifications, email_preferences_model.subscription_notifications))
    return result

def set_email_preferences_for_exploration(user_id: str, exploration_id: str, mute_feedback_notifications: Optional[bool]=None, mute_suggestion_notifications: Optional[bool]=None) -> None:
    if False:
        while True:
            i = 10
    'Sets mute preferences for exploration with given exploration_id of user\n    with given user_id.\n\n    If no ExplorationUserDataModel exists for this user and exploration,\n    a new one will be created.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        exploration_id: str. The exploration id.\n        mute_feedback_notifications: bool. Whether the given user has muted\n            feedback emails. Defaults to None.\n        mute_suggestion_notifications: bool. Whether the given user has muted\n            suggestion emails. Defaults to None.\n    '
    exploration_user_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
    if exploration_user_model is None:
        exploration_user_model = user_models.ExplorationUserDataModel.create(user_id, exploration_id)
    if mute_feedback_notifications is not None:
        exploration_user_model.mute_feedback_notifications = mute_feedback_notifications
    if mute_suggestion_notifications is not None:
        exploration_user_model.mute_suggestion_notifications = mute_suggestion_notifications
    exploration_user_model.update_timestamps()
    exploration_user_model.put()

def get_email_preferences_for_exploration(user_id: str, exploration_id: str) -> user_domain.UserExplorationPrefs:
    if False:
        while True:
            i = 10
    'Gives mute preferences for exploration with given exploration_id of user\n    with given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        exploration_id: str. The exploration id.\n\n    Returns:\n        UserExplorationPrefs. Representing whether the user has chosen to\n        receive email updates for particular exploration.\n    '
    exploration_user_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
    if exploration_user_model is None:
        return user_domain.UserExplorationPrefs.create_default_prefs()
    else:
        return user_domain.UserExplorationPrefs(exploration_user_model.mute_feedback_notifications, exploration_user_model.mute_suggestion_notifications)

def get_users_email_preferences_for_exploration(user_ids: List[str], exploration_id: str) -> List[user_domain.UserExplorationPrefs]:
    if False:
        for i in range(10):
            print('nop')
    'Gives mute preferences for exploration with given exploration_id of user\n    with given user_id.\n\n    Args:\n        user_ids: list(str). A list of user IDs for whom we want to get email\n            preferences.\n        exploration_id: str. The exploration id.\n\n    Returns:\n        list(UserExplorationPrefs). Representing whether the users has chosen to\n        receive email updates for particular exploration.\n    '
    user_id_exp_id_combinations = list(itertools.product(user_ids, [exploration_id]))
    exploration_user_models = user_models.ExplorationUserDataModel.get_multi(user_id_exp_id_combinations)
    result = []
    for exploration_user_model in exploration_user_models:
        if exploration_user_model is None:
            result.append(user_domain.UserExplorationPrefs.create_default_prefs())
        else:
            result.append(user_domain.UserExplorationPrefs(exploration_user_model.mute_feedback_notifications, exploration_user_model.mute_suggestion_notifications))
    return result

@overload
def get_user_contributions(user_id: str, *, strict: Literal[True]) -> user_domain.UserContributions:
    if False:
        print('Hello World!')
    ...

@overload
def get_user_contributions(user_id: str) -> Optional[user_domain.UserContributions]:
    if False:
        return 10
    ...

@overload
def get_user_contributions(user_id: str, *, strict: Literal[False]) -> Optional[user_domain.UserContributions]:
    if False:
        for i in range(10):
            print('nop')
    ...

def get_user_contributions(user_id: str, strict: bool=False) -> Optional[user_domain.UserContributions]:
    if False:
        i = 10
        return i + 15
    'Gets domain object representing the contributions for the given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        strict: bool. Whether to fail noisily if no user with the given\n            id exists in the datastore. Defaults to False.\n\n    Returns:\n        UserContributions or None. If the given user_id does not exist, return\n        None. Otherwise, return the corresponding UserContributions domain\n        object.\n    '
    model = user_models.UserContributionsModel.get(user_id, strict=strict)
    if model is None:
        return None
    result = user_domain.UserContributions(model.id, model.created_exploration_ids, model.edited_exploration_ids)
    return result

def get_or_create_new_user_contributions(user_id: str) -> user_domain.UserContributions:
    if False:
        print('Hello World!')
    'Gets domain object representing the contributions for the given user_id.\n    If the domain object does not exist, it is created.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        UserContributions. The UserContributions domain object corresponding to\n        the given user_id.\n    '
    user_contributions = get_user_contributions(user_id, strict=False)
    if user_contributions is None:
        user_contributions = user_domain.UserContributions(user_id, [], [])
    return user_contributions

def save_user_contributions(user_contributions: user_domain.UserContributions) -> None:
    if False:
        i = 10
        return i + 15
    'Saves a user contributions object to the datastore.\n\n    Args:\n        user_contributions: UserContributions. The user contributions object to\n            be saved.\n    '
    user_contributions_model = get_validated_user_contributions_model(user_contributions)
    user_contributions_model.update_timestamps()
    user_contributions_model.put()

def update_user_contributions(user_id: str, created_exploration_ids: List[str], edited_exploration_ids: List[str]) -> None:
    if False:
        print('Hello World!')
    'Updates an existing UserContributionsModel with new calculated\n    contributions.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        created_exploration_ids: list(str). IDs of explorations that this\n            user has created.\n        edited_exploration_ids: list(str). IDs of explorations that this\n            user has edited.\n\n    Raises:\n        Exception. The UserContributionsModel for the given user_id does not\n            exist.\n    '
    user_contributions = get_user_contributions(user_id, strict=False)
    if not user_contributions:
        raise Exception('User contributions model for user %s does not exist.' % user_id)
    user_contributions.created_exploration_ids = created_exploration_ids
    user_contributions.edited_exploration_ids = edited_exploration_ids
    get_validated_user_contributions_model(user_contributions).put()

def get_validated_user_contributions_model(user_contributions: user_domain.UserContributions) -> user_models.UserContributionsModel:
    if False:
        print('Hello World!')
    "Constructs a valid UserContributionsModel from the given domain object.\n\n    This function does not save anything to the datastore.\n\n    Args:\n        user_contributions: UserContributions. Value object representing\n            a user's contributions.\n\n    Returns:\n        UserContributionsModel. The UserContributionsModel object that was\n        updated.\n    "
    user_contributions.validate()
    return user_models.UserContributionsModel(id=user_contributions.user_id, created_exploration_ids=user_contributions.created_exploration_ids, edited_exploration_ids=user_contributions.edited_exploration_ids)

def migrate_dashboard_stats_to_latest_schema(versioned_dashboard_stats: user_models.UserStatsModel) -> None:
    if False:
        while True:
            i = 10
    'Holds responsibility of updating the structure of dashboard stats.\n\n    Args:\n        versioned_dashboard_stats: UserStatsModel. Value object representing\n            user-specific statistics.\n\n    Raises:\n        Exception. If schema_version > CURRENT_DASHBOARD_STATS_SCHEMA_VERSION.\n    '
    stats_schema_version = versioned_dashboard_stats.schema_version
    if not 1 <= stats_schema_version <= feconf.CURRENT_DASHBOARD_STATS_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d dashboard stats schemas at present.' % feconf.CURRENT_DASHBOARD_STATS_SCHEMA_VERSION)

def get_current_date_as_string() -> str:
    if False:
        return 10
    "Gets the current date.\n\n    Returns:\n        str. Current date as a string of format 'YYYY-MM-DD'.\n    "
    return datetime.datetime.utcnow().strftime(feconf.DASHBOARD_STATS_DATETIME_STRING_FORMAT)

def parse_date_from_string(datetime_str: str) -> Dict[str, int]:
    if False:
        print('Hello World!')
    'Parses the given string, and returns the year, month and day of the\n    date that it represents.\n\n    Args:\n        datetime_str: str. String representing datetime.\n\n    Returns:\n        dict. Representing date with year, month and day as keys.\n    '
    datetime_obj = datetime.datetime.strptime(datetime_str, feconf.DASHBOARD_STATS_DATETIME_STRING_FORMAT)
    return {'year': datetime_obj.year, 'month': datetime_obj.month, 'day': datetime_obj.day}

def get_user_impact_score(user_id: str) -> float:
    if False:
        for i in range(10):
            print('nop')
    'Gets the user impact score for the given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        float. The user impact score associated with the given user_id.\n        Returns 0 if UserStatsModel does not exist for the given user_id.\n    '
    model = user_models.UserStatsModel.get(user_id, strict=False)
    if model:
        impact_score: float = model.impact_score
        return impact_score
    else:
        return 0

def get_weekly_dashboard_stats(user_id: str) -> List[Dict[str, DashboardStatsDict]]:
    if False:
        return 10
    "Gets weekly dashboard stats for a given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        list(dict): The weekly dashboard stats for the given user. Each dict in\n        the list denotes the dashboard stats of the user, keyed by a datetime\n        string. The stats currently being saved are:\n            - 'average ratings': Average of ratings across all explorations of\n                a user.\n            - 'total plays': Total number of plays across all explorations of\n                a user.\n\n        The format of returned value:\n        [\n            {\n                {{datetime_string_1}}: {\n                    'num_ratings': (value),\n                    'average_ratings': (value),\n                    'total_plays': (value)\n                }\n            },\n            {\n                {{datetime_string_2}}: {\n                    'num_ratings': (value),\n                    'average_ratings': (value),\n                    'total_plays': (value)\n                }\n            }\n        ]\n        If the user doesn't exist, then this function returns None.\n    "
    model = user_models.UserStatsModel.get(user_id, strict=False)
    if model and model.weekly_creator_stats_list:
        weekly_creator_stats_list: List[Dict[str, DashboardStatsDict]] = model.weekly_creator_stats_list
        return weekly_creator_stats_list
    else:
        return []

def get_last_week_dashboard_stats(user_id: str) -> Optional[Dict[str, DashboardStatsDict]]:
    if False:
        print('Hello World!')
    "Gets last week's dashboard stats for a given user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        dict or None: The dict denotes last week dashboard stats of the user,\n        and contains a single key-value pair. The key is the datetime string and\n        the value is the dashboard stats in the format:\n        {\n            'num_ratings': (value),\n            'average_ratings': (value),\n            'total_plays': (value)\n        }\n        If the user doesn't exist, then this function returns None.\n    "
    weekly_dashboard_stats = get_weekly_dashboard_stats(user_id)
    if weekly_dashboard_stats:
        return weekly_dashboard_stats[-1]
    else:
        return None

def update_dashboard_stats_log(user_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Save statistics for creator dashboard of a user by appending to a list\n    keyed by a datetime string.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    '
    model = user_models.UserStatsModel.get_or_create(user_id)
    if model.schema_version != feconf.CURRENT_DASHBOARD_STATS_SCHEMA_VERSION:
        migrate_dashboard_stats_to_latest_schema(model)
    weekly_dashboard_stats = {get_current_date_as_string(): {'num_ratings': model.num_ratings or 0, 'average_ratings': model.average_ratings, 'total_plays': model.total_plays or 0}}
    model.weekly_creator_stats_list.append(weekly_dashboard_stats)
    model.update_timestamps()
    model.put()

def is_moderator(user_id: str) -> bool:
    if False:
        print('Hello World!')
    'Checks if a user with given user_id is a moderator.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. True if user is a moderator, False otherwise.\n    '
    return feconf.ROLE_ID_MODERATOR in get_user_roles_from_id(user_id)

def is_curriculum_admin(user_id: str) -> bool:
    if False:
        return 10
    'Checks if a user with given user_id is an admin.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. True if user is an admin, False otherwise.\n    '
    return feconf.ROLE_ID_CURRICULUM_ADMIN in get_user_roles_from_id(user_id)

def is_topic_manager(user_id: str) -> bool:
    if False:
        while True:
            i = 10
    'Checks if a user with given user_id is a topic manager.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. Whether the user is a topic manager.\n    '
    return feconf.ROLE_ID_TOPIC_MANAGER in get_user_roles_from_id(user_id)

def can_review_translation_suggestions(user_id: str, language_code: Optional[str]=None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Returns whether the user can review translation suggestions in any\n    language or in the given language.\n\n    NOTE: If the language_code is provided then this method will check whether\n    the user can review translations in the given language code. Otherwise, it\n    will check whether the user can review in any language.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        language_code: str. The code of the language.\n\n    Returns:\n        bool. Whether the user can review translation suggestions in any\n        language or in the given language.\n    '
    user_contribution_rights = get_user_contribution_rights(user_id)
    reviewable_language_codes = user_contribution_rights.can_review_translation_for_language_codes
    if language_code is not None:
        return language_code in reviewable_language_codes
    else:
        return bool(reviewable_language_codes)

def can_review_question_suggestions(user_id: str) -> bool:
    if False:
        print('Hello World!')
    'Checks whether the user can review question suggestions.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. Whether the user can review question suggestions.\n    '
    user_contribution_rights = get_user_contribution_rights(user_id)
    return user_contribution_rights.can_review_questions

def can_submit_question_suggestions(user_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks whether the user can submit question suggestions.\n\n    Args:\n        user_id: str. The unique ID of the user.\n\n    Returns:\n        bool. Whether the user can submit question suggestions.\n    '
    user_contribution_rights = get_user_contribution_rights(user_id)
    return user_contribution_rights.can_submit_questions

def allow_user_to_review_translation_in_language(user_id: str, language_code: str) -> None:
    if False:
        while True:
            i = 10
    'Allows the user with the given user id to review translation in the given\n    language_code.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        language_code: str. The code of the language. Callers should ensure that\n            the user does not have rights to review translations in the given\n            language code.\n    '
    user_contribution_rights = get_user_contribution_rights(user_id)
    allowed_language_codes = set(user_contribution_rights.can_review_translation_for_language_codes)
    if language_code is not None:
        allowed_language_codes.add(language_code)
    user_contribution_rights.can_review_translation_for_language_codes = sorted(list(allowed_language_codes))
    _save_user_contribution_rights(user_contribution_rights)

def remove_translation_review_rights_in_language(user_id: str, language_code_to_remove: str) -> None:
    if False:
        i = 10
        return i + 15
    "Removes the user's review rights to translation suggestions in the given\n    language_code.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        language_code_to_remove: str. The code of the language. Callers should\n            ensure that the user already has rights to review translations in\n            the given language code.\n    "
    user_contribution_rights = get_user_contribution_rights(user_id)
    user_contribution_rights.can_review_translation_for_language_codes = [lang_code for lang_code in user_contribution_rights.can_review_translation_for_language_codes if lang_code != language_code_to_remove]
    _update_user_contribution_rights(user_contribution_rights)

def allow_user_to_review_voiceover_in_language(user_id: str, language_code: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Allows the user with the given user id to review voiceover applications\n    in the given language_code.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        language_code: str. The code of the language. Callers should ensure that\n            the user does not have rights to review voiceovers in the given\n            language code.\n    '
    user_contribution_rights = get_user_contribution_rights(user_id)
    allowed_language_codes = set(user_contribution_rights.can_review_voiceover_for_language_codes)
    allowed_language_codes.add(language_code)
    user_contribution_rights.can_review_voiceover_for_language_codes = sorted(list(allowed_language_codes))
    _save_user_contribution_rights(user_contribution_rights)

def remove_voiceover_review_rights_in_language(user_id: str, language_code: str) -> None:
    if False:
        return 10
    "Removes the user's review rights to voiceover applications in the given\n    language_code.\n\n    Args:\n        user_id: str. The unique ID of the user.\n        language_code: str. The code of the language. Callers should ensure that\n            the user already has rights to review voiceovers in the given\n            language code.\n    "
    user_contribution_rights = get_user_contribution_rights(user_id)
    user_contribution_rights.can_review_voiceover_for_language_codes.remove(language_code)
    _update_user_contribution_rights(user_contribution_rights)

def allow_user_to_review_question(user_id: str) -> None:
    if False:
        return 10
    'Allows the user with the given user id to review question suggestions.\n\n    Args:\n        user_id: str. The unique ID of the user. Callers should ensure that\n            the given user does not have rights to review questions.\n    '
    user_contribution_rights = get_user_contribution_rights(user_id)
    user_contribution_rights.can_review_questions = True
    _save_user_contribution_rights(user_contribution_rights)

def remove_question_review_rights(user_id: str) -> None:
    if False:
        while True:
            i = 10
    "Removes the user's review rights to question suggestions.\n\n    Args:\n        user_id: str. The unique ID of the user. Callers should ensure that\n            the given user already has rights to review questions.\n    "
    user_contribution_rights = get_user_contribution_rights(user_id)
    user_contribution_rights.can_review_questions = False
    _update_user_contribution_rights(user_contribution_rights)

def allow_user_to_submit_question(user_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Allows the user with the given user id to submit question suggestions.\n\n    Args:\n        user_id: str. The unique ID of the user. Callers should ensure that\n            the given user does not have rights to submit questions.\n    '
    user_contribution_rights = get_user_contribution_rights(user_id)
    user_contribution_rights.can_submit_questions = True
    _save_user_contribution_rights(user_contribution_rights)

def remove_question_submit_rights(user_id: str) -> None:
    if False:
        i = 10
        return i + 15
    "Removes the user's submit rights to question suggestions.\n\n    Args:\n        user_id: str. The unique ID of the user. Callers should ensure that\n            the given user already has rights to submit questions.\n    "
    user_contribution_rights = get_user_contribution_rights(user_id)
    user_contribution_rights.can_submit_questions = False
    _update_user_contribution_rights(user_contribution_rights)

def remove_contribution_reviewer(user_id: str) -> None:
    if False:
        while True:
            i = 10
    'Deletes the UserContributionRightsModel corresponding to the given\n    user_id.\n\n    Args:\n        user_id: str. The unique ID of the user.\n    '
    user_contribution_rights_model = user_models.UserContributionRightsModel.get_by_id(user_id)
    if user_contribution_rights_model is not None:
        user_contribution_rights = _create_user_contribution_rights_from_model(user_contribution_rights_model)
        user_contribution_rights.can_review_questions = False
        user_contribution_rights.can_review_translation_for_language_codes = []
        _update_reviewer_counts_in_community_contribution_stats(user_contribution_rights)
        user_contribution_rights_model.delete()

def get_contributor_usernames(category: str, language_code: Optional[str]=None) -> Sequence[str]:
    if False:
        i = 10
        return i + 15
    "Returns a list of usernames of users who has contribution rights of given\n    category.\n\n    Args:\n        category: str. The review category to find the list of reviewers\n            for.\n        language_code: None|str. The language code for translation or voiceover\n            review category.\n\n    Returns:\n        Sequence(str). A list of usernames.\n\n    Raises:\n        Exception. The language code is not of None for question review\n            contribution.\n        Exception. Invalid category.\n        Exception. The language_code cannot be None if review category is\n            'translation' or 'voiceover'.\n    "
    user_ids = []
    if category in (constants.CONTRIBUTION_RIGHT_CATEGORY_REVIEW_TRANSLATION, constants.CONTRIBUTION_RIGHT_CATEGORY_REVIEW_VOICEOVER) and language_code is None:
        raise Exception("The language_code cannot be None if review category is 'translation' or 'voiceover'.")
    if category == constants.CONTRIBUTION_RIGHT_CATEGORY_REVIEW_TRANSLATION:
        assert language_code is not None
        user_ids = user_models.UserContributionRightsModel.get_translation_reviewer_user_ids(language_code)
    elif category == constants.CONTRIBUTION_RIGHT_CATEGORY_REVIEW_VOICEOVER:
        assert language_code is not None
        user_ids = user_models.UserContributionRightsModel.get_voiceover_reviewer_user_ids(language_code)
    elif category == constants.CONTRIBUTION_RIGHT_CATEGORY_REVIEW_QUESTION:
        if language_code is not None:
            raise Exception('Expected language_code to be None, found: %s' % language_code)
        user_ids = user_models.UserContributionRightsModel.get_question_reviewer_user_ids()
    elif category == constants.CONTRIBUTION_RIGHT_CATEGORY_SUBMIT_QUESTION:
        user_ids = user_models.UserContributionRightsModel.get_question_submitter_user_ids()
    else:
        raise Exception('Invalid category: %s' % category)
    usernames = get_usernames(user_ids, strict=True)
    return usernames

def log_username_change(committer_id: str, old_username: str, new_username: str) -> None:
    if False:
        print('Hello World!')
    'Stores the query to role structure in UsernameChangeAuditModel.\n\n    Args:\n        committer_id: str. The ID of the user that is making the change.\n        old_username: str. The current username that is being changed.\n        new_username: str. The new username that the current one is being\n            changed to.\n    '
    model_id = '%s.%d' % (committer_id, utils.get_current_time_in_millisecs())
    audit_models.UsernameChangeAuditModel(id=model_id, committer_id=committer_id, old_username=old_username, new_username=new_username).put()

def create_login_url(return_url: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Creates a login url.\n\n    Args:\n        return_url: str. The URL to redirect to after login.\n\n    Returns:\n        str. The correct login URL that includes the page to redirect to.\n    '
    return '/login?%s' % urllib.parse.urlencode({'return_url': return_url})

def mark_user_banned(user_id: str) -> None:
    if False:
        return 10
    'Marks a user banned.\n\n    Args:\n        user_id: str. The Id of the user.\n    '
    user_settings = get_user_settings(user_id)
    user_settings.mark_banned()
    save_user_settings(user_settings)

def unmark_user_banned(user_id: str) -> None:
    if False:
        return 10
    'Unmarks a banned user.\n\n    Args:\n        user_id: str. The Id of the user.\n    '
    user_auth_details = auth_services.get_user_auth_details_from_model(auth_models.UserAuthDetailsModel.get(user_id))
    user_settings = get_user_settings(user_id)
    user_settings.unmark_banned(feconf.ROLE_ID_FULL_USER if user_auth_details.is_full_user() else feconf.ROLE_ID_MOBILE_LEARNER)
    save_user_settings(user_settings)

def get_dashboard_stats(user_id: str) -> DashboardStatsDict:
    if False:
        print('Hello World!')
    "Returns the dashboard stats associated with the given user_id.\n\n    Args:\n        user_id: str. The id of the user.\n\n    Returns:\n        dict. Has the keys:\n            total_plays: int. Number of times the user's explorations were\n                played.\n            num_ratings: int. Number of times the explorations have been\n                rated.\n            average_ratings: float. Average of average ratings across all\n                explorations.\n    "
    user_stats_model = user_models.UserStatsModel.get(user_id, strict=False)
    if user_stats_model is None:
        total_plays = 0
        num_ratings = 0
        average_ratings = None
    else:
        total_plays = user_stats_model.total_plays
        num_ratings = user_stats_model.num_ratings
        average_ratings = user_stats_model.average_ratings
    return {'total_plays': total_plays, 'num_ratings': num_ratings, 'average_ratings': average_ratings}

def get_checkpoints_in_order(init_state_name: str, states: Dict[str, state_domain.State]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the checkpoints of an exploration in sequential order by a\n    BFS traversal.\n\n    Args:\n        init_state_name: str. The name of the first state of the exploration.\n        states: dict(state). All states of the exploration.\n\n    Returns:\n        list(str). List of all checkpoints of the exploration in sequential\n        order.\n\n    Raises:\n        Exception. States with a null destination can never be a checkpoint.\n    '
    queue = [init_state_name]
    checkpoint_state_names = []
    visited_state_names = []
    while len(queue) > 0:
        current_state_name = queue.pop()
        if current_state_name not in visited_state_names:
            visited_state_names.append(current_state_name)
            current_state = states[current_state_name]
            if current_state.card_is_checkpoint and current_state_name not in checkpoint_state_names:
                checkpoint_state_names.append(current_state_name)
            for answer_group in current_state.interaction.answer_groups:
                if answer_group.outcome.dest is None:
                    raise Exception('States with a null destination can never be a checkpoint.')
                queue.append(answer_group.outcome.dest)
            if current_state.interaction.default_outcome is not None:
                if current_state.interaction.default_outcome.dest is None:
                    raise Exception('States with a null destination can never be a checkpoint.')
                queue.append(current_state.interaction.default_outcome.dest)
    return checkpoint_state_names

def get_most_distant_reached_checkpoint_in_current_exploration(checkpoints_in_current_exploration: List[str], checkpoints_in_older_exploration: List[str], most_distant_reached_checkpoint_state_name_in_older_exploration: str) -> Optional[str]:
    if False:
        while True:
            i = 10
    'Returns the most distant reached checkpoint in current exploration after\n    comparing current exploration with older exploration.\n\n    Args:\n        checkpoints_in_current_exploration: list(str). The checkpoints of\n            current exploration in sequential order.\n        checkpoints_in_older_exploration: list(str). The checkpoints\n            of older exploration in sequential order.\n        most_distant_reached_checkpoint_state_name_in_older_exploration: str.\n            The state name of the most distant reached checkpoint in the older\n            exploration.\n\n    Returns:\n        str or None. The most distant checkpoint in current exploration or\n        None if most distant reached checkpoint of older exploration is not\n        present in current exploration.\n    '
    mdrc_index = checkpoints_in_older_exploration.index(most_distant_reached_checkpoint_state_name_in_older_exploration)
    while mdrc_index >= 0:
        checkpoint_in_old_exp = checkpoints_in_older_exploration[mdrc_index]
        if checkpoint_in_old_exp in checkpoints_in_current_exploration:
            return checkpoint_in_old_exp
        mdrc_index -= 1
    return None

def update_learner_checkpoint_progress(user_id: str, exploration_id: str, state_name: str, exp_version: int) -> None:
    if False:
        return 10
    'Sets the furthest reached and most recently reached checkpoint in\n    an exploration by the user.\n\n    Args:\n        user_id: str. The Id of the user.\n        exploration_id: str. The Id of the exploration.\n        state_name: str. The state name of the most recently reached checkpoint.\n        exp_version: int. The exploration version of the most recently reached\n            checkpoint.\n    '
    exp_user_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
    if exp_user_model is None:
        exp_user_model = user_models.ExplorationUserDataModel.create(user_id, exploration_id)
    current_exploration = exp_fetchers.get_exploration_by_id(exploration_id, strict=True, version=exp_version)
    if exp_user_model.furthest_reached_checkpoint_state_name is None:
        exp_user_model.furthest_reached_checkpoint_exp_version = exp_version
        exp_user_model.furthest_reached_checkpoint_state_name = state_name
    elif exp_user_model.furthest_reached_checkpoint_exp_version < exp_version:
        furthest_reached_checkpoint_exp = exp_fetchers.get_exploration_by_id(exploration_id, strict=True, version=exp_user_model.furthest_reached_checkpoint_exp_version)
        checkpoints_in_current_exp = get_checkpoints_in_order(current_exploration.init_state_name, current_exploration.states)
        checkpoints_in_older_exp = get_checkpoints_in_order(furthest_reached_checkpoint_exp.init_state_name, furthest_reached_checkpoint_exp.states)
        furthest_reached_checkpoint_in_current_exp = get_most_distant_reached_checkpoint_in_current_exploration(checkpoints_in_current_exp, checkpoints_in_older_exp, exp_user_model.furthest_reached_checkpoint_state_name)
        if furthest_reached_checkpoint_in_current_exp is None:
            exp_user_model.furthest_reached_checkpoint_exp_version = exp_version
            exp_user_model.furthest_reached_checkpoint_state_name = state_name
        else:
            frc_index = checkpoints_in_current_exp.index(furthest_reached_checkpoint_in_current_exp)
            if frc_index <= checkpoints_in_current_exp.index(state_name):
                exp_user_model.furthest_reached_checkpoint_exp_version = exp_version
                exp_user_model.furthest_reached_checkpoint_state_name = state_name
    exp_user_model.most_recently_reached_checkpoint_exp_version = exp_version
    exp_user_model.most_recently_reached_checkpoint_state_name = state_name
    exp_user_model.update_timestamps()
    exp_user_model.put()

def set_user_has_viewed_lesson_info_modal_once(user_id: str) -> None:
    if False:
        while True:
            i = 10
    "Updates the user's settings once he has viewed the lesson info modal.\n\n    Args:\n        user_id: str. The Id of the user.\n    "
    user_settings = get_user_settings(user_id)
    user_settings.mark_lesson_info_modal_viewed()
    save_user_settings(user_settings)

def clear_learner_checkpoint_progress(user_id: str, exploration_id: str) -> None:
    if False:
        print('Hello World!')
    "Clears learner's checkpoint progress through the exploration by\n    clearing the most recently reached checkpoint fields of the exploration.\n\n    Args:\n        user_id: str. The Id of the user.\n        exploration_id: str. The Id of the exploration.\n    "
    exp_user_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
    if exp_user_model is not None:
        exp_user_model.most_recently_reached_checkpoint_exp_version = None
        exp_user_model.most_recently_reached_checkpoint_state_name = None
        exp_user_model.update_timestamps()
        exp_user_model.put()

@overload
def sync_logged_in_learner_checkpoint_progress_with_current_exp_version(user_id: str, exploration_id: str) -> Optional[user_domain.ExplorationUserData]:
    if False:
        while True:
            i = 10
    ...

@overload
def sync_logged_in_learner_checkpoint_progress_with_current_exp_version(user_id: str, exploration_id: str, *, strict: Literal[True]) -> user_domain.ExplorationUserData:
    if False:
        while True:
            i = 10
    ...

@overload
def sync_logged_in_learner_checkpoint_progress_with_current_exp_version(user_id: str, exploration_id: str, *, strict: Literal[False]) -> Optional[user_domain.ExplorationUserData]:
    if False:
        for i in range(10):
            print('nop')
    ...

def sync_logged_in_learner_checkpoint_progress_with_current_exp_version(user_id: str, exploration_id: str, strict: bool=False) -> Optional[user_domain.ExplorationUserData]:
    if False:
        for i in range(10):
            print('nop')
    'Synchronizes the most recently reached checkpoint and the furthest\n    reached checkpoint with the latest exploration.\n\n    Args:\n        user_id: str. The Id of the user.\n        exploration_id: str. The Id of the exploration.\n        strict: bool. Whether to fail noisily if no ExplorationUserDataModel\n            with the given user_id exists in the datastore.\n\n    Returns:\n        ExplorationUserData. The domain object corresponding to the given user\n        and exploration.\n\n    Raises:\n        Exception. No ExplorationUserDataModel found for the given user and\n            exploration ids.\n    '
    exp_user_model = user_models.ExplorationUserDataModel.get(user_id, exploration_id)
    if exp_user_model is None:
        if strict:
            raise Exception('No ExplorationUserDataModel found for the given user and exploration ids: %s, %s' % (user_id, exploration_id))
        return None
    latest_exploration = exp_fetchers.get_exploration_by_id(exploration_id)
    most_recently_interacted_exploration = exp_fetchers.get_exploration_by_id(exploration_id, strict=True, version=exp_user_model.most_recently_reached_checkpoint_exp_version)
    furthest_reached_exploration = exp_fetchers.get_exploration_by_id(exploration_id, strict=True, version=exp_user_model.furthest_reached_checkpoint_exp_version)
    most_recently_reached_checkpoint_in_current_exploration = get_most_distant_reached_checkpoint_in_current_exploration(get_checkpoints_in_order(latest_exploration.init_state_name, latest_exploration.states), get_checkpoints_in_order(most_recently_interacted_exploration.init_state_name, most_recently_interacted_exploration.states), exp_user_model.most_recently_reached_checkpoint_state_name)
    furthest_reached_checkpoint_in_current_exploration = get_most_distant_reached_checkpoint_in_current_exploration(get_checkpoints_in_order(latest_exploration.init_state_name, latest_exploration.states), get_checkpoints_in_order(furthest_reached_exploration.init_state_name, furthest_reached_exploration.states), exp_user_model.furthest_reached_checkpoint_state_name)
    if most_recently_reached_checkpoint_in_current_exploration != exp_user_model.most_recently_reached_checkpoint_state_name:
        exp_user_model.most_recently_reached_checkpoint_state_name = most_recently_reached_checkpoint_in_current_exploration
        exp_user_model.most_recently_reached_checkpoint_exp_version = latest_exploration.version
        exp_user_model.update_timestamps()
        exp_user_model.put()
    if furthest_reached_checkpoint_in_current_exploration != exp_user_model.furthest_reached_checkpoint_state_name:
        exp_user_model.furthest_reached_checkpoint_state_name = furthest_reached_checkpoint_in_current_exploration
        exp_user_model.furthest_reached_checkpoint_exp_version = latest_exploration.version
        exp_user_model.update_timestamps()
        exp_user_model.put()
    return exp_fetchers.get_exploration_user_data(user_id, exploration_id)

def is_user_blog_post_author(user_id: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Checks whether user can write blog posts.\n\n    Args:\n        user_id: str. The user id of the user.\n\n    Returns:\n        bool. Whether the user can author blog posts.\n    '
    user_settings = get_user_settings(user_id, strict=True)
    author_roles = [feconf.ROLE_ID_BLOG_ADMIN, feconf.ROLE_ID_BLOG_POST_EDITOR]
    return any((role in author_roles for role in user_settings.roles))

def assign_coordinator(committer: user_domain.UserActionsInfo, assignee: user_domain.UserActionsInfo, language_id: str) -> None:
    if False:
        print('Hello World!')
    'Assigns a new role to the user.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user\n            who is performing the action.\n        assignee: UserActionsInfo. UserActionsInfo object for the user\n            whose role is being changed.\n        language_id: str. ID of the language.\n\n    Raises:\n        Exception. The committer does not have rights to modify a role.\n        Exception. The assignee is already coordinator for this language.\n        Exception. Guest user is not allowed to assign roles to a user.\n        Exception. The role of the Guest user cannot be changed.\n    '
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to assign roles to a user.')
    if role_services.ACTION_MODIFY_CORE_ROLES_FOR_ANY_ACTIVITY not in committer.actions:
        logging.error('User %s tried to allow user %s to be a coordinator of language %s but was refused permission.' % (committer_id, assignee.user_id, language_id))
        raise Exception('UnauthorizedUserException: Could not assign new role.')
    if assignee.user_id is None:
        raise Exception('Cannot change the role of the Guest user.')
    language_rights = suggestion_models.TranslationCoordinatorsModel.get(language_id, strict=False)
    if language_rights is None:
        model = suggestion_models.TranslationCoordinatorsModel(id=language_id, coordinator_ids=[assignee.user_id], coordinators_count=1)
        model.update_timestamps()
        model.put()
    else:
        if assignee.user_id in language_rights.coordinator_ids:
            raise Exception('This user already is a coordinator for this language.')
        language_rights.coordinator_ids.append(assignee.user_id)
        language_rights.coordinators_count += 1
        suggestion_models.TranslationCoordinatorsModel.update_timestamps(language_rights, update_last_updated_time=True)
        suggestion_models.TranslationCoordinatorsModel.put(language_rights)

def deassign_coordinator(committer: user_domain.UserActionsInfo, assignee: user_domain.UserActionsInfo, language_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Removes the user as a coordinator of that language.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user\n            who is performing the action.\n        assignee: UserActionsInfo. UserActionsInfo object for the user\n            whose role is being changed.\n        language_id: str. ID of the language.\n\n    Raises:\n        Exception. The committer does not have rights to modify a role.\n        Exception. The assignee is already coordinator for this language.\n        Exception. Guest user is not allowed to assign roles to a user.\n        Exception. The role of the Guest user cannot be changed.\n    '
    committer_id = committer.user_id
    if committer_id is None:
        raise Exception('Guest user is not allowed to deassign roles to a user.')
    language_rights = suggestion_models.TranslationCoordinatorsModel.get(language_id, strict=False)
    if role_services.ACTION_MODIFY_CORE_ROLES_FOR_ANY_ACTIVITY not in committer.actions:
        logging.error('User %s tried to allow user %s to be a coordinator of language %s but was refused permission.' % (committer_id, assignee.user_id, language_id))
        raise Exception('UnauthorizedUserException: Could not assign new role.')
    if assignee.user_id is None:
        raise Exception('Cannot change the role of the Guest user.')
    if language_rights is None:
        raise Exception('No model exists for provided language.')
    if assignee.user_id not in language_rights.coordinator_ids:
        raise Exception('This user is not a coordinator for this language')
    language_rights.coordinator_ids.remove(assignee.user_id)
    language_rights.coordinators_count -= 1
    suggestion_models.TranslationCoordinatorsModel.update_timestamps(language_rights, update_last_updated_time=True)
    suggestion_models.TranslationCoordinatorsModel.put(language_rights)

def get_translation_rights_from_model(translation_coordinator_model: suggestion_models.TranslationCoordinatorsModel) -> user_domain.TranslationCoordinatorStats:
    if False:
        while True:
            i = 10
    'Constructs a TranslationCoordinatorStats object from the given\n    translation coordinator model.\n\n    Args:\n        translation_coordinator_model: TranslationCoordinatorsModel. The\n            model which is to be converted to an object.\n\n    Returns:\n        TranslationCoordinatorStats. The TranslationCoordinatorStats object\n        created from the model.\n    '
    return user_domain.TranslationCoordinatorStats(translation_coordinator_model.id, translation_coordinator_model.coordinator_ids, translation_coordinator_model.coordinators_count)

def get_translation_rights_with_user(user_id: str) -> List[user_domain.TranslationCoordinatorStats]:
    if False:
        while True:
            i = 10
    'Retrieves the rights object for all languages assigned to given user.\n\n    Args:\n        user_id: str. ID of the user.\n\n    Returns:\n        list(TranslationCoordinatorStats). The rights objects associated with\n        the languagesassigned to given user.\n    '
    translation_coordinator_models: Sequence[suggestion_models.TranslationCoordinatorsModel] = suggestion_models.TranslationCoordinatorsModel.get_by_user(user_id)
    return [get_translation_rights_from_model(model) for model in translation_coordinator_models if model is not None]

def deassign_user_from_all_languages(committer: user_domain.UserActionsInfo, user_id: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Deassigns given user from all languages assigned to them.\n\n    Args:\n        committer: UserActionsInfo. UserActionsInfo object for the user\n            who is performing the action.\n        user_id: str. The ID of the user.\n\n    Raises:\n        Exception. Guest users are not allowed to deassign users from\n            all languages.\n    '
    translation_rights_list = get_translation_rights_with_user(user_id)
    if committer.user_id is None:
        raise Exception('Guest users are not allowed to deassign users from all languages.')
    for translation_rights in translation_rights_list:
        translation_rights.coordinator_ids.remove(user_id)
        translation_rights.coordinators_count -= 1
        language_rights = suggestion_models.TranslationCoordinatorsModel(id=translation_rights.language_id, coordinator_ids=translation_rights.coordinator_ids, coordinators_count=translation_rights.coordinators_count)
        suggestion_models.TranslationCoordinatorsModel.update_timestamps(language_rights, update_last_updated_time=True)
        suggestion_models.TranslationCoordinatorsModel.put(language_rights)

def check_user_is_coordinator(user_id: str, language_id: str) -> bool:
    if False:
        while True:
            i = 10
    'Check if the given user is coordinator of provided language.\n\n    Args:\n        user_id:str. User ID.\n        language_id: str. ID of the language.\n\n    Returns:\n        bool. True if the user is coordinator or else False.\n    '
    model = suggestion_models.TranslationCoordinatorsModel.get(language_id, strict=False)
    if model is None:
        return False
    return user_id in model.coordinator_ids