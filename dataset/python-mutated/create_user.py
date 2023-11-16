from datetime import datetime
from email.headerregistry import Address
from typing import Optional, Union
import orjson
from django.contrib.auth.models import UserManager
from django.utils.timezone import now as timezone_now
from zerver.lib.hotspots import copy_hotspots
from zerver.lib.i18n import get_default_language_for_new_user
from zerver.lib.timezone import canonicalize_timezone
from zerver.lib.upload import copy_avatar
from zerver.models import Realm, RealmUserDefault, Recipient, Stream, Subscription, UserBaseSettings, UserProfile, get_fake_email_domain

def copy_default_settings(settings_source: Union[UserProfile, RealmUserDefault], target_profile: UserProfile) -> None:
    if False:
        for i in range(10):
            print('nop')
    for settings_name in UserBaseSettings.property_types:
        if settings_name in ['default_language', 'enable_login_emails'] and isinstance(settings_source, RealmUserDefault):
            continue
        if settings_name == 'email_address_visibility':
            continue
        value = getattr(settings_source, settings_name)
        setattr(target_profile, settings_name, value)
    if isinstance(settings_source, RealmUserDefault):
        target_profile.save()
        return
    target_profile.full_name = settings_source.full_name
    target_profile.timezone = canonicalize_timezone(settings_source.timezone)
    target_profile.save()
    if settings_source.avatar_source == UserProfile.AVATAR_FROM_USER:
        from zerver.actions.user_settings import do_change_avatar_fields
        do_change_avatar_fields(target_profile, UserProfile.AVATAR_FROM_USER, skip_notify=True, acting_user=target_profile)
        copy_avatar(settings_source, target_profile)
    copy_hotspots(settings_source, target_profile)

def get_display_email_address(user_profile: UserProfile) -> str:
    if False:
        print('Hello World!')
    if not user_profile.email_address_is_realm_public():
        return Address(username=f'user{user_profile.id}', domain=get_fake_email_domain(user_profile.realm)).addr_spec
    return user_profile.delivery_email

def create_user_profile(realm: Realm, email: str, password: Optional[str], active: bool, bot_type: Optional[int], full_name: str, bot_owner: Optional[UserProfile], is_mirror_dummy: bool, tos_version: Optional[str], timezone: str, default_language: str, tutorial_status: str=UserProfile.TUTORIAL_WAITING, force_id: Optional[int]=None, force_date_joined: Optional[datetime]=None, *, email_address_visibility: int) -> UserProfile:
    if False:
        return 10
    if force_date_joined is None:
        date_joined = timezone_now()
    else:
        date_joined = force_date_joined
    email = UserManager.normalize_email(email)
    extra_kwargs = {}
    if force_id is not None:
        extra_kwargs['id'] = force_id
    user_profile = UserProfile(is_staff=False, is_active=active, full_name=full_name, last_login=date_joined, date_joined=date_joined, realm=realm, is_bot=bool(bot_type), bot_type=bot_type, bot_owner=bot_owner, is_mirror_dummy=is_mirror_dummy, tos_version=tos_version, timezone=timezone, tutorial_status=tutorial_status, onboarding_steps=orjson.dumps([]).decode(), default_language=default_language, delivery_email=email, email_address_visibility=email_address_visibility, **extra_kwargs)
    if bot_type or not active:
        password = None
    if user_profile.email_address_is_realm_public():
        user_profile.email = get_display_email_address(user_profile)
    user_profile.set_password(password)
    return user_profile

def create_user(email: str, password: Optional[str], realm: Realm, full_name: str, active: bool=True, role: Optional[int]=None, bot_type: Optional[int]=None, bot_owner: Optional[UserProfile]=None, tos_version: Optional[str]=None, timezone: str='', avatar_source: str=UserProfile.AVATAR_FROM_GRAVATAR, is_mirror_dummy: bool=False, default_language: Optional[str]=None, default_sending_stream: Optional[Stream]=None, default_events_register_stream: Optional[Stream]=None, default_all_public_streams: Optional[bool]=None, source_profile: Optional[UserProfile]=None, force_id: Optional[int]=None, force_date_joined: Optional[datetime]=None, create_personal_recipient: bool=True, enable_marketing_emails: Optional[bool]=None, email_address_visibility: Optional[int]=None) -> UserProfile:
    if False:
        for i in range(10):
            print('nop')
    realm_user_default = RealmUserDefault.objects.get(realm=realm)
    if bot_type is None:
        if email_address_visibility is not None:
            user_email_address_visibility = email_address_visibility
        else:
            user_email_address_visibility = realm_user_default.email_address_visibility
    else:
        user_email_address_visibility = UserProfile.EMAIL_ADDRESS_VISIBILITY_EVERYONE
    if default_language is None:
        default_language = get_default_language_for_new_user(realm, request=None)
    user_profile = create_user_profile(realm, email, password, active, bot_type, full_name, bot_owner, is_mirror_dummy, tos_version, timezone, default_language, force_id=force_id, force_date_joined=force_date_joined, email_address_visibility=user_email_address_visibility)
    user_profile.avatar_source = avatar_source
    user_profile.timezone = timezone
    user_profile.default_sending_stream = default_sending_stream
    user_profile.default_events_register_stream = default_events_register_stream
    if role is not None:
        user_profile.role = role
    if default_all_public_streams is not None:
        user_profile.default_all_public_streams = default_all_public_streams
    if source_profile is not None:
        copy_default_settings(source_profile, user_profile)
    elif bot_type is None:
        copy_default_settings(realm_user_default, user_profile)
    else:
        user_profile.save()
    if bot_type is None and enable_marketing_emails is not None:
        user_profile.enable_marketing_emails = enable_marketing_emails
        user_profile.save(update_fields=['enable_marketing_emails'])
    if not user_profile.email_address_is_realm_public():
        user_profile.email = get_display_email_address(user_profile)
        user_profile.save(update_fields=['email'])
    if not create_personal_recipient:
        return user_profile
    recipient = Recipient.objects.create(type_id=user_profile.id, type=Recipient.PERSONAL)
    user_profile.recipient = recipient
    user_profile.save(update_fields=['recipient'])
    Subscription.objects.create(user_profile=user_profile, recipient=recipient, is_user_active=user_profile.is_active)
    return user_profile