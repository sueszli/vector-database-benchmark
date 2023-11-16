from typing import Dict, Optional, Sequence
from django.core.exceptions import ValidationError
from django.utils.translation import gettext as _
from zerver.models import Recipient, UserProfile, get_or_create_huddle, is_cross_realm_bot_email

def get_recipient_from_user_profiles(recipient_profiles: Sequence[UserProfile], forwarded_mirror_message: bool, forwarder_user_profile: Optional[UserProfile], sender: UserProfile) -> Recipient:
    if False:
        return 10
    recipient_profiles_map = {user_profile.id: user_profile for user_profile in recipient_profiles}
    if forwarded_mirror_message:
        assert forwarder_user_profile is not None
        if forwarder_user_profile.id not in recipient_profiles_map:
            raise ValidationError(_('User not authorized for this query'))
    if len(recipient_profiles_map) == 2 and sender.id in recipient_profiles_map:
        del recipient_profiles_map[sender.id]
    assert recipient_profiles_map
    if len(recipient_profiles_map) == 1:
        [user_profile] = recipient_profiles_map.values()
        return Recipient(id=user_profile.recipient_id, type=Recipient.PERSONAL, type_id=user_profile.id)
    recipient_profiles_map[sender.id] = sender
    user_ids = list(recipient_profiles_map)
    huddle = get_or_create_huddle(user_ids)
    return Recipient(id=huddle.recipient_id, type=Recipient.HUDDLE, type_id=huddle.id)

def validate_recipient_user_profiles(user_profiles: Sequence[UserProfile], sender: UserProfile, allow_deactivated: bool=False) -> Sequence[UserProfile]:
    if False:
        for i in range(10):
            print('nop')
    recipient_profiles_map: Dict[int, UserProfile] = {}
    realms = set()
    if not is_cross_realm_bot_email(sender.email):
        realms.add(sender.realm_id)
    for user_profile in user_profiles:
        if not user_profile.is_active and (not user_profile.is_mirror_dummy) and (not allow_deactivated) or user_profile.realm.deactivated:
            raise ValidationError(_("'{email}' is no longer using Zulip.").format(email=user_profile.email))
        recipient_profiles_map[user_profile.id] = user_profile
        if not is_cross_realm_bot_email(user_profile.email):
            realms.add(user_profile.realm_id)
    if len(realms) > 1:
        raise ValidationError(_("You can't send direct messages outside of your organization."))
    return list(recipient_profiles_map.values())

def recipient_for_user_profiles(user_profiles: Sequence[UserProfile], forwarded_mirror_message: bool, forwarder_user_profile: Optional[UserProfile], sender: UserProfile, allow_deactivated: bool=False) -> Recipient:
    if False:
        i = 10
        return i + 15
    recipient_profiles = validate_recipient_user_profiles(user_profiles, sender, allow_deactivated=allow_deactivated)
    return get_recipient_from_user_profiles(recipient_profiles, forwarded_mirror_message, forwarder_user_profile, sender)