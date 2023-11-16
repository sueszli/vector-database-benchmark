from django.http.request import HttpRequest
from django.http.response import HttpResponse
from django.utils.translation import gettext as _
from zerver.lib.exceptions import JsonableError
from zerver.lib.message import access_message
from zerver.lib.request import REQ, has_request_variables
from zerver.lib.response import json_success
from zerver.lib.validator import to_non_negative_int
from zerver.models import UserMessage, UserProfile

@has_request_variables
def read_receipts(request: HttpRequest, user_profile: UserProfile, message_id: int=REQ(converter=to_non_negative_int, path_only=True)) -> HttpResponse:
    if False:
        return 10
    message = access_message(user_profile, message_id)[0]
    if not user_profile.realm.enable_read_receipts:
        raise JsonableError(_('Read receipts are disabled in this organization.'))
    user_ids = UserMessage.objects.filter(message_id=message.id, user_profile__is_active=True, user_profile__send_read_receipts=True).exclude(user_profile_id=message.sender_id).exclude(user_profile__muter__muted_user_id=user_profile.id).exclude(user_profile__muted__user_profile_id=user_profile.id).extra(where=[UserMessage.where_read()]).values_list('user_profile_id', flat=True)
    return json_success(request, {'user_ids': list(user_ids)})