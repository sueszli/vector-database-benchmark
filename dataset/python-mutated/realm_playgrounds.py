import re
from django.http import HttpRequest, HttpResponse
from django.utils.translation import gettext as _
from zerver.actions.realm_playgrounds import check_add_realm_playground, do_remove_realm_playground
from zerver.decorator import require_realm_admin
from zerver.lib.exceptions import JsonableError
from zerver.lib.request import REQ, has_request_variables
from zerver.lib.response import json_success
from zerver.lib.validator import check_capped_string
from zerver.models import Realm, RealmPlayground, UserProfile

def check_pygments_language(var_name: str, val: object) -> str:
    if False:
        return 10
    s = check_capped_string(RealmPlayground.MAX_PYGMENTS_LANGUAGE_LENGTH)(var_name, val)
    valid_pygments_language = re.compile('^[ a-zA-Z0-9_+-./#]*$')
    matched_results = valid_pygments_language.match(s)
    if not matched_results:
        raise JsonableError(_('Invalid characters in pygments language'))
    return s

def access_playground_by_id(realm: Realm, playground_id: int) -> RealmPlayground:
    if False:
        i = 10
        return i + 15
    try:
        realm_playground = RealmPlayground.objects.get(id=playground_id, realm=realm)
    except RealmPlayground.DoesNotExist:
        raise JsonableError(_('Invalid playground'))
    return realm_playground

@require_realm_admin
@has_request_variables
def add_realm_playground(request: HttpRequest, user_profile: UserProfile, name: str=REQ(), url_template: str=REQ(), pygments_language: str=REQ(str_validator=check_pygments_language)) -> HttpResponse:
    if False:
        print('Hello World!')
    playground_id = check_add_realm_playground(realm=user_profile.realm, acting_user=user_profile, name=name.strip(), pygments_language=pygments_language.strip(), url_template=url_template.strip())
    return json_success(request, data={'id': playground_id})

@require_realm_admin
@has_request_variables
def delete_realm_playground(request: HttpRequest, user_profile: UserProfile, playground_id: int) -> HttpResponse:
    if False:
        i = 10
        return i + 15
    realm_playground = access_playground_by_id(user_profile.realm, playground_id)
    do_remove_realm_playground(user_profile.realm, realm_playground, acting_user=user_profile)
    return json_success(request)