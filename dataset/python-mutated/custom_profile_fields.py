from typing import List, Optional, cast
import orjson
from django.core.exceptions import ValidationError
from django.db import IntegrityError
from django.http import HttpRequest, HttpResponse
from django.utils.translation import gettext as _
from zerver.actions.custom_profile_fields import check_remove_custom_profile_field_value, do_remove_realm_custom_profile_field, do_update_user_custom_profile_data_if_changed, try_add_realm_custom_profile_field, try_add_realm_default_custom_profile_field, try_reorder_realm_custom_profile_fields, try_update_realm_custom_profile_field
from zerver.decorator import human_users_only, require_realm_admin
from zerver.lib.exceptions import JsonableError
from zerver.lib.external_accounts import validate_external_account_field_data
from zerver.lib.request import REQ, has_request_variables
from zerver.lib.response import json_success
from zerver.lib.types import ProfileDataElementUpdateDict, ProfileFieldData, Validator
from zerver.lib.users import validate_user_custom_profile_data
from zerver.lib.validator import check_bool, check_capped_string, check_dict, check_dict_only, check_int, check_list, check_string, check_union, validate_select_field_data
from zerver.models import CustomProfileField, Realm, UserProfile, custom_profile_fields_for_realm

def list_realm_custom_profile_fields(request: HttpRequest, user_profile: UserProfile) -> HttpResponse:
    if False:
        for i in range(10):
            print('nop')
    fields = custom_profile_fields_for_realm(user_profile.realm_id)
    return json_success(request, data={'custom_fields': [f.as_dict() for f in fields]})
hint_validator = check_capped_string(CustomProfileField.HINT_MAX_LENGTH)
name_validator = check_capped_string(CustomProfileField.NAME_MAX_LENGTH)

def validate_field_name_and_hint(name: str, hint: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    if not name.strip():
        raise JsonableError(_('Label cannot be blank.'))
    try:
        hint_validator('hint', hint)
        name_validator('name', name)
    except ValidationError as error:
        raise JsonableError(error.message)

def validate_custom_field_data(field_type: int, field_data: ProfileFieldData) -> None:
    if False:
        print('Hello World!')
    try:
        if field_type == CustomProfileField.SELECT:
            if len(field_data) < 1:
                raise JsonableError(_('Field must have at least one choice.'))
            validate_select_field_data(field_data)
        elif field_type == CustomProfileField.EXTERNAL_ACCOUNT:
            validate_external_account_field_data(field_data)
    except ValidationError as error:
        raise JsonableError(error.message)

def validate_display_in_profile_summary_field(field_type: int, display_in_profile_summary: bool) -> None:
    if False:
        for i in range(10):
            print('nop')
    if not display_in_profile_summary:
        return
    if field_type in (CustomProfileField.LONG_TEXT, CustomProfileField.USER):
        raise JsonableError(_('Field type not supported for display in profile summary.'))

def is_default_external_field(field_type: int, field_data: ProfileFieldData) -> bool:
    if False:
        for i in range(10):
            print('nop')
    if field_type != CustomProfileField.EXTERNAL_ACCOUNT:
        return False
    if field_data['subtype'] == 'custom':
        return False
    return True

def validate_custom_profile_field(name: str, hint: str, field_type: int, field_data: ProfileFieldData, display_in_profile_summary: bool) -> None:
    if False:
        i = 10
        return i + 15
    validate_custom_field_data(field_type, field_data)
    if not is_default_external_field(field_type, field_data):
        validate_field_name_and_hint(name, hint)
    field_types = [i[0] for i in CustomProfileField.FIELD_TYPE_CHOICES]
    if field_type not in field_types:
        raise JsonableError(_('Invalid field type.'))
    validate_display_in_profile_summary_field(field_type, display_in_profile_summary)
check_profile_field_data: Validator[ProfileFieldData] = check_dict(value_validator=check_union([check_dict(value_validator=check_string), check_string]))

def update_only_display_in_profile_summary(requested_name: str, requested_hint: str, requested_field_data: ProfileFieldData, existing_field: CustomProfileField) -> bool:
    if False:
        return 10
    if requested_name != existing_field.name or requested_hint != existing_field.hint or requested_field_data != orjson.loads(existing_field.field_data):
        return False
    return True

def display_in_profile_summary_limit_reached(realm: Realm, profile_field_id: Optional[int]=None) -> bool:
    if False:
        i = 10
        return i + 15
    query = CustomProfileField.objects.filter(realm=realm, display_in_profile_summary=True)
    if profile_field_id is not None:
        query = query.exclude(id=profile_field_id)
    return query.count() >= CustomProfileField.MAX_DISPLAY_IN_PROFILE_SUMMARY_FIELDS

@require_realm_admin
@has_request_variables
def create_realm_custom_profile_field(request: HttpRequest, user_profile: UserProfile, name: str=REQ(default='', converter=lambda var_name, x: x.strip()), hint: str=REQ(default=''), field_data: ProfileFieldData=REQ(default={}, json_validator=check_profile_field_data), field_type: int=REQ(json_validator=check_int), display_in_profile_summary: bool=REQ(default=False, json_validator=check_bool)) -> HttpResponse:
    if False:
        for i in range(10):
            print('nop')
    if display_in_profile_summary and display_in_profile_summary_limit_reached(user_profile.realm):
        raise JsonableError(_('Only 2 custom profile fields can be displayed in the profile summary.'))
    validate_custom_profile_field(name, hint, field_type, field_data, display_in_profile_summary)
    try:
        if is_default_external_field(field_type, field_data):
            field_subtype = field_data['subtype']
            assert isinstance(field_subtype, str)
            field = try_add_realm_default_custom_profile_field(realm=user_profile.realm, field_subtype=field_subtype, display_in_profile_summary=display_in_profile_summary)
            return json_success(request, data={'id': field.id})
        else:
            field = try_add_realm_custom_profile_field(realm=user_profile.realm, name=name, field_data=field_data, field_type=field_type, hint=hint, display_in_profile_summary=display_in_profile_summary)
            return json_success(request, data={'id': field.id})
    except IntegrityError:
        raise JsonableError(_('A field with that label already exists.'))

@require_realm_admin
def delete_realm_custom_profile_field(request: HttpRequest, user_profile: UserProfile, field_id: int) -> HttpResponse:
    if False:
        while True:
            i = 10
    try:
        field = CustomProfileField.objects.get(id=field_id)
    except CustomProfileField.DoesNotExist:
        raise JsonableError(_('Field id {id} not found.').format(id=field_id))
    do_remove_realm_custom_profile_field(realm=user_profile.realm, field=field)
    return json_success(request)

@require_realm_admin
@has_request_variables
def update_realm_custom_profile_field(request: HttpRequest, user_profile: UserProfile, field_id: int, name: str=REQ(default='', converter=lambda var_name, x: x.strip()), hint: str=REQ(default=''), field_data: ProfileFieldData=REQ(default={}, json_validator=check_profile_field_data), display_in_profile_summary: bool=REQ(default=False, json_validator=check_bool)) -> HttpResponse:
    if False:
        while True:
            i = 10
    realm = user_profile.realm
    try:
        field = CustomProfileField.objects.get(realm=realm, id=field_id)
    except CustomProfileField.DoesNotExist:
        raise JsonableError(_('Field id {id} not found.').format(id=field_id))
    if display_in_profile_summary and display_in_profile_summary_limit_reached(user_profile.realm, field.id):
        raise JsonableError(_('Only 2 custom profile fields can be displayed in the profile summary.'))
    if field.field_type == CustomProfileField.EXTERNAL_ACCOUNT and is_default_external_field(field.field_type, orjson.loads(field.field_data)) and (not update_only_display_in_profile_summary(name, hint, field_data, field)):
        raise JsonableError(_('Default custom field cannot be updated.'))
    validate_custom_profile_field(name, hint, field.field_type, field_data, display_in_profile_summary)
    try:
        try_update_realm_custom_profile_field(realm, field, name, hint=hint, field_data=field_data, display_in_profile_summary=display_in_profile_summary)
    except IntegrityError:
        raise JsonableError(_('A field with that label already exists.'))
    return json_success(request)

@require_realm_admin
@has_request_variables
def reorder_realm_custom_profile_fields(request: HttpRequest, user_profile: UserProfile, order: List[int]=REQ(json_validator=check_list(check_int))) -> HttpResponse:
    if False:
        print('Hello World!')
    try_reorder_realm_custom_profile_fields(user_profile.realm, order)
    return json_success(request)

@human_users_only
@has_request_variables
def remove_user_custom_profile_data(request: HttpRequest, user_profile: UserProfile, data: List[int]=REQ(json_validator=check_list(check_int))) -> HttpResponse:
    if False:
        i = 10
        return i + 15
    for field_id in data:
        check_remove_custom_profile_field_value(user_profile, field_id)
    return json_success(request)
check_profile_data_element_update_dict = cast(Validator[ProfileDataElementUpdateDict], check_dict_only([('id', check_int), ('value', check_union([check_string, check_list(check_int)]))]))

@human_users_only
@has_request_variables
def update_user_custom_profile_data(request: HttpRequest, user_profile: UserProfile, data: List[ProfileDataElementUpdateDict]=REQ(json_validator=check_list(check_profile_data_element_update_dict))) -> HttpResponse:
    if False:
        return 10
    validate_user_custom_profile_data(user_profile.realm.id, data)
    do_update_user_custom_profile_data_if_changed(user_profile, data)
    return json_success(request)