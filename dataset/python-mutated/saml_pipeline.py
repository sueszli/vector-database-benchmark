import re
import logging
from django.conf import settings
from awx.main.models import Team
from awx.sso.common import create_org_and_teams, reconcile_users_org_team_mappings, get_orgs_by_ids
logger = logging.getLogger('awx.sso.saml_pipeline')

def populate_user(backend, details, user=None, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if not user:
        return
    desired_org_state = {}
    desired_team_state = {}
    orgs_to_create = []
    teams_to_create = {}
    _update_user_orgs_by_saml_attr(backend, desired_org_state, orgs_to_create, **kwargs)
    _update_user_teams_by_saml_attr(desired_team_state, teams_to_create, **kwargs)
    _update_user_orgs(backend, desired_org_state, orgs_to_create, user)
    _update_user_teams(backend, desired_team_state, teams_to_create, user)
    create_org_and_teams(orgs_to_create, teams_to_create, 'SAML', settings.SAML_AUTO_CREATE_OBJECTS)
    reconcile_users_org_team_mappings(user, desired_org_state, desired_team_state, 'SAML')

def _update_m2m_from_expression(user, expr, remove=True):
    if False:
        return 10
    '\n    Helper function to update m2m relationship based on user matching one or\n    more expressions.\n    '
    should_add = False
    if expr is None or not expr:
        pass
    elif expr is True:
        should_add = True
    else:
        if isinstance(expr, (str, type(re.compile('')))):
            expr = [expr]
        for ex in expr:
            if isinstance(ex, str):
                if user.username == ex or user.email == ex:
                    should_add = True
            elif isinstance(ex, type(re.compile(''))):
                if ex.match(user.username) or ex.match(user.email):
                    should_add = True
    if should_add:
        return True
    elif remove:
        return False
    else:
        return None

def _update_user_orgs(backend, desired_org_state, orgs_to_create, user=None):
    if False:
        print('Hello World!')
    '\n    Update organization memberships for the given user based on mapping rules\n    defined in settings.\n    '
    org_map = backend.setting('ORGANIZATION_MAP') or {}
    for (org_name, org_opts) in org_map.items():
        organization_alias = org_opts.get('organization_alias')
        if organization_alias:
            organization_name = organization_alias
        else:
            organization_name = org_name
        if organization_name not in orgs_to_create:
            orgs_to_create.append(organization_name)
        remove = bool(org_opts.get('remove', True))
        if organization_name not in desired_org_state:
            desired_org_state[organization_name] = {}
        for (role_name, user_type) in (('admin_role', 'admins'), ('member_role', 'users'), ('auditor_role', 'auditors')):
            is_member_expression = org_opts.get(user_type, None)
            remove_members = bool(org_opts.get('remove_{}'.format(user_type), remove))
            has_role = _update_m2m_from_expression(user, is_member_expression, remove_members)
            desired_org_state[organization_name][role_name] = desired_org_state[organization_name].get(role_name, False) or has_role

def _update_user_teams(backend, desired_team_state, teams_to_create, user=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update team memberships for the given user based on mapping rules defined\n    in settings.\n    '
    team_map = backend.setting('TEAM_MAP') or {}
    for (team_name, team_opts) in team_map.items():
        if 'organization' not in team_opts:
            continue
        teams_to_create[team_name] = team_opts['organization']
        users_expr = team_opts.get('users', None)
        remove = bool(team_opts.get('remove', True))
        add_or_remove = _update_m2m_from_expression(user, users_expr, remove)
        if add_or_remove is not None:
            org_name = team_opts['organization']
            if org_name not in desired_team_state:
                desired_team_state[org_name] = {}
            desired_team_state[org_name][team_name] = {'member_role': add_or_remove}

def _update_user_orgs_by_saml_attr(backend, desired_org_state, orgs_to_create, **kwargs):
    if False:
        return 10
    org_map = settings.SOCIAL_AUTH_SAML_ORGANIZATION_ATTR
    roles_and_flags = (('member_role', 'remove', 'saml_attr'), ('admin_role', 'remove_admins', 'saml_admin_attr'), ('auditor_role', 'remove_auditors', 'saml_auditor_attr'))
    all_orgs = None
    for (role, remove_flag, _) in roles_and_flags:
        remove = bool(org_map.get(remove_flag, True))
        if remove:
            if all_orgs is None:
                all_orgs = get_orgs_by_ids()
            for org_name in all_orgs.keys():
                if org_name not in desired_org_state:
                    desired_org_state[org_name] = {}
                desired_org_state[org_name][role] = False
    for (role, _, attr_flag) in roles_and_flags:
        if org_map.get(attr_flag) is None:
            continue
        saml_attr_values = kwargs.get('response', {}).get('attributes', {}).get(org_map.get(attr_flag), [])
        for org_name in saml_attr_values:
            try:
                organization_alias = backend.setting('ORGANIZATION_MAP').get(org_name).get('organization_alias')
                if organization_alias is not None:
                    organization_name = organization_alias
                else:
                    organization_name = org_name
            except Exception:
                organization_name = org_name
            if organization_name not in orgs_to_create:
                orgs_to_create.append(organization_name)
            if organization_name not in desired_org_state:
                desired_org_state[organization_name] = {}
            desired_org_state[organization_name][role] = True

def _update_user_teams_by_saml_attr(desired_team_state, teams_to_create, **kwargs):
    if False:
        print('Hello World!')
    team_map = settings.SOCIAL_AUTH_SAML_TEAM_ATTR
    if team_map.get('saml_attr') is None:
        return
    all_teams = None
    for (role, remove_flag) in [('member_role', 'remove')]:
        remove = bool(team_map.get(remove_flag, True))
        if remove:
            if all_teams is None:
                all_teams = Team.objects.all().values_list('name', 'organization__name')
            for (team_name, organization_name) in all_teams:
                if organization_name not in desired_team_state:
                    desired_team_state[organization_name] = {}
                desired_team_state[organization_name][team_name] = {role: False}
    saml_team_names = set(kwargs.get('response', {}).get('attributes', {}).get(team_map['saml_attr'], []))
    for team_name_map in team_map.get('team_org_map', []):
        team_name = team_name_map.get('team', None)
        team_alias = team_name_map.get('team_alias', None)
        organization_name = team_name_map.get('organization', None)
        if team_name in saml_team_names:
            if not organization_name:
                logger.error('organization name invalid for team {}'.format(team_name))
                continue
            if team_alias:
                team_name = team_alias
            teams_to_create[team_name] = organization_name
            user_is_member_of_team = True
        else:
            user_is_member_of_team = False
        if organization_name not in desired_team_state:
            desired_team_state[organization_name] = {}
        desired_team_state[organization_name][team_name] = {'member_role': user_is_member_of_team}

def _get_matches(list1, list2):
    if False:
        return 10
    if not isinstance(list1, (list, tuple)):
        list1 = [list1]
    if not isinstance(list2, (list, tuple)):
        list2 = [list2]
    return set(list1).intersection(set(list2))

def _check_flag(user, flag, attributes, user_flags_settings):
    if False:
        return 10
    '\n    Helper function to set the is_superuser is_system_auditor flags for the SAML adapter\n    Returns the new flag and whether or not it changed the flag\n    '
    new_flag = False
    is_role_key = 'is_%s_role' % flag
    is_attr_key = 'is_%s_attr' % flag
    is_value_key = 'is_%s_value' % flag
    remove_setting = 'remove_%ss' % flag
    required_roles = user_flags_settings.get(is_role_key, None)
    if required_roles:
        matching_roles = _get_matches(required_roles, attributes.get('Role', []))
        if matching_roles:
            logger.debug('User %s has %s role(s) %s' % (user.username, flag, ', '.join(matching_roles)))
            new_flag = True
        else:
            logger.debug('User %s is missing the %s role(s) %s' % (user.username, flag, ', '.join(required_roles)))
    attr_setting = user_flags_settings.get(is_attr_key, None)
    if attr_setting and attributes.get(attr_setting, None):
        required_value = user_flags_settings.get(is_value_key, None)
        if required_value:
            saml_user_attribute_value = attributes.get(attr_setting, None)
            matching_values = _get_matches(required_value, saml_user_attribute_value)
            if matching_values:
                logger.debug('Giving %s %s from attribute %s with matching values %s' % (user.username, flag, attr_setting, ', '.join(matching_values)))
                new_flag = True
            else:
                logger.debug('Refusing %s for %s because attr %s (%s) did not match value(s) %s' % (flag, user.username, attr_setting, ', '.join(saml_user_attribute_value), ', '.join(required_value)))
                new_flag = False
        else:
            logger.debug('Giving %s %s from attribute %s' % (user.username, flag, attr_setting))
            new_flag = True
    old_value = getattr(user, 'is_%s' % flag)
    remove_flag = user_flags_settings.get(remove_setting, True)
    if not remove_flag and (old_value and (not new_flag)):
        logger.debug('Remove flag %s preventing removal of %s for %s' % (remove_flag, flag, user.username))
        return (old_value, False)
    if old_value and (not new_flag):
        logger.debug('Revoking %s from %s' % (flag, user.username))
    return (new_flag, old_value != new_flag)

def update_user_flags(backend, details, user=None, *args, **kwargs):
    if False:
        print('Hello World!')
    user_flags_settings = settings.SOCIAL_AUTH_SAML_USER_FLAGS_BY_ATTR
    attributes = kwargs.get('response', {}).get('attributes', {})
    logger.debug('User attributes for %s: %s' % (user.username, attributes))
    (user.is_superuser, superuser_changed) = _check_flag(user, 'superuser', attributes, user_flags_settings)
    (user.is_system_auditor, auditor_changed) = _check_flag(user, 'system_auditor', attributes, user_flags_settings)
    if superuser_changed or auditor_changed:
        user.save()