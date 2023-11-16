import re
import logging
from awx.sso.common import get_or_create_org_with_default_galaxy_cred
logger = logging.getLogger('awx.sso.social_pipeline')

def _update_m2m_from_expression(user, related, expr, remove=True):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to update m2m relationship based on user matching one or\n    more expressions.\n    '
    should_add = False
    if expr is None:
        return
    elif not expr:
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
        related.add(user)
    elif remove:
        related.remove(user)

def update_user_orgs(backend, details, user=None, *args, **kwargs):
    if False:
        print('Hello World!')
    '\n    Update organization memberships for the given user based on mapping rules\n    defined in settings.\n    '
    if not user:
        return
    org_map = backend.setting('ORGANIZATION_MAP') or {}
    for (org_name, org_opts) in org_map.items():
        organization_alias = org_opts.get('organization_alias')
        if organization_alias:
            organization_name = organization_alias
        else:
            organization_name = org_name
        org = get_or_create_org_with_default_galaxy_cred(name=organization_name)
        remove = bool(org_opts.get('remove', True))
        admins_expr = org_opts.get('admins', None)
        remove_admins = bool(org_opts.get('remove_admins', remove))
        _update_m2m_from_expression(user, org.admin_role.members, admins_expr, remove_admins)
        users_expr = org_opts.get('users', None)
        remove_users = bool(org_opts.get('remove_users', remove))
        _update_m2m_from_expression(user, org.member_role.members, users_expr, remove_users)

def update_user_teams(backend, details, user=None, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Update team memberships for the given user based on mapping rules defined\n    in settings.\n    '
    if not user:
        return
    from awx.main.models import Team
    team_map = backend.setting('TEAM_MAP') or {}
    for (team_name, team_opts) in team_map.items():
        if 'organization' not in team_opts:
            continue
        org = get_or_create_org_with_default_galaxy_cred(name=team_opts['organization'])
        team = Team.objects.get_or_create(name=team_name, organization=org)[0]
        users_expr = team_opts.get('users', None)
        remove = bool(team_opts.get('remove', True))
        _update_m2m_from_expression(user, team.member_role.members, users_expr, remove)