"""
Module to centralize event reporting on the server-side.
"""
from typing import Dict, List, Optional
import posthoganalytics
from posthog.models import Organization, User
from posthog.models.team import Team
from posthog.settings import SITE_URL
from posthog.utils import get_instance_realm

def report_user_signed_up(user: User, is_instance_first_user: bool, is_organization_first_user: bool, new_onboarding_enabled: bool=False, backend_processor: str='', social_provider: str='', user_analytics_metadata: Optional[dict]=None, org_analytics_metadata: Optional[dict]=None, role_at_organization: str='', referral_source: str='') -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Reports that a new user has joined. Only triggered when a new user is actually created (i.e. when an existing user\n    joins a new organization, this event is **not** triggered; see `report_user_joined_organization`).\n    '
    props = {'is_first_user': is_instance_first_user, 'is_organization_first_user': is_organization_first_user, 'new_onboarding_enabled': new_onboarding_enabled, 'signup_backend_processor': backend_processor, 'signup_social_provider': social_provider, 'realm': get_instance_realm(), 'role_at_organization': role_at_organization, 'referral_source': referral_source, 'is_email_verified': user.is_email_verified}
    if user_analytics_metadata is not None:
        props.update(user_analytics_metadata)
    if org_analytics_metadata is not None:
        for (k, v) in org_analytics_metadata.items():
            props[f'org__{k}'] = v
    props = {**props, '$set': {**props, **user.get_analytics_metadata()}}
    posthoganalytics.capture(user.distinct_id, 'user signed up', properties=props, groups=groups(user.organization, user.team))

def report_user_verified_email(current_user: User) -> None:
    if False:
        return 10
    '\n    Triggered after a user verifies their email address.\n    '
    posthoganalytics.capture(current_user.distinct_id, 'user verified email', properties={'$set': current_user.get_analytics_metadata()})

def alias_invite_id(user: User, invite_id: str) -> None:
    if False:
        return 10
    posthoganalytics.alias(user.distinct_id, f'invite_{invite_id}')

def report_user_joined_organization(organization: Organization, current_user: User) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Triggered after an already existing user joins an already existing organization.\n    '
    posthoganalytics.capture(current_user.distinct_id, 'user joined organization', properties={'organization_id': str(organization.id), 'user_number_of_org_membership': current_user.organization_memberships.count(), 'org_current_invite_count': organization.active_invites.count(), 'org_current_project_count': organization.teams.count(), 'org_current_members_count': organization.memberships.count(), '$set': current_user.get_analytics_metadata()}, groups=groups(organization))

def report_user_logged_in(user: User, social_provider: str='') -> None:
    if False:
        i = 10
        return i + 15
    '\n    Reports that a user has logged in to PostHog.\n    '
    posthoganalytics.capture(user.distinct_id, 'user logged in', properties={'social_provider': social_provider}, groups=groups(user.current_organization, user.current_team))

def report_user_updated(user: User, updated_attrs: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Reports a user has been updated. This includes current_team, current_organization & password.\n    '
    updated_attrs.sort()
    posthoganalytics.capture(user.distinct_id, 'user updated', properties={'updated_attrs': updated_attrs}, groups=groups(user.current_organization, user.current_team))

def report_user_password_reset(user: User) -> None:
    if False:
        return 10
    '\n    Reports a user resetting their password.\n    '
    posthoganalytics.capture(user.distinct_id, 'user password reset', groups=groups(user.current_organization, user.current_team))

def report_team_member_invited(inviting_user: User, invite_id: str, name_provided: bool, current_invite_count: int, current_member_count: int, is_bulk: bool, email_available: bool) -> None:
    if False:
        return 10
    '\n    Triggered after a user creates an **individual** invite for a new team member. See `report_bulk_invited`\n    for bulk invite creation.\n    '
    properties = {'name_provided': name_provided, 'current_invite_count': current_invite_count, 'current_member_count': current_member_count, 'email_available': email_available, 'is_bulk': is_bulk}
    posthoganalytics.capture(inviting_user.distinct_id, 'team invite executed', properties=properties, groups=groups(inviting_user.current_organization, inviting_user.current_team))
    posthoganalytics.capture(f'invite_{invite_id}', 'user invited', properties=properties, groups=groups(inviting_user.current_organization, None))

def report_bulk_invited(user: User, invitee_count: int, name_count: int, current_invite_count: int, current_member_count: int, email_available: bool) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Triggered after a user bulk creates invites for another user.\n    '
    posthoganalytics.capture(user.distinct_id, 'bulk invite executed', properties={'invitee_count': invitee_count, 'name_count': name_count, 'current_invite_count': current_invite_count, 'current_member_count': current_member_count, 'email_available': email_available}, groups=groups(user.current_organization, user.current_team))

def report_user_action(user: User, event: str, properties: Dict={}):
    if False:
        while True:
            i = 10
    posthoganalytics.capture(user.distinct_id, event, properties=properties, groups=groups(user.current_organization, user.current_team))

def report_organization_deleted(user: User, organization: Organization):
    if False:
        return 10
    posthoganalytics.capture(user.distinct_id, 'organization deleted', organization.get_analytics_metadata(), groups=groups(organization))

def groups(organization: Optional[Organization]=None, team: Optional[Team]=None):
    if False:
        return 10
    result = {'instance': SITE_URL}
    if organization is not None:
        result['organization'] = str(organization.pk)
        if organization.customer_id:
            result['customer'] = organization.customer_id
    elif team is not None and team.organization_id:
        result['organization'] = str(team.organization_id)
    if team is not None:
        result['project'] = str(team.uuid)
    return result

def report_team_action(team: Team, event: str, properties: Dict={}, group_properties: Optional[Dict]=None):
    if False:
        while True:
            i = 10
    '\n    For capturing events where it is unclear which user was the core actor we can use the team instead\n    '
    posthoganalytics.capture(str(team.uuid), event, properties=properties, groups=groups(team=team))
    if group_properties:
        posthoganalytics.group_identify('team', str(team.id), properties=group_properties)

def report_organization_action(organization: Organization, event: str, properties: Dict={}, group_properties: Optional[Dict]=None):
    if False:
        return 10
    '\n    For capturing events where it is unclear which user was the core actor we can use the organization instead\n    '
    posthoganalytics.capture(str(organization.id), event, properties=properties, groups=groups(organization=organization))
    if group_properties:
        posthoganalytics.group_identify('organization', str(organization.id), properties=group_properties)