from __future__ import annotations
from abc import ABCMeta
from typing import TYPE_CHECKING, Iterable, MutableMapping, Optional
from sentry import roles
from sentry.models.organizationmember import OrganizationMember
from sentry.roles.manager import OrganizationRole
from sentry.services.hybrid_cloud.actor import ActorType, RpcActor
from sentry.services.hybrid_cloud.user import RpcUser
from sentry.services.hybrid_cloud.user.service import user_service
if TYPE_CHECKING:
    from sentry.models.organization import Organization

class RoleBasedRecipientStrategy(metaclass=ABCMeta):
    member_by_user_id: MutableMapping[int, OrganizationMember] = {}
    role: Optional[OrganizationRole] = None
    scope: Optional[str] = None

    def __init__(self, organization: Organization):
        if False:
            for i in range(10):
                print('nop')
        self.organization = organization

    def get_member(self, user: RpcUser | RpcActor) -> OrganizationMember:
        if False:
            return 10
        actor = RpcActor.from_object(user)
        if actor.actor_type != ActorType.USER:
            raise OrganizationMember.DoesNotExist()
        user_id = actor.id
        if user_id not in self.member_by_user_id:
            self.member_by_user_id[user_id] = OrganizationMember.objects.get(user_id=user_id, organization=self.organization)
        return self.member_by_user_id[user_id]

    def set_member_in_cache(self, member: OrganizationMember) -> None:
        if False:
            i = 10
            return i + 15
        '\n        A way to set a member in a cache to avoid a query.\n        '
        self.member_by_user_id[member.user_id] = member

    def determine_recipients(self) -> list[RpcUser]:
        if False:
            while True:
                i = 10
        members = self.determine_member_recipients()
        for member in members:
            self.set_member_in_cache(member)
        return user_service.get_many(filter={'user_ids': [member.user_id for member in members]})

    def determine_member_recipients(self) -> Iterable[OrganizationMember]:
        if False:
            return 10
        '\n        Depending on the type of request this might be all organization owners,\n        a specific person, or something in between.\n        '
        members: Iterable[OrganizationMember] = OrganizationMember.objects.get_contactable_members_for_org(self.organization.id)
        if not self.scope and (not self.role):
            return members
        valid_roles = []
        if self.role and (not self.scope):
            valid_roles = [self.role.id]
        elif self.scope:
            valid_roles = [r.id for r in roles.get_all() if r.has_scope(self.scope)]
        member_ids = self.organization.get_members_with_org_roles(roles=valid_roles).values_list('id', flat=True)
        members = members.filter(id__in=member_ids)
        return members

    def build_notification_footer_from_settings_url(self, settings_url: str) -> str:
        if False:
            i = 10
            return i + 15
        if self.scope and (not self.role):
            return f'You are receiving this notification because you have the scope {self.scope} | {settings_url}'
        role_name = 'Member'
        if self.role:
            role_name = self.role.name
        return f"You are receiving this notification because you're listed as an organization {role_name} | {settings_url}"