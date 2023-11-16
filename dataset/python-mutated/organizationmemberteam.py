from __future__ import annotations
from typing import Any, ClassVar, FrozenSet, Mapping
from django.db import models
from sentry import features, roles
from sentry.backup.scopes import RelocationScope
from sentry.db.models import BoundedAutoField, FlexibleForeignKey, region_silo_only_model, sane_repr
from sentry.db.models.outboxes import RegionOutboxProducingManager, ReplicatedRegionModel
from sentry.models.outbox import OutboxCategory, RegionOutboxBase
from sentry.roles import team_roles
from sentry.roles.manager import TeamRole

@region_silo_only_model
class OrganizationMemberTeam(ReplicatedRegionModel):
    """
    Identifies relationships between organization members and the teams they are on.
    """
    objects: ClassVar[RegionOutboxProducingManager[OrganizationMemberTeam]] = RegionOutboxProducingManager()
    __relocation_scope__ = RelocationScope.Organization
    category = OutboxCategory.ORGANIZATION_MEMBER_TEAM_UPDATE
    id = BoundedAutoField(primary_key=True)
    team = FlexibleForeignKey('sentry.Team')
    organizationmember = FlexibleForeignKey('sentry.OrganizationMember')
    is_active = models.BooleanField(default=True)
    role = models.CharField(max_length=32, null=True, blank=True)

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_organizationmember_teams'
        unique_together = (('team', 'organizationmember'),)
    __repr__ = sane_repr('team_id', 'organizationmember_id')

    def outbox_for_update(self, shard_identifier: int | None=None) -> RegionOutboxBase:
        if False:
            return 10
        return super().outbox_for_update(shard_identifier=self.organizationmember.organization_id if shard_identifier is None else shard_identifier)

    def handle_async_replication(self, shard_identifier: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        from sentry.services.hybrid_cloud.organization.serial import serialize_rpc_organization_member_team
        from sentry.services.hybrid_cloud.replica.service import control_replica_service
        control_replica_service.upsert_replicated_organization_member_team(omt=serialize_rpc_organization_member_team(self))

    @classmethod
    def handle_async_deletion(cls, identifier: int, shard_identifier: int, payload: Mapping[str, Any] | None) -> None:
        if False:
            while True:
                i = 10
        from sentry.services.hybrid_cloud.replica.service import control_replica_service
        control_replica_service.remove_replicated_organization_member_team(organization_id=shard_identifier, organization_member_team_id=identifier)

    def get_audit_log_data(self):
        if False:
            i = 10
            return i + 15
        return {'team_slug': self.team.slug, 'member_id': self.organizationmember_id, 'email': self.organizationmember.get_email(), 'is_active': self.is_active}

    def get_team_role(self) -> TeamRole:
        if False:
            for i in range(10):
                print('nop')
        "Get this member's team-level role.\n\n        If the role field is null, resolve to the minimum team role given by this\n        member's organization role.\n        "
        highest_org_role = self.organizationmember.get_all_org_roles_sorted()[0].id
        minimum_role = roles.get_minimum_team_role(highest_org_role)
        if self.role and features.has('organizations:team-roles', self.organizationmember.organization):
            team_role = team_roles.get(self.role)
            if team_role.priority > minimum_role.priority:
                return team_role
        return minimum_role

    def get_scopes(self) -> FrozenSet[str]:
        if False:
            i = 10
            return i + 15
        "Get the scopes belonging to this member's team-level role."
        if features.has('organizations:team-roles', self.organizationmember.organization):
            return self.organizationmember.organization.get_scopes(self.get_team_role())
        return frozenset()