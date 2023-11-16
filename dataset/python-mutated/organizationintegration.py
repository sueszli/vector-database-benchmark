from sentry.constants import ObjectStatus
from sentry.models.integrations.organization_integration import OrganizationIntegration
from sentry.services.hybrid_cloud.repository import repository_service
from ..base import ModelDeletionTask, ModelRelation

class OrganizationIntegrationDeletionTask(ModelDeletionTask):

    def should_proceed(self, instance):
        if False:
            return 10
        return instance.status in {ObjectStatus.DELETION_IN_PROGRESS, ObjectStatus.PENDING_DELETION}

    def get_child_relations(self, instance):
        if False:
            i = 10
            return i + 15
        from sentry.models.identity import Identity
        relations = []
        if instance.default_auth_id:
            relations.append(ModelRelation(Identity, {'id': instance.default_auth_id}))
        return relations

    def delete_instance(self, instance: OrganizationIntegration):
        if False:
            for i in range(10):
                print('nop')
        repository_service.disassociate_organization_integration(organization_id=instance.organization_id, organization_integration_id=instance.id, integration_id=instance.integration_id)
        return super().delete_instance(instance)