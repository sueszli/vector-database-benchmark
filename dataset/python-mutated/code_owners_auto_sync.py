from rest_framework.exceptions import NotFound
from sentry.models.commit import Commit
from sentry.models.integrations.repository_project_path_config import RepositoryProjectPathConfig
from sentry.models.organization import Organization
from sentry.models.projectcodeowners import ProjectCodeOwners
from sentry.models.projectownership import ProjectOwnership
from sentry.notifications.notifications.codeowners_auto_sync import AutoSyncNotification
from sentry.silo import SiloMode
from sentry.tasks.base import instrumented_task, retry

@instrumented_task(name='sentry.tasks.code_owners_auto_sync', queue='code_owners', default_retry_delay=60 * 5, max_retries=1, silo_mode=SiloMode.REGION)
@retry(on=(Commit.DoesNotExist,))
def code_owners_auto_sync(commit_id: int, **kwargs):
    if False:
        while True:
            i = 10
    from django.db.models import BooleanField, Case, Exists, OuterRef, Subquery, When
    from sentry.api.endpoints.organization_code_mapping_codeowners import get_codeowner_contents
    commit = Commit.objects.get(id=commit_id)
    code_mappings = RepositoryProjectPathConfig.objects.filter(repository_id=commit.repository_id, project__organization_id=commit.organization_id).annotate(ownership_exists=Exists(ProjectOwnership.objects.filter(project=OuterRef('project')))).annotate(codeowners_auto_sync=Case(When(ownership_exists=False, then=True), When(ownership_exists=True, then=Subquery(ProjectOwnership.objects.filter(project=OuterRef('project')).values_list('codeowners_auto_sync', flat=True)[:1])), default=True, output_field=BooleanField())).annotate(has_codeowners=Exists(ProjectCodeOwners.objects.filter(repository_project_path_config=OuterRef('pk')))).filter(codeowners_auto_sync=True, has_codeowners=True)
    for code_mapping in code_mappings:
        try:
            codeowner_contents = get_codeowner_contents(code_mapping)
        except (NotImplementedError, NotFound):
            codeowner_contents = None
        if not codeowner_contents:
            return AutoSyncNotification(code_mapping.project).send()
        codeowners = ProjectCodeOwners.objects.get(repository_project_path_config=code_mapping)
        organization = Organization.objects.get(id=code_mapping.organization_id)
        codeowners.update_schema(organization=organization, raw=codeowner_contents['raw'])