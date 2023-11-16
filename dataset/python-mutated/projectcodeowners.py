from __future__ import annotations
import logging
from typing import Sequence
from django.db import models
from django.db.models.signals import post_delete, post_save, pre_save
from django.utils import timezone
from rest_framework.exceptions import ValidationError
from sentry import analytics
from sentry.backup.scopes import RelocationScope
from sentry.db.models import FlexibleForeignKey, JSONField, Model, region_silo_only_model, sane_repr
from sentry.models.organization import Organization
from sentry.ownership.grammar import convert_codeowners_syntax, create_schema_from_issue_owners
from sentry.utils.cache import cache
logger = logging.getLogger(__name__)
READ_CACHE_DURATION = 3600

@region_silo_only_model
class ProjectCodeOwners(Model):
    __relocation_scope__ = RelocationScope.Excluded
    project = FlexibleForeignKey('sentry.Project', db_constraint=False)
    repository_project_path_config = FlexibleForeignKey('sentry.RepositoryProjectPathConfig', unique=True, on_delete=models.PROTECT)
    raw = models.TextField(null=True)
    schema = JSONField(null=True)
    date_updated = models.DateTimeField(default=timezone.now)
    date_added = models.DateTimeField(default=timezone.now)

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_projectcodeowners'
    __repr__ = sane_repr('project_id', 'id')

    @classmethod
    def get_cache_key(self, project_id: int) -> str:
        if False:
            return 10
        return f'projectcodeowners_project_id:1:{project_id}'

    @classmethod
    def get_codeowners_cached(self, project_id: int) -> ProjectCodeOwners | None:
        if False:
            return 10
        "\n        Cached read access to sentry_projectcodeowners.\n\n        This method implements a negative cache which saves us\n        a pile of read queries in post_processing as most projects\n        don't have CODEOWNERS.\n        "
        cache_key = self.get_cache_key(project_id)
        code_owners = cache.get(cache_key)
        if code_owners is None:
            query = self.objects.filter(project_id=project_id).order_by('-date_added') or ()
            code_owners = self.merge_code_owners_list(code_owners_list=query) if query else query
            cache.set(cache_key, code_owners, READ_CACHE_DURATION)
        return code_owners or None

    @classmethod
    def merge_code_owners_list(self, code_owners_list: Sequence[ProjectCodeOwners]) -> ProjectCodeOwners | None:
        if False:
            i = 10
            return i + 15
        '\n        Merge list of code_owners into a single code_owners object concatenating\n        all the rules. We assume schema version is constant.\n        '
        merged_code_owners: ProjectCodeOwners | None = None
        for code_owners in code_owners_list:
            if code_owners.schema:
                if merged_code_owners is None:
                    merged_code_owners = code_owners
                    continue
                merged_code_owners.schema['rules'] = [*merged_code_owners.schema['rules'], *code_owners.schema['rules']]
        return merged_code_owners

    def update_schema(self, organization: Organization, raw: str | None=None) -> None:
        if False:
            return 10
        '\n        Updating the schema goes through the following steps:\n        1. parsing the original codeowner file to get the associations\n        2. convert the codeowner file to the ownership syntax\n        3. convert the ownership syntax to the schema\n        '
        from sentry.api.validators.project_codeowners import validate_codeowners_associations
        from sentry.utils.codeowners import MAX_RAW_LENGTH
        if raw and self.raw != raw:
            self.raw = raw
        if not self.raw:
            return
        if len(self.raw) > MAX_RAW_LENGTH:
            analytics.record('codeowners.max_length_exceeded', organization_id=organization.id)
            logger.warning({'raw': f'Raw needs to be <= {MAX_RAW_LENGTH} characters in length'})
            return
        (associations, _) = validate_codeowners_associations(self.raw, self.project)
        issue_owner_rules = convert_codeowners_syntax(codeowners=self.raw, associations=associations, code_mapping=self.repository_project_path_config)
        try:
            schema = create_schema_from_issue_owners(issue_owners=issue_owner_rules, project_id=self.project.id)
            if schema:
                self.schema = schema
                self.save()
        except ValidationError:
            return

def modify_date_updated(instance, **kwargs):
    if False:
        while True:
            i = 10
    if instance.id is None:
        return
    instance.date_updated = timezone.now()

def process_resource_change(instance, change, **kwargs):
    if False:
        while True:
            i = 10
    from sentry.models.groupowner import GroupOwner
    from sentry.models.projectownership import ProjectOwnership
    cache.set(ProjectCodeOwners.get_cache_key(instance.project_id), None, READ_CACHE_DURATION)
    ownership = ProjectOwnership.get_ownership_cached(instance.project_id)
    if not ownership:
        ownership = ProjectOwnership(project_id=instance.project_id)
    autoassignment_types = ProjectOwnership._get_autoassignment_types(ownership)
    if ownership.auto_assignment:
        GroupOwner.invalidate_autoassigned_owner_cache(instance.project_id, autoassignment_types)
    GroupOwner.invalidate_debounce_issue_owners_evaluation_cache(instance.project_id)
pre_save.connect(modify_date_updated, sender=ProjectCodeOwners, dispatch_uid='projectcodeowners_modify_date_updated', weak=False)
post_save.connect(lambda instance, **kwargs: process_resource_change(instance, 'updated', **kwargs), sender=ProjectCodeOwners, weak=False)
post_delete.connect(lambda instance, **kwargs: process_resource_change(instance, 'deleted', **kwargs), sender=ProjectCodeOwners, weak=False)