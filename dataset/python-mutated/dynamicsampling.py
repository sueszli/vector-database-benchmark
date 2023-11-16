import hashlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from django.db import connections, models, router, transaction
from django.db.models import Q
from django.utils import timezone
from sentry.backup.scopes import RelocationScope
from sentry.constants import ObjectStatus
from sentry.db.models import FlexibleForeignKey, Model, region_silo_only_model
from sentry.db.models.fields.hybrid_cloud_foreign_key import HybridCloudForeignKey
from sentry.utils import json, metrics
if TYPE_CHECKING:
    from sentry.models.organization import Organization
    from sentry.models.project import Project
MAX_CUSTOM_RULES = 2000
CUSTOM_RULE_START = 3000
MAX_CUSTOM_RULES_PER_PROJECT = 50
CUSTOM_RULE_DATE_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

class TooManyRules(ValueError):
    """
    Raised when a there is already the max number of rules active for an organization
    """
    pass

def get_rule_hash(condition: Any, project_ids: Sequence[int]) -> str:
    if False:
        while True:
            i = 10
    '\n    Returns the hash of the rule based on the condition and projects\n    '
    condition_string = to_order_independent_string(condition)
    project_string = to_order_independent_string(list(project_ids))
    rule_string = f'{condition_string}-{project_string}'
    return hashlib.sha1(rule_string.encode('utf-8')).hexdigest()

def to_order_independent_string(val: Any) -> str:
    if False:
        print('Hello World!')
    "\n    Converts a value in an order independent string and then hashes it\n\n    Note: this will insure the same repr is generated for ['x', 'y'] and ['y', 'x']\n        Also the same repr is generated for {'x': 1, 'y': 2} and {'y': 2, 'x': 1}\n    "
    ret_val = ''
    if isinstance(val, Mapping):
        for key in sorted(val.keys()):
            ret_val += f'{key}:{to_order_independent_string(val[key])}-'
    elif isinstance(val, (list, tuple)):
        vals = sorted([to_order_independent_string(item) for item in val])
        for item in vals:
            ret_val += f'{item}-'
    else:
        ret_val = str(val)
    return ret_val

@region_silo_only_model
class CustomDynamicSamplingRuleProject(Model):
    """
    Many-to-many relationship between a custom dynamic sampling rule and a project.
    """
    __relocation_scope__ = RelocationScope.Organization
    custom_dynamic_sampling_rule = FlexibleForeignKey('sentry.CustomDynamicSamplingRule', on_delete=models.CASCADE)
    project = FlexibleForeignKey('sentry.Project', on_delete=models.CASCADE)

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_customdynamicsamplingruleproject'
        unique_together = (('custom_dynamic_sampling_rule', 'project'),)

@region_silo_only_model
class CustomDynamicSamplingRule(Model):
    """
    This represents a custom dynamic sampling rule that is created by the user based
    on a query (a.k.a. investigation rule).

    """
    __relocation_scope__ = RelocationScope.Organization
    date_added = models.DateTimeField(default=timezone.now)
    organization = FlexibleForeignKey('sentry.Organization', on_delete=models.CASCADE)
    projects = models.ManyToManyField('sentry.Project', related_name='custom_dynamic_sampling_rules', through=CustomDynamicSamplingRuleProject)
    is_active = models.BooleanField(default=True)
    is_org_level = models.BooleanField(default=False)
    rule_id = models.IntegerField(default=0)
    condition = models.TextField()
    sample_rate = models.FloatField(default=0.0)
    start_date = models.DateTimeField(default=timezone.now)
    end_date = models.DateTimeField()
    num_samples = models.IntegerField()
    condition_hash = models.CharField(max_length=40)
    query = models.TextField(null=True)
    created_by_id = HybridCloudForeignKey('sentry.User', on_delete='CASCADE', null=True, blank=True)
    notification_sent = models.BooleanField(null=True, blank=True)

    @property
    def external_rule_id(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the external rule id\n\n        For external users, i.e. Relay, we need to shift the ids since the slot we\n        have allocated starts at the offset specified in RESERVED_IDS.\n        '
        return self.rule_id + CUSTOM_RULE_START

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_customdynamicsamplingrule'
        indexes = [models.Index(fields=['organization'], name='org_idx', condition=Q(is_active=True)), models.Index(fields=['end_date'], name='end_date_idx', condition=Q(is_active=True)), models.Index(fields=['condition_hash'], name='condition_hash_idx', condition=Q(is_active=True))]

    @staticmethod
    def get_rule_for_org(condition: Any, organization_id: int, project_ids: Sequence[int]) -> Optional['CustomDynamicSamplingRule']:
        if False:
            i = 10
            return i + 15
        "\n        Returns an active rule for the given condition and organization if it exists otherwise None\n\n        Note: There should not be more than one active rule for a given condition and organization\n        This function doesn't verify this condition, it just returns the first one.\n        "
        rule_hash = get_rule_hash(condition, project_ids)
        rules = CustomDynamicSamplingRule.objects.filter(organization_id=organization_id, condition_hash=rule_hash, is_active=True, end_date__gt=timezone.now())[:1]
        return rules[0] if rules else None

    @staticmethod
    def update_or_create(condition: Any, start: datetime, end: datetime, project_ids: Sequence[int], organization_id: int, num_samples: int, sample_rate: float, query: str, created_by_id: Optional[int]=None) -> 'CustomDynamicSamplingRule':
        if False:
            return 10
        from sentry.models.organization import Organization
        from sentry.models.project import Project
        with transaction.atomic(router.db_for_write(CustomDynamicSamplingRule)):
            existing_rule = CustomDynamicSamplingRule.get_rule_for_org(condition, organization_id, project_ids)
            if existing_rule is not None:
                existing_rule.end_date = max(end, existing_rule.end_date)
                existing_rule.num_samples = max(num_samples, existing_rule.num_samples)
                existing_rule.sample_rate = max(sample_rate, existing_rule.sample_rate)
                existing_rule.save()
                return existing_rule
            else:
                projects = Project.objects.get_many_from_cache(project_ids)
                projects = list(projects)
                organization = Organization.objects.get_from_cache(id=organization_id)
                if CustomDynamicSamplingRule.per_project_limit_reached(projects, organization):
                    raise TooManyRules()
                rule_hash = get_rule_hash(condition, project_ids)
                is_org_level = len(project_ids) == 0
                condition_str = json.dumps(condition)
                rule = CustomDynamicSamplingRule.objects.create(organization_id=organization_id, condition=condition_str, sample_rate=sample_rate, start_date=start, end_date=end, num_samples=num_samples, condition_hash=rule_hash, is_active=True, is_org_level=is_org_level, query=query, notification_sent=False, created_by_id=created_by_id)
                rule.save()
                id = rule.assign_rule_id()
                if id > MAX_CUSTOM_RULES:
                    rule.delete()
                    raise TooManyRules()
                for project in projects:
                    CustomDynamicSamplingRuleProject.objects.create(custom_dynamic_sampling_rule=rule, project=project)
                return rule

    def assign_rule_id(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assigns the smallest rule id that is not taken in the\n        current organization.\n        '
        table_name = self._meta.db_table
        if self.id is None:
            raise ValueError('Cannot assign rule id to unsaved object')
        if self.rule_id != 0:
            raise ValueError('Cannot assign rule id to object that already has a rule id')
        now = timezone.now()
        raw_sql = f'UPDATE {table_name} SET rule_id = (    SELECT COALESCE ((SELECT MIN(rule_id) + 1  FROM {table_name} WHERE rule_id + 1 NOT IN (       SELECT rule_id FROM {table_name} WHERE organization_id = %s AND end_date > %s AND is_active)),1))  WHERE id = %s'
        with connections['default'].cursor() as cursor:
            cursor.execute(raw_sql, (self.organization.id, now, self.id))
        self.refresh_from_db()
        return self.rule_id

    @staticmethod
    def deactivate_old_rules() -> None:
        if False:
            print('Hello World!')
        '\n        Deactivates all rules expired rules (this is just an optimization to remove old rules from indexes).\n\n        This should be called periodically to clean up old rules (it is not necessary to call it for correctness,\n        just for performance)\n        '
        CustomDynamicSamplingRule.objects.filter(end_date__lt=timezone.now() - timedelta(minutes=1)).update(is_active=False)

    @staticmethod
    def get_project_rules(project: 'Project') -> Sequence['CustomDynamicSamplingRule']:
        if False:
            print('Hello World!')
        '\n        Returns all active project rules\n        '
        now = timezone.now()
        org_rules = CustomDynamicSamplingRule.objects.filter(is_active=True, is_org_level=True, organization=project.organization, end_date__gt=now, start_date__lt=now)[:MAX_CUSTOM_RULES_PER_PROJECT + 1]
        project_rules = CustomDynamicSamplingRule.objects.filter(is_active=True, projects__in=[project], end_date__gt=now, start_date__lt=now)[:MAX_CUSTOM_RULES_PER_PROJECT + 1]
        rules = project_rules.union(org_rules)[:MAX_CUSTOM_RULES_PER_PROJECT + 1]
        rules = list(rules)
        if len(rules) > MAX_CUSTOM_RULES_PER_PROJECT:
            metrics.incr('dynamic_sampling.custom_rules.overflow')
        return rules[:MAX_CUSTOM_RULES_PER_PROJECT]

    @staticmethod
    def deactivate_expired_rules():
        if False:
            print('Hello World!')
        '\n        Deactivates all rules that have expired\n        '
        CustomDynamicSamplingRule.objects.filter(end_date__lt=timezone.now(), is_active=True).update(is_active=False)

    @staticmethod
    def num_active_rules_for_project(project: 'Project') -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the number of active rules for the given project\n        '
        now = timezone.now()
        num_org_rules = CustomDynamicSamplingRule.objects.filter(is_active=True, is_org_level=True, organization=project.organization, end_date__gt=now, start_date__lte=now).count()
        num_proj_rules = CustomDynamicSamplingRule.objects.filter(is_active=True, is_org_level=False, projects__in=[project], end_date__gt=now, start_date__lte=now).count()
        return num_proj_rules + num_org_rules

    @staticmethod
    def per_project_limit_reached(projects: Sequence['Project'], organization: 'Organization') -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns True if the rule limit is reached for any of the given projects (or all\n        the projects in the organization if org level rule)\n        '
        projects = list(projects)
        if len(projects) == 0:
            org_projects = organization.project_set.filter(status=ObjectStatus.ACTIVE)
            projects = list(org_projects)
        for project in projects:
            num_rules = CustomDynamicSamplingRule.num_active_rules_for_project(project)
            if num_rules >= MAX_CUSTOM_RULES_PER_PROJECT:
                return True
        return False