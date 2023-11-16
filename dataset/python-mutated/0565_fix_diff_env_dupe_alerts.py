from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

class ObjectStatus:
    ACTIVE = 0
    HIDDEN = 1
    PENDING_DELETION = 2
    DELETION_IN_PROGRESS = 3
    DISABLED = 1

def is_incorrectly_disabled(apps, rule):
    if False:
        print('Hello World!')
    Rule = apps.get_model('sentry', 'Rule')
    matchers = {key for key in list(rule.data.keys()) if key not in ('name', 'user_id')}
    extra_fields = ['actions', 'environment']
    matchers.update(extra_fields)
    existing_rules = Rule.objects.exclude(id=rule.id).filter(project=rule.project)
    for existing_rule in existing_rules:
        keys = 0
        matches = 0
        for matcher in matchers:
            if existing_rule.data.get(matcher) and rule.data.get(matcher):
                keys += 1
                if existing_rule.data[matcher] == rule.data[matcher]:
                    matches += 1
            elif matcher in extra_fields:
                if matcher == 'environment':
                    if existing_rule.environment_id and rule.environment_id:
                        keys += 1
                        if existing_rule.environment_id == rule.environment_id:
                            matches += 1
                    else:
                        keys += 1
                elif not existing_rule.data.get(matcher) and (not rule.data.get(matcher)):
                    continue
                else:
                    keys += 1
        if keys == matches:
            return False
    return True

def fix_diff_env_rules(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Rule = apps.get_model('sentry', 'Rule')
    for rule in RangeQuerySetWrapperWithProgressBar(Rule.objects.all()):
        if rule.status == ObjectStatus.DISABLED and rule.environment_id:
            if is_incorrectly_disabled(apps, rule):
                rule.status = ObjectStatus.ACTIVE
                rule.save(update_fields=['status'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0564_commitfilechange_delete_language_column')]
    operations = [migrations.RunPython(fix_diff_env_rules, migrations.RunPython.noop, hints={'tables': ['sentry_rule']})]