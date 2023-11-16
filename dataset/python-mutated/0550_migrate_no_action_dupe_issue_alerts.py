from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

class ObjectStatus:
    ACTIVE = 0
    HIDDEN = 1
    PENDING_DELETION = 2
    DELETION_IN_PROGRESS = 3
    DISABLED = 1

def has_duplicate_rule(apps, rule_data, project, rule_id):
    if False:
        i = 10
        return i + 15
    Rule = apps.get_model('sentry', 'Rule')
    matchers = {key for key in list(rule_data.keys()) if key not in ('name', 'user_id')}
    extra_fields = ['actions', 'environment']
    matchers.update(extra_fields)
    existing_rules = Rule.objects.exclude(id=rule_id).filter(project=project, status=ObjectStatus.ACTIVE)
    for existing_rule in existing_rules:
        keys = 0
        matches = 0
        for matcher in matchers:
            if existing_rule.data.get(matcher) and rule_data.get(matcher):
                keys += 1
                if existing_rule.data[matcher] == rule_data[matcher]:
                    matches += 1
            elif matcher in extra_fields:
                if not existing_rule.data.get(matcher) and (not rule_data.get(matcher)):
                    continue
                elif matcher == 'environment':
                    if existing_rule.environment_id and rule_data.get(matcher):
                        keys += 1
                        if existing_rule.environment_id == rule_data.get(matcher):
                            matches += 1
                    else:
                        keys += 1
                else:
                    keys += 1
        if keys == matches:
            return True
    return False

def migrate_bad_rules(apps, schema_editor):
    if False:
        return 10
    Rule = apps.get_model('sentry', 'Rule')
    for rule in RangeQuerySetWrapperWithProgressBar(Rule.objects.all()):
        if not rule.data.get('actions', []) or has_duplicate_rule(apps, rule.data, rule.project, rule.id):
            rule.status = ObjectStatus.DISABLED
            rule.save(update_fields=['status'])

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0549_re_add_groupsubscription_columns')]
    operations = [migrations.RunPython(migrate_bad_rules, migrations.RunPython.noop, hints={'tables': ['sentry_rule']})]