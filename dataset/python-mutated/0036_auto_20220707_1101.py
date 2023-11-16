from django.db import migrations

def update_build_reference(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    "Update the build order reference.\n\n    Ref: https://github.com/inventree/InvenTree/pull/3267\n\n    Performs the following steps:\n    - Extract existing 'prefix' value\n    - Generate a build order pattern based on the prefix value\n    - Update any existing build order references with the specified prefix\n    "
    InvenTreeSetting = apps.get_model('common', 'inventreesetting')
    try:
        prefix = InvenTreeSetting.objects.get(key='BUILDORDER_REFERENCE_PREFIX').value
    except Exception:
        prefix = 'BO-'
    pattern = prefix + '{ref:04d}'
    try:
        setting = InvenTreeSetting.objects.get(key='BUILDORDER_REFERENCE_PATTERN')
        setting.value = pattern
        setting.save()
    except InvenTreeSetting.DoesNotExist:
        setting = InvenTreeSetting.objects.create(key='BUILDORDER_REFERENCE_PATTERN', value=pattern)
    Build = apps.get_model('build', 'build')
    n = 0
    for build in Build.objects.all():
        if not build.reference.startswith(prefix):
            build.reference = prefix + build.reference
            build.save()
            n += 1
    if n > 0:
        print(f'Updated reference field for {n} BuildOrder objects')

class Migration(migrations.Migration):
    dependencies = [('build', '0035_alter_build_notes')]
    operations = [migrations.RunPython(update_build_reference, reverse_code=migrations.RunPython.noop)]