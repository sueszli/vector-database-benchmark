import re
from django.db import migrations

def build_refs(apps, schema_editor):
    if False:
        while True:
            i = 10
    '\n    Rebuild the integer "reference fields" for existing Build objects\n    '
    BuildOrder = apps.get_model('build', 'build')
    for build in BuildOrder.objects.all():
        ref = 0
        result = re.match('^(\\d+)', build.reference)
        if result and len(result.groups()) == 1:
            try:
                ref = int(result.groups()[0])
            except Exception:
                ref = 0
        if ref > 2147483647:
            ref = 2147483647
        build.reference_int = ref
        build.save()

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('build', '0031_build_reference_int')]
    operations = [migrations.RunPython(build_refs, reverse_code=migrations.RunPython.noop)]