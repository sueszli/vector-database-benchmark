from django.db import migrations

def get_bom_items_for_part(part, Part, BomItem):
    if False:
        for i in range(10):
            print('nop')
    ' Return a list of all BOM items for a given part.\n\n    Note that we cannot use the ORM here (as we are inside a data migration),\n    so we *copy* the logic from the Part class.\n\n    This is a snapshot of the Part.get_bom_items() method as of 2023-05-29\n    '
    bom_items = set()
    for bom_item in BomItem.objects.filter(part=part):
        bom_items.add(bom_item)
    parents = Part.objects.filter(tree_id=part.tree_id, level__lt=part.level, lft__lt=part.lft, rght__gt=part.rght)
    for bom_item in BomItem.objects.filter(part__in=parents, inherited=True):
        bom_items.add(bom_item)
    return list(bom_items)

def add_lines_to_builds(apps, schema_editor):
    if False:
        print('Hello World!')
    'Create BuildOrderLine objects for existing build orders'
    Build = apps.get_model('build', 'Build')
    BuildLine = apps.get_model('build', 'BuildLine')
    Part = apps.get_model('part', 'Part')
    BomItem = apps.get_model('part', 'BomItem')
    build_lines = []
    builds = Build.objects.all()
    if builds.count() > 0:
        print(f'Creating BuildOrderLine objects for {builds.count()} existing builds')
    for build in builds:
        bom_items = get_bom_items_for_part(build.part, Part, BomItem)
        for item in bom_items:
            build_lines.append(BuildLine(build=build, bom_item=item, quantity=item.quantity * build.quantity))
    if len(build_lines) > 0:
        BuildLine.objects.bulk_create(build_lines)
        print(f'Created {len(build_lines)} BuildOrderLine objects for existing builds')

def remove_build_lines(apps, schema_editor):
    if False:
        print('Hello World!')
    'Remove BuildOrderLine objects from the database'
    BuildLine = apps.get_model('build', 'BuildLine')
    n = BuildLine.objects.all().count()
    BuildLine.objects.all().delete()
    if n > 0:
        print(f'Removed {n} BuildOrderLine objects')

class Migration(migrations.Migration):
    dependencies = [('build', '0043_buildline')]
    operations = [migrations.RunPython(add_lines_to_builds, reverse_code=remove_build_lines)]