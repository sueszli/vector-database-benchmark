from django.db import migrations

def grow_tail(x, y):
    if False:
        while True:
            i = 10
    pass

def feed(x, y):
    if False:
        return 10
    'Feed salamander.'
    pass

class Migration(migrations.Migration):
    dependencies = [('migrations', '0004_fourth')]
    operations = [migrations.RunPython(migrations.RunPython.noop), migrations.RunPython(grow_tail), migrations.RunPython(feed, migrations.RunPython.noop)]