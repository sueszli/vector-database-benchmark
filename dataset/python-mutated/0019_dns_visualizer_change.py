from django.db import migrations

def migrate(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    VisualizerConfig = apps.get_model('visualizers_manager', 'VisualizerConfig')
    visualizer = VisualizerConfig.objects.get(name='DNS')
    visualizer.analyzers.remove(*['GoogleSafebrowsing', 'GoogleWebRisk'])
    visualizer.full_clean()
    visualizer.save()

def reverse_migrate(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    VisualizerConfig = apps.get_model('visualizers_manager', 'VisualizerConfig')
    visualizer = VisualizerConfig.objects.get(name='DNS')
    visualizer.analyzers.add(*['GoogleSafebrowsing', 'GoogleWebRisk'])
    visualizer.full_clean()
    visualizer.save()

class Migration(migrations.Migration):
    dependencies = [('visualizers_manager', '0018_visualizer_config')]
    operations = [migrations.RunPython(migrate, reverse_migrate)]