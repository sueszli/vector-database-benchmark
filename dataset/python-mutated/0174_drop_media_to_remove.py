from django.db import migrations
from ...core.tasks import delete_files_from_storage_task

def iter_media_to_delete(qs):
    if False:
        print('Hello World!')
    for media in qs.iterator():
        yield media.image.name
        for thumbnail in media.thumbnails.all():
            yield thumbnail.image.name

def drop_media_to_remove(apps, _schema_editor):
    if False:
        print('Hello World!')
    ProductMedia = apps.get_model('product', 'ProductMedia')
    image_paths_to_delete = []
    item_per_task = 100
    for path in iter_media_to_delete(ProductMedia.objects.filter(to_remove=True)):
        image_paths_to_delete.append(path)
        if len(image_paths_to_delete) == item_per_task:
            delete_files_from_storage_task.delay(image_paths_to_delete)
            image_paths_to_delete.clear()
    if image_paths_to_delete:
        delete_files_from_storage_task.delay(image_paths_to_delete)
    ProductMedia.objects.filter(to_remove=True).delete()

class Migration(migrations.Migration):
    dependencies = [('product', '0173_create_default_category_and_product_type'), ('thumbnail', '0001_initial')]
    operations = [migrations.RunPython(drop_media_to_remove, reverse_code=migrations.RunPython.noop)]