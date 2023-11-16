from django.db import migrations
import saleor.core.db.fields
import saleor.core.utils.editorjs

def update_empty_content_field(apps, schema_editor):
    if False:
        return 10
    Page = apps.get_model('page', 'Page')
    PageTranslation = apps.get_model('page', 'PageTranslation')
    Page.objects.filter(content={}).update(content=None)
    PageTranslation.objects.filter(content={}).update(content=None)

class Migration(migrations.Migration):
    dependencies = [('page', '0019_auto_20210125_0905')]
    operations = [migrations.AlterField(model_name='page', name='content', field=saleor.core.db.fields.SanitizedJSONField(blank=True, null=True, sanitizer=saleor.core.utils.editorjs.clean_editor_js)), migrations.AlterField(model_name='pagetranslation', name='content', field=saleor.core.db.fields.SanitizedJSONField(blank=True, null=True, sanitizer=saleor.core.utils.editorjs.clean_editor_js)), migrations.RunPython(update_empty_content_field, migrations.RunPython.noop)]