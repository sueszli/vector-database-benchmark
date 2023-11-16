import django.contrib.postgres.indexes
import django.contrib.postgres.search
from django.db import migrations, models

def parse_draftjs_content_to_string(definitions):
    if False:
        while True:
            i = 10
    string = ''
    blocks = definitions.get('blocks')
    if not blocks or not isinstance(blocks, list):
        return ''
    for block in blocks:
        text = block.get('text')
        if not text:
            continue
        string += f'{text} '
    return string

def parse_description_json_field(apps, schema):
    if False:
        while True:
            i = 10
    Product = apps.get_model('product', 'Product')
    for product in Product.objects.iterator():
        product.description_plaintext = parse_draftjs_content_to_string(product.description_json)
        product.save()

class Migration(migrations.Migration):
    dependencies = [('product', '0129_add_product_types_and_attributes_perm')]
    operations = [migrations.AddField(model_name='product', name='description_plaintext', field=models.TextField(blank=True)), migrations.AddField(model_name='product', name='search_vector', field=django.contrib.postgres.search.SearchVectorField(blank=True, null=True)), migrations.AddIndex(model_name='product', index=django.contrib.postgres.indexes.GinIndex(fields=['search_vector'], name='product_pro_search__e78047_gin')), migrations.RunSQL("\n            CREATE TRIGGER title_vector_update BEFORE INSERT OR UPDATE\n            ON product_product FOR EACH ROW EXECUTE PROCEDURE\n            tsvector_update_trigger(\n                'search_vector', 'pg_catalog.english', 'description_plaintext', 'name'\n            )\n\n\n        "), migrations.RunSQL("\n            CREATE FUNCTION messages_trigger() RETURNS trigger AS $$\n            begin\n              new.search_vector :=\n                 setweight(\n                 to_tsvector('pg_catalog.english', coalesce(new.name,'')), 'A'\n                 ) ||\n                 setweight(\n                 to_tsvector(\n                 'pg_catalog.english', coalesce(new.description_plaintext,'')),\n                 'B'\n                 );\n              return new;\n            end\n            $$ LANGUAGE plpgsql;\n\n            CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE\n                ON product_product FOR EACH ROW EXECUTE FUNCTION messages_trigger();\n            "), migrations.RunPython(parse_description_json_field, migrations.RunPython.noop)]