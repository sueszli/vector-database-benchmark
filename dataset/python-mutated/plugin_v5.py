from ckan.types import Schema
import ckan.plugins as p
import ckan.plugins.toolkit as tk

class ExampleIDatasetFormPlugin(tk.DefaultDatasetForm, p.SingletonPlugin):
    p.implements(p.IDatasetForm)

    def create_package_schema(self) -> Schema:
        if False:
            while True:
                i = 10
        schema: Schema = super(ExampleIDatasetFormPlugin, self).create_package_schema()
        schema.update({u'custom_text': [tk.get_validator(u'ignore_missing'), tk.get_converter(u'convert_to_extras')]})
        return schema

    def update_package_schema(self) -> Schema:
        if False:
            i = 10
            return i + 15
        schema: Schema = super(ExampleIDatasetFormPlugin, self).update_package_schema()
        schema.update({u'custom_text': [tk.get_validator(u'ignore_missing'), tk.get_converter(u'convert_to_extras')]})
        return schema

    def show_package_schema(self) -> Schema:
        if False:
            while True:
                i = 10
        schema: Schema = super(ExampleIDatasetFormPlugin, self).show_package_schema()
        schema.update({u'custom_text': [tk.get_converter(u'convert_from_extras'), tk.get_validator(u'ignore_missing')], u'custom_text_2': [tk.get_converter(u'convert_from_extras'), tk.get_validator(u'ignore_missing')]})
        return schema

    def is_fallback(self):
        if False:
            return 10
        return True

    def package_types(self):
        if False:
            while True:
                i = 10
        return [u'fancy_type']