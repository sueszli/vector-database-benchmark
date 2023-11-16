from __future__ import annotations
from ckan.common import CKANConfig
from ckan.types import Schema
import ckan.plugins as p
import ckan.plugins.toolkit as tk

class ExampleIDatasetFormPlugin(tk.DefaultDatasetForm, p.SingletonPlugin):
    p.implements(p.IDatasetForm)
    p.implements(p.IConfigurer)

    def create_package_schema(self) -> Schema:
        if False:
            print('Hello World!')
        schema: Schema = super(ExampleIDatasetFormPlugin, self).create_package_schema()
        schema.update({'custom_text': [tk.get_validator('ignore_missing'), tk.get_converter('convert_to_extras')]})
        return schema

    def update_package_schema(self) -> Schema:
        if False:
            print('Hello World!')
        schema: Schema = super(ExampleIDatasetFormPlugin, self).update_package_schema()
        schema.update({'custom_text': [tk.get_validator('ignore_missing'), tk.get_converter('convert_to_extras')]})
        return schema

    def show_package_schema(self) -> Schema:
        if False:
            print('Hello World!')
        schema: Schema = super(ExampleIDatasetFormPlugin, self).show_package_schema()
        schema.update({'custom_text': [tk.get_converter('convert_from_extras'), tk.get_validator('ignore_missing')]})
        return schema

    def is_fallback(self):
        if False:
            while True:
                i = 10
        return True

    def package_types(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        return []

    def update_config(self, config: CKANConfig):
        if False:
            print('Hello World!')
        tk.add_template_directory(config, 'templates')