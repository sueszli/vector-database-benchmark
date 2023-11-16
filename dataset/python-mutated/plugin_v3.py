"""Example IDatasetFormPlugin"""
from __future__ import annotations
from ckan.types import Schema
import ckan.plugins as p
import ckan.plugins.toolkit as tk

class ExampleIDatasetFormPlugin(tk.DefaultDatasetForm, p.SingletonPlugin):
    p.implements(p.IDatasetForm)

    def _modify_package_schema(self, schema: Schema) -> Schema:
        if False:
            print('Hello World!')
        schema.update({'custom_text': [tk.get_validator('ignore_missing'), tk.get_converter('convert_to_extras')]})
        return schema

    def create_package_schema(self):
        if False:
            print('Hello World!')
        schema: Schema = super(ExampleIDatasetFormPlugin, self).create_package_schema()
        schema = self._modify_package_schema(schema)
        return schema

    def update_package_schema(self):
        if False:
            i = 10
            return i + 15
        schema: Schema = super(ExampleIDatasetFormPlugin, self).update_package_schema()
        schema = self._modify_package_schema(schema)
        return schema

    def show_package_schema(self) -> Schema:
        if False:
            i = 10
            return i + 15
        schema: Schema = super(ExampleIDatasetFormPlugin, self).show_package_schema()
        schema.update({'custom_text': [tk.get_converter('convert_from_extras'), tk.get_validator('ignore_missing')]})
        return schema

    def is_fallback(self):
        if False:
            i = 10
            return i + 15
        return True

    def package_types(self) -> list[str]:
        if False:
            return 10
        return []