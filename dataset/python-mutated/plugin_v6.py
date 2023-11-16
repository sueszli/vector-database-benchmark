from flask import Blueprint
import ckan.plugins as p
import ckan.plugins.toolkit as tk

def fancy_route(package_type: str):
    if False:
        while True:
            i = 10
    return u'Hello, {}'.format(package_type)

def fancy_new_route(package_type: str):
    if False:
        for i in range(10):
            print('nop')
    return u'Hello, new {}'.format(package_type)

def fancy_resource_route(package_type: str, id: str):
    if False:
        print('Hello World!')
    return u'Hello, {}:{}'.format(package_type, id)

class ExampleIDatasetFormPlugin(p.SingletonPlugin, tk.DefaultDatasetForm):
    p.implements(p.IDatasetForm)

    def is_fallback(self):
        if False:
            while True:
                i = 10
        return False

    def package_types(self):
        if False:
            return 10
        return [u'fancy_type']

    def prepare_dataset_blueprint(self, package_type: str, bp: Blueprint):
        if False:
            for i in range(10):
                print('nop')
        bp.add_url_rule(u'/fancy-route', view_func=fancy_route)
        bp.add_url_rule(u'/new', view_func=fancy_new_route)
        return bp

    def prepare_resource_blueprint(self, package_type: str, bp: Blueprint):
        if False:
            i = 10
            return i + 15
        bp.add_url_rule(u'/new', view_func=fancy_resource_route)
        return bp