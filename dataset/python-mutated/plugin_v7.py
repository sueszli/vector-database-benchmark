from ckan.common import CKANConfig
import ckan.plugins as p
import ckan.plugins.toolkit as tk

class ExampleIDatasetFormPlugin(p.SingletonPlugin, tk.DefaultDatasetForm):
    p.implements(p.IConfigurer)
    p.implements(p.IDatasetForm)

    def update_config(self, config: CKANConfig):
        if False:
            print('Hello World!')
        tk.add_template_directory(config, u'templates')

    def is_fallback(self):
        if False:
            return 10
        return False

    def package_types(self):
        if False:
            for i in range(10):
                print('nop')
        return [u'first', u'second']

    def read_template(self, package_type: str):
        if False:
            for i in range(10):
                print('nop')
        return u'{}/read.html'.format(package_type)

    def new_template(self):
        if False:
            i = 10
            return i + 15
        return [u'first/new.html', u'first/read.html']