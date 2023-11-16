from ckan.common import CKANConfig
from ckan import plugins
from ckan.plugins import toolkit

class ExampleITranslationPlugin(plugins.SingletonPlugin):
    plugins.implements(plugins.IConfigurer)

    def update_config(self, config: CKANConfig):
        if False:
            while True:
                i = 10
        toolkit.add_template_directory(config, 'templates')