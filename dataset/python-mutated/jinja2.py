from django.apps import apps
from jinja2 import BaseLoader, TemplateNotFound
from jinja2.meta import find_referenced_templates
__all__ = ('ConfigTemplateLoader',)

class ConfigTemplateLoader(BaseLoader):
    """
    Custom Jinja2 loader to facilitate populating template content from DataFiles.
    """

    def __init__(self, data_source):
        if False:
            return 10
        self.data_source = data_source
        self._template_cache = {}

    def get_source(self, environment, template):
        if False:
            while True:
                i = 10
        DataFile = apps.get_model('core', 'DataFile')
        try:
            template_source = self._template_cache[template]
        except KeyError:
            raise TemplateNotFound(template)
        if (referenced_templates := find_referenced_templates(environment.parse(template_source))):
            self.cache_templates({df.path: df.data_as_string for df in DataFile.objects.filter(source=self.data_source, path__in=referenced_templates)})
        return (template_source, template, lambda : True)

    def cache_templates(self, templates):
        if False:
            return 10
        self._template_cache.update(templates)