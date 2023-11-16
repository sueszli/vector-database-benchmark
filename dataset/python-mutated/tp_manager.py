import hashlib
import inspect
from mako.exceptions import TemplateLookupException
from mako.template import Template as mTemplate
from pylons import app_globals as g
NULL_TEMPLATE = mTemplate('')
NULL_TEMPLATE.is_null = True

class tp_manager:

    def __init__(self, template_cls=mTemplate):
        if False:
            while True:
                i = 10
        self.templates = {}
        self.Template = template_cls
        self.cache_override_styles = set()

    def add_handler(self, name, style, handler):
        if False:
            print('Hello World!')
        key = (name.lower(), style.lower())
        self.templates[key] = handler
        self.cache_override_styles.add(style.lower())

    def cache_template(self, cls, style, template):
        if False:
            return 10
        use_cache = not g.reload_templates
        if use_cache:
            if not hasattr(template, 'hash') and getattr(template, 'filename', None):
                with open(template.filename, 'r') as handle:
                    template.hash = hashlib.sha1(handle.read()).hexdigest()
            key = (cls.__name__.lower(), style)
            self.templates[key] = template

    def get_template(self, cls, style):
        if False:
            while True:
                i = 10
        name = cls.__name__.lower()
        use_cache = not g.reload_templates
        if use_cache or style.lower() in self.cache_override_styles:
            key = (name, style)
            template = self.templates.get(key)
            if template:
                return template
        filename = '/%s.%s' % (name, style)
        try:
            template = g.mako_lookup.get_template(filename)
        except TemplateLookupException:
            return
        self.cache_template(cls, style, template)
        return template

    def get(self, thing, style):
        if False:
            i = 10
            return i + 15
        if not isinstance(thing, type(object)):
            thing = thing.__class__
        style = style.lower()
        template = self.get_template(thing, style)
        if template:
            return template
        for cls in inspect.getmro(thing)[1:]:
            template = self.get_template(cls, style)
            if template:
                break
        else:
            template = NULL_TEMPLATE
        self.cache_template(thing, style, template)
        return template