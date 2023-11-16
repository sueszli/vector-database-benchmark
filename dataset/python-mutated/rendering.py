from pyramid import renderers
from pyramid.config.actions import action_method
from pyramid.interfaces import PHASE1_CONFIG, IRendererFactory
DEFAULT_RENDERERS = (('json', renderers.json_renderer_factory), ('string', renderers.string_renderer_factory))

class RenderingConfiguratorMixin:

    def add_default_renderers(self):
        if False:
            print('Hello World!')
        for (name, renderer) in DEFAULT_RENDERERS:
            self.add_renderer(name, renderer)

    @action_method
    def add_renderer(self, name, factory):
        if False:
            return 10
        '\n        Add a :app:`Pyramid` :term:`renderer` factory to the\n        current configuration state.\n\n        The ``name`` argument is the renderer name.  Use ``None`` to\n        represent the default renderer (a renderer which will be used for all\n        views unless they name another renderer specifically).\n\n        The ``factory`` argument is Python reference to an\n        implementation of a :term:`renderer` factory or a\n        :term:`dotted Python name` to same.\n        '
        factory = self.maybe_dotted(factory)
        if not name:
            name = ''

        def register():
            if False:
                print('Hello World!')
            self.registry.registerUtility(factory, IRendererFactory, name=name)
        intr = self.introspectable('renderer factories', name, self.object_description(factory), 'renderer factory')
        intr['factory'] = factory
        intr['name'] = name
        self.action((IRendererFactory, name), register, order=PHASE1_CONFIG, introspectables=(intr,))