from wagtail import hooks

class FeatureRegistry:
    """
    A central store of information about optional features that can be enabled in rich text
    editors by passing a ``features`` list to the RichTextField, such as how to
    whitelist / convert HTML tags, and how to enable the feature on various editors.

    This information may come from diverse sources - for example, wagtailimages might define
    an 'images' feature and a Draftail plugin for it, while a third-party module might
    define a TinyMCE plugin for the same feature. The information is therefore collected into
    this registry via the 'register_rich_text_features' hook.
    """

    def __init__(self):
        if False:
            return 10
        self.has_scanned_for_features = False
        self.plugins_by_editor = {}
        self.default_features = []
        self.link_types = {}
        self.embed_types = {}
        self.converter_rules_by_converter = {}

    def get_default_features(self):
        if False:
            return 10
        if not self.has_scanned_for_features:
            self._scan_for_features()
        return self.default_features

    def _scan_for_features(self):
        if False:
            return 10
        for fn in hooks.get_hooks('register_rich_text_features'):
            fn(self)
        self.has_scanned_for_features = True

    def register_editor_plugin(self, editor_name, feature_name, plugin):
        if False:
            print('Hello World!')
        plugins = self.plugins_by_editor.setdefault(editor_name, {})
        plugins[feature_name] = plugin

    def get_editor_plugin(self, editor_name, feature_name):
        if False:
            return 10
        if not self.has_scanned_for_features:
            self._scan_for_features()
        try:
            return self.plugins_by_editor[editor_name][feature_name]
        except KeyError:
            return None

    def register_link_type(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self.link_types[handler.identifier] = handler

    def get_link_types(self):
        if False:
            i = 10
            return i + 15
        if not self.has_scanned_for_features:
            self._scan_for_features()
        return self.link_types

    def register_embed_type(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self.embed_types[handler.identifier] = handler

    def get_embed_types(self):
        if False:
            i = 10
            return i + 15
        if not self.has_scanned_for_features:
            self._scan_for_features()
        return self.embed_types

    def register_converter_rule(self, converter_name, feature_name, rule):
        if False:
            while True:
                i = 10
        rules = self.converter_rules_by_converter.setdefault(converter_name, {})
        rules[feature_name] = rule

    def get_converter_rule(self, converter_name, feature_name):
        if False:
            while True:
                i = 10
        if not self.has_scanned_for_features:
            self._scan_for_features()
        try:
            return self.converter_rules_by_converter[converter_name][feature_name]
        except KeyError:
            return None

    @staticmethod
    def function_as_entity_handler(identifier, fn):
        if False:
            i = 10
            return i + 15
        'Supports legacy registering of entity handlers as functions.'
        return type('EntityHandlerRegisteredAsFunction', (object,), {'identifier': identifier, 'expand_db_attributes': staticmethod(fn)})