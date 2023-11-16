import json
from django.forms import Media, widgets
from wagtail.utils.widgets import WidgetWithScript

class CustomRichTextArea(WidgetWithScript, widgets.Textarea):

    def render_js_init(self, id_, name, value):
        if False:
            for i in range(10):
                print('nop')
        return f'customEditorInitScript({json.dumps(id_)});'

    @property
    def media(self):
        if False:
            return 10
        return Media(js=['vendor/custom_editor.js'])

class LegacyRichTextArea(WidgetWithScript, widgets.Textarea):

    def render_js_init(self, id_, name, value):
        if False:
            for i in range(10):
                print('nop')
        return f'legacyEditorInitScript({json.dumps(id_)});'

    @property
    def media(self):
        if False:
            i = 10
            return i + 15
        return Media(js=['vendor/legacy_editor.js'])