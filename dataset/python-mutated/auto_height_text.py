from django.forms import widgets

class AdminAutoHeightTextInput(widgets.Textarea):

    def __init__(self, attrs=None):
        if False:
            return 10
        default_attrs = {'rows': 1, 'data-controller': 'w-autosize'}
        if attrs:
            default_attrs.update(attrs)
        try:
            default_attrs['class'] += ' w-field__autosize'
        except KeyError:
            default_attrs['class'] = 'w-field__autosize'
        super().__init__(default_attrs)