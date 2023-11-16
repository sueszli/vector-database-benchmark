import datetime
from decimal import Decimal
from django import forms
from django.db.models import Model
from django.db.models.fields import BLANK_CHOICE_DASH
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.encoding import force_str
from django.utils.functional import cached_property
from django.utils.html import format_html
from django.utils.safestring import mark_safe
from django.utils.translation import gettext as _
from wagtail.admin.staticfiles import versioned_static
from wagtail.coreutils import camelcase_to_underscore, resolve_model_string
from wagtail.rich_text import RichText, RichTextMaxLengthValidator, extract_references_from_rich_text, get_text_for_indexing
from wagtail.telepath import Adapter, register
from .base import Block
try:
    from django.utils.choices import CallableChoiceIterator
except ImportError:
    from django.forms.fields import CallableChoiceIterator

class FieldBlock(Block):
    """A block that wraps a Django form field"""

    def id_for_label(self, prefix):
        if False:
            i = 10
            return i + 15
        return self.field.widget.id_for_label(prefix)

    def value_from_form(self, value):
        if False:
            return 10
        "\n        The value that we get back from the form field might not be the type\n        that this block works with natively; for example, the block may want to\n        wrap a simple value such as a string in an object that provides a fancy\n        HTML rendering (e.g. EmbedBlock).\n\n        We therefore provide this method to perform any necessary conversion\n        from the form field value to the block's native value. As standard,\n        this returns the form field value unchanged.\n        "
        return value

    def value_for_form(self, value):
        if False:
            while True:
                i = 10
        "\n        Reverse of value_from_form; convert a value of this block's native value type\n        to one that can be rendered by the form field\n        "
        return value

    def value_from_datadict(self, data, files, prefix):
        if False:
            for i in range(10):
                print('nop')
        return self.value_from_form(self.field.widget.value_from_datadict(data, files, prefix))

    def value_omitted_from_data(self, data, files, prefix):
        if False:
            print('Hello World!')
        return self.field.widget.value_omitted_from_data(data, files, prefix)

    def clean(self, value):
        if False:
            i = 10
            return i + 15
        return self.value_from_form(self.field.clean(self.value_for_form(value)))

    @property
    def required(self):
        if False:
            while True:
                i = 10
        return self.field.required

    def get_form_state(self, value):
        if False:
            for i in range(10):
                print('nop')
        return self.field.widget.format_value(self.field.prepare_value(self.value_for_form(value)))

    class Meta:
        icon = 'placeholder'
        default = None

class FieldBlockAdapter(Adapter):
    js_constructor = 'wagtail.blocks.FieldBlock'

    def js_args(self, block):
        if False:
            return 10
        classname = ['w-field', f'w-field--{camelcase_to_underscore(block.field.__class__.__name__)}', f'w-field--{camelcase_to_underscore(block.field.widget.__class__.__name__)}']
        form_classname = getattr(block.meta, 'form_classname', '')
        if form_classname:
            classname.append(form_classname)
        legacy_classname = getattr(block.meta, 'classname', '')
        if legacy_classname:
            classname.append(legacy_classname)
        meta = {'label': block.label, 'required': block.required, 'icon': block.meta.icon, 'classname': ' '.join(classname), 'showAddCommentButton': getattr(block.field.widget, 'show_add_comment_button', True), 'strings': {'ADD_COMMENT': _('Add Comment')}}
        if block.field.help_text:
            meta['helpText'] = block.field.help_text
        return [block.name, block.field.widget, meta]

    @cached_property
    def media(self):
        if False:
            while True:
                i = 10
        return forms.Media(js=[versioned_static('wagtailadmin/js/telepath/blocks.js')])
register(FieldBlockAdapter(), FieldBlock)

class CharBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, max_length=None, min_length=None, validators=(), search_index=True, **kwargs):
        if False:
            return 10
        self.search_index = search_index
        self.field = forms.CharField(required=required, help_text=help_text, max_length=max_length, min_length=min_length, validators=validators)
        super().__init__(**kwargs)

    def get_searchable_content(self, value):
        if False:
            while True:
                i = 10
        return [force_str(value)] if self.search_index else []

class TextBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, rows=1, max_length=None, min_length=None, search_index=True, validators=(), **kwargs):
        if False:
            i = 10
            return i + 15
        self.field_options = {'required': required, 'help_text': help_text, 'max_length': max_length, 'min_length': min_length, 'validators': validators}
        self.rows = rows
        self.search_index = search_index
        super().__init__(**kwargs)

    @cached_property
    def field(self):
        if False:
            for i in range(10):
                print('nop')
        from wagtail.admin.widgets import AdminAutoHeightTextInput
        field_kwargs = {'widget': AdminAutoHeightTextInput(attrs={'rows': self.rows})}
        field_kwargs.update(self.field_options)
        return forms.CharField(**field_kwargs)

    def get_searchable_content(self, value):
        if False:
            print('Hello World!')
        return [force_str(value)] if self.search_index else []

    class Meta:
        icon = 'pilcrow'

class BlockQuoteBlock(TextBlock):

    def render_basic(self, value, context=None):
        if False:
            return 10
        if value:
            return format_html('<blockquote>{0}</blockquote>', value)
        else:
            return ''

    class Meta:
        icon = 'openquote'

class FloatBlock(FieldBlock):

    def __init__(self, required=True, max_value=None, min_value=None, validators=(), *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.field = forms.FloatField(required=required, max_value=max_value, min_value=min_value, validators=validators)
        super().__init__(*args, **kwargs)

    class Meta:
        icon = 'decimal'

class DecimalBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, max_value=None, min_value=None, max_digits=None, decimal_places=None, validators=(), *args, **kwargs):
        if False:
            return 10
        self.field = forms.DecimalField(required=required, help_text=help_text, max_value=max_value, min_value=min_value, max_digits=max_digits, decimal_places=decimal_places, validators=validators)
        super().__init__(*args, **kwargs)

    def to_python(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return value
        else:
            return Decimal(value)

    class Meta:
        icon = 'decimal'

class RegexBlock(FieldBlock):

    def __init__(self, regex, required=True, help_text=None, max_length=None, min_length=None, error_messages=None, validators=(), *args, **kwargs):
        if False:
            while True:
                i = 10
        self.field = forms.RegexField(regex=regex, required=required, help_text=help_text, max_length=max_length, min_length=min_length, error_messages=error_messages, validators=validators)
        super().__init__(*args, **kwargs)

    class Meta:
        icon = 'regex'

class URLBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, max_length=None, min_length=None, validators=(), **kwargs):
        if False:
            print('Hello World!')
        self.field = forms.URLField(required=required, help_text=help_text, max_length=max_length, min_length=min_length, validators=validators)
        super().__init__(**kwargs)

    class Meta:
        icon = 'link-external'

class BooleanBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, **kwargs):
        if False:
            print('Hello World!')
        self.field = forms.BooleanField(required=required, help_text=help_text)
        super().__init__(**kwargs)

    def get_form_state(self, value):
        if False:
            for i in range(10):
                print('nop')
        return bool(value)

    class Meta:
        icon = 'tick-inverse'

class DateBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, format=None, validators=(), **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.field_options = {'required': required, 'help_text': help_text, 'validators': validators}
        try:
            self.field_options['input_formats'] = kwargs.pop('input_formats')
        except KeyError:
            pass
        self.format = format
        super().__init__(**kwargs)

    @cached_property
    def field(self):
        if False:
            print('Hello World!')
        from wagtail.admin.widgets import AdminDateInput
        field_kwargs = {'widget': AdminDateInput(format=self.format)}
        field_kwargs.update(self.field_options)
        return forms.DateField(**field_kwargs)

    def to_python(self, value):
        if False:
            print('Hello World!')
        if value is None or isinstance(value, datetime.date):
            return value
        else:
            return parse_date(value)

    class Meta:
        icon = 'date'

class TimeBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, format=None, validators=(), **kwargs):
        if False:
            i = 10
            return i + 15
        self.field_options = {'required': required, 'help_text': help_text, 'validators': validators}
        self.format = format
        super().__init__(**kwargs)

    @cached_property
    def field(self):
        if False:
            for i in range(10):
                print('nop')
        from wagtail.admin.widgets import AdminTimeInput
        field_kwargs = {'widget': AdminTimeInput(format=self.format)}
        field_kwargs.update(self.field_options)
        return forms.TimeField(**field_kwargs)

    def to_python(self, value):
        if False:
            return 10
        if value is None or isinstance(value, datetime.time):
            return value
        else:
            return parse_time(value)

    class Meta:
        icon = 'time'

class DateTimeBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, format=None, validators=(), **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.field_options = {'required': required, 'help_text': help_text, 'validators': validators}
        self.format = format
        super().__init__(**kwargs)

    @cached_property
    def field(self):
        if False:
            while True:
                i = 10
        from wagtail.admin.widgets import AdminDateTimeInput
        field_kwargs = {'widget': AdminDateTimeInput(format=self.format)}
        field_kwargs.update(self.field_options)
        return forms.DateTimeField(**field_kwargs)

    def to_python(self, value):
        if False:
            while True:
                i = 10
        if value is None or isinstance(value, datetime.datetime):
            return value
        else:
            return parse_datetime(value)

    class Meta:
        icon = 'date'

class EmailBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, validators=(), **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.field = forms.EmailField(required=required, help_text=help_text, validators=validators)
        super().__init__(**kwargs)

    class Meta:
        icon = 'mail'

class IntegerBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, min_value=None, max_value=None, validators=(), **kwargs):
        if False:
            while True:
                i = 10
        self.field = forms.IntegerField(required=required, help_text=help_text, min_value=min_value, max_value=max_value, validators=validators)
        super().__init__(**kwargs)

    class Meta:
        icon = 'placeholder'

class BaseChoiceBlock(FieldBlock):
    choices = ()

    def __init__(self, choices=None, default=None, required=True, help_text=None, search_index=True, widget=None, validators=(), **kwargs):
        if False:
            return 10
        self._required = required
        self._default = default
        self.search_index = search_index
        if choices is None:
            choices = self.choices
        if callable(choices):
            choices_for_constructor = choices
            choices = CallableChoiceIterator(choices)
        else:
            choices_for_constructor = choices = list(choices)
        self._constructor_kwargs = kwargs.copy()
        self._constructor_kwargs['choices'] = choices_for_constructor
        if required is not True:
            self._constructor_kwargs['required'] = required
        if help_text is not None:
            self._constructor_kwargs['help_text'] = help_text
        callable_choices = self._get_callable_choices(choices)
        self.field = self.get_field(choices=callable_choices, required=required, help_text=help_text, validators=validators, widget=widget)
        super().__init__(default=default, **kwargs)

    def _get_callable_choices(self, choices, blank_choice=True):
        if False:
            while True:
                i = 10
        '\n        Return a callable that we can pass into `forms.ChoiceField`, which will provide the\n        choices list with the addition of a blank choice (if blank_choice=True and one does not\n        already exist).\n        '

        def choices_callable():
            if False:
                print('Hello World!')
            local_choices = list(choices)
            if not blank_choice:
                return local_choices
            has_blank_choice = False
            for (v1, v2) in local_choices:
                if isinstance(v2, (list, tuple)):
                    has_blank_choice = any((value in ('', None) for (value, label) in v2))
                    if has_blank_choice:
                        break
                elif v1 in ('', None):
                    has_blank_choice = True
                    break
            if not has_blank_choice:
                return BLANK_CHOICE_DASH + local_choices
            return local_choices
        return choices_callable

    class Meta:
        icon = 'placeholder'

class ChoiceBlock(BaseChoiceBlock):

    def get_field(self, **kwargs):
        if False:
            print('Hello World!')
        return forms.ChoiceField(**kwargs)

    def _get_callable_choices(self, choices, blank_choice=None):
        if False:
            while True:
                i = 10
        if blank_choice is None:
            blank_choice = not (self._default and self._required)
        return super()._get_callable_choices(choices, blank_choice=blank_choice)

    def deconstruct(self):
        if False:
            i = 10
            return i + 15
        '\n        Always deconstruct ChoiceBlock instances as if they were plain ChoiceBlocks with their\n        choice list passed in the constructor, even if they are actually subclasses. This allows\n        users to define subclasses of ChoiceBlock in their models.py, with specific choice lists\n        passed in, without references to those classes ending up frozen into migrations.\n        '
        return ('wagtail.blocks.ChoiceBlock', [], self._constructor_kwargs)

    def get_searchable_content(self, value):
        if False:
            while True:
                i = 10
        if not self.search_index:
            return []
        text_value = force_str(value)
        for (k, v) in self.field.choices:
            if isinstance(v, (list, tuple)):
                for (k2, v2) in v:
                    if value == k2 or text_value == force_str(k2):
                        return [force_str(k), force_str(v2)]
            elif value == k or text_value == force_str(k):
                return [force_str(v)]
        return []

class MultipleChoiceBlock(BaseChoiceBlock):

    def get_field(self, **kwargs):
        if False:
            print('Hello World!')
        return forms.MultipleChoiceField(**kwargs)

    def _get_callable_choices(self, choices, blank_choice=False):
        if False:
            return 10
        'Override to default blank choice to False'
        return super()._get_callable_choices(choices, blank_choice=blank_choice)

    def deconstruct(self):
        if False:
            print('Hello World!')
        '\n        Always deconstruct MultipleChoiceBlock instances as if they were plain\n        MultipleChoiceBlocks with their choice list passed in the constructor,\n        even if they are actually subclasses. This allows users to define\n        subclasses of MultipleChoiceBlock in their models.py, with specific choice\n        lists passed in, without references to those classes ending up frozen\n        into migrations.\n        '
        return ('wagtail.blocks.MultipleChoiceBlock', [], self._constructor_kwargs)

    def get_searchable_content(self, value):
        if False:
            i = 10
            return i + 15
        if not self.search_index:
            return []
        content = []
        text_value = force_str(value)
        for (k, v) in self.field.choices:
            if isinstance(v, (list, tuple)):
                for (k2, v2) in v:
                    if value == k2 or text_value == force_str(k2):
                        content.append(force_str(k))
                        content.append(force_str(v2))
            elif value == k or text_value == force_str(k):
                content.append(force_str(v))
        return content

class RichTextBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, editor='default', features=None, max_length=None, validators=(), search_index=True, **kwargs):
        if False:
            while True:
                i = 10
        if max_length is not None:
            validators = list(validators) + [RichTextMaxLengthValidator(max_length)]
        self.field_options = {'required': required, 'help_text': help_text, 'validators': validators}
        self.editor = editor
        self.features = features
        self.search_index = search_index
        super().__init__(**kwargs)

    def get_default(self):
        if False:
            while True:
                i = 10
        if isinstance(self.meta.default, RichText):
            return self.meta.default
        else:
            return RichText(self.meta.default)

    def to_python(self, value):
        if False:
            while True:
                i = 10
        return RichText(value)

    def get_prep_value(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value.source

    @cached_property
    def field(self):
        if False:
            print('Hello World!')
        from wagtail.admin.rich_text import get_rich_text_editor_widget
        return forms.CharField(widget=get_rich_text_editor_widget(self.editor, features=self.features), **self.field_options)

    def value_for_form(self, value):
        if False:
            while True:
                i = 10
        return value.source

    def value_from_form(self, value):
        if False:
            i = 10
            return i + 15
        return RichText(value)

    def get_searchable_content(self, value):
        if False:
            while True:
                i = 10
        if not self.search_index:
            return []
        source = force_str(value.source)
        return [get_text_for_indexing(source)]

    def extract_references(self, value):
        if False:
            print('Hello World!')
        yield from extract_references_from_rich_text(force_str(value.source))

    class Meta:
        icon = 'pilcrow'

class RawHTMLBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, max_length=None, min_length=None, validators=(), **kwargs):
        if False:
            while True:
                i = 10
        self.field = forms.CharField(required=required, help_text=help_text, max_length=max_length, min_length=min_length, validators=validators, widget=forms.Textarea)
        super().__init__(**kwargs)

    def get_default(self):
        if False:
            while True:
                i = 10
        return mark_safe(self.meta.default or '')

    def to_python(self, value):
        if False:
            return 10
        return mark_safe(value)

    def get_prep_value(self, value):
        if False:
            print('Hello World!')
        return str(value) + ''

    def value_for_form(self, value):
        if False:
            i = 10
            return i + 15
        return str(value) + ''

    def value_from_form(self, value):
        if False:
            i = 10
            return i + 15
        return mark_safe(value)

    class Meta:
        icon = 'code'

class ChooserBlock(FieldBlock):

    def __init__(self, required=True, help_text=None, validators=(), **kwargs):
        if False:
            print('Hello World!')
        self._required = required
        self._help_text = help_text
        self._validators = validators
        super().__init__(**kwargs)
    'Abstract superclass for fields that implement a chooser interface (page, image, snippet etc)'

    @cached_property
    def model_class(self):
        if False:
            print('Hello World!')
        return resolve_model_string(self.target_model)

    @cached_property
    def field(self):
        if False:
            i = 10
            return i + 15
        return forms.ModelChoiceField(queryset=self.model_class.objects.all(), widget=self.widget, required=self._required, validators=self._validators, help_text=self._help_text)

    def to_python(self, value):
        if False:
            while True:
                i = 10
        if value is None:
            return value
        else:
            try:
                return self.model_class.objects.get(pk=value)
            except self.model_class.DoesNotExist:
                return None

    def bulk_to_python(self, values):
        if False:
            for i in range(10):
                print('nop')
        'Return the model instances for the given list of primary keys.\n\n        The instances must be returned in the same order as the values and keep None values.\n        '
        objects = self.model_class.objects.in_bulk(values)
        return [objects.get(id) for id in values]

    def get_prep_value(self, value):
        if False:
            print('Hello World!')
        if value is None:
            return None
        else:
            return value.pk

    def value_from_form(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value is None or isinstance(value, self.model_class):
            return value
        else:
            try:
                return self.model_class.objects.get(pk=value)
            except self.model_class.DoesNotExist:
                return None

    def get_form_state(self, value):
        if False:
            i = 10
            return i + 15
        return self.widget.get_value_data(value)

    def clean(self, value):
        if False:
            while True:
                i = 10
        if isinstance(value, self.model_class):
            value = value.pk
        return super().clean(value)

    def extract_references(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value is not None and issubclass(self.model_class, Model):
            yield (self.model_class, str(value.pk), '', '')

    class Meta:
        icon = 'placeholder'

class PageChooserBlock(ChooserBlock):

    def __init__(self, page_type=None, can_choose_root=False, target_model=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if target_model:
            page_type = target_model
        if page_type:
            if not isinstance(page_type, (list, tuple)):
                page_type = [page_type]
        else:
            page_type = []
        self.page_type = page_type
        self.can_choose_root = can_choose_root
        super().__init__(**kwargs)

    @cached_property
    def target_model(self):
        if False:
            print('Hello World!')
        '\n        Defines the model used by the base ChooserBlock for ID <-> instance\n        conversions. If a single page type is specified in target_model,\n        we can use that to get the more specific instance "for free"; otherwise\n        use the generic Page model.\n        '
        if len(self.target_models) == 1:
            return self.target_models[0]
        return resolve_model_string('wagtailcore.Page')

    @cached_property
    def target_models(self):
        if False:
            for i in range(10):
                print('nop')
        target_models = []
        for target_model in self.page_type:
            target_models.append(resolve_model_string(target_model))
        return target_models

    @cached_property
    def widget(self):
        if False:
            for i in range(10):
                print('nop')
        from wagtail.admin.widgets import AdminPageChooser
        return AdminPageChooser(target_models=self.target_models, can_choose_root=self.can_choose_root)

    def get_form_state(self, value):
        if False:
            while True:
                i = 10
        value_data = self.widget.get_value_data(value)
        if value_data is None:
            return None
        else:
            return {'id': value_data['id'], 'parentId': value_data['parent_id'], 'adminTitle': value_data['display_title'], 'editUrl': value_data['edit_url']}

    def render_basic(self, value, context=None):
        if False:
            while True:
                i = 10
        if value:
            return format_html('<a href="{0}">{1}</a>', value.url, value.title)
        else:
            return ''

    def deconstruct(self):
        if False:
            print('Hello World!')
        (name, args, kwargs) = super().deconstruct()
        if 'target_model' in kwargs or 'page_type' in kwargs:
            target_models = []
            for target_model in self.target_models:
                opts = target_model._meta
                target_models.append(f'{opts.app_label}.{opts.object_name}')
            kwargs.pop('target_model', None)
            kwargs['page_type'] = target_models
        return (name, args, kwargs)

    class Meta:
        icon = 'doc-empty-inverse'
block_classes = [FieldBlock, CharBlock, URLBlock, RichTextBlock, RawHTMLBlock, ChooserBlock, PageChooserBlock, TextBlock, BooleanBlock, DateBlock, TimeBlock, DateTimeBlock, ChoiceBlock, MultipleChoiceBlock, EmailBlock, IntegerBlock, FloatBlock, DecimalBlock, RegexBlock, BlockQuoteBlock]
DECONSTRUCT_ALIASES = {cls: 'wagtail.blocks.%s' % cls.__name__ for cls in block_classes}
__all__ = [cls.__name__ for cls in block_classes]