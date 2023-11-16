import collections
from django import forms
from django.core.exceptions import ValidationError
from django.forms.utils import ErrorList
from django.template.loader import render_to_string
from django.utils.functional import cached_property
from django.utils.html import format_html, format_html_join
from django.utils.safestring import mark_safe
from wagtail.admin.staticfiles import versioned_static
from wagtail.telepath import Adapter, register
from .base import Block, BoundBlock, DeclarativeSubBlocksMetaclass, get_error_json_data, get_error_list_json_data, get_help_icon
__all__ = ['BaseStructBlock', 'StructBlock', 'StructValue', 'StructBlockValidationError']

class StructBlockValidationError(ValidationError):

    def __init__(self, block_errors=None, non_block_errors=None):
        if False:
            while True:
                i = 10
        self.non_block_errors = ErrorList(non_block_errors)
        self.block_errors = {}
        if block_errors is None:
            pass
        else:
            for (name, val) in block_errors.items():
                if isinstance(val, ErrorList):
                    self.block_errors[name] = val.as_data()[0]
                elif isinstance(val, list):
                    self.block_errors[name] = val[0]
                else:
                    self.block_errors[name] = val
        super().__init__('Validation error in StructBlock')

    def as_json_data(self):
        if False:
            print('Hello World!')
        result = {}
        if self.non_block_errors:
            result['messages'] = get_error_list_json_data(self.non_block_errors)
        if self.block_errors:
            result['blockErrors'] = {name: get_error_json_data(error) for (name, error) in self.block_errors.items()}
        return result

class StructValue(collections.OrderedDict):
    """A class that generates a StructBlock value from provided sub-blocks"""

    def __init__(self, block, *args):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args)
        self.block = block

    def __html__(self):
        if False:
            return 10
        return self.block.render(self)

    def render_as_block(self, context=None):
        if False:
            i = 10
            return i + 15
        return self.block.render(self, context=context)

    @cached_property
    def bound_blocks(self):
        if False:
            i = 10
            return i + 15
        return collections.OrderedDict([(name, block.bind(self.get(name))) for (name, block) in self.block.child_blocks.items()])

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        return (self.__class__, (self.block,), None, None, iter(self.items()))

class PlaceholderBoundBlock(BoundBlock):
    """
    Provides a render_form method that outputs a block placeholder, for use in custom form_templates
    """

    def render_form(self):
        if False:
            print('Hello World!')
        return format_html('<div data-structblock-child="{}"></div>', self.block.name)

class BaseStructBlock(Block):

    def __init__(self, local_blocks=None, search_index=True, **kwargs):
        if False:
            print('Hello World!')
        self._constructor_kwargs = kwargs
        self.search_index = search_index
        super().__init__(**kwargs)
        self.child_blocks = self.base_blocks.copy()
        if local_blocks:
            for (name, block) in local_blocks:
                block.set_name(name)
                self.child_blocks[name] = block

    def get_default(self):
        if False:
            return 10
        '\n        Any default value passed in the constructor or self.meta is going to be a dict\n        rather than a StructValue; for consistency, we need to convert it to a StructValue\n        for StructBlock to work with\n        '
        return self._to_struct_value([(name, self.meta.default[name] if name in self.meta.default else block.get_default()) for (name, block) in self.child_blocks.items()])

    def value_from_datadict(self, data, files, prefix):
        if False:
            return 10
        return self._to_struct_value([(name, block.value_from_datadict(data, files, f'{prefix}-{name}')) for (name, block) in self.child_blocks.items()])

    def value_omitted_from_data(self, data, files, prefix):
        if False:
            i = 10
            return i + 15
        return all((block.value_omitted_from_data(data, files, f'{prefix}-{name}') for (name, block) in self.child_blocks.items()))

    def clean(self, value):
        if False:
            print('Hello World!')
        result = []
        errors = {}
        for (name, val) in value.items():
            try:
                result.append((name, self.child_blocks[name].clean(val)))
            except ValidationError as e:
                errors[name] = e
        if errors:
            raise StructBlockValidationError(errors)
        return self._to_struct_value(result)

    def to_python(self, value):
        if False:
            while True:
                i = 10
        'Recursively call to_python on children and return as a StructValue'
        return self._to_struct_value([(name, child_block.to_python(value[name]) if name in value else child_block.get_default()) for (name, child_block) in self.child_blocks.items()])

    def bulk_to_python(self, values):
        if False:
            i = 10
            return i + 15
        values_by_subfield = {}
        for (name, child_block) in self.child_blocks.items():
            indexes = []
            raw_values = []
            for (i, val) in enumerate(values):
                if name in val:
                    indexes.append(i)
                    raw_values.append(val[name])
            converted_values = child_block.bulk_to_python(raw_values)
            converted_values_by_index = dict(zip(indexes, converted_values))
            values_by_subfield[name] = []
            for i in range(0, len(values)):
                try:
                    converted_value = converted_values_by_index[i]
                except KeyError:
                    converted_value = child_block.get_default()
                values_by_subfield[name].append(converted_value)
        return [self._to_struct_value({name: values_by_subfield[name][i] for name in self.child_blocks.keys()}) for i in range(0, len(values))]

    def _to_struct_value(self, block_items):
        if False:
            return 10
        'Return a Structvalue representation of the sub-blocks in this block'
        return self.meta.value_class(self, block_items)

    def get_prep_value(self, value):
        if False:
            while True:
                i = 10
        'Recursively call get_prep_value on children and return as a plain dict'
        return {name: self.child_blocks[name].get_prep_value(val) for (name, val) in value.items()}

    def get_form_state(self, value):
        if False:
            i = 10
            return i + 15
        return {name: self.child_blocks[name].get_form_state(val) for (name, val) in value.items()}

    def get_api_representation(self, value, context=None):
        if False:
            return 10
        'Recursively call get_api_representation on children and return as a plain dict'
        return {name: self.child_blocks[name].get_api_representation(val, context=context) for (name, val) in value.items()}

    def get_searchable_content(self, value):
        if False:
            for i in range(10):
                print('nop')
        if not self.search_index:
            return []
        content = []
        for (name, block) in self.child_blocks.items():
            content.extend(block.get_searchable_content(value.get(name, block.get_default())))
        return content

    def extract_references(self, value):
        if False:
            for i in range(10):
                print('nop')
        for (name, block) in self.child_blocks.items():
            for (model, object_id, model_path, content_path) in block.extract_references(value.get(name, block.get_default())):
                model_path = f'{name}.{model_path}' if model_path else name
                content_path = f'{name}.{content_path}' if content_path else name
                yield (model, object_id, model_path, content_path)

    def get_block_by_content_path(self, value, path_elements):
        if False:
            return 10
        '\n        Given a list of elements from a content path, retrieve the block at that path\n        as a BoundBlock object, or None if the path does not correspond to a valid block.\n        '
        if path_elements:
            (name, *remaining_elements) = path_elements
            try:
                child_block = self.child_blocks[name]
            except KeyError:
                return None
            child_value = value.get(name, child_block.get_default())
            return child_block.get_block_by_content_path(child_value, remaining_elements)
        else:
            return self.bind(value)

    def deconstruct(self):
        if False:
            while True:
                i = 10
        "\n        Always deconstruct StructBlock instances as if they were plain StructBlocks with all of the\n        field definitions passed to the constructor - even if in reality this is a subclass of StructBlock\n        with the fields defined declaratively, or some combination of the two.\n\n        This ensures that the field definitions get frozen into migrations, rather than leaving a reference\n        to a custom subclass in the user's models.py that may or may not stick around.\n        "
        path = 'wagtail.blocks.StructBlock'
        args = [list(self.child_blocks.items())]
        kwargs = self._constructor_kwargs
        return (path, args, kwargs)

    def check(self, **kwargs):
        if False:
            while True:
                i = 10
        errors = super().check(**kwargs)
        for (name, child_block) in self.child_blocks.items():
            errors.extend(child_block.check(**kwargs))
            errors.extend(child_block._check_name(**kwargs))
        return errors

    def render_basic(self, value, context=None):
        if False:
            return 10
        return format_html('<dl>\n{}\n</dl>', format_html_join('\n', '    <dt>{}</dt>\n    <dd>{}</dd>', value.items()))

    def render_form_template(self):
        if False:
            return 10
        context = self.get_form_context(self.get_default(), prefix='__PREFIX__', errors=None)
        return mark_safe(render_to_string(self.meta.form_template, context))

    def get_form_context(self, value, prefix='', errors=None):
        if False:
            i = 10
            return i + 15
        return {'children': collections.OrderedDict([(name, PlaceholderBoundBlock(block, value.get(name), prefix=f'{prefix}-{name}')) for (name, block) in self.child_blocks.items()]), 'help_text': getattr(self.meta, 'help_text', None), 'classname': self.meta.form_classname, 'block_definition': self, 'prefix': prefix}

    class Meta:
        default = {}
        form_classname = 'struct-block'
        form_template = None
        value_class = StructValue
        label_format = None
        icon = 'placeholder'

class StructBlock(BaseStructBlock, metaclass=DeclarativeSubBlocksMetaclass):
    pass

class StructBlockAdapter(Adapter):
    js_constructor = 'wagtail.blocks.StructBlock'

    def js_args(self, block):
        if False:
            print('Hello World!')
        meta = {'label': block.label, 'required': block.required, 'icon': block.meta.icon, 'classname': block.meta.form_classname}
        help_text = getattr(block.meta, 'help_text', None)
        if help_text:
            meta['helpText'] = help_text
            meta['helpIcon'] = get_help_icon()
        if block.meta.form_template:
            meta['formTemplate'] = block.render_form_template()
        if block.meta.label_format:
            meta['labelFormat'] = block.meta.label_format
        return [block.name, block.child_blocks.values(), meta]

    @cached_property
    def media(self):
        if False:
            print('Hello World!')
        return forms.Media(js=[versioned_static('wagtailadmin/js/telepath/blocks.js')])
register(StructBlockAdapter(), StructBlock)