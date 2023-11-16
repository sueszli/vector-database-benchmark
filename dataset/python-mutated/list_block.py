import uuid
from collections.abc import Mapping, MutableSequence
from django import forms
from django.core.exceptions import ValidationError
from django.forms.utils import ErrorList
from django.utils.functional import cached_property
from django.utils.html import format_html, format_html_join
from django.utils.translation import gettext as _
from wagtail.admin.staticfiles import versioned_static
from wagtail.telepath import Adapter, register
from .base import Block, BoundBlock, get_error_json_data, get_error_list_json_data, get_help_icon
__all__ = ['ListBlock', 'ListBlockValidationError']

class ListBlockValidationError(ValidationError):

    def __init__(self, block_errors=None, non_block_errors=None):
        if False:
            return 10
        self.non_block_errors = ErrorList(non_block_errors)
        if block_errors is None:
            block_errors_dict = {}
        elif isinstance(block_errors, Mapping):
            block_errors_dict = block_errors
        elif isinstance(block_errors, list):
            block_errors_dict = {index: val for (index, val) in enumerate(block_errors) if val is not None}
        else:
            raise ValueError('Expected dict or list for block_errors, got %r' % block_errors)
        self.block_errors = {}
        for (index, val) in block_errors_dict.items():
            if isinstance(val, ErrorList):
                self.block_errors[index] = val.as_data()[0]
            elif isinstance(val, list):
                self.block_errors[index] = val[0]
            else:
                self.block_errors[index] = val
        super().__init__('Validation error in ListBlock')

    def as_json_data(self):
        if False:
            return 10
        result = {}
        if self.non_block_errors:
            result['messages'] = get_error_list_json_data(self.non_block_errors)
        if self.block_errors:
            result['blockErrors'] = {index: get_error_json_data(error) for (index, error) in self.block_errors.items()}
        return result

class ListValue(MutableSequence):
    """
    The native data type used by ListBlock. Behaves as a list of values, but also provides
    a bound_blocks property giving access to block IDs
    """

    class ListChild(BoundBlock):

        def __init__(self, *args, **kwargs):
            if False:
                return 10
            self.original_id = kwargs.pop('id', None)
            self.id = self.original_id or str(uuid.uuid4())
            super().__init__(*args, **kwargs)

        def get_prep_value(self):
            if False:
                i = 10
                return i + 15
            return {'type': 'item', 'value': self.block.get_prep_value(self.value), 'id': self.id}

    def __init__(self, list_block, values=None, bound_blocks=None):
        if False:
            return 10
        self.list_block = list_block
        if bound_blocks is not None:
            self.bound_blocks = bound_blocks
        elif values is not None:
            self.bound_blocks = [ListValue.ListChild(self.list_block.child_block, value) for value in values]
        else:
            self.bound_blocks = []

    def __getitem__(self, i):
        if False:
            return 10
        return self.bound_blocks[i].value

    def __setitem__(self, i, item):
        if False:
            i = 10
            return i + 15
        self.bound_blocks[i] = ListValue.ListChild(self.list_block.child_block, item)

    def __delitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        del self.bound_blocks[i]

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.bound_blocks)

    def insert(self, i, item):
        if False:
            i = 10
            return i + 15
        self.bound_blocks.insert(i, ListValue.ListChild(self.list_block.child_block, item))

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'<ListValue: {[bb.value for bb in self.bound_blocks]!r}>'

class ListBlock(Block):

    def __init__(self, child_block, search_index=True, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.search_index = search_index
        if isinstance(child_block, type):
            self.child_block = child_block()
        else:
            self.child_block = child_block
        if not hasattr(self.meta, 'default'):
            self.meta.default = [self.child_block.get_default()]

    def get_default(self):
        if False:
            return 10
        return ListValue(self, values=list(self.meta.default))

    def value_from_datadict(self, data, files, prefix):
        if False:
            return 10
        count = int(data['%s-count' % prefix])
        child_blocks_with_indexes = []
        for i in range(0, count):
            if data['%s-%d-deleted' % (prefix, i)]:
                continue
            child_blocks_with_indexes.append((int(data['%s-%d-order' % (prefix, i)]), ListValue.ListChild(self.child_block, self.child_block.value_from_datadict(data, files, '%s-%d-value' % (prefix, i)), id=data.get('%s-%d-id' % (prefix, i)))))
        child_blocks_with_indexes.sort()
        return ListValue(self, bound_blocks=[b for (i, b) in child_blocks_with_indexes])

    def value_omitted_from_data(self, data, files, prefix):
        if False:
            while True:
                i = 10
        return '%s-count' % prefix not in data

    def clean(self, value):
        if False:
            print('Hello World!')
        if not isinstance(value, ListValue):
            value = ListValue(self, values=value)
        result = []
        block_errors = {}
        non_block_errors = ErrorList()
        for (index, bound_block) in enumerate(value.bound_blocks):
            try:
                result.append(ListValue.ListChild(self.child_block, self.child_block.clean(bound_block.value), id=bound_block.id))
            except ValidationError as e:
                block_errors[index] = e
        if self.meta.min_num is not None and self.meta.min_num > len(value):
            non_block_errors.append(ValidationError(_('The minimum number of items is %(min_num)d') % {'min_num': self.meta.min_num}))
        if self.meta.max_num is not None and self.meta.max_num < len(value):
            non_block_errors.append(ValidationError(_('The maximum number of items is %(max_num)d') % {'max_num': self.meta.max_num}))
        if block_errors or non_block_errors:
            raise ListBlockValidationError(block_errors=block_errors, non_block_errors=non_block_errors)
        return ListValue(self, bound_blocks=result)

    def _item_is_in_block_format(self, item):
        if False:
            return 10
        return isinstance(item, dict) and 'id' in item and ('value' in item) and (item.get('type') == 'item')

    def to_python(self, value):
        if False:
            for i in range(10):
                print('nop')
        raw_values = [item['value'] if self._item_is_in_block_format(item) else item for item in value]
        converted_values = self.child_block.bulk_to_python(raw_values)
        bound_blocks = []
        for (i, item) in enumerate(value):
            if self._item_is_in_block_format(item):
                list_item_id = item['id']
            else:
                list_item_id = None
            bound_blocks.append(ListValue.ListChild(self.child_block, converted_values[i], id=list_item_id))
        return ListValue(self, bound_blocks=bound_blocks)

    def bulk_to_python(self, values):
        if False:
            return 10
        lengths = []
        raw_values = []
        for list_stream in values:
            lengths.append(len(list_stream))
            for list_child in list_stream:
                if self._item_is_in_block_format(list_child):
                    raw_values.append(list_child['value'])
                else:
                    raw_values.append(list_child)
        converted_values = self.child_block.bulk_to_python(raw_values)
        result = []
        offset = 0
        values = list(values)
        for (i, sublist_len) in enumerate(lengths):
            bound_blocks = []
            for j in range(sublist_len):
                if self._item_is_in_block_format(values[i][j]):
                    list_item_id = values[i][j]['id']
                else:
                    list_item_id = None
                bound_blocks.append(ListValue.ListChild(self.child_block, converted_values[offset + j], id=list_item_id))
            result.append(ListValue(self, bound_blocks=bound_blocks))
            offset += sublist_len
        return result

    def get_prep_value(self, value):
        if False:
            return 10
        if not isinstance(value, ListValue):
            value = ListValue(self, values=value)
        prep_value = []
        for item in value.bound_blocks:
            if not item.id:
                item.id = str(uuid.uuid4())
            prep_value.append(item.get_prep_value())
        return prep_value

    def get_form_state(self, value):
        if False:
            i = 10
            return i + 15
        if not isinstance(value, ListValue):
            value = ListValue(self, values=value)
        return [{'value': self.child_block.get_form_state(block.value), 'id': block.id} for block in value.bound_blocks]

    def get_api_representation(self, value, context=None):
        if False:
            return 10
        return [self.child_block.get_api_representation(item, context=context) for item in value]

    def render_basic(self, value, context=None):
        if False:
            for i in range(10):
                print('nop')
        children = format_html_join('\n', '<li>{0}</li>', [(self.child_block.render(child_value, context=context),) for child_value in value])
        return format_html('<ul>{0}</ul>', children)

    def get_searchable_content(self, value):
        if False:
            print('Hello World!')
        if not self.search_index:
            return []
        content = []
        for child_value in value:
            content.extend(self.child_block.get_searchable_content(child_value))
        return content

    def extract_references(self, value):
        if False:
            while True:
                i = 10
        for child in value.bound_blocks:
            for (model, object_id, model_path, content_path) in child.block.extract_references(child.value):
                model_path = f'item.{model_path}' if model_path else 'item'
                content_path = f'{child.id}.{content_path}' if content_path else child.id
                yield (model, object_id, model_path, content_path)

    def get_block_by_content_path(self, value, path_elements):
        if False:
            print('Hello World!')
        '\n        Given a list of elements from a content path, retrieve the block at that path\n        as a BoundBlock object, or None if the path does not correspond to a valid block.\n        '
        if path_elements:
            (id, *remaining_elements) = path_elements
            for child in value.bound_blocks:
                if child.id == id:
                    return child.block.get_block_by_content_path(child.value, remaining_elements)
        else:
            return self.bind(value)

    def check(self, **kwargs):
        if False:
            print('Hello World!')
        errors = super().check(**kwargs)
        errors.extend(self.child_block.check(**kwargs))
        return errors

    class Meta:
        icon = 'placeholder'
        form_classname = None
        min_num = None
        max_num = None
        collapsed = False
    MUTABLE_META_ATTRIBUTES = ['min_num', 'max_num']

class ListBlockAdapter(Adapter):
    js_constructor = 'wagtail.blocks.ListBlock'

    def js_args(self, block):
        if False:
            for i in range(10):
                print('nop')
        meta = {'label': block.label, 'icon': block.meta.icon, 'classname': block.meta.form_classname, 'collapsed': block.meta.collapsed, 'strings': {'MOVE_UP': _('Move up'), 'MOVE_DOWN': _('Move down'), 'DUPLICATE': _('Duplicate'), 'DELETE': _('Delete'), 'ADD': _('Add')}}
        help_text = getattr(block.meta, 'help_text', None)
        if help_text:
            meta['helpText'] = help_text
            meta['helpIcon'] = get_help_icon()
        if block.meta.min_num is not None:
            meta['minNum'] = block.meta.min_num
        if block.meta.max_num is not None:
            meta['maxNum'] = block.meta.max_num
        return [block.name, block.child_block, block.child_block.get_form_state(block.child_block.get_default()), meta]

    @cached_property
    def media(self):
        if False:
            while True:
                i = 10
        return forms.Media(js=[versioned_static('wagtailadmin/js/telepath/blocks.js')])
register(ListBlockAdapter(), ListBlock)
DECONSTRUCT_ALIASES = {ListBlock: 'wagtail.blocks.ListBlock'}