import re
from html.parser import HTMLParser
from wagtail.admin.rich_text.converters.contentstate_models import Block, ContentState, Entity, EntityRange, InlineStyleRange
from wagtail.admin.rich_text.converters.html_ruleset import HTMLRuleset
from wagtail.models import Page
from wagtail.rich_text import features as feature_registry
STRIP_WHITESPACE = 0
KEEP_WHITESPACE = 1
FORCE_WHITESPACE = 2
WHITESPACE_RE = re.compile('[ \\t\\n\\f\\r]+')
BLOCK_KEY_NAME = 'data-block-key'

class HandlerState:

    def __init__(self):
        if False:
            print('Hello World!')
        self.current_block = None
        self.current_inline_styles = []
        self.current_entity_ranges = []
        self.leading_whitespace = STRIP_WHITESPACE
        self.list_depth = 0
        self.list_item_type = None
        self.has_preceding_nonatomic_block = False
        self.pushed_states = []

    def push(self):
        if False:
            i = 10
            return i + 15
        self.pushed_states.append({'current_block': self.current_block, 'current_inline_styles': self.current_inline_styles, 'current_entity_ranges': self.current_entity_ranges, 'leading_whitespace': self.leading_whitespace, 'list_depth': self.list_depth, 'list_item_type': self.list_item_type})

    def pop(self):
        if False:
            print('Hello World!')
        last_state = self.pushed_states.pop()
        self.current_block = last_state['current_block']
        self.current_inline_styles = last_state['current_inline_styles']
        self.current_entity_ranges = last_state['current_entity_ranges']
        self.leading_whitespace = last_state['leading_whitespace']
        self.list_depth = last_state['list_depth']
        self.list_item_type = last_state['list_item_type']

def add_paragraph_block(state, contentstate):
    if False:
        for i in range(10):
            print('nop')
    "\n    Utility function for adding an unstyled (paragraph) block to contentstate;\n    useful for element handlers that aren't paragraph elements themselves, but need\n    to insert paragraphs to ensure correctness\n    "
    block = Block('unstyled', depth=state.list_depth)
    contentstate.blocks.append(block)
    state.current_block = block
    state.leading_whitespace = STRIP_WHITESPACE
    state.has_preceding_nonatomic_block = True

class ListElementHandler:
    """Handler for <ul> / <ol> tags"""

    def __init__(self, list_item_type):
        if False:
            i = 10
            return i + 15
        self.list_item_type = list_item_type

    def handle_starttag(self, name, attrs, state, contentstate):
        if False:
            print('Hello World!')
        state.push()
        if state.list_item_type is None:
            pass
        else:
            state.list_depth += 1
        state.list_item_type = self.list_item_type

    def handle_endtag(self, name, state, contentstate):
        if False:
            print('Hello World!')
        state.pop()

class BlockElementHandler:

    def __init__(self, block_type):
        if False:
            while True:
                i = 10
        self.block_type = block_type

    def create_block(self, name, attrs, state, contentstate):
        if False:
            for i in range(10):
                print('nop')
        return Block(self.block_type, depth=state.list_depth, key=attrs.get(BLOCK_KEY_NAME))

    def handle_starttag(self, name, attrs, state, contentstate):
        if False:
            while True:
                i = 10
        attr_dict = dict(attrs)
        block = self.create_block(name, attr_dict, state, contentstate)
        contentstate.blocks.append(block)
        state.current_block = block
        state.leading_whitespace = STRIP_WHITESPACE
        state.has_preceding_nonatomic_block = True

    def handle_endtag(self, name, state, contentState):
        if False:
            return 10
        assert not state.current_inline_styles, 'End of block reached without closing inline style elements'
        assert not state.current_entity_ranges, 'End of block reached without closing entity elements'
        state.current_block = None

class ListItemElementHandler(BlockElementHandler):
    """Handler for <li> tag"""

    def __init__(self):
        if False:
            return 10
        pass

    def create_block(self, name, attrs, state, contentstate):
        if False:
            return 10
        assert state.list_item_type is not None, '%s element found outside of an enclosing list element' % name
        return Block(state.list_item_type, depth=state.list_depth, key=attrs.get(BLOCK_KEY_NAME))

class InlineStyleElementHandler:

    def __init__(self, style):
        if False:
            print('Hello World!')
        self.style = style

    def handle_starttag(self, name, attrs, state, contentstate):
        if False:
            while True:
                i = 10
        if state.current_block is None:
            add_paragraph_block(state, contentstate)
        if state.leading_whitespace == FORCE_WHITESPACE:
            state.current_block.text += ' '
            state.leading_whitespace = STRIP_WHITESPACE
        inline_style_range = InlineStyleRange(self.style)
        inline_style_range.offset = len(state.current_block.text)
        state.current_block.inline_style_ranges.append(inline_style_range)
        state.current_inline_styles.append(inline_style_range)

    def handle_endtag(self, name, state, contentstate):
        if False:
            return 10
        inline_style_range = state.current_inline_styles.pop()
        assert inline_style_range.style == self.style
        inline_style_range.length = len(state.current_block.text) - inline_style_range.offset

class InlineEntityElementHandler:
    """
    Abstract superclass for elements that will be represented as inline entities.
    Subclasses should define a `mutability` property
    """

    def __init__(self, entity_type):
        if False:
            for i in range(10):
                print('nop')
        self.entity_type = entity_type

    def handle_starttag(self, name, attrs, state, contentstate):
        if False:
            for i in range(10):
                print('nop')
        if state.current_block is None:
            add_paragraph_block(state, contentstate)
        if state.leading_whitespace == FORCE_WHITESPACE:
            state.current_block.text += ' '
            state.leading_whitespace = STRIP_WHITESPACE
        attrs = dict(attrs)
        entity = Entity(self.entity_type, self.mutability, self.get_attribute_data(attrs))
        key = contentstate.add_entity(entity)
        entity_range = EntityRange(key)
        entity_range.offset = len(state.current_block.text)
        state.current_block.entity_ranges.append(entity_range)
        state.current_entity_ranges.append(entity_range)

    def get_attribute_data(self, attrs):
        if False:
            while True:
                i = 10
        '\n        Given a dict of attributes found on the source element, return the data dict\n        to be associated with the resulting entity\n        '
        return {}

    def handle_endtag(self, name, state, contentstate):
        if False:
            i = 10
            return i + 15
        entity_range = state.current_entity_ranges.pop()
        entity_range.length = len(state.current_block.text) - entity_range.offset

class LinkElementHandler(InlineEntityElementHandler):
    mutability = 'MUTABLE'

class ExternalLinkElementHandler(LinkElementHandler):

    def get_attribute_data(self, attrs):
        if False:
            i = 10
            return i + 15
        return {'url': attrs['href']}

class PageLinkElementHandler(LinkElementHandler):

    def get_attribute_data(self, attrs):
        if False:
            print('Hello World!')
        try:
            page = Page.objects.get(id=attrs['id']).specific
        except Page.DoesNotExist:
            return {'id': int(attrs['id']), 'url': None, 'parentId': None}
        parent_page = page.get_parent()
        return {'id': page.id, 'url': page.url, 'parentId': parent_page.id if parent_page else None}

class AtomicBlockEntityElementHandler:
    """
    Handler for elements like <img> that exist as a single immutable item at the block level
    """

    def handle_starttag(self, name, attrs, state, contentstate):
        if False:
            while True:
                i = 10
        if state.current_block:
            next_block = Block(state.current_block.type, depth=state.current_block.depth)
            for inline_style_range in state.current_inline_styles:
                inline_style_range.length = len(state.current_block.text) - inline_style_range.offset
                new_inline_style = InlineStyleRange(inline_style_range.style)
                new_inline_style.offset = 0
                next_block.inline_style_ranges.append(new_inline_style)
            for entity_range in state.current_entity_ranges:
                entity_range.length = len(state.current_block.text) - entity_range.offset
                new_entity_range = EntityRange(entity_range.key)
                new_entity_range.offset = 0
                next_block.entity_ranges.append(new_entity_range)
            state.current_block = None
        else:
            next_block = None
        if not state.has_preceding_nonatomic_block:
            add_paragraph_block(state, contentstate)
            state.current_block = None
        attr_dict = dict(attrs)
        entity = self.create_entity(name, attr_dict, state, contentstate)
        key = contentstate.add_entity(entity)
        block = Block('atomic', depth=state.list_depth)
        contentstate.blocks.append(block)
        block.text = ' '
        entity_range = EntityRange(key)
        entity_range.offset = 0
        entity_range.length = 1
        block.entity_ranges.append(entity_range)
        state.has_preceding_nonatomic_block = False
        if next_block:
            contentstate.blocks.append(next_block)
            state.current_block = next_block
            state.current_inline_styles = next_block.inline_style_ranges.copy()
            state.current_entity_ranges = next_block.entity_ranges.copy()
            state.has_preceding_nonatomic_block = True
            state.leading_whitespace = STRIP_WHITESPACE

    def handle_endtag(self, name, state, contentstate):
        if False:
            return 10
        pass

class HorizontalRuleHandler(AtomicBlockEntityElementHandler):

    def create_entity(self, name, attrs, state, contentstate):
        if False:
            while True:
                i = 10
        return Entity('HORIZONTAL_RULE', 'IMMUTABLE', {})

class LineBreakHandler:

    def handle_starttag(self, name, attrs, state, contentstate):
        if False:
            i = 10
            return i + 15
        if state.current_block is None:
            return
        state.current_block.text += '\n'

    def handle_endtag(self, name, state, contentstate):
        if False:
            return 10
        pass

class HtmlToContentStateHandler(HTMLParser):

    def __init__(self, features=()):
        if False:
            i = 10
            return i + 15
        self.paragraph_handler = BlockElementHandler('unstyled')
        self.element_handlers = HTMLRuleset({'p': self.paragraph_handler, 'br': LineBreakHandler()})
        for feature in features:
            rule = feature_registry.get_converter_rule('contentstate', feature)
            if rule is not None:
                self.element_handlers.add_rules(rule['from_database_format'])
        super().__init__(convert_charrefs=True)

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.state = HandlerState()
        self.contentstate = ContentState()
        self.open_elements = []
        super().reset()

    def handle_starttag(self, name, attrs):
        if False:
            print('Hello World!')
        attr_dict = dict(attrs)
        element_handler = self.element_handlers.match(name, attr_dict)
        if element_handler is None and (not self.open_elements):
            element_handler = self.paragraph_handler
        self.open_elements.append((name, element_handler))
        if element_handler:
            element_handler.handle_starttag(name, attrs, self.state, self.contentstate)

    def handle_endtag(self, name):
        if False:
            for i in range(10):
                print('nop')
        if not self.open_elements:
            return
        (expected_name, element_handler) = self.open_elements.pop()
        assert name == expected_name, 'Unmatched tags: expected {}, got {}'.format(expected_name, name)
        if element_handler:
            element_handler.handle_endtag(name, self.state, self.contentstate)

    def handle_data(self, content):
        if False:
            i = 10
            return i + 15
        content = re.sub(WHITESPACE_RE, ' ', content)
        if self.state.current_block is None:
            if content == ' ':
                return
            else:
                add_paragraph_block(self.state, self.contentstate)
        if content == ' ':
            if self.state.leading_whitespace != STRIP_WHITESPACE:
                self.state.leading_whitespace = FORCE_WHITESPACE
        else:
            if self.state.leading_whitespace == STRIP_WHITESPACE:
                content = content.lstrip()
            elif self.state.leading_whitespace == FORCE_WHITESPACE and (not content.startswith(' ')):
                content = ' ' + content
            if content.endswith(' '):
                content = content.rstrip()
                self.state.leading_whitespace = FORCE_WHITESPACE
            else:
                self.state.leading_whitespace = KEEP_WHITESPACE
            self.state.current_block.text += content

    def close(self):
        if False:
            print('Hello World!')
        if not self.state.has_preceding_nonatomic_block:
            add_paragraph_block(self.state, self.contentstate)
        super().close()