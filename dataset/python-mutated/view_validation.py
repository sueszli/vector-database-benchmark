""" View validation code (using assertions, not the RNG schema). """
import collections
import logging
_logger = logging.getLogger(__name__)
_validators = collections.defaultdict(list)

def valid_view(arch):
    if False:
        while True:
            i = 10
    for pred in _validators[arch.tag]:
        if not pred(arch):
            _logger.error('Invalid XML: %s', pred.__doc__)
            return False
    return True

def validate(*view_types):
    if False:
        while True:
            i = 10
    ' Registers a view-validation function for the specific view types\n    '

    def decorator(fn):
        if False:
            while True:
                i = 10
        for arch in view_types:
            _validators[arch].append(fn)
        return fn
    return decorator

@validate('form')
def valid_page_in_book(arch):
    if False:
        while True:
            i = 10
    'A `page` node must be below a `notebook` node.'
    return not arch.xpath('//page[not(ancestor::notebook)]')

@validate('graph')
def valid_field_in_graph(arch):
    if False:
        i = 10
        return i + 15
    ' Children of ``graph`` can only be ``field`` '
    return all((child.tag == 'field' for child in arch.xpath('/graph/*')))

@validate('tree')
def valid_field_in_tree(arch):
    if False:
        for i in range(10):
            print('nop')
    ' Children of ``tree`` view must be ``field`` or ``button``.'
    return all((child.tag in ('field', 'button') for child in arch.xpath('/tree/*')))

@validate('form', 'graph', 'tree')
def valid_att_in_field(arch):
    if False:
        for i in range(10):
            print('nop')
    ' ``field`` nodes must all have a ``@name`` '
    return not arch.xpath('//field[not(@name)]')

@validate('form')
def valid_att_in_label(arch):
    if False:
        print('Hello World!')
    ' ``label`` nodes must have a ``@for`` or a ``@string`` '
    return not arch.xpath('//label[not(@for or @string)]')

@validate('form')
def valid_att_in_form(arch):
    if False:
        for i in range(10):
            print('nop')
    return True

@validate('form')
def valid_type_in_colspan(arch):
    if False:
        print('Hello World!')
    'A `colspan` attribute must be an `integer` type.'
    return all((attrib.isdigit() for attrib in arch.xpath('//@colspan')))

@validate('form')
def valid_type_in_col(arch):
    if False:
        for i in range(10):
            print('nop')
    'A `col` attribute must be an `integer` type.'
    return all((attrib.isdigit() for attrib in arch.xpath('//@col')))