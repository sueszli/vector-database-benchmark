"""Validate properties expanders."""
import functools
from tinycss2.ast import DimensionToken, IdentToken, NumberToken
from tinycss2.color3 import parse_color
from ..properties import INITIAL_VALUES
from ..utils import InvalidValues, get_keyword, get_single_keyword, split_on_comma
from .descriptors import expand_font_variant
from .properties import background_attachment, background_image, background_position, background_repeat, background_size, block_ellipsis, border_style, border_width, box, column_count, column_width, flex_basis, flex_direction, flex_grow_shrink, flex_wrap, font_family, font_size, font_stretch, font_style, font_weight, line_height, list_style_image, list_style_position, list_style_type, other_colors, overflow_wrap, validate_non_shorthand
EXPANDERS = {}

def expander(property_name):
    if False:
        print('Hello World!')
    'Decorator adding a function to the ``EXPANDERS``.'

    def expander_decorator(function):
        if False:
            i = 10
            return i + 15
        'Add ``function`` to the ``EXPANDERS``.'
        assert property_name not in EXPANDERS, property_name
        EXPANDERS[property_name] = function
        return function
    return expander_decorator

@expander('border-color')
@expander('border-style')
@expander('border-width')
@expander('margin')
@expander('padding')
@expander('bleed')
def expand_four_sides(base_url, name, tokens):
    if False:
        for i in range(10):
            print('nop')
    'Expand properties setting a token for the four sides of a box.'
    if len(tokens) == 1:
        tokens *= 4
    elif len(tokens) == 2:
        tokens *= 2
    elif len(tokens) == 3:
        tokens += (tokens[1],)
    elif len(tokens) != 4:
        raise InvalidValues(f'Expected 1 to 4 token components got {len(tokens)}')
    for (suffix, token) in zip(('-top', '-right', '-bottom', '-left'), tokens):
        i = name.rfind('-')
        if i == -1:
            new_name = name + suffix
        else:
            new_name = name[:i] + suffix + name[i:]
        (result,) = validate_non_shorthand(base_url, new_name, [token], required=True)
        yield result

def generic_expander(*expanded_names, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "Decorator helping expanders to handle ``inherit`` and ``initial``.\n\n    Wrap an expander so that it does not have to handle the 'inherit' and\n    'initial' cases, and can just yield name suffixes. Missing suffixes\n    get the initial value.\n\n    "
    wants_base_url = kwargs.pop('wants_base_url', False)
    assert not kwargs

    def generic_expander_decorator(wrapped):
        if False:
            return 10
        'Decorate the ``wrapped`` expander.'

        @functools.wraps(wrapped)
        def generic_expander_wrapper(base_url, name, tokens):
            if False:
                print('Hello World!')
            'Wrap the expander.'
            keyword = get_single_keyword(tokens)
            if keyword in ('inherit', 'initial'):
                results = dict.fromkeys(expanded_names, keyword)
                skip_validation = True
            else:
                skip_validation = False
                results = {}
                if wants_base_url:
                    result = wrapped(name, tokens, base_url)
                else:
                    result = wrapped(name, tokens)
                for (new_name, new_token) in result:
                    assert new_name in expanded_names, new_name
                    if new_name in results:
                        raise InvalidValues(f"got multiple {new_name.strip('-')} values in a {name} shorthand")
                    results[new_name] = new_token
            for new_name in expanded_names:
                if new_name.startswith('-'):
                    actual_new_name = name + new_name
                else:
                    actual_new_name = new_name
                if new_name in results:
                    value = results[new_name]
                    if not skip_validation:
                        ((actual_new_name, value),) = validate_non_shorthand(base_url, actual_new_name, value, required=True)
                else:
                    value = 'initial'
                yield (actual_new_name, value)
        return generic_expander_wrapper
    return generic_expander_decorator

@expander('border-radius')
@generic_expander('border-top-left-radius', 'border-top-right-radius', 'border-bottom-right-radius', 'border-bottom-left-radius', wants_base_url=True)
def border_radius(name, tokens, base_url):
    if False:
        for i in range(10):
            print('nop')
    'Validator for the ``border-radius`` property.'
    current = horizontal = []
    vertical = []
    for token in tokens:
        if token.type == 'literal' and token.value == '/':
            if current is horizontal:
                if token == tokens[-1]:
                    raise InvalidValues('Expected value after "/" separator')
                else:
                    current = vertical
            else:
                raise InvalidValues('Expected only one "/" separator')
        else:
            current.append(token)
    if not vertical:
        vertical = horizontal[:]
    for values in (horizontal, vertical):
        if len(values) == 1:
            values *= 4
        elif len(values) == 2:
            values *= 2
        elif len(values) == 3:
            values.append(values[1])
        elif len(values) != 4:
            raise InvalidValues(f'Expected 1 to 4 token components got {len(values)}')
    corners = ('top-left', 'top-right', 'bottom-right', 'bottom-left')
    for (corner, tokens) in zip(corners, zip(horizontal, vertical)):
        new_name = f'border-{corner}-radius'
        validate_non_shorthand(base_url, new_name, tokens, required=True)
        yield (new_name, tokens)

@expander('list-style')
@generic_expander('-type', '-position', '-image', wants_base_url=True)
def expand_list_style(name, tokens, base_url):
    if False:
        return 10
    'Expand the ``list-style`` shorthand property.\n\n    See https://www.w3.org/TR/CSS21/generate.html#propdef-list-style\n\n    '
    type_specified = image_specified = False
    none_count = 0
    for token in tokens:
        if get_keyword(token) == 'none':
            none_count += 1
            none_token = token
            continue
        if list_style_image([token], base_url) is not None:
            suffix = '-image'
            image_specified = True
        elif list_style_position([token]) is not None:
            suffix = '-position'
        elif list_style_type([token]) is not None:
            suffix = '-type'
            type_specified = True
        else:
            raise InvalidValues
        yield (suffix, [token])
    if not type_specified and none_count:
        yield ('-type', [none_token])
        none_count -= 1
    if not image_specified and none_count:
        yield ('-image', [none_token])
        none_count -= 1
    if none_count:
        raise InvalidValues

@expander('border')
def expand_border(base_url, name, tokens):
    if False:
        print('Hello World!')
    'Expand the ``border`` shorthand property.\n\n    See https://www.w3.org/TR/CSS21/box.html#propdef-border\n\n    '
    for suffix in ('-top', '-right', '-bottom', '-left'):
        for new_prop in expand_border_side(base_url, name + suffix, tokens):
            yield new_prop

@expander('border-top')
@expander('border-right')
@expander('border-bottom')
@expander('border-left')
@expander('column-rule')
@expander('outline')
@generic_expander('-width', '-color', '-style')
def expand_border_side(name, tokens):
    if False:
        while True:
            i = 10
    'Expand the ``border-*`` shorthand properties.\n\n    See https://www.w3.org/TR/CSS21/box.html#propdef-border-top\n\n    '
    for token in tokens:
        if parse_color(token) is not None:
            suffix = '-color'
        elif border_width([token]) is not None:
            suffix = '-width'
        elif border_style([token]) is not None:
            suffix = '-style'
        else:
            raise InvalidValues
        yield (suffix, [token])

@expander('background')
def expand_background(base_url, name, tokens):
    if False:
        while True:
            i = 10
    'Expand the ``background`` shorthand property.\n\n    See https://drafts.csswg.org/css-backgrounds-3/#the-background\n\n    '
    properties = ['background_color', 'background_image', 'background_repeat', 'background_attachment', 'background_position', 'background_size', 'background_clip', 'background_origin']
    keyword = get_single_keyword(tokens)
    if keyword in ('initial', 'inherit'):
        for name in properties:
            yield (name, keyword)
        return

    def parse_layer(tokens, final_layer=False):
        if False:
            for i in range(10):
                print('nop')
        results = {}

        def add(name, value):
            if False:
                return 10
            if value is None:
                return False
            name = f'background_{name}'
            if name in results:
                raise InvalidValues
            results[name] = value
            return True
        tokens = tokens[::-1]
        while tokens:
            if add('repeat', background_repeat.single_value(tokens[-2:][::-1])):
                del tokens[-2:]
                continue
            token = tokens[-1:]
            if final_layer and add('color', other_colors(token)):
                tokens.pop()
                continue
            if add('image', background_image.single_value(token, base_url)):
                tokens.pop()
                continue
            if add('repeat', background_repeat.single_value(token)):
                tokens.pop()
                continue
            if add('attachment', background_attachment.single_value(token)):
                tokens.pop()
                continue
            for n in (4, 3, 2, 1)[-len(tokens):]:
                n_tokens = tokens[-n:][::-1]
                position = background_position.single_value(n_tokens)
                if position is not None:
                    assert add('position', position)
                    del tokens[-n:]
                    if tokens and tokens[-1].type == 'literal' and (tokens[-1].value == '/'):
                        for n in (3, 2)[-len(tokens):]:
                            n_tokens = tokens[-n:-1][::-1]
                            size = background_size.single_value(n_tokens)
                            if size is not None:
                                assert add('size', size)
                                del tokens[-n:]
                    break
            if position is not None:
                continue
            if add('origin', box.single_value(token)):
                tokens.pop()
                next_token = tokens[-1:]
                if add('clip', box.single_value(next_token)):
                    tokens.pop()
                else:
                    add('clip', box.single_value(token))
                continue
            raise InvalidValues
        color = results.pop('background_color', INITIAL_VALUES['background_color'])
        for name in properties:
            if name not in results:
                results[name] = INITIAL_VALUES[name][0]
        return (color, results)
    layers = reversed(split_on_comma(tokens))
    (color, last_layer) = parse_layer(next(layers), final_layer=True)
    results = dict(((k, [v]) for (k, v) in last_layer.items()))
    for tokens in layers:
        (_, layer) = parse_layer(tokens)
        for (name, value) in layer.items():
            results[name].append(value)
    for (name, values) in results.items():
        yield (name, values[::-1])
    yield ('background-color', color)

@expander('text-decoration')
@generic_expander('-line', '-color', '-style')
def expand_text_decoration(name, tokens):
    if False:
        i = 10
        return i + 15
    'Expand the ``text-decoration`` shorthand property.'
    text_decoration_line = []
    text_decoration_color = []
    text_decoration_style = []
    none_in_line = False
    for token in tokens:
        keyword = get_keyword(token)
        if keyword in ('none', 'underline', 'overline', 'line-through', 'blink'):
            text_decoration_line.append(token)
            if none_in_line:
                raise InvalidValues
            elif keyword == 'none':
                none_in_line = True
        elif keyword in ('solid', 'double', 'dotted', 'dashed', 'wavy'):
            if text_decoration_style:
                raise InvalidValues
            else:
                text_decoration_style.append(token)
        else:
            color = parse_color(token)
            if color is None:
                raise InvalidValues
            elif text_decoration_color:
                raise InvalidValues
            else:
                text_decoration_color.append(token)
    if text_decoration_line:
        yield ('-line', text_decoration_line)
    if text_decoration_color:
        yield ('-color', text_decoration_color)
    if text_decoration_style:
        yield ('-style', text_decoration_style)

def expand_page_break_before_after(name, tokens):
    if False:
        return 10
    'Expand legacy ``page-break-before`` and ``page-break-after`` properties.\n\n    See https://www.w3.org/TR/css-break-3/#page-break-properties\n\n    '
    keyword = get_single_keyword(tokens)
    new_name = name.split('-', 1)[1]
    if keyword in ('auto', 'left', 'right', 'avoid'):
        yield (new_name, tokens)
    elif keyword == 'always':
        token = IdentToken(tokens[0].source_line, tokens[0].source_column, 'page')
        yield (new_name, [token])
    else:
        raise InvalidValues

@expander('page-break-after')
@generic_expander('break-after')
def expand_page_break_after(name, tokens):
    if False:
        return 10
    'Expand legacy ``page-break-after`` property.\n\n    See https://www.w3.org/TR/css-break-3/#page-break-properties\n\n    '
    return expand_page_break_before_after(name, tokens)

@expander('page-break-before')
@generic_expander('break-before')
def expand_page_break_before(name, tokens):
    if False:
        print('Hello World!')
    'Expand legacy ``page-break-before`` property.\n\n    See https://www.w3.org/TR/css-break-3/#page-break-properties\n\n    '
    return expand_page_break_before_after(name, tokens)

@expander('page-break-inside')
@generic_expander('break-inside')
def expand_page_break_inside(name, tokens):
    if False:
        for i in range(10):
            print('nop')
    'Expand the legacy ``page-break-inside`` property.\n\n    See https://www.w3.org/TR/css-break-3/#page-break-properties\n\n    '
    keyword = get_single_keyword(tokens)
    if keyword in ('auto', 'avoid'):
        yield ('break-inside', tokens)
    else:
        raise InvalidValues

@expander('columns')
@generic_expander('column-width', 'column-count')
def expand_columns(name, tokens):
    if False:
        while True:
            i = 10
    'Expand the ``columns`` shorthand property.'
    name = None
    if len(tokens) == 2 and get_keyword(tokens[0]) == 'auto':
        tokens = tokens[::-1]
    for token in tokens:
        if column_width([token]) is not None and name != 'column-width':
            name = 'column-width'
        elif column_count([token]) is not None:
            name = 'column-count'
        else:
            raise InvalidValues
        yield (name, [token])
    if len(tokens) == 1:
        name = 'column-width' if name == 'column-count' else 'column-count'
        token = IdentToken(tokens[0].source_line, tokens[0].source_column, 'auto')
        yield (name, [token])

@expander('font-variant')
@generic_expander('-alternates', '-caps', '-east-asian', '-ligatures', '-numeric', '-position')
def font_variant(name, tokens):
    if False:
        while True:
            i = 10
    'Expand the ``font-variant`` shorthand property.\n\n    https://www.w3.org/TR/css-fonts-3/#font-variant-prop\n\n    '
    return expand_font_variant(tokens)

@expander('font')
@generic_expander('-style', '-variant-caps', '-weight', '-stretch', '-size', 'line-height', '-family')
def expand_font(name, tokens):
    if False:
        return 10
    'Expand the ``font`` shorthand property.\n\n    https://www.w3.org/TR/css-fonts-3/#font-prop\n\n    '
    expand_font_keyword = get_single_keyword(tokens)
    if expand_font_keyword in ('caption', 'icon', 'menu', 'message-box', 'small-caption', 'status-bar'):
        raise InvalidValues('System fonts are not supported')
    tokens = list(reversed(tokens))
    for _ in range(4):
        token = tokens.pop()
        if get_keyword(token) == 'normal':
            continue
        if font_style([token]) is not None:
            suffix = '-style'
        elif get_keyword(token) in ('normal', 'small-caps'):
            suffix = '-variant-caps'
        elif font_weight([token]) is not None:
            suffix = '-weight'
        elif font_stretch([token]) is not None:
            suffix = '-stretch'
        else:
            break
        yield (suffix, [token])
        if not tokens:
            raise InvalidValues
    else:
        if not tokens:
            raise InvalidValues
        token = tokens.pop()
    if font_size([token]) is None:
        raise InvalidValues
    yield ('-size', [token])
    if not tokens:
        raise InvalidValues
    token = tokens.pop()
    if token.type == 'literal' and token.value == '/':
        token = tokens.pop()
        if line_height([token]) is None:
            raise InvalidValues
        yield ('line-height', [token])
    else:
        tokens.append(token)
    tokens.reverse()
    if font_family(tokens) is None:
        raise InvalidValues
    yield ('-family', tokens)

@expander('word-wrap')
@generic_expander('overflow-wrap')
def expand_word_wrap(name, tokens):
    if False:
        print('Hello World!')
    'Expand the ``word-wrap`` legacy property.\n\n    See https://www.w3.org/TR/css-text-3/#overflow-wrap\n\n    '
    keyword = overflow_wrap(tokens)
    if keyword is None:
        raise InvalidValues
    yield ('overflow-wrap', tokens)

@expander('flex')
@generic_expander('-grow', '-shrink', '-basis')
def expand_flex(name, tokens):
    if False:
        return 10
    'Expand the ``flex`` property.'
    keyword = get_single_keyword(tokens)
    if keyword == 'none':
        (line, column) = (tokens[0].source_line, tokens[0].source_column)
        zero_token = NumberToken(line, column, 0, 0, '0')
        auto_token = IdentToken(line, column, 'auto')
        yield ('-grow', [zero_token])
        yield ('-shrink', [zero_token])
        yield ('-basis', [auto_token])
    else:
        (grow, shrink, basis) = (1, 1, None)
        (grow_found, shrink_found, basis_found) = (False, False, False)
        for token in tokens:
            forced_flex_factor = token.type == 'number' and token.int_value == 0 and (not all((grow_found, shrink_found)))
            if not basis_found and (not forced_flex_factor):
                new_basis = flex_basis([token])
                if new_basis is not None:
                    basis = token
                    basis_found = True
                    continue
            if not grow_found:
                new_grow = flex_grow_shrink([token])
                if new_grow is None:
                    raise InvalidValues
                else:
                    grow = new_grow
                    grow_found = True
                    continue
            elif not shrink_found:
                new_shrink = flex_grow_shrink([token])
                if new_shrink is None:
                    raise InvalidValues
                else:
                    shrink = new_shrink
                    shrink_found = True
                    continue
            else:
                raise InvalidValues
        (line, column) = (tokens[0].source_line, tokens[0].source_column)
        int_grow = int(grow) if float(grow).is_integer() else None
        int_shrink = int(shrink) if float(shrink).is_integer() else None
        grow_token = NumberToken(line, column, grow, int_grow, str(grow))
        shrink_token = NumberToken(line, column, shrink, int_shrink, str(shrink))
        if not basis_found:
            basis = DimensionToken(line, column, 0, 0, '0', 'px')
        yield ('-grow', [grow_token])
        yield ('-shrink', [shrink_token])
        yield ('-basis', [basis])

@expander('flex-flow')
@generic_expander('flex-direction', 'flex-wrap')
def expand_flex_flow(name, tokens):
    if False:
        return 10
    'Expand the ``flex-flow`` property.'
    if len(tokens) == 2:
        for sorted_tokens in (tokens, tokens[::-1]):
            direction = flex_direction([sorted_tokens[0]])
            wrap = flex_wrap([sorted_tokens[1]])
            if direction and wrap:
                yield ('flex-direction', [sorted_tokens[0]])
                yield ('flex-wrap', [sorted_tokens[1]])
                break
        else:
            raise InvalidValues
    elif len(tokens) == 1:
        direction = flex_direction([tokens[0]])
        if direction:
            yield ('flex-direction', [tokens[0]])
        else:
            wrap = flex_wrap([tokens[0]])
            if wrap:
                yield ('flex-wrap', [tokens[0]])
            else:
                raise InvalidValues
    else:
        raise InvalidValues

@expander('line-clamp')
@generic_expander('max-lines', 'continue', 'block-ellipsis')
def expand_line_clamp(name, tokens):
    if False:
        while True:
            i = 10
    'Expand the ``line-clamp`` property.'
    if len(tokens) == 1:
        keyword = get_single_keyword(tokens)
        if keyword == 'none':
            (line, column) = (tokens[0].source_line, tokens[0].source_column)
            none_token = IdentToken(line, column, 'none')
            auto_token = IdentToken(line, column, 'auto')
            yield ('max-lines', [none_token])
            yield ('continue', [auto_token])
            yield ('block-ellipsis', [none_token])
        elif tokens[0].type == 'number' and tokens[0].int_value is not None:
            (line, column) = (tokens[0].source_line, tokens[0].source_column)
            auto_token = IdentToken(line, column, 'auto')
            discard_token = IdentToken(line, column, 'discard')
            yield ('max-lines', [tokens[0]])
            yield ('continue', [discard_token])
            yield ('block-ellipsis', [auto_token])
        else:
            raise InvalidValues
    elif len(tokens) == 2:
        if tokens[0].type == 'number':
            max_lines = tokens[0].int_value
            ellipsis = block_ellipsis([tokens[1]])
            if max_lines and ellipsis is not None:
                (line, column) = (tokens[0].source_line, tokens[0].source_column)
                discard_token = IdentToken(line, column, 'discard')
                yield ('max-lines', [tokens[0]])
                yield ('continue', [discard_token])
                yield ('block-ellipsis', [tokens[1]])
            else:
                raise InvalidValues
        else:
            raise InvalidValues
    else:
        raise InvalidValues

@expander('text-align')
@generic_expander('-all', '-last')
def expand_text_align(name, tokens):
    if False:
        i = 10
        return i + 15
    'Expand the ``text-align`` property.'
    if len(tokens) == 1:
        keyword = get_single_keyword(tokens)
        if keyword is None:
            raise InvalidValues
        if keyword == 'justify-all':
            (line, column) = (tokens[0].source_line, tokens[0].source_column)
            align_all = IdentToken(line, column, 'justify')
        else:
            align_all = tokens[0]
        yield ('-all', [align_all])
        if keyword == 'justify':
            (line, column) = (tokens[0].source_line, tokens[0].source_column)
            align_last = IdentToken(line, column, 'start')
        else:
            align_last = align_all
        yield ('-last', [align_last])
    else:
        raise InvalidValues