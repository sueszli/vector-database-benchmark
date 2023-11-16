"""Decide where to break text lines."""
import re
from math import inf
import pyphen
from .constants import LST_TO_ISO, PANGO_WRAP_MODE
from .ffi import ffi, gobject, pango, pangoft2, unicode_to_char_p, units_from_double, units_to_double
from .fonts import font_features, get_font_description

def line_size(line, style):
    if False:
        return 10
    'Get logical width and height of the given ``line``.\n\n    ``style`` is used to add letter spacing (if needed).\n\n    '
    logical_extents = ffi.new('PangoRectangle *')
    pango.pango_layout_line_get_extents(line, ffi.NULL, logical_extents)
    width = units_to_double(logical_extents.width)
    height = units_to_double(logical_extents.height)
    ffi.release(logical_extents)
    if style['letter_spacing'] != 'normal':
        width += style['letter_spacing']
    return (width, height)

def first_line_metrics(first_line, text, layout, resume_at, space_collapse, style, hyphenated=False, hyphenation_character=None):
    if False:
        return 10
    length = first_line.length
    if hyphenated:
        length -= len(hyphenation_character.encode())
    elif resume_at:
        pango.pango_layout_set_width(layout.layout, -1)
        first_line_text = text.encode()[:length].decode()
        if space_collapse:
            first_line_text = first_line_text.rstrip(' ')
        layout.set_text(first_line_text)
        (first_line, _) = layout.get_first_line()
        length = first_line.length if first_line is not None else 0
    (width, height) = line_size(first_line, style)
    baseline = units_to_double(pango.pango_layout_get_baseline(layout.layout))
    layout.deactivate()
    return (layout, length, resume_at, width, height, baseline)

class Layout:
    """Object holding PangoLayout-related cdata pointers."""

    def __init__(self, context, style, justification_spacing=0, max_width=None):
        if False:
            print('Hello World!')
        self.justification_spacing = justification_spacing
        self.setup(context, style)
        self.max_width = max_width

    def setup(self, context, style):
        if False:
            for i in range(10):
                print('nop')
        self.context = context
        self.style = style
        self.first_line_direction = 0
        if context is None:
            font_map = ffi.gc(pangoft2.pango_ft2_font_map_new(), gobject.g_object_unref)
        else:
            font_map = context.font_config.font_map
        pango_context = ffi.gc(pango.pango_font_map_create_context(font_map), gobject.g_object_unref)
        pango.pango_context_set_round_glyph_positions(pango_context, False)
        if style['font_language_override'] != 'normal':
            (lang_p, lang) = unicode_to_char_p(LST_TO_ISO.get(style['font_language_override'].lower(), style['font_language_override']))
        elif style['lang']:
            (lang_p, lang) = unicode_to_char_p(style['lang'])
        else:
            lang = None
            self.language = pango.pango_language_get_default()
        if lang:
            self.language = pango.pango_language_from_string(lang_p)
            pango.pango_context_set_language(pango_context, self.language)
        assert not isinstance(style['font_family'], str), 'font_family should be a list'
        font_description = get_font_description(style)
        self.layout = ffi.gc(pango.pango_layout_new(pango_context), gobject.g_object_unref)
        pango.pango_layout_set_font_description(self.layout, font_description)
        text_decoration = style['text_decoration_line']
        if text_decoration != 'none':
            metrics = ffi.gc(pango.pango_context_get_metrics(pango_context, font_description, self.language), pango.pango_font_metrics_unref)
            self.ascent = units_to_double(pango.pango_font_metrics_get_ascent(metrics))
            self.underline_position = units_to_double(pango.pango_font_metrics_get_underline_position(metrics))
            self.strikethrough_position = units_to_double(pango.pango_font_metrics_get_strikethrough_position(metrics))
            self.underline_thickness = units_to_double(pango.pango_font_metrics_get_underline_thickness(metrics))
            self.strikethrough_thickness = units_to_double(pango.pango_font_metrics_get_strikethrough_thickness(metrics))
        else:
            self.ascent = None
            self.underline_position = None
            self.strikethrough_position = None
        features = font_features(style['font_kerning'], style['font_variant_ligatures'], style['font_variant_position'], style['font_variant_caps'], style['font_variant_numeric'], style['font_variant_alternates'], style['font_variant_east_asian'], style['font_feature_settings'])
        if features and context:
            features = ','.join((f'{key} {value}' for (key, value) in features.items())).encode()
            attr = context.font_features.setdefault(features, pango.pango_attr_font_features_new(features))
            attr_list = pango.pango_attr_list_new()
            pango.pango_attr_list_insert(attr_list, attr)
            pango.pango_layout_set_attributes(self.layout, attr_list)

    def get_first_line(self):
        if False:
            while True:
                i = 10
        first_line = pango.pango_layout_get_line_readonly(self.layout, 0)
        second_line = pango.pango_layout_get_line_readonly(self.layout, 1)
        index = None if second_line == ffi.NULL else second_line.start_index
        self.first_line_direction = first_line.resolved_dir
        return (first_line, index)

    def set_text(self, text, justify=False):
        if False:
            return 10
        index = text.find('\n')
        if index != -1:
            text = text[:index + 2]
        self.text = text
        (text, bytestring) = unicode_to_char_p(text)
        pango.pango_layout_set_text(self.layout, text, -1)
        word_spacing = self.style['word_spacing']
        if justify:
            word_spacing += self.justification_spacing
        letter_spacing = self.style['letter_spacing']
        if letter_spacing == 'normal':
            letter_spacing = 0
        word_breaking = self.style['overflow_wrap'] in ('anywhere', 'break-word')
        if self.text and (word_spacing or letter_spacing or word_breaking):
            attr_list = pango.pango_layout_get_attributes(self.layout)
            if attr_list == ffi.NULL:
                attr_list = ffi.gc(pango.pango_attr_list_new(), pango.pango_attr_list_unref)

            def add_attr(start, end, spacing):
                if False:
                    return 10
                attr = pango.pango_attr_letter_spacing_new(spacing)
                (attr.start_index, attr.end_index) = (start, end)
                pango.pango_attr_list_change(attr_list, attr)
            if letter_spacing:
                letter_spacing = units_from_double(letter_spacing)
                add_attr(0, len(bytestring), letter_spacing)
            if word_spacing:
                if bytestring == b' ':
                    self.text = ' \u200b'
                    (text, bytestring) = unicode_to_char_p(self.text)
                    pango.pango_layout_set_text(self.layout, text, -1)
                space_spacing = units_from_double(word_spacing) + letter_spacing
                position = bytestring.find(b' ')
                boundary_positions = (0, len(bytestring) - 1)
                while position != -1:
                    factor = 1 + (position in boundary_positions)
                    add_attr(position, position + 1, factor * space_spacing)
                    position = bytestring.find(b' ', position + 1)
            if word_breaking:
                attr = pango.pango_attr_insert_hyphens_new(False)
                (attr.start_index, attr.end_index) = (0, len(bytestring))
                pango.pango_attr_list_change(attr_list, attr)
            pango.pango_layout_set_attributes(self.layout, attr_list)
        if b'\t' in bytestring:
            self.set_tabs()

    def set_tabs(self):
        if False:
            print('Hello World!')
        if isinstance(self.style['tab_size'], int):
            layout = Layout(self.context, self.style, self.justification_spacing)
            layout.set_text(' ' * self.style['tab_size'])
            (line, _) = layout.get_first_line()
            (width, _) = line_size(line, self.style)
            width = int(round(width))
        else:
            width = int(self.style['tab_size'].value)
        array = ffi.gc(pango.pango_tab_array_new_with_positions(1, True, pango.PANGO_TAB_LEFT, width or 1), pango.pango_tab_array_free)
        pango.pango_layout_set_tabs(self.layout, array)

    def deactivate(self):
        if False:
            while True:
                i = 10
        del self.layout, self.language, self.style

    def reactivate(self, style):
        if False:
            i = 10
            return i + 15
        self.setup(self.context, style)
        self.set_text(self.text, justify=True)

def create_layout(text, style, context, max_width, justification_spacing):
    if False:
        while True:
            i = 10
    'Return an opaque Pango layout with default Pango line-breaks.'
    layout = Layout(context, style, justification_spacing, max_width)
    text_wrap = style['white_space'] in ('normal', 'pre-wrap', 'pre-line')
    if max_width is not None and text_wrap and (max_width < 2 ** 21):
        pango.pango_layout_set_width(layout.layout, units_from_double(max(0, max_width)))
    layout.set_text(text)
    return layout

def split_first_line(text, style, context, max_width, justification_spacing, is_line_start=True, minimum=False):
    if False:
        i = 10
        return i + 15
    'Fit as much as possible in the available width for one line of text.\n\n    Return ``(layout, length, resume_index, width, height, baseline)``.\n\n    ``layout``: a pango Layout with the first line\n    ``length``: length in UTF-8 bytes of the first line\n    ``resume_index``: The number of UTF-8 bytes to skip for the next line.\n                      May be ``None`` if the whole text fits in one line.\n                      This may be greater than ``length`` in case of preserved\n                      newline characters.\n    ``width``: width in pixels of the first line\n    ``height``: height in pixels of the first line\n    ``baseline``: baseline in pixels of the first line\n\n    '
    text_wrap = style['white_space'] in ('normal', 'pre-wrap', 'pre-line')
    space_collapse = style['white_space'] in ('normal', 'nowrap', 'pre-line')
    original_max_width = max_width
    if not text_wrap:
        max_width = None
    if max_width is not None and max_width != inf and style['font_size']:
        short_text = text
        if max_width == 0:
            space_index = text.find(' ')
            if space_index != -1:
                short_text = text[:space_index + 2]
        else:
            short_text = text[:int(max_width / style['font_size'] * 2.5)]
        layout = create_layout(short_text, style, context, max_width, justification_spacing)
        (first_line, resume_index) = layout.get_first_line()
        if resume_index is None and short_text != text:
            layout.set_text(text)
            (first_line, resume_index) = layout.get_first_line()
    else:
        layout = create_layout(text, style, context, original_max_width, justification_spacing)
        (first_line, resume_index) = layout.get_first_line()
    if max_width is None:
        return first_line_metrics(first_line, text, layout, resume_index, space_collapse, style)
    (first_line_width, _) = line_size(first_line, style)
    if resume_index is None and first_line_width <= max_width:
        return first_line_metrics(first_line, text, layout, resume_index, space_collapse, style)
    first_line_text = text.encode()[:resume_index].decode()
    first_line_fits = first_line_width <= max_width or ' ' in first_line_text.strip() or can_break_text(first_line_text.strip(), style['lang'])
    if first_line_fits:
        second_line_text = text.encode()[resume_index:].decode()
    else:
        first_line_text = ''
        second_line_text = text
    next_word = second_line_text.split(' ', 1)[0]
    if next_word:
        if space_collapse:
            new_first_line_text = first_line_text + next_word
            layout.set_text(new_first_line_text)
            (first_line, resume_index) = layout.get_first_line()
            if resume_index is None:
                if first_line_text:
                    resume_index = len(new_first_line_text.encode()) + 1
                    return first_line_metrics(first_line, text, layout, resume_index, space_collapse, style)
                else:
                    resume_index = first_line.length + 1
                    if resume_index >= len(text.encode()):
                        resume_index = None
    elif first_line_text:
        return first_line_metrics(first_line, text, layout, resume_index, space_collapse, style)
    hyphens = style['hyphens']
    lang = style['lang'] and pyphen.language_fallback(style['lang'])
    (total, left, right) = style['hyphenate_limit_chars']
    hyphenated = False
    soft_hyphen = '\xad'
    auto_hyphenation = manual_hyphenation = False
    if hyphens != 'none':
        manual_hyphenation = soft_hyphen in first_line_text + next_word
    if hyphens == 'auto' and lang:
        next_word_boundaries = get_next_word_boundaries(second_line_text, lang)
        if next_word_boundaries:
            (start_word, stop_word) = next_word_boundaries
            next_word = second_line_text[start_word:stop_word]
            if stop_word - start_word >= total:
                (first_line_width, _) = line_size(first_line, style)
                space = max_width - first_line_width
                if style['hyphenate_limit_zone'].unit == '%':
                    limit_zone = max_width * style['hyphenate_limit_zone'].value / 100
                else:
                    limit_zone = style['hyphenate_limit_zone'].value
                if space > limit_zone or space < 0:
                    auto_hyphenation = True
    if manual_hyphenation:
        if first_line_text.endswith(soft_hyphen):
            if ' ' in first_line_text:
                (first_line_text, next_word) = first_line_text.rsplit(' ', 1)
                next_word = f' {next_word}'
                layout.set_text(first_line_text)
                (first_line, _) = layout.get_first_line()
                resume_index = len(f'{first_line_text} '.encode())
            else:
                (first_line_text, next_word) = ('', first_line_text)
        soft_hyphen_indexes = [match.start() for match in re.finditer(soft_hyphen, next_word)]
        soft_hyphen_indexes.reverse()
        dictionary_iterations = [next_word[:i + 1] for i in soft_hyphen_indexes]
        start_word = 0
    elif auto_hyphenation:
        dictionary_key = (lang, left, right, total)
        dictionary = context.dictionaries.get(dictionary_key)
        if dictionary is None:
            dictionary = pyphen.Pyphen(lang=lang, left=left, right=right)
            context.dictionaries[dictionary_key] = dictionary
        dictionary_iterations = [start for (start, end) in dictionary.iterate(next_word)]
    else:
        dictionary_iterations = []
    if dictionary_iterations:
        for first_word_part in dictionary_iterations:
            new_first_line_text = first_line_text + second_line_text[:start_word] + first_word_part
            hyphenated_first_line_text = new_first_line_text + style['hyphenate_character']
            new_layout = create_layout(hyphenated_first_line_text, style, context, max_width, justification_spacing)
            (new_first_line, index) = new_layout.get_first_line()
            (new_first_line_width, _) = line_size(new_first_line, style)
            new_space = max_width - new_first_line_width
            hyphenated = index is None and (new_space >= 0 or first_word_part == dictionary_iterations[-1])
            if hyphenated:
                layout = new_layout
                first_line = new_first_line
                resume_index = len(new_first_line_text.encode())
                break
        if not hyphenated and (not first_line_text):
            hyphenated = True
            layout.set_text(hyphenated_first_line_text)
            pango.pango_layout_set_width(layout.layout, -1)
            (first_line, _) = layout.get_first_line()
            resume_index = len(new_first_line_text.encode())
            if text[len(first_line_text)] == soft_hyphen:
                resume_index += len(soft_hyphen.encode())
    if not hyphenated and first_line_text.endswith(soft_hyphen):
        hyphenated = True
        hyphenated_first_line_text = first_line_text + style['hyphenate_character']
        layout.set_text(hyphenated_first_line_text)
        pango.pango_layout_set_width(layout.layout, -1)
        (first_line, _) = layout.get_first_line()
        resume_index = len(first_line_text.encode())
    overflow_wrap = style['overflow_wrap']
    (first_line_width, _) = line_size(first_line, style)
    space = max_width - first_line_width
    can_break = style['word_break'] == 'break-all' or (is_line_start and (overflow_wrap == 'anywhere' or (overflow_wrap == 'break-word' and (not minimum))))
    if space < 0 and can_break:
        hyphenated = False
        layout.set_text(text)
        pango.pango_layout_set_width(layout.layout, units_from_double(max_width))
        pango.pango_layout_set_wrap(layout.layout, PANGO_WRAP_MODE['WRAP_CHAR'])
        (first_line, index) = layout.get_first_line()
        resume_index = index or first_line.length
        if resume_index >= len(text.encode()):
            resume_index = None
    return first_line_metrics(first_line, text, layout, resume_index, space_collapse, style, hyphenated, style['hyphenate_character'])

def get_log_attrs(text, lang):
    if False:
        while True:
            i = 10
    if lang:
        (lang_p, lang) = unicode_to_char_p(lang)
    else:
        lang = None
        language = pango.pango_language_get_default()
    if lang:
        language = pango.pango_language_from_string(lang_p)
    for char in ('\u202a', '\u202b', '\u202c', '\u202d', '\u202e'):
        text = text.replace(char, '\u200b')
    (text_p, bytestring) = unicode_to_char_p(text)
    length = len(text) + 1
    log_attrs = ffi.new('PangoLogAttr[]', length)
    pango.pango_get_log_attrs(text_p, len(bytestring), -1, language, log_attrs, length)
    return (bytestring, log_attrs)

def can_break_text(text, lang):
    if False:
        return 10
    if not text or len(text) < 2:
        return None
    (bytestring, log_attrs) = get_log_attrs(text, lang)
    length = len(text) + 1
    return any((attr.is_line_break for attr in log_attrs[1:length - 1]))

def get_next_word_boundaries(text, lang):
    if False:
        return 10
    if not text or len(text) < 2:
        return None
    (bytestring, log_attrs) = get_log_attrs(text, lang)
    for (i, attr) in enumerate(log_attrs):
        if attr.is_word_end:
            word_end = i
            break
        if attr.is_word_boundary:
            word_start = i
    else:
        return None
    return (word_start, word_end)

def get_last_word_end(text, lang):
    if False:
        for i in range(10):
            print('nop')
    if not text or len(text) < 2:
        return None
    (bytestring, log_attrs) = get_log_attrs(text, lang)
    for (i, attr) in enumerate(list(log_attrs)[::-1]):
        if i and attr.is_word_end:
            return len(text) - i