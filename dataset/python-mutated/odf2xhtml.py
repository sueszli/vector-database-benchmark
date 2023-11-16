from collections import defaultdict
from xml.sax import handler
from xml.sax.saxutils import escape, quoteattr
from xml.dom import Node
from .opendocument import load
from .namespaces import ANIMNS, CHARTNS, CONFIGNS, DCNS, DR3DNS, DRAWNS, FONS, FORMNS, MATHNS, METANS, NUMBERNS, OFFICENS, PRESENTATIONNS, SCRIPTNS, SMILNS, STYLENS, SVGNS, TABLENS, TEXTNS, XLINKNS
from polyglot.builtins import unicode_type
if False:
    (DR3DNS, MATHNS, CHARTNS, CONFIGNS, ANIMNS, FORMNS, SMILNS, SCRIPTNS)

class StyleToCSS:
    """ The purpose of the StyleToCSS class is to contain the rules to convert
        ODF styles to CSS2. Since it needs the generic fonts, it would probably
        make sense to also contain the Styles in a dict as well..
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.fontdict = {}
        self.fillimages = {}
        self.ruleconversions = {(DRAWNS, 'fill-image-name'): self.c_drawfillimage, (FONS, 'background-color'): self.c_fo, (FONS, 'border'): self.c_fo, (FONS, 'border-bottom'): self.c_fo, (FONS, 'border-left'): self.c_fo, (FONS, 'border-right'): self.c_fo, (FONS, 'border-top'): self.c_fo, (FONS, 'break-after'): self.c_break, (FONS, 'break-before'): self.c_break, (FONS, 'color'): self.c_fo, (FONS, 'font-family'): self.c_fo, (FONS, 'font-size'): self.c_fo, (FONS, 'font-style'): self.c_fo, (FONS, 'font-variant'): self.c_fo, (FONS, 'font-weight'): self.c_fo, (FONS, 'line-height'): self.c_fo, (FONS, 'margin'): self.c_fo, (FONS, 'margin-bottom'): self.c_fo, (FONS, 'margin-left'): self.c_fo, (FONS, 'margin-right'): self.c_fo, (FONS, 'margin-top'): self.c_fo, (FONS, 'min-height'): self.c_fo, (FONS, 'padding'): self.c_fo, (FONS, 'padding-bottom'): self.c_fo, (FONS, 'padding-left'): self.c_fo, (FONS, 'padding-right'): self.c_fo, (FONS, 'padding-top'): self.c_fo, (FONS, 'page-width'): self.c_page_width, (FONS, 'page-height'): self.c_page_height, (FONS, 'text-align'): self.c_text_align, (FONS, 'text-indent'): self.c_fo, (TABLENS, 'border-model'): self.c_border_model, (STYLENS, 'column-width'): self.c_width, (STYLENS, 'font-name'): self.c_fn, (STYLENS, 'horizontal-pos'): self.c_hp, (STYLENS, 'text-position'): self.c_text_position, (STYLENS, 'text-line-through-style'): self.c_text_line_through_style, (STYLENS, 'text-underline-style'): self.c_text_underline_style, (STYLENS, 'width'): self.c_width}

    def save_font(self, name, family, generic):
        if False:
            i = 10
            return i + 15
        " It is possible that the HTML browser doesn't know how to\n            show a particular font. Fortunately ODF provides generic fallbacks.\n            Unfortunately they are not the same as CSS2.\n            CSS2: serif, sans-serif, cursive, fantasy, monospace\n            ODF: roman, swiss, modern, decorative, script, system\n            This method put the font and fallback into a dictionary\n        "
        htmlgeneric = 'sans-serif'
        if generic == 'roman':
            htmlgeneric = 'serif'
        elif generic == 'swiss':
            htmlgeneric = 'sans-serif'
        elif generic == 'modern':
            htmlgeneric = 'monospace'
        elif generic == 'decorative':
            htmlgeneric = 'sans-serif'
        elif generic == 'script':
            htmlgeneric = 'monospace'
        elif generic == 'system':
            htmlgeneric = 'serif'
        self.fontdict[name] = (family, htmlgeneric)

    def c_drawfillimage(self, ruleset, sdict, rule, val):
        if False:
            i = 10
            return i + 15
        " Fill a figure with an image. Since CSS doesn't let you resize images\n            this should really be implemented as an absolutely position <img>\n            with a width and a height\n        "
        sdict['background-image'] = "url('%s')" % self.fillimages[val]

    def c_fo(self, ruleset, sdict, rule, val):
        if False:
            for i in range(10):
                print('nop')
        ' XSL formatting attributes '
        selector = rule[1]
        sdict[selector] = val

    def c_break(self, ruleset, sdict, rule, val):
        if False:
            for i in range(10):
                print('nop')
        property = 'page-' + rule[1]
        values = {'auto': 'auto', 'column': 'always', 'page': 'always', 'even-page': 'left', 'odd-page': 'right', 'inherit': 'inherit'}
        sdict[property] = values.get(val, 'auto')

    def c_border_model(self, ruleset, sdict, rule, val):
        if False:
            while True:
                i = 10
        ' Convert to CSS2 border model '
        if val == 'collapsing':
            sdict['border-collapse'] = 'collapse'
        else:
            sdict['border-collapse'] = 'separate'

    def c_width(self, ruleset, sdict, rule, val):
        if False:
            i = 10
            return i + 15
        ' Set width of box '
        sdict['width'] = val

    def c_text_align(self, ruleset, sdict, rule, align):
        if False:
            for i in range(10):
                print('nop')
        ' Text align '
        if align == 'start':
            align = 'left'
        if align == 'end':
            align = 'right'
        sdict['text-align'] = align

    def c_fn(self, ruleset, sdict, rule, fontstyle):
        if False:
            return 10
        ' Generate the CSS font family\n            A generic font can be found in two ways. In a <style:font-face>\n            element or as a font-family-generic attribute in text-properties.\n        '
        generic = ruleset.get((STYLENS, 'font-family-generic'))
        if generic is not None:
            self.save_font(fontstyle, fontstyle, generic)
        (family, htmlgeneric) = self.fontdict.get(fontstyle, (fontstyle, 'serif'))
        sdict['font-family'] = '%s, %s' % (family, htmlgeneric)

    def c_text_position(self, ruleset, sdict, rule, tp):
        if False:
            print('Hello World!')
        ' Text position. This is used e.g. to make superscript and subscript\n            This attribute can have one or two values.\n\n            The first value must be present and specifies the vertical\n            text position as a percentage that relates to the current font\n            height or it takes one of the values sub or super. Negative\n            percentages or the sub value place the text below the\n            baseline. Positive percentages or the super value place\n            the text above the baseline. If sub or super is specified,\n            the application can choose an appropriate text position.\n\n            The second value is optional and specifies the font height\n            as a percentage that relates to the current font-height. If\n            this value is not specified, an appropriate font height is\n            used. Although this value may change the font height that\n            is displayed, it never changes the current font height that\n            is used for additional calculations.\n        '
        textpos = tp.split(' ')
        if len(textpos) == 2 and textpos[0] != '0%':
            sdict['font-size'] = textpos[1]
        if textpos[0] == 'super':
            sdict['vertical-align'] = '33%'
        elif textpos[0] == 'sub':
            sdict['vertical-align'] = '-33%'
        else:
            sdict['vertical-align'] = textpos[0]

    def c_hp(self, ruleset, sdict, rule, hpos):
        if False:
            return 10
        wrap = ruleset.get((STYLENS, 'wrap'), 'parallel')
        if hpos == 'center':
            sdict['margin-left'] = 'auto'
            sdict['margin-right'] = 'auto'
        if hpos in ('right', 'outside'):
            if wrap in ('left', 'parallel', 'dynamic'):
                sdict['float'] = 'right'
            elif wrap == 'run-through':
                sdict['position'] = 'absolute'
                sdict['top'] = '0'
                sdict['right'] = '0'
            else:
                sdict['margin-left'] = 'auto'
                sdict['margin-right'] = '0px'
        elif hpos in ('left', 'inside'):
            if wrap in ('right', 'parallel', 'dynamic'):
                sdict['float'] = 'left'
            elif wrap == 'run-through':
                sdict['position'] = 'absolute'
                sdict['top'] = '0'
                sdict['left'] = '0'
            else:
                sdict['margin-left'] = '0px'
                sdict['margin-right'] = 'auto'
        elif hpos in ('from-left', 'from-inside'):
            if wrap in ('right', 'parallel'):
                sdict['float'] = 'left'
            else:
                sdict['position'] = 'relative'
                if (SVGNS, 'x') in ruleset:
                    sdict['left'] = ruleset[SVGNS, 'x']

    def c_page_width(self, ruleset, sdict, rule, val):
        if False:
            i = 10
            return i + 15
        " Set width of box\n            HTML doesn't really have a page-width. It is always 100% of the browser width\n        "
        sdict['width'] = val

    def c_text_underline_style(self, ruleset, sdict, rule, val):
        if False:
            return 10
        " Set underline decoration\n            HTML doesn't really have a page-width. It is always 100% of the browser width\n        "
        if val and val != 'none':
            sdict['text-decoration'] = 'underline'

    def c_text_line_through_style(self, ruleset, sdict, rule, val):
        if False:
            i = 10
            return i + 15
        " Set underline decoration\n            HTML doesn't really have a page-width. It is always 100% of the browser width\n        "
        if val and val != 'none':
            sdict['text-decoration'] = 'line-through'

    def c_page_height(self, ruleset, sdict, rule, val):
        if False:
            print('Hello World!')
        ' Set height of box '
        sdict['height'] = val

    def convert_styles(self, ruleset):
        if False:
            while True:
                i = 10
        " Rule is a tuple of (namespace, name). If the namespace is '' then\n            it is already CSS2\n        "
        sdict = {}
        for (rule, val) in ruleset.items():
            if rule[0] == '':
                sdict[rule[1]] = val
                continue
            method = self.ruleconversions.get(rule, None)
            if method:
                method(ruleset, sdict, rule, val)
        return sdict

class TagStack:

    def __init__(self):
        if False:
            return 10
        self.stack = []

    def push(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.stack.append((tag, attrs))

    def pop(self):
        if False:
            return 10
        item = self.stack.pop()
        return item

    def stackparent(self):
        if False:
            return 10
        item = self.stack[-1]
        return item[1]

    def rfindattr(self, attr):
        if False:
            print('Hello World!')
        ' Find a tag with the given attribute '
        for (tag, attrs) in self.stack:
            if attr in attrs:
                return attrs[attr]
        return None

    def count_tags(self, tag):
        if False:
            print('Hello World!')
        c = 0
        for (ttag, tattrs) in self.stack:
            if ttag == tag:
                c = c + 1
        return c
special_styles = {'S-Emphasis': 'em', 'S-Citation': 'cite', 'S-Strong_20_Emphasis': 'strong', 'S-Variable': 'var', 'S-Definition': 'dfn', 'S-Teletype': 'tt', 'P-Heading_20_1': 'h1', 'P-Heading_20_2': 'h2', 'P-Heading_20_3': 'h3', 'P-Heading_20_4': 'h4', 'P-Heading_20_5': 'h5', 'P-Heading_20_6': 'h6', 'P-Addressee': 'address', 'P-Preformatted_20_Text': 'pre'}

class ODF2XHTML(handler.ContentHandler):
    """ The ODF2XHTML parses an ODF file and produces XHTML"""

    def __init__(self, generate_css=True, embedable=False):
        if False:
            for i in range(10):
                print('nop')
        self.generate_css = generate_css
        self.frame_stack = []
        self.list_number_map = defaultdict(lambda : 1)
        self.list_id_map = {}
        self.list_class_stack = []
        self.elements = {(DCNS, 'title'): (self.s_processcont, self.e_dc_title), (DCNS, 'language'): (self.s_processcont, self.e_dc_contentlanguage), (DCNS, 'creator'): (self.s_processcont, self.e_dc_creator), (DCNS, 'description'): (self.s_processcont, self.e_dc_metatag), (DCNS, 'date'): (self.s_processcont, self.e_dc_metatag), (DRAWNS, 'custom-shape'): (self.s_custom_shape, self.e_custom_shape), (DRAWNS, 'frame'): (self.s_draw_frame, self.e_draw_frame), (DRAWNS, 'image'): (self.s_draw_image, None), (DRAWNS, 'fill-image'): (self.s_draw_fill_image, None), (DRAWNS, 'layer-set'): (self.s_ignorexml, None), (DRAWNS, 'object'): (self.s_draw_object, None), (DRAWNS, 'object-ole'): (self.s_draw_object_ole, None), (DRAWNS, 'page'): (self.s_draw_page, self.e_draw_page), (DRAWNS, 'text-box'): (self.s_draw_textbox, self.e_draw_textbox), (METANS, 'creation-date'): (self.s_processcont, self.e_dc_metatag), (METANS, 'generator'): (self.s_processcont, self.e_dc_metatag), (METANS, 'initial-creator'): (self.s_processcont, self.e_dc_metatag), (METANS, 'keyword'): (self.s_processcont, self.e_dc_metatag), (NUMBERNS, 'boolean-style'): (self.s_ignorexml, None), (NUMBERNS, 'currency-style'): (self.s_ignorexml, None), (NUMBERNS, 'date-style'): (self.s_ignorexml, None), (NUMBERNS, 'number-style'): (self.s_ignorexml, None), (NUMBERNS, 'text-style'): (self.s_ignorexml, None), (OFFICENS, 'annotation'): (self.s_ignorexml, None), (OFFICENS, 'automatic-styles'): (self.s_office_automatic_styles, None), (OFFICENS, 'document'): (self.s_office_document_content, self.e_office_document_content), (OFFICENS, 'document-content'): (self.s_office_document_content, self.e_office_document_content), (OFFICENS, 'forms'): (self.s_ignorexml, None), (OFFICENS, 'master-styles'): (self.s_office_master_styles, None), (OFFICENS, 'meta'): (self.s_ignorecont, None), (OFFICENS, 'presentation'): (self.s_office_presentation, self.e_office_presentation), (OFFICENS, 'spreadsheet'): (self.s_office_spreadsheet, self.e_office_spreadsheet), (OFFICENS, 'styles'): (self.s_office_styles, None), (OFFICENS, 'text'): (self.s_office_text, self.e_office_text), (OFFICENS, 'scripts'): (self.s_ignorexml, None), (OFFICENS, 'settings'): (self.s_ignorexml, None), (PRESENTATIONNS, 'notes'): (self.s_ignorexml, None), (STYLENS, 'default-page-layout'): (self.s_ignorexml, None), (STYLENS, 'default-style'): (self.s_style_default_style, self.e_style_default_style), (STYLENS, 'drawing-page-properties'): (self.s_style_handle_properties, None), (STYLENS, 'font-face'): (self.s_style_font_face, None), (STYLENS, 'graphic-properties'): (self.s_style_handle_properties, None), (STYLENS, 'handout-master'): (self.s_ignorexml, None), (STYLENS, 'master-page'): (self.s_style_master_page, None), (STYLENS, 'page-layout-properties'): (self.s_style_handle_properties, None), (STYLENS, 'page-layout'): (self.s_style_page_layout, self.e_style_page_layout), (STYLENS, 'paragraph-properties'): (self.s_style_handle_properties, None), (STYLENS, 'style'): (self.s_style_style, self.e_style_style), (STYLENS, 'table-cell-properties'): (self.s_style_handle_properties, None), (STYLENS, 'table-column-properties'): (self.s_style_handle_properties, None), (STYLENS, 'table-properties'): (self.s_style_handle_properties, None), (STYLENS, 'text-properties'): (self.s_style_handle_properties, None), (SVGNS, 'desc'): (self.s_ignorexml, None), (TABLENS, 'covered-table-cell'): (self.s_ignorexml, None), (TABLENS, 'table-cell'): (self.s_table_table_cell, self.e_table_table_cell), (TABLENS, 'table-column'): (self.s_table_table_column, None), (TABLENS, 'table-row'): (self.s_table_table_row, self.e_table_table_row), (TABLENS, 'table'): (self.s_table_table, self.e_table_table), (TEXTNS, 'a'): (self.s_text_a, self.e_text_a), (TEXTNS, 'alphabetical-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'bibliography-configuration'): (self.s_ignorexml, None), (TEXTNS, 'bibliography-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'bookmark'): (self.s_text_bookmark, None), (TEXTNS, 'bookmark-start'): (self.s_text_bookmark, None), (TEXTNS, 'reference-mark-start'): (self.s_text_bookmark, None), (TEXTNS, 'bookmark-ref'): (self.s_text_bookmark_ref, self.e_text_a), (TEXTNS, 'reference-ref'): (self.s_text_bookmark_ref, self.e_text_a), (TEXTNS, 'bookmark-ref-start'): (self.s_text_bookmark_ref, None), (TEXTNS, 'h'): (self.s_text_h, self.e_text_h), (TEXTNS, 'illustration-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'line-break'): (self.s_text_line_break, None), (TEXTNS, 'linenumbering-configuration'): (self.s_ignorexml, None), (TEXTNS, 'list'): (self.s_text_list, self.e_text_list), (TEXTNS, 'list-item'): (self.s_text_list_item, self.e_text_list_item), (TEXTNS, 'list-level-style-bullet'): (self.s_text_list_level_style_bullet, self.e_text_list_level_style_bullet), (TEXTNS, 'list-level-style-number'): (self.s_text_list_level_style_number, self.e_text_list_level_style_number), (TEXTNS, 'list-style'): (None, None), (TEXTNS, 'note'): (self.s_text_note, None), (TEXTNS, 'note-body'): (self.s_text_note_body, self.e_text_note_body), (TEXTNS, 'note-citation'): (None, self.e_text_note_citation), (TEXTNS, 'notes-configuration'): (self.s_ignorexml, None), (TEXTNS, 'object-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'p'): (self.s_text_p, self.e_text_p), (TEXTNS, 's'): (self.s_text_s, None), (TEXTNS, 'span'): (self.s_text_span, self.e_text_span), (TEXTNS, 'tab'): (self.s_text_tab, None), (TEXTNS, 'table-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'table-of-content-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'user-index-source'): (self.s_text_x_source, self.e_text_x_source)}
        if embedable:
            self.make_embedable()
        self._resetobject()

    def set_plain(self):
        if False:
            print('Hello World!')
        ' Tell the parser to not generate CSS '
        self.generate_css = False

    def set_embedable(self):
        if False:
            print('Hello World!')
        ' Tells the converter to only output the parts inside the <body>'
        self.elements[OFFICENS, 'text'] = (None, None)
        self.elements[OFFICENS, 'spreadsheet'] = (None, None)
        self.elements[OFFICENS, 'presentation'] = (None, None)
        self.elements[OFFICENS, 'document-content'] = (None, None)

    def add_style_file(self, stylefilename, media=None):
        if False:
            for i in range(10):
                print('nop')
        ' Add a link to an external style file.\n            Also turns of the embedding of styles in the HTML\n        '
        self.use_internal_css = False
        self.stylefilename = stylefilename
        if media:
            self.metatags.append(f'<link rel="stylesheet" type="text/css" href="{stylefilename}" media="{media}"/>\n')
        else:
            self.metatags.append('<link rel="stylesheet" type="text/css" href="%s"/>\n' % stylefilename)

    def _resetfootnotes(self):
        if False:
            i = 10
            return i + 15
        self.notedict = {}
        self.currentnote = 0
        self.notebody = ''

    def _resetobject(self):
        if False:
            i = 10
            return i + 15
        self.lines = []
        self._wfunc = self._wlines
        self.xmlfile = ''
        self.title = ''
        self.language = ''
        self.creator = ''
        self.data = []
        self.tagstack = TagStack()
        self.htmlstack = []
        self.pstack = []
        self.processelem = True
        self.processcont = True
        self.listtypes = {}
        self.headinglevels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.use_internal_css = True
        self.cs = StyleToCSS()
        self.anchors = {}
        self.stylestack = []
        self.styledict = {}
        self.currentstyle = None
        self.list_starts = {}
        self._resetfootnotes()
        self.metatags = []

    def writeout(self, s):
        if False:
            while True:
                i = 10
        if s != '':
            self._wfunc(s)

    def writedata(self):
        if False:
            while True:
                i = 10
        d = ''.join(self.data)
        if d != '':
            self.writeout(escape(d))

    def opentag(self, tag, attrs={}, block=False):
        if False:
            while True:
                i = 10
        ' Create an open HTML tag '
        self.htmlstack.append((tag, attrs, block))
        a = []
        for (key, val) in attrs.items():
            a.append(f'{key}={quoteattr(val)}')
        if len(a) == 0:
            self.writeout('<%s>' % tag)
        else:
            self.writeout('<{} {}>'.format(tag, ' '.join(a)))
        if block:
            self.writeout('\n')

    def closetag(self, tag, block=True):
        if False:
            while True:
                i = 10
        ' Close an open HTML tag '
        self.htmlstack.pop()
        self.writeout('</%s>' % tag)
        if block:
            self.writeout('\n')

    def emptytag(self, tag, attrs={}):
        if False:
            i = 10
            return i + 15
        a = []
        for (key, val) in attrs.items():
            a.append(f'{key}={quoteattr(val)}')
        self.writeout('<{} {}/>\n'.format(tag, ' '.join(a)))

    def characters(self, data):
        if False:
            while True:
                i = 10
        if self.processelem and self.processcont:
            self.data.append(data)

    def startElementNS(self, tag, qname, attrs):
        if False:
            while True:
                i = 10
        self.pstack.append((self.processelem, self.processcont))
        if self.processelem:
            method = self.elements.get(tag, (None, None))[0]
            if method:
                self.handle_starttag(tag, method, attrs)
            else:
                self.unknown_starttag(tag, attrs)
        self.tagstack.push(tag, attrs)

    def endElementNS(self, tag, qname):
        if False:
            print('Hello World!')
        (stag, attrs) = self.tagstack.pop()
        if self.processelem:
            method = self.elements.get(tag, (None, None))[1]
            if method:
                self.handle_endtag(tag, attrs, method)
            else:
                self.unknown_endtag(tag, attrs)
        (self.processelem, self.processcont) = self.pstack.pop()

    def handle_starttag(self, tag, method, attrs):
        if False:
            for i in range(10):
                print('nop')
        method(tag, attrs)

    def handle_endtag(self, tag, attrs, method):
        if False:
            for i in range(10):
                print('nop')
        method(tag, attrs)

    def unknown_starttag(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def unknown_endtag(self, tag, attrs):
        if False:
            while True:
                i = 10
        pass

    def s_ignorexml(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Ignore this xml element and all children of it\n            It will automatically stop ignoring\n        '
        self.processelem = False

    def s_ignorecont(self, tag, attrs):
        if False:
            return 10
        ' Stop processing the text nodes '
        self.processcont = False

    def s_processcont(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' Start processing the text nodes '
        self.processcont = True

    def classname(self, attrs):
        if False:
            while True:
                i = 10
        ' Generate a class name from a style name '
        c = attrs.get((TEXTNS, 'style-name'), '')
        c = c.replace('.', '_')
        return c

    def get_anchor(self, name):
        if False:
            for i in range(10):
                print('nop')
        ' Create a unique anchor id for a href name '
        if name not in self.anchors:
            self.anchors[name] = 'anchor%d' % (len(self.anchors) + 1)
        return self.anchors.get(name)

    def purgedata(self):
        if False:
            while True:
                i = 10
        self.data = []

    def e_dc_title(self, tag, attrs):
        if False:
            return 10
        ' Get the title from the meta data and create a HTML <title>\n        '
        self.title = ''.join(self.data)
        self.data = []

    def e_dc_metatag(self, tag, attrs):
        if False:
            return 10
        ' Any other meta data is added as a <meta> element\n        '
        self.metatags.append('<meta name="{}" content={}/>\n'.format(tag[1], quoteattr(''.join(self.data))))
        self.data = []

    def e_dc_contentlanguage(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Set the content language. Identifies the targeted audience\n        '
        self.language = ''.join(self.data)
        self.metatags.append('<meta http-equiv="content-language" content="%s"/>\n' % escape(self.language))
        self.data = []

    def e_dc_creator(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' Set the content creator. Identifies the targeted audience\n        '
        self.creator = ''.join(self.data)
        self.metatags.append('<meta http-equiv="creator" content="%s"/>\n' % escape(self.creator))
        self.data = []

    def s_custom_shape(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' A <draw:custom-shape> is made into a <div> in HTML which is then styled\n        '
        anchor_type = attrs.get((TEXTNS, 'anchor-type'), 'notfound')
        htmltag = 'div'
        name = 'G-' + attrs.get((DRAWNS, 'style-name'), '')
        if name == 'G-':
            name = 'PR-' + attrs.get((PRESENTATIONNS, 'style-name'), '')
        name = name.replace('.', '_')
        if anchor_type == 'paragraph':
            style = 'position:absolute;'
        elif anchor_type == 'char':
            style = 'position:absolute;'
        elif anchor_type == 'as-char':
            htmltag = 'div'
            style = ''
        else:
            style = 'position: absolute;'
        if (SVGNS, 'width') in attrs:
            style = style + 'width:' + attrs[SVGNS, 'width'] + ';'
        if (SVGNS, 'height') in attrs:
            style = style + 'height:' + attrs[SVGNS, 'height'] + ';'
        if (SVGNS, 'x') in attrs:
            style = style + 'left:' + attrs[SVGNS, 'x'] + ';'
        if (SVGNS, 'y') in attrs:
            style = style + 'top:' + attrs[SVGNS, 'y'] + ';'
        if self.generate_css:
            self.opentag(htmltag, {'class': name, 'style': style})
        else:
            self.opentag(htmltag)

    def e_custom_shape(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' End the <draw:frame>\n        '
        self.closetag('div')

    def s_draw_frame(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' A <draw:frame> is made into a <div> in HTML which is then styled\n        '
        self.frame_stack.append([])
        anchor_type = attrs.get((TEXTNS, 'anchor-type'), 'notfound')
        htmltag = 'div'
        name = 'G-' + attrs.get((DRAWNS, 'style-name'), '')
        if name == 'G-':
            name = 'PR-' + attrs.get((PRESENTATIONNS, 'style-name'), '')
        name = name.replace('.', '_')
        if anchor_type == 'paragraph':
            style = 'position:relative;'
        elif anchor_type == 'char':
            style = 'position:relative;'
        elif anchor_type == 'as-char':
            htmltag = 'div'
            style = ''
        else:
            style = 'position:absolute;'
        if (SVGNS, 'width') in attrs:
            style = style + 'width:' + attrs[SVGNS, 'width'] + ';'
        if (SVGNS, 'height') in attrs:
            style = style + 'height:' + attrs[SVGNS, 'height'] + ';'
        if (SVGNS, 'x') in attrs:
            style = style + 'left:' + attrs[SVGNS, 'x'] + ';'
        if (SVGNS, 'y') in attrs:
            style = style + 'top:' + attrs[SVGNS, 'y'] + ';'
        if self.generate_css:
            self.opentag(htmltag, {'class': name, 'style': style})
        else:
            self.opentag(htmltag)

    def e_draw_frame(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' End the <draw:frame>\n        '
        self.closetag('div')
        self.frame_stack.pop()

    def s_draw_fill_image(self, tag, attrs):
        if False:
            while True:
                i = 10
        name = attrs.get((DRAWNS, 'name'), 'NoName')
        imghref = attrs[XLINKNS, 'href']
        imghref = self.rewritelink(imghref)
        self.cs.fillimages[name] = imghref

    def rewritelink(self, imghref):
        if False:
            return 10
        " Intended to be overloaded if you don't store your pictures\n            in a Pictures subfolder\n        "
        return imghref

    def s_draw_image(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' A <draw:image> becomes an <img/> element\n        '
        if self.frame_stack:
            if self.frame_stack[-1]:
                return
            self.frame_stack[-1].append('img')
        parent = self.tagstack.stackparent()
        anchor_type = parent.get((TEXTNS, 'anchor-type'))
        imghref = attrs[XLINKNS, 'href']
        imghref = self.rewritelink(imghref)
        htmlattrs = {'alt': '', 'src': imghref}
        if self.generate_css:
            if anchor_type != 'char':
                htmlattrs['style'] = 'display: block;'
        self.emptytag('img', htmlattrs)

    def s_draw_object(self, tag, attrs):
        if False:
            print('Hello World!')
        ' A <draw:object> is embedded object in the document (e.g. spreadsheet in presentation).\n        '
        return
        objhref = attrs[XLINKNS, 'href']
        for c in self.document.childnodes:
            if c.folder == objhref:
                self._walknode(c.topnode)

    def s_draw_object_ole(self, tag, attrs):
        if False:
            return 10
        ' A <draw:object-ole> is embedded OLE object in the document (e.g. MS Graph).\n        '
        try:
            class_id = attrs[DRAWNS, 'class-id']
        except KeyError:
            return
        if class_id and class_id.lower() == '00020803-0000-0000-c000-000000000046':
            tagattrs = {'name': 'object_ole_graph', 'class': 'ole-graph'}
            self.opentag('a', tagattrs)
            self.closetag('a', tagattrs)

    def s_draw_page(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' A <draw:page> is a slide in a presentation. We use a <fieldset> element in HTML.\n            Therefore if you convert a ODP file, you get a series of <fieldset>s.\n            Override this for your own purpose.\n        '
        name = attrs.get((DRAWNS, 'name'), 'NoName')
        stylename = attrs.get((DRAWNS, 'style-name'), '')
        stylename = stylename.replace('.', '_')
        masterpage = attrs.get((DRAWNS, 'master-page-name'), '')
        masterpage = masterpage.replace('.', '_')
        if self.generate_css:
            self.opentag('fieldset', {'class': f'DP-{stylename} MP-{masterpage}'})
        else:
            self.opentag('fieldset')
        self.opentag('legend')
        self.writeout(escape(name))
        self.closetag('legend')

    def e_draw_page(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.closetag('fieldset')

    def s_draw_textbox(self, tag, attrs):
        if False:
            return 10
        style = ''
        if (FONS, 'min-height') in attrs:
            style = style + 'min-height:' + attrs[FONS, 'min-height'] + ';'
        self.opentag('div')

    def e_draw_textbox(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' End the <draw:text-box>\n        '
        self.closetag('div')

    def html_body(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        self.writedata()
        if self.generate_css and self.use_internal_css:
            self.opentag('style', {'type': 'text/css'}, True)
            self.writeout('/*<![CDATA[*/\n')
            self.generate_stylesheet()
            self.writeout('/*]]>*/\n')
            self.closetag('style')
        self.purgedata()
        self.closetag('head')
        self.opentag('body', block=True)
    default_styles = '\nimg { width: 100%; height: 100%; }\n* { padding: 0; margin: 0; }\nbody { margin: 0 1em; }\nol, ul { padding-left: 2em; }\na.citation { text-decoration: none }\nh1.notes-header { page-break-before: always }\ndl.notes dt { font-size: large }\ndl.notes dt a { text-decoration: none }\ndl.notes dd { page-break-after: always }\ndl.notes dd:last-of-type { page-break-after: avoid }\n'

    def generate_stylesheet(self):
        if False:
            while True:
                i = 10
        for name in self.stylestack:
            styles = self.styledict.get(name)
            if '__style-family' in styles and styles['__style-family'] in self.styledict:
                familystyle = self.styledict[styles['__style-family']].copy()
                del styles['__style-family']
                for (style, val) in styles.items():
                    familystyle[style] = val
                styles = familystyle
            while '__parent-style-name' in styles and styles['__parent-style-name'] in self.styledict:
                parentstyle = self.styledict[styles['__parent-style-name']].copy()
                del styles['__parent-style-name']
                for (style, val) in styles.items():
                    parentstyle[style] = val
                styles = parentstyle
            self.styledict[name] = styles
        self.writeout(self.default_styles)
        css_styles = {}
        for name in self.stylestack:
            styles = self.styledict.get(name)
            css2 = tuple(self.cs.convert_styles(styles).items())
            if css2 in css_styles:
                css_styles[css2].append(name)
            else:
                css_styles[css2] = [name]

        def filter_margins(css2):
            if False:
                i = 10
                return i + 15
            names = {k for (k, v) in css2}
            ignore = set()
            if {'margin-left', 'margin-right', 'margin-top', 'margin-bottom'}.issubset(names):
                ignore.add('margin')
            css2 = sorted(css2, key=lambda x: {'margin': 0}.get(x[0], 1))
            for (k, v) in css2:
                if k not in ignore:
                    yield (k, v)
        for (css2, names) in css_styles.items():
            self.writeout('%s {\n' % ', '.join(names))
            for (style, val) in filter_margins(css2):
                self.writeout(f'\t{style}: {val};\n')
            self.writeout('}\n')

    def generate_footnotes(self):
        if False:
            while True:
                i = 10
        if self.currentnote == 0:
            return
        from builtins import _
        self.opentag('h1', {'class': 'notes-header'})
        self.writeout(_('Notes'))
        self.closetag('h1')
        self.opentag('dl', {'class': 'notes'})
        for key in range(1, self.currentnote + 1):
            note = self.notedict[key]
            self.opentag('dt', {'id': 'footnote-%d' % key})
            self.writeout('[')
            self.opentag('a', {'href': '#citation-%d' % key})
            self.writeout('←%d' % key)
            self.closetag('a')
            self.writeout(']\xa0')
            self.closetag('dt')
            self.opentag('dd')
            self.writeout(note['body'])
            self.closetag('dd')
        self.closetag('dl')

    def s_office_automatic_styles(self, tag, attrs):
        if False:
            return 10
        if self.xmlfile == 'styles.xml':
            self.autoprefix = 'A'
        else:
            self.autoprefix = ''

    def s_office_document_content(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' First tag in the content.xml file'
        self.writeout('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN" ')
        self.writeout('"http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n')
        self.opentag('html', {'xmlns': 'http://www.w3.org/1999/xhtml'}, True)
        self.opentag('head', block=True)
        self.emptytag('meta', {'http-equiv': 'Content-Type', 'content': 'text/html;charset=UTF-8'})
        for metaline in self.metatags:
            self.writeout(metaline)
        self.writeout('<title>%s</title>\n' % escape(self.title))

    def e_office_document_content(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Last tag '
        self.closetag('html')

    def s_office_master_styles(self, tag, attrs):
        if False:
            return 10
        ' '

    def s_office_presentation(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        " For some odd reason, OpenOffice Impress doesn't define a default-style\n            for the 'paragraph'. We therefore force a standard when we see\n            it is a presentation\n        "
        self.styledict['p'] = {(FONS, 'font-size'): '24pt'}
        self.styledict['presentation'] = {(FONS, 'font-size'): '24pt'}
        self.html_body(tag, attrs)

    def e_office_presentation(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.generate_footnotes()
        self.closetag('body')

    def s_office_spreadsheet(self, tag, attrs):
        if False:
            print('Hello World!')
        self.html_body(tag, attrs)

    def e_office_spreadsheet(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        self.generate_footnotes()
        self.closetag('body')

    def s_office_styles(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.autoprefix = ''

    def s_office_text(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' OpenDocument text '
        self.styledict['frame'] = {(STYLENS, 'wrap'): 'parallel'}
        self.html_body(tag, attrs)

    def e_office_text(self, tag, attrs):
        if False:
            return 10
        self.generate_footnotes()
        self.closetag('body')

    def s_style_handle_properties(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Copy all attributes to a struct.\n            We will later convert them to CSS2\n        '
        if self.currentstyle is None:
            return
        for (key, attr) in attrs.items():
            self.styledict[self.currentstyle][key] = attr
    familymap = {'frame': 'frame', 'paragraph': 'p', 'presentation': 'presentation', 'text': 'span', 'section': 'div', 'table': 'table', 'table-cell': 'td', 'table-column': 'col', 'table-row': 'tr', 'graphic': 'graphic'}

    def s_style_default_style(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' A default style is like a style on an HTML tag\n        '
        family = attrs[STYLENS, 'family']
        htmlfamily = self.familymap.get(family, 'unknown')
        self.currentstyle = htmlfamily
        self.styledict[self.currentstyle] = {}

    def e_style_default_style(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        self.currentstyle = None

    def s_style_font_face(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        " It is possible that the HTML browser doesn't know how to\n            show a particular font. Luckily ODF provides generic fallbacks\n            Unfortunately they are not the same as CSS2.\n            CSS2: serif, sans-serif, cursive, fantasy, monospace\n            ODF: roman, swiss, modern, decorative, script, system\n        "
        name = attrs[STYLENS, 'name']
        family = attrs[SVGNS, 'font-family']
        generic = attrs.get((STYLENS, 'font-family-generic'), '')
        self.cs.save_font(name, family, generic)

    def s_style_footer(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.opentag('div', {'id': 'footer'})
        self.purgedata()

    def e_style_footer(self, tag, attrs):
        if False:
            return 10
        self.writedata()
        self.closetag('div')
        self.purgedata()

    def s_style_footer_style(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.currentstyle = '@print #footer'
        self.stylestack.append(self.currentstyle)
        self.styledict[self.currentstyle] = {}

    def s_style_header(self, tag, attrs):
        if False:
            return 10
        self.opentag('div', {'id': 'header'})
        self.purgedata()

    def e_style_header(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        self.writedata()
        self.closetag('div')
        self.purgedata()

    def s_style_header_style(self, tag, attrs):
        if False:
            while True:
                i = 10
        self.currentstyle = '@print #header'
        self.stylestack.append(self.currentstyle)
        self.styledict[self.currentstyle] = {}

    def s_style_default_page_layout(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' Collect the formatting for the default page layout style.\n        '
        self.currentstyle = '@page'
        self.stylestack.append(self.currentstyle)
        self.styledict[self.currentstyle] = {}

    def s_style_page_layout(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        " Collect the formatting for the page layout style.\n            This won't work in CSS 2.1, as page identifiers are not allowed.\n            It is legal in CSS3, but the rest of the application doesn't specify when to use what page layout\n        "
        name = attrs[STYLENS, 'name']
        name = name.replace('.', '_')
        self.currentstyle = '.PL-' + name
        self.stylestack.append(self.currentstyle)
        self.styledict[self.currentstyle] = {}

    def e_style_page_layout(self, tag, attrs):
        if False:
            print('Hello World!')
        ' End this style\n        '
        self.currentstyle = None

    def s_style_master_page(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Collect the formatting for the page layout style.\n        '
        name = attrs[STYLENS, 'name']
        name = name.replace('.', '_')
        self.currentstyle = '.MP-' + name
        self.stylestack.append(self.currentstyle)
        self.styledict[self.currentstyle] = {('', 'position'): 'relative'}
        pagelayout = attrs.get((STYLENS, 'page-layout-name'), None)
        if pagelayout:
            pagelayout = '.PL-' + pagelayout
            if pagelayout in self.styledict:
                styles = self.styledict[pagelayout]
                for (style, val) in styles.items():
                    self.styledict[self.currentstyle][style] = val
            else:
                self.styledict[self.currentstyle]['__parent-style-name'] = pagelayout
        self.s_ignorexml(tag, attrs)
    _familyshort = {'drawing-page': 'DP', 'paragraph': 'P', 'presentation': 'PR', 'text': 'S', 'section': 'D', 'table': 'T', 'table-cell': 'TD', 'table-column': 'TC', 'table-row': 'TR', 'graphic': 'G'}

    def s_style_style(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' Collect the formatting for the style.\n            Styles have scope. The same name can be used for both paragraph and\n            character styles Since CSS has no scope we use a prefix. (Not elegant)\n            In ODF a style can have a parent, these parents can be chained.\n            We may not have encountered the parent yet, but if we have, we resolve it.\n        '
        name = attrs[STYLENS, 'name']
        name = name.replace('.', '_')
        family = attrs[STYLENS, 'family']
        htmlfamily = self.familymap.get(family, 'unknown')
        sfamily = self._familyshort.get(family, 'X')
        name = f'{self.autoprefix}{sfamily}-{name}'
        parent = attrs.get((STYLENS, 'parent-style-name'))
        self.currentstyle = special_styles.get(name, '.' + name)
        self.stylestack.append(self.currentstyle)
        if self.currentstyle not in self.styledict:
            self.styledict[self.currentstyle] = {}
        self.styledict[self.currentstyle]['__style-family'] = htmlfamily
        if parent:
            parent = parent.replace('.', '_')
            parent = f'{sfamily}-{parent}'
            parent = special_styles.get(parent, '.' + parent)
            if parent in self.styledict:
                styles = self.styledict[parent]
                for (style, val) in styles.items():
                    self.styledict[self.currentstyle][style] = val
            else:
                self.styledict[self.currentstyle]['__parent-style-name'] = parent

    def e_style_style(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' End this style\n        '
        self.currentstyle = None

    def s_table_table(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' Start a table\n        '
        c = attrs.get((TABLENS, 'style-name'), None)
        if c and self.generate_css:
            c = c.replace('.', '_')
            self.opentag('table', {'class': 'T-%s' % c})
        else:
            self.opentag('table')
        self.purgedata()

    def e_table_table(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' End a table\n        '
        self.writedata()
        self.closetag('table')
        self.purgedata()

    def s_table_table_cell(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' Start a table cell '
        htmlattrs = {}
        rowspan = attrs.get((TABLENS, 'number-rows-spanned'))
        if rowspan:
            htmlattrs['rowspan'] = rowspan
        colspan = attrs.get((TABLENS, 'number-columns-spanned'))
        if colspan:
            htmlattrs['colspan'] = colspan
        c = attrs.get((TABLENS, 'style-name'))
        if c:
            htmlattrs['class'] = 'TD-%s' % c.replace('.', '_')
        self.opentag('td', htmlattrs)
        self.purgedata()

    def e_table_table_cell(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' End a table cell '
        self.writedata()
        self.closetag('td')
        self.purgedata()

    def s_table_table_column(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Start a table column '
        c = attrs.get((TABLENS, 'style-name'), None)
        repeated = int(attrs.get((TABLENS, 'number-columns-repeated'), 1))
        htmlattrs = {}
        if c:
            htmlattrs['class'] = 'TC-%s' % c.replace('.', '_')
        for x in range(repeated):
            self.emptytag('col', htmlattrs)
        self.purgedata()

    def s_table_table_row(self, tag, attrs):
        if False:
            return 10
        ' Start a table row '
        c = attrs.get((TABLENS, 'style-name'), None)
        htmlattrs = {}
        if c:
            htmlattrs['class'] = 'TR-%s' % c.replace('.', '_')
        self.opentag('tr', htmlattrs)
        self.purgedata()

    def e_table_table_row(self, tag, attrs):
        if False:
            return 10
        ' End a table row '
        self.writedata()
        self.closetag('tr')
        self.purgedata()

    def s_text_a(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' Anchors start '
        self.writedata()
        href = attrs[XLINKNS, 'href'].split('|')[0]
        if href[:1] == '#':
            href = '#' + self.get_anchor(href[1:])
        self.opentag('a', {'href': href})
        self.purgedata()

    def e_text_a(self, tag, attrs):
        if False:
            print('Hello World!')
        ' End an anchor or bookmark reference '
        self.writedata()
        self.closetag('a', False)
        self.purgedata()

    def s_text_bookmark(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' Bookmark definition '
        name = attrs[TEXTNS, 'name']
        html_id = self.get_anchor(name)
        self.writedata()
        self.opentag('span', {'id': html_id})
        self.closetag('span', False)
        self.purgedata()

    def s_text_bookmark_ref(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Bookmark reference '
        name = attrs[TEXTNS, 'ref-name']
        html_id = '#' + self.get_anchor(name)
        self.writedata()
        self.opentag('a', {'href': html_id})
        self.purgedata()

    def s_text_h(self, tag, attrs):
        if False:
            return 10
        ' Headings start '
        level = int(attrs[TEXTNS, 'outline-level'])
        if level > 6:
            level = 6
        if level < 1:
            level = 1
        self.headinglevels[level] = self.headinglevels[level] + 1
        name = self.classname(attrs)
        for x in range(level + 1, 10):
            self.headinglevels[x] = 0
        special = special_styles.get('P-' + name)
        if special or not self.generate_css:
            self.opentag('h%s' % level)
        else:
            self.opentag('h%s' % level, {'class': 'P-%s' % name})
        self.purgedata()

    def e_text_h(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' Headings end\n            Side-effect: If there is no title in the metadata, then it is taken\n            from the first heading of any level.\n        '
        self.writedata()
        level = int(attrs[TEXTNS, 'outline-level'])
        if level > 6:
            level = 6
        if level < 1:
            level = 1
        lev = self.headinglevels[1:level + 1]
        outline = '.'.join(map(str, lev))
        heading = ''.join(self.data)
        if self.title == '':
            self.title = heading
        tail = ''.join(self.data)
        anchor = self.get_anchor(f'{outline}.{tail}')
        anchor2 = self.get_anchor(tail)
        self.opentag('a', {'id': anchor})
        self.closetag('a', False)
        self.opentag('a', {'id': anchor2})
        self.closetag('a', False)
        self.closetag('h%s' % level)
        self.purgedata()

    def s_text_line_break(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Force a line break (<br/>) '
        self.writedata()
        self.emptytag('br')
        self.purgedata()

    def s_text_list(self, tag, attrs):
        if False:
            print('Hello World!')
        " Start a list (<ul> or <ol>)\n            To know which level we're at, we have to count the number\n            of <text:list> elements on the tagstack.\n        "
        name = attrs.get((TEXTNS, 'style-name'))
        continue_numbering = attrs.get((TEXTNS, 'continue-numbering')) == 'true'
        continue_list = attrs.get((TEXTNS, 'continue-list'))
        list_id = attrs.get(('http://www.w3.org/XML/1998/namespace', 'id'))
        level = self.tagstack.count_tags(tag) + 1
        if name:
            name = name.replace('.', '_')
        else:
            name = self.tagstack.rfindattr((TEXTNS, 'style-name'))
        list_class = '%s_%d' % (name, level)
        tag_name = self.listtypes.get(list_class, 'ul')
        number_class = tag_name + list_class
        if list_id:
            self.list_id_map[list_id] = number_class
        if continue_list:
            if continue_list in self.list_id_map:
                tglc = self.list_id_map[continue_list]
                self.list_number_map[number_class] = self.list_number_map[tglc]
            else:
                self.list_number_map.pop(number_class, None)
        elif not continue_numbering:
            self.list_number_map.pop(number_class, None)
        self.list_class_stack.append(number_class)
        attrs = {}
        if tag_name == 'ol' and self.list_number_map[number_class] != 1:
            attrs = {'start': unicode_type(self.list_number_map[number_class])}
        if self.generate_css:
            attrs['class'] = list_class
        self.opentag('%s' % tag_name, attrs)
        self.purgedata()

    def e_text_list(self, tag, attrs):
        if False:
            print('Hello World!')
        ' End a list '
        self.writedata()
        if self.list_class_stack:
            self.list_class_stack.pop()
        name = attrs.get((TEXTNS, 'style-name'))
        level = self.tagstack.count_tags(tag) + 1
        if name:
            name = name.replace('.', '_')
        else:
            name = self.tagstack.rfindattr((TEXTNS, 'style-name'))
        list_class = '%s_%d' % (name, level)
        self.closetag(self.listtypes.get(list_class, 'ul'))
        self.purgedata()

    def s_text_list_item(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' Start list item '
        number_class = self.list_class_stack[-1] if self.list_class_stack else None
        if number_class:
            self.list_number_map[number_class] += 1
        self.opentag('li')
        self.purgedata()

    def e_text_list_item(self, tag, attrs):
        if False:
            print('Hello World!')
        ' End list item '
        self.writedata()
        self.closetag('li')
        self.purgedata()

    def s_text_list_level_style_bullet(self, tag, attrs):
        if False:
            return 10
        " CSS doesn't have the ability to set the glyph\n            to a particular character, so we just go through\n            the available glyphs\n        "
        name = self.tagstack.rfindattr((STYLENS, 'name'))
        level = attrs[TEXTNS, 'level']
        self.prevstyle = self.currentstyle
        list_class = f'{name}_{level}'
        self.listtypes[list_class] = 'ul'
        self.currentstyle = '.{}_{}'.format(name.replace('.', '_'), level)
        self.stylestack.append(self.currentstyle)
        self.styledict[self.currentstyle] = {}
        level = int(level)
        listtype = ('square', 'disc', 'circle')[level % 3]
        self.styledict[self.currentstyle]['', 'list-style-type'] = listtype

    def e_text_list_level_style_bullet(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        self.currentstyle = self.prevstyle
        del self.prevstyle

    def s_text_list_level_style_number(self, tag, attrs):
        if False:
            while True:
                i = 10
        name = self.tagstack.stackparent()[STYLENS, 'name']
        level = attrs[TEXTNS, 'level']
        num_format = attrs.get((STYLENS, 'num-format'), '1')
        start_value = attrs.get((TEXTNS, 'start-value'), '1')
        list_class = f'{name}_{level}'
        self.prevstyle = self.currentstyle
        self.currentstyle = '.{}_{}'.format(name.replace('.', '_'), level)
        if start_value != '1':
            self.list_starts[self.currentstyle] = start_value
        self.listtypes[list_class] = 'ol'
        self.stylestack.append(self.currentstyle)
        self.styledict[self.currentstyle] = {}
        if num_format == '1':
            listtype = 'decimal'
        elif num_format == 'I':
            listtype = 'upper-roman'
        elif num_format == 'i':
            listtype = 'lower-roman'
        elif num_format == 'A':
            listtype = 'upper-alpha'
        elif num_format == 'a':
            listtype = 'lower-alpha'
        else:
            listtype = 'decimal'
        self.styledict[self.currentstyle]['', 'list-style-type'] = listtype

    def e_text_list_level_style_number(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        self.currentstyle = self.prevstyle
        del self.prevstyle

    def s_text_note(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        self.writedata()
        self.purgedata()
        self.currentnote = self.currentnote + 1
        self.notedict[self.currentnote] = {}
        self.notebody = []

    def e_text_note(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        pass

    def collectnote(self, s):
        if False:
            for i in range(10):
                print('nop')
        if s != '':
            self.notebody.append(s)

    def s_text_note_body(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        self._orgwfunc = self._wfunc
        self._wfunc = self.collectnote

    def e_text_note_body(self, tag, attrs):
        if False:
            print('Hello World!')
        self._wfunc = self._orgwfunc
        self.notedict[self.currentnote]['body'] = ''.join(self.notebody)
        self.notebody = ''
        del self._orgwfunc

    def e_text_note_citation(self, tag, attrs):
        if False:
            while True:
                i = 10
        mark = ''.join(self.data)
        self.notedict[self.currentnote]['citation'] = mark
        self.opentag('sup')
        self.opentag('a', {'href': '#footnote-%s' % self.currentnote, 'class': 'citation', 'id': 'citation-%s' % self.currentnote})
        self.writeout(str(self.currentnote))
        self.closetag('a')
        self.closetag('sup')

    def s_text_p(self, tag, attrs):
        if False:
            return 10
        ' Paragraph\n        '
        htmlattrs = {}
        specialtag = 'p'
        c = attrs.get((TEXTNS, 'style-name'), None)
        if c:
            c = c.replace('.', '_')
            specialtag = special_styles.get('P-' + c)
            if specialtag is None:
                specialtag = 'p'
                if self.generate_css:
                    htmlattrs['class'] = 'P-%s' % c
        self.opentag(specialtag, htmlattrs)
        self.purgedata()

    def e_text_p(self, tag, attrs):
        if False:
            while True:
                i = 10
        ' End Paragraph\n        '
        specialtag = 'p'
        c = attrs.get((TEXTNS, 'style-name'), None)
        if c:
            c = c.replace('.', '_')
            specialtag = special_styles.get('P-' + c)
            if specialtag is None:
                specialtag = 'p'
        self.writedata()
        if not self.data:
            self.writeout('&#160;')
        self.closetag(specialtag)
        self.purgedata()

    def s_text_s(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Generate a number of spaces. We use the non breaking space for\n        the text:s ODF element.\n        '
        try:
            c = int(attrs.get((TEXTNS, 'c'), 1))
        except:
            c = 0
        if c > 0:
            self.data.append('\xa0' * c)

    def s_text_span(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' The <text:span> element matches the <span> element in HTML. It is\n            typically used to properties of the text.\n        '
        self.writedata()
        c = attrs.get((TEXTNS, 'style-name'), None)
        htmlattrs = {}
        special = 'span'
        if c:
            c = c.replace('.', '_')
            special = special_styles.get('S-' + c)
            if special is None:
                special = 'span'
                if self.generate_css:
                    htmlattrs['class'] = 'S-%s' % c
        self.opentag(special, htmlattrs)
        self.purgedata()

    def e_text_span(self, tag, attrs):
        if False:
            for i in range(10):
                print('nop')
        ' End the <text:span> '
        self.writedata()
        c = attrs.get((TEXTNS, 'style-name'), None)
        special = 'span'
        if c:
            c = c.replace('.', '_')
            special = special_styles.get('S-' + c)
            if special is None:
                special = 'span'
        self.closetag(special, False)
        self.purgedata()

    def s_text_tab(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' Move to the next tabstop. We ignore this in HTML\n        '
        self.writedata()
        self.writeout(' ')
        self.purgedata()

    def s_text_x_source(self, tag, attrs):
        if False:
            i = 10
            return i + 15
        ' Various indexes and tables of contents. We ignore those.\n        '
        self.writedata()
        self.purgedata()
        self.s_ignorexml(tag, attrs)

    def e_text_x_source(self, tag, attrs):
        if False:
            print('Hello World!')
        ' Various indexes and tables of contents. We ignore those.\n        '
        self.writedata()
        self.purgedata()

    def load(self, odffile):
        if False:
            return 10
        ' Loads a document into the parser and parses it.\n            The argument can either be a filename or a document in memory.\n        '
        self.lines = []
        self._wfunc = self._wlines
        if isinstance(odffile, (bytes, str)) or hasattr(odffile, 'read'):
            self.document = load(odffile)
        else:
            self.document = odffile
        self._walknode(self.document.topnode)

    def _walknode(self, node):
        if False:
            for i in range(10):
                print('nop')
        if node.nodeType == Node.ELEMENT_NODE:
            self.startElementNS(node.qname, node.tagName, node.attributes)
            for c in node.childNodes:
                self._walknode(c)
            self.endElementNS(node.qname, node.tagName)
        if node.nodeType == Node.TEXT_NODE or node.nodeType == Node.CDATA_SECTION_NODE:
            self.characters(str(node))

    def odf2xhtml(self, odffile):
        if False:
            return 10
        ' Load a file and return the XHTML\n        '
        self.load(odffile)
        return self.xhtml()

    def _wlines(self, s):
        if False:
            return 10
        if s:
            self.lines.append(s)

    def xhtml(self):
        if False:
            print('Hello World!')
        ' Returns the xhtml\n        '
        return ''.join(self.lines)

    def _writecss(self, s):
        if False:
            while True:
                i = 10
        if s:
            self._csslines.append(s)

    def _writenothing(self, s):
        if False:
            while True:
                i = 10
        pass

    def css(self):
        if False:
            i = 10
            return i + 15
        ' Returns the CSS content '
        self._csslines = []
        self._wfunc = self._writecss
        self.generate_stylesheet()
        res = ''.join(self._csslines)
        self._wfunc = self._wlines
        del self._csslines
        return res

    def save(self, outputfile, addsuffix=False):
        if False:
            return 10
        " Save the HTML under the filename.\n            If the filename is '-' then save to stdout\n            We have the last style filename in self.stylefilename\n        "
        if outputfile == '-':
            import sys
            outputfp = sys.stdout
        else:
            if addsuffix:
                outputfile = outputfile + '.html'
            outputfp = open(outputfile, 'wb')
        outputfp.write(self.xhtml().encode('us-ascii', 'xmlcharrefreplace'))
        outputfp.close()

class ODF2XHTMLembedded(ODF2XHTML):
    """ The ODF2XHTML parses an ODF file and produces XHTML"""

    def __init__(self, lines, generate_css=True, embedable=False):
        if False:
            return 10
        self._resetobject()
        self.lines = lines
        self.generate_css = generate_css
        self.elements = {(DRAWNS, 'frame'): (self.s_draw_frame, self.e_draw_frame), (DRAWNS, 'image'): (self.s_draw_image, None), (DRAWNS, 'fill-image'): (self.s_draw_fill_image, None), (DRAWNS, 'layer-set'): (self.s_ignorexml, None), (DRAWNS, 'page'): (self.s_draw_page, self.e_draw_page), (DRAWNS, 'object'): (self.s_draw_object, None), (DRAWNS, 'object-ole'): (self.s_draw_object_ole, None), (DRAWNS, 'text-box'): (self.s_draw_textbox, self.e_draw_textbox), (NUMBERNS, 'boolean-style'): (self.s_ignorexml, None), (NUMBERNS, 'currency-style'): (self.s_ignorexml, None), (NUMBERNS, 'date-style'): (self.s_ignorexml, None), (NUMBERNS, 'number-style'): (self.s_ignorexml, None), (NUMBERNS, 'text-style'): (self.s_ignorexml, None), (OFFICENS, 'forms'): (self.s_ignorexml, None), (OFFICENS, 'meta'): (self.s_ignorecont, None), (OFFICENS, 'scripts'): (self.s_ignorexml, None), (PRESENTATIONNS, 'notes'): (self.s_ignorexml, None), (SVGNS, 'desc'): (self.s_ignorexml, None), (TABLENS, 'covered-table-cell'): (self.s_ignorexml, None), (TABLENS, 'table-cell'): (self.s_table_table_cell, self.e_table_table_cell), (TABLENS, 'table-column'): (self.s_table_table_column, None), (TABLENS, 'table-row'): (self.s_table_table_row, self.e_table_table_row), (TABLENS, 'table'): (self.s_table_table, self.e_table_table), (TEXTNS, 'a'): (self.s_text_a, self.e_text_a), (TEXTNS, 'alphabetical-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'bibliography-configuration'): (self.s_ignorexml, None), (TEXTNS, 'bibliography-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'h'): (self.s_text_h, self.e_text_h), (TEXTNS, 'illustration-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'line-break'): (self.s_text_line_break, None), (TEXTNS, 'linenumbering-configuration'): (self.s_ignorexml, None), (TEXTNS, 'list'): (self.s_text_list, self.e_text_list), (TEXTNS, 'list-item'): (self.s_text_list_item, self.e_text_list_item), (TEXTNS, 'list-level-style-bullet'): (self.s_text_list_level_style_bullet, self.e_text_list_level_style_bullet), (TEXTNS, 'list-level-style-number'): (self.s_text_list_level_style_number, self.e_text_list_level_style_number), (TEXTNS, 'list-style'): (None, None), (TEXTNS, 'note'): (self.s_text_note, None), (TEXTNS, 'note-body'): (self.s_text_note_body, self.e_text_note_body), (TEXTNS, 'note-citation'): (None, self.e_text_note_citation), (TEXTNS, 'notes-configuration'): (self.s_ignorexml, None), (TEXTNS, 'object-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'p'): (self.s_text_p, self.e_text_p), (TEXTNS, 's'): (self.s_text_s, None), (TEXTNS, 'span'): (self.s_text_span, self.e_text_span), (TEXTNS, 'tab'): (self.s_text_tab, None), (TEXTNS, 'table-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'table-of-content-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'user-index-source'): (self.s_text_x_source, self.e_text_x_source), (TEXTNS, 'page-number'): (None, None)}