import zipfile, xml.dom.minidom
from .namespaces import nsdict
from .elementtypes import empty_elements, inline_elements
from polyglot.builtins import unicode_type
IGNORED_TAGS = ['draw:adraw:g', 'draw:line', 'draw:object-ole', 'office:annotation', 'presentation:notes', 'svg:desc'] + [nsdict[item[0]] + ':' + item[1] for item in empty_elements]
INLINE_TAGS = [nsdict[item[0]] + ':' + item[1] for item in inline_elements]

class TextProps:
    """ Holds properties for a text style. """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.italic = False
        self.bold = False
        self.fixed = False
        self.underlined = False
        self.strikethrough = False
        self.superscript = False
        self.subscript = False

    def setItalic(self, value):
        if False:
            return 10
        if value == 'italic':
            self.italic = True
        elif value == 'normal':
            self.italic = False

    def setBold(self, value):
        if False:
            i = 10
            return i + 15
        if value == 'bold':
            self.bold = True
        elif value == 'normal':
            self.bold = False

    def setFixed(self, value):
        if False:
            while True:
                i = 10
        self.fixed = value

    def setUnderlined(self, value):
        if False:
            while True:
                i = 10
        if value and value != 'none':
            self.underlined = True

    def setStrikethrough(self, value):
        if False:
            i = 10
            return i + 15
        if value and value != 'none':
            self.strikethrough = True

    def setPosition(self, value):
        if False:
            i = 10
            return i + 15
        if value is None or value == '':
            return
        posisize = value.split(' ')
        textpos = posisize[0]
        if textpos.find('%') == -1:
            if textpos == 'sub':
                self.superscript = False
                self.subscript = True
            elif textpos == 'super':
                self.superscript = True
                self.subscript = False
        else:
            itextpos = int(textpos[:textpos.find('%')])
            if itextpos > 10:
                self.superscript = False
                self.subscript = True
            elif itextpos < -10:
                self.superscript = True
                self.subscript = False

    def __unicode__(self):
        if False:
            for i in range(10):
                print('nop')
        return '[italic={}, bold=i{}, fixed={}]'.format(unicode_type(self.italic), unicode_type(self.bold), unicode_type(self.fixed))
    __str__ = __unicode__

class ParagraphProps:
    """ Holds properties of a paragraph style. """

    def __init__(self):
        if False:
            print('Hello World!')
        self.blockquote = False
        self.headingLevel = 0
        self.code = False
        self.title = False
        self.indented = 0

    def setIndented(self, value):
        if False:
            print('Hello World!')
        self.indented = value

    def setHeading(self, level):
        if False:
            for i in range(10):
                print('nop')
        self.headingLevel = level

    def setTitle(self, value):
        if False:
            i = 10
            return i + 15
        self.title = value

    def setCode(self, value):
        if False:
            while True:
                i = 10
        self.code = value

    def __unicode__(self):
        if False:
            print('Hello World!')
        return '[bq=%s, h=%d, code=%s]' % (unicode_type(self.blockquote), self.headingLevel, unicode_type(self.code))
    __str__ = __unicode__

class ListProperties:
    """ Holds properties for a list style. """

    def __init__(self):
        if False:
            print('Hello World!')
        self.ordered = False

    def setOrdered(self, value):
        if False:
            return 10
        self.ordered = value

class ODF2MoinMoin:

    def __init__(self, filepath):
        if False:
            for i in range(10):
                print('nop')
        self.footnotes = []
        self.footnoteCounter = 0
        self.textStyles = {'Standard': TextProps()}
        self.paragraphStyles = {'Standard': ParagraphProps()}
        self.listStyles = {}
        self.fixedFonts = []
        self.hasTitle = 0
        self.lastsegment = None
        self.elements = {'draw:page': self.textToString, 'draw:frame': self.textToString, 'draw:image': self.draw_image, 'draw:text-box': self.textToString, 'text:a': self.text_a, 'text:note': self.text_note}
        for tag in IGNORED_TAGS:
            self.elements[tag] = self.do_nothing
        for tag in INLINE_TAGS:
            self.elements[tag] = self.inline_markup
        self.elements['text:line-break'] = self.text_line_break
        self.elements['text:s'] = self.text_s
        self.elements['text:tab'] = self.text_tab
        self.load(filepath)

    def processFontDeclarations(self, fontDecl):
        if False:
            for i in range(10):
                print('nop')
        ' Extracts necessary font information from a font-declaration\n            element.\n            '
        for fontFace in fontDecl.getElementsByTagName('style:font-face'):
            if fontFace.getAttribute('style:font-pitch') == 'fixed':
                self.fixedFonts.append(fontFace.getAttribute('style:name'))

    def extractTextProperties(self, style, parent=None):
        if False:
            for i in range(10):
                print('nop')
        ' Extracts text properties from a style element. '
        textProps = TextProps()
        textPropEl = style.getElementsByTagName('style:text-properties')
        if not textPropEl:
            return textProps
        textPropEl = textPropEl[0]
        textProps.setItalic(textPropEl.getAttribute('fo:font-style'))
        textProps.setBold(textPropEl.getAttribute('fo:font-weight'))
        textProps.setUnderlined(textPropEl.getAttribute('style:text-underline-style'))
        textProps.setStrikethrough(textPropEl.getAttribute('style:text-line-through-style'))
        textProps.setPosition(textPropEl.getAttribute('style:text-position'))
        if textPropEl.getAttribute('style:font-name') in self.fixedFonts:
            textProps.setFixed(True)
        return textProps

    def extractParagraphProperties(self, style, parent=None):
        if False:
            while True:
                i = 10
        ' Extracts paragraph properties from a style element. '
        paraProps = ParagraphProps()
        name = style.getAttribute('style:name')
        if name.startswith('Heading_20_'):
            level = name[11:]
            try:
                level = int(level)
                paraProps.setHeading(level)
            except:
                level = 0
        if name == 'Title':
            paraProps.setTitle(True)
        paraPropEl = style.getElementsByTagName('style:paragraph-properties')
        if paraPropEl:
            paraPropEl = paraPropEl[0]
            leftMargin = paraPropEl.getAttribute('fo:margin-left')
            if leftMargin:
                try:
                    leftMargin = float(leftMargin[:-2])
                    if leftMargin > 0.01:
                        paraProps.setIndented(True)
                except:
                    pass
        textProps = self.extractTextProperties(style)
        if textProps.fixed:
            paraProps.setCode(True)
        return paraProps

    def processStyles(self, styleElements):
        if False:
            return 10
        ' Runs through "style" elements extracting necessary information.\n        '
        for style in styleElements:
            name = style.getAttribute('style:name')
            if name == 'Standard':
                continue
            family = style.getAttribute('style:family')
            parent = style.getAttribute('style:parent-style-name')
            if family == 'text':
                self.textStyles[name] = self.extractTextProperties(style, parent)
            elif family == 'paragraph':
                self.paragraphStyles[name] = self.extractParagraphProperties(style, parent)
                self.textStyles[name] = self.extractTextProperties(style, parent)

    def processListStyles(self, listStyleElements):
        if False:
            print('Hello World!')
        for style in listStyleElements:
            name = style.getAttribute('style:name')
            prop = ListProperties()
            if style.hasChildNodes():
                subitems = [el for el in style.childNodes if el.nodeType == xml.dom.Node.ELEMENT_NODE and el.tagName == 'text:list-level-style-number']
                if len(subitems) > 0:
                    prop.setOrdered(True)
            self.listStyles[name] = prop

    def load(self, filepath):
        if False:
            for i in range(10):
                print('nop')
        ' Loads an ODT file. '
        zip = zipfile.ZipFile(filepath)
        styles_doc = xml.dom.minidom.parseString(zip.read('styles.xml'))
        fontfacedecls = styles_doc.getElementsByTagName('office:font-face-decls')
        if fontfacedecls:
            self.processFontDeclarations(fontfacedecls[0])
        self.processStyles(styles_doc.getElementsByTagName('style:style'))
        self.processListStyles(styles_doc.getElementsByTagName('text:list-style'))
        self.content = xml.dom.minidom.parseString(zip.read('content.xml'))
        fontfacedecls = self.content.getElementsByTagName('office:font-face-decls')
        if fontfacedecls:
            self.processFontDeclarations(fontfacedecls[0])
        self.processStyles(self.content.getElementsByTagName('style:style'))
        self.processListStyles(self.content.getElementsByTagName('text:list-style'))

    def compressCodeBlocks(self, text):
        if False:
            while True:
                i = 10
        ' Removes extra blank lines from code blocks. '
        return text
        lines = text.split('\n')
        buffer = []
        numLines = len(lines)
        for i in range(numLines):
            if lines[i].strip() or i == numLines - 1 or i == 0 or (not (lines[i - 1].startswith('    ') and lines[i + 1].startswith('    '))):
                buffer.append('\n' + lines[i])
        return ''.join(buffer)

    def do_nothing(self, node):
        if False:
            i = 10
            return i + 15
        return ''

    def draw_image(self, node):
        if False:
            i = 10
            return i + 15
        '\n        '
        link = node.getAttribute('xlink:href')
        if link and link[:2] == './':
            return '%s\n' % link
        if link and link[:9] == 'Pictures/':
            link = link[9:]
        return '[[Image(%s)]]\n' % link

    def text_a(self, node):
        if False:
            print('Hello World!')
        text = self.textToString(node)
        link = node.getAttribute('xlink:href')
        if link.strip() == text.strip():
            return '[%s] ' % link.strip()
        else:
            return f'[{link.strip()} {text.strip()}] '

    def text_line_break(self, node):
        if False:
            i = 10
            return i + 15
        return '[[BR]]'

    def text_note(self, node):
        if False:
            i = 10
            return i + 15
        cite = node.getElementsByTagName('text:note-citation')[0].childNodes[0].nodeValue
        body = node.getElementsByTagName('text:note-body')[0].childNodes[0]
        self.footnotes.append((cite, self.textToString(body)))
        return '^%s^' % cite

    def text_s(self, node):
        if False:
            print('Hello World!')
        try:
            num = int(node.getAttribute('text:c'))
            return ' ' * num
        except:
            return ' '

    def text_tab(self, node):
        if False:
            return 10
        return '    '

    def inline_markup(self, node):
        if False:
            i = 10
            return i + 15
        text = self.textToString(node)
        if not text.strip():
            return ''
        styleName = node.getAttribute('text:style-name')
        style = self.textStyles.get(styleName, TextProps())
        if style.fixed:
            return '`' + text + '`'
        mark = []
        if style:
            if style.italic:
                mark.append("''")
            if style.bold:
                mark.append("'''")
            if style.underlined:
                mark.append('__')
            if style.strikethrough:
                mark.append('~~')
            if style.superscript:
                mark.append('^')
            if style.subscript:
                mark.append(',,')
        revmark = mark[:]
        revmark.reverse()
        return '{}{}{}'.format(''.join(mark), text, ''.join(revmark))

    def listToString(self, listElement, indent=0):
        if False:
            for i in range(10):
                print('nop')
        self.lastsegment = listElement.tagName
        buffer = []
        styleName = listElement.getAttribute('text:style-name')
        props = self.listStyles.get(styleName, ListProperties())
        i = 0
        for item in listElement.childNodes:
            buffer.append(' ' * indent)
            i += 1
            if props.ordered:
                number = unicode_type(i)
                number = ' ' + number + '. '
                buffer.append(' 1. ')
            else:
                buffer.append(' * ')
            subitems = [el for el in item.childNodes if el.tagName in ['text:p', 'text:h', 'text:list']]
            for subitem in subitems:
                if subitem.tagName == 'text:list':
                    buffer.append('\n')
                    buffer.append(self.listToString(subitem, indent + 3))
                else:
                    buffer.append(self.paragraphToString(subitem, indent + 3))
                self.lastsegment = subitem.tagName
            self.lastsegment = item.tagName
            buffer.append('\n')
        return ''.join(buffer)

    def tableToString(self, tableElement):
        if False:
            while True:
                i = 10
        ' MoinMoin uses || to delimit table cells\n        '
        self.lastsegment = tableElement.tagName
        buffer = []
        for item in tableElement.childNodes:
            self.lastsegment = item.tagName
            if item.tagName == 'table:table-header-rows':
                buffer.append(self.tableToString(item))
            if item.tagName == 'table:table-row':
                buffer.append('\n||')
                for cell in item.childNodes:
                    buffer.append(self.inline_markup(cell))
                    buffer.append('||')
                    self.lastsegment = cell.tagName
        return ''.join(buffer)

    def toString(self):
        if False:
            while True:
                i = 10
        ' Converts the document to a string.\n            FIXME: Result from second call differs from first call\n        '
        body = self.content.getElementsByTagName('office:body')[0]
        text = body.childNodes[0]
        buffer = []
        paragraphs = [el for el in text.childNodes if el.tagName in ['draw:page', 'text:p', 'text:h', 'text:section', 'text:list', 'table:table']]
        for paragraph in paragraphs:
            if paragraph.tagName == 'text:list':
                text = self.listToString(paragraph)
            elif paragraph.tagName == 'text:section':
                text = self.textToString(paragraph)
            elif paragraph.tagName == 'table:table':
                text = self.tableToString(paragraph)
            else:
                text = self.paragraphToString(paragraph)
            if text:
                buffer.append(text)
        if self.footnotes:
            buffer.append('----')
            for (cite, body) in self.footnotes:
                buffer.append(f'{cite}: {body}')
        buffer.append('')
        return self.compressCodeBlocks('\n'.join(buffer))

    def textToString(self, element):
        if False:
            for i in range(10):
                print('nop')
        buffer = []
        for node in element.childNodes:
            if node.nodeType == xml.dom.Node.TEXT_NODE:
                buffer.append(node.nodeValue)
            elif node.nodeType == xml.dom.Node.ELEMENT_NODE:
                tag = node.tagName
                if tag in ('draw:text-box', 'draw:frame'):
                    buffer.append(self.textToString(node))
                elif tag in ('text:p', 'text:h'):
                    text = self.paragraphToString(node)
                    if text:
                        buffer.append(text)
                elif tag == 'text:list':
                    buffer.append(self.listToString(node))
                else:
                    method = self.elements.get(tag)
                    if method:
                        buffer.append(method(node))
                    else:
                        buffer.append(' {' + tag + '} ')
        return ''.join(buffer)

    def paragraphToString(self, paragraph, indent=0):
        if False:
            print('Hello World!')
        dummyParaProps = ParagraphProps()
        style_name = paragraph.getAttribute('text:style-name')
        paraProps = self.paragraphStyles.get(style_name, dummyParaProps)
        text = self.inline_markup(paragraph)
        if paraProps and (not paraProps.code):
            text = text.strip()
        if paragraph.tagName == 'text:p' and self.lastsegment == 'text:p':
            text = '\n' + text
        self.lastsegment = paragraph.tagName
        if paraProps.title:
            self.hasTitle = 1
            return '= ' + text + ' =\n'
        outlinelevel = paragraph.getAttribute('text:outline-level')
        if outlinelevel:
            level = int(outlinelevel)
            if self.hasTitle:
                level += 1
            if level >= 1:
                return '=' * level + ' ' + text + ' ' + '=' * level + '\n'
        elif paraProps.code:
            return '{{{\n' + text + '\n}}}\n'
        if paraProps.indented:
            return self.wrapParagraph(text, indent=indent, blockquote=True)
        else:
            return self.wrapParagraph(text, indent=indent)

    def wrapParagraph(self, text, indent=0, blockquote=False):
        if False:
            print('Hello World!')
        counter = 0
        buffer = []
        LIMIT = 50
        if blockquote:
            buffer.append('  ')
        return ''.join(buffer) + text
        for token in text.split():
            if counter > LIMIT - indent:
                buffer.append('\n' + ' ' * indent)
                if blockquote:
                    buffer.append('  ')
                counter = 0
            buffer.append(token + ' ')
            counter += len(token)
        return ''.join(buffer)