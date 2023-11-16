import os, re, codecs, operator, io
from xml.sax.saxutils import escape
from datetime import date
from xml.etree.ElementTree import Element, SubElement, ElementTree
from .pylrf import LrfWriter, LrfObject, LrfTag, LrfToc, STREAM_COMPRESSED, LrfTagStream, LrfStreamBase, IMAGE_TYPE_ENCODING, BINDING_DIRECTION_ENCODING, LINE_TYPE_ENCODING, LrfFileStream, STREAM_FORCE_COMPRESSED
from calibre.utils.date import isoformat
DEFAULT_SOURCE_ENCODING = 'cp1252'
DEFAULT_GENREADING = 'fs'
from calibre import __appname__, __version__
from calibre import entity_to_unicode
from polyglot.builtins import string_or_bytes, iteritems, native_string_type

class LrsError(Exception):
    pass

class ContentError(Exception):
    pass

def _checkExists(filename):
    if False:
        return 10
    if not os.path.exists(filename):
        raise LrsError("file '%s' not found" % filename)

def _formatXml(root):
    if False:
        while True:
            i = 10
    ' A helper to make the LRS output look nicer. '
    for elem in root.iter():
        if len(elem) > 0 and (not elem.text or not elem.text.strip()):
            elem.text = '\n'
        if not elem.tail or not elem.tail.strip():
            elem.tail = '\n'

def ElementWithText(tag, text, **extra):
    if False:
        i = 10
        return i + 15
    ' A shorthand function to create Elements with text. '
    e = Element(tag, **extra)
    e.text = text
    return e

def ElementWithReading(tag, text, reading=False):
    if False:
        return 10
    ' A helper function that creates reading attributes. '
    if text is None:
        readingText = ''
    elif isinstance(text, string_or_bytes):
        readingText = text
    else:
        readingText = text[1]
        text = text[0]
    if not reading:
        readingText = ''
    return ElementWithText(tag, text, reading=readingText)

def appendTextElements(e, contentsList, se):
    if False:
        while True:
            i = 10
    ' A helper function to convert text streams into the proper elements. '

    def uconcat(text, newText, se):
        if False:
            i = 10
            return i + 15
        if isinstance(text, bytes):
            text = text.decode(se)
        if isinstance(newText, bytes):
            newText = newText.decode(se)
        return text + newText
    e.text = ''
    lastElement = None
    for content in contentsList:
        if not isinstance(content, Text):
            newElement = content.toElement(se)
            if newElement is None:
                continue
            lastElement = newElement
            lastElement.tail = ''
            e.append(lastElement)
        elif lastElement is None:
            e.text = uconcat(e.text, content.text, se)
        else:
            lastElement.tail = uconcat(lastElement.tail, content.text, se)

class Delegator:
    """ A mixin class to create delegated methods that create elements. """

    def __init__(self, delegates):
        if False:
            return 10
        self.delegates = delegates
        self.delegatedMethods = []
        for d in delegates:
            d.parent = self
            methods = d.getMethods()
            self.delegatedMethods += methods
            for m in methods:
                setattr(self, m, getattr(d, m))
            '\n            for setting in d.getSettings():\n                if isinstance(setting, string_or_bytes):\n                    setting = (d, setting)\n                delegates =                         self.delegatedSettingsDict.setdefault(setting[1], [])\n                delegates.append(setting[0])\n                self.delegatedSettings.append(setting)\n            '

    def applySetting(self, name, value, testValid=False):
        if False:
            while True:
                i = 10
        applied = False
        if name in self.getSettings():
            setattr(self, name, value)
            applied = True
        for d in self.delegates:
            if hasattr(d, 'applySetting'):
                applied = applied or d.applySetting(name, value)
            elif name in d.getSettings():
                setattr(d, name, value)
                applied = True
        if testValid and (not applied):
            raise LrsError('setting %s not valid' % name)
        return applied

    def applySettings(self, settings, testValid=False):
        if False:
            return 10
        for (setting, value) in settings.items():
            self.applySetting(setting, value, testValid)
            '\n            if setting not in self.delegatedSettingsDict:\n                raise LrsError, "setting %s not valid" % setting\n            delegates = self.delegatedSettingsDict[setting]\n            for d in delegates:\n                setattr(d, setting, value)\n            '

    def appendDelegates(self, element, sourceEncoding):
        if False:
            while True:
                i = 10
        for d in self.delegates:
            e = d.toElement(sourceEncoding)
            if e is not None:
                if isinstance(e, list):
                    for e1 in e:
                        element.append(e1)
                else:
                    element.append(e)

    def appendReferencedObjects(self, parent):
        if False:
            return 10
        for d in self.delegates:
            d.appendReferencedObjects(parent)

    def getMethods(self):
        if False:
            return 10
        return self.delegatedMethods

    def getSettings(self):
        if False:
            print('Hello World!')
        return []

    def toLrfDelegates(self, lrfWriter):
        if False:
            print('Hello World!')
        for d in self.delegates:
            d.toLrf(lrfWriter)

    def toLrf(self, lrfWriter):
        if False:
            return 10
        self.toLrfDelegates(lrfWriter)

class LrsAttributes:
    """ A mixin class to handle default and user supplied attributes. """

    def __init__(self, defaults, alsoAllow=None, **settings):
        if False:
            i = 10
            return i + 15
        if alsoAllow is None:
            alsoAllow = []
        self.attrs = defaults.copy()
        for (name, value) in settings.items():
            if name not in self.attrs and name not in alsoAllow:
                raise LrsError('%s does not support setting %s' % (self.__class__.__name__, name))
            if isinstance(value, int):
                value = str(value)
            self.attrs[name] = value

class LrsContainer:
    """ This class is a mixin class for elements that are contained in or
        contain an unknown number of other elements.
    """

    def __init__(self, validChildren):
        if False:
            return 10
        self.parent = None
        self.contents = []
        self.validChildren = validChildren
        self.must_append = False

    def has_text(self):
        if False:
            print('Hello World!')
        ' Return True iff this container has non whitespace text '
        if hasattr(self, 'text'):
            if self.text.strip():
                return True
        if hasattr(self, 'contents'):
            for child in self.contents:
                if child.has_text():
                    return True
        for item in self.contents:
            if isinstance(item, (Plot, ImageBlock, Canvas, CR)):
                return True
        return False

    def append_to(self, parent):
        if False:
            i = 10
            return i + 15
        '\n        Append self to C{parent} iff self has non whitespace textual content\n        @type parent: LrsContainer\n        '
        if self.contents or self.must_append:
            parent.append(self)

    def appendReferencedObjects(self, parent):
        if False:
            print('Hello World!')
        for c in self.contents:
            c.appendReferencedObjects(parent)

    def setParent(self, parent):
        if False:
            for i in range(10):
                print('nop')
        if self.parent is not None:
            raise LrsError('object already has parent')
        self.parent = parent

    def append(self, content, convertText=True):
        if False:
            return 10
        '\n            Appends valid objects to container.  Can auto-covert text strings\n            to Text objects.\n        '
        for validChild in self.validChildren:
            if isinstance(content, validChild):
                break
        else:
            raise LrsError("can't append %s to %s" % (content.__class__.__name__, self.__class__.__name__))
        if convertText and isinstance(content, string_or_bytes):
            content = Text(content)
        content.setParent(self)
        if isinstance(content, LrsObject):
            content.assignId()
        self.contents.append(content)
        return self

    def get_all(self, predicate=lambda x: x):
        if False:
            print('Hello World!')
        for child in self.contents:
            if predicate(child):
                yield child
            if hasattr(child, 'get_all'):
                yield from child.get_all(predicate)

class LrsObject:
    """ A mixin class for elements that need an object id. """
    nextObjId = 0

    @classmethod
    def getNextObjId(selfClass):
        if False:
            while True:
                i = 10
        selfClass.nextObjId += 1
        return selfClass.nextObjId

    def __init__(self, assignId=False):
        if False:
            return 10
        if assignId:
            self.objId = LrsObject.getNextObjId()
        else:
            self.objId = 0

    def assignId(self):
        if False:
            return 10
        if self.objId != 0:
            raise LrsError('id already assigned to ' + self.__class__.__name__)
        self.objId = LrsObject.getNextObjId()

    def lrsObjectElement(self, name, objlabel='objlabel', labelName=None, labelDecorate=True, **settings):
        if False:
            while True:
                i = 10
        element = Element(name)
        element.attrib['objid'] = str(self.objId)
        if labelName is None:
            labelName = name
        if labelDecorate:
            label = '%s.%d' % (labelName, self.objId)
        else:
            label = str(self.objId)
        element.attrib[objlabel] = label
        element.attrib.update(settings)
        return element

class Book(Delegator):
    """
        Main class for any lrs or lrf.  All objects must be appended to
        the Book class in some way or another in order to be rendered as
        an LRS or LRF file.

        The following settings are available on the constructor of Book:

        author="book author" or author=("book author", "sort as")
        Author of the book.

        title="book title" or title=("book title", "sort as")
        Title of the book.

        sourceencoding="codec"
        Gives the assumed encoding for all non-unicode strings.


        thumbnail="thumbnail file name"
        A small (80x80?) graphics file with a thumbnail of the book's cover.

        bookid="book id"
        A unique id for the book.

        textstyledefault=<dictionary of settings>
        Sets the default values for all TextStyles.

        pagetstyledefault=<dictionary of settings>
        Sets the default values for all PageStyles.

        blockstyledefault=<dictionary of settings>
        Sets the default values for all BlockStyles.

        booksetting=BookSetting()
        Override the default BookSetting.

        setdefault=StyleDefault()
        Override the default SetDefault.

        There are several other settings -- see the BookInfo class for more.
    """

    def __init__(self, textstyledefault=None, blockstyledefault=None, pagestyledefault=None, optimizeTags=False, optimizeCompression=False, **settings):
        if False:
            print('Hello World!')
        self.parent = None
        if 'thumbnail' in settings:
            _checkExists(settings['thumbnail'])
        self.optimizeTags = optimizeTags
        self.optimizeCompression = optimizeCompression
        pageStyle = PageStyle(**PageStyle.baseDefaults.copy())
        blockStyle = BlockStyle(**BlockStyle.baseDefaults.copy())
        textStyle = TextStyle(**TextStyle.baseDefaults.copy())
        if textstyledefault is not None:
            textStyle.update(textstyledefault)
        if blockstyledefault is not None:
            blockStyle.update(blockstyledefault)
        if pagestyledefault is not None:
            pageStyle.update(pagestyledefault)
        self.defaultPageStyle = pageStyle
        self.defaultTextStyle = textStyle
        self.defaultBlockStyle = blockStyle
        LrsObject.nextObjId += 1
        styledefault = StyleDefault()
        if 'setdefault' in settings:
            styledefault = settings.pop('setdefault')
        Delegator.__init__(self, [BookInformation(), Main(), Template(), Style(styledefault), Solos(), Objects()])
        self.sourceencoding = None
        self.applySetting('genreading', DEFAULT_GENREADING)
        self.applySetting('sourceencoding', DEFAULT_SOURCE_ENCODING)
        self.applySettings(settings, testValid=True)
        self.allow_new_page = True
        self.gc_count = 0

    def set_title(self, title):
        if False:
            return 10
        ot = self.delegates[0].delegates[0].delegates[0].title
        self.delegates[0].delegates[0].delegates[0].title = (title, ot[1])

    def set_author(self, author):
        if False:
            i = 10
            return i + 15
        ot = self.delegates[0].delegates[0].delegates[0].author
        self.delegates[0].delegates[0].delegates[0].author = (author, ot[1])

    def create_text_style(self, **settings):
        if False:
            for i in range(10):
                print('nop')
        ans = TextStyle(**self.defaultTextStyle.attrs.copy())
        ans.update(settings)
        return ans

    def create_block_style(self, **settings):
        if False:
            return 10
        ans = BlockStyle(**self.defaultBlockStyle.attrs.copy())
        ans.update(settings)
        return ans

    def create_page_style(self, **settings):
        if False:
            for i in range(10):
                print('nop')
        if not self.allow_new_page:
            raise ContentError
        ans = PageStyle(**self.defaultPageStyle.attrs.copy())
        ans.update(settings)
        return ans

    def create_page(self, pageStyle=None, **settings):
        if False:
            return 10
        '\n        Return a new L{Page}. The page has not been appended to this book.\n        @param pageStyle: If None the default pagestyle is used.\n        @type pageStyle: L{PageStyle}\n        '
        if not pageStyle:
            pageStyle = self.defaultPageStyle
        return Page(pageStyle=pageStyle, **settings)

    def create_text_block(self, textStyle=None, blockStyle=None, **settings):
        if False:
            i = 10
            return i + 15
        '\n        Return a new L{TextBlock}. The block has not been appended to this\n        book.\n        @param textStyle: If None the default text style is used\n        @type textStyle: L{TextStyle}\n        @param blockStyle: If None the default block style is used.\n        @type blockStyle: L{BlockStyle}\n        '
        if not textStyle:
            textStyle = self.defaultTextStyle
        if not blockStyle:
            blockStyle = self.defaultBlockStyle
        return TextBlock(textStyle=textStyle, blockStyle=blockStyle, **settings)

    def pages(self):
        if False:
            i = 10
            return i + 15
        'Return list of Page objects in this book '
        ans = []
        for item in self.delegates:
            if isinstance(item, Main):
                for candidate in item.contents:
                    if isinstance(candidate, Page):
                        ans.append(candidate)
                break
        return ans

    def last_page(self):
        if False:
            for i in range(10):
                print('nop')
        'Return last Page in this book '
        for item in self.delegates:
            if isinstance(item, Main):
                temp = list(item.contents)
                temp.reverse()
                for candidate in temp:
                    if isinstance(candidate, Page):
                        return candidate

    def embed_font(self, file, facename):
        if False:
            for i in range(10):
                print('nop')
        f = Font(file, facename)
        self.append(f)

    def getSettings(self):
        if False:
            i = 10
            return i + 15
        return ['sourceencoding']

    def append(self, content):
        if False:
            return 10
        ' Find and invoke the correct appender for this content. '
        className = content.__class__.__name__
        try:
            method = getattr(self, 'append' + className)
        except AttributeError:
            raise LrsError("can't append %s to Book" % className)
        method(content)

    def rationalize_font_sizes(self, base_font_size=10):
        if False:
            while True:
                i = 10
        base_font_size *= 10.0
        main = None
        for obj in self.delegates:
            if isinstance(obj, Main):
                main = obj
                break
        fonts = {}
        for text in main.get_all(lambda x: isinstance(x, Text)):
            fs = base_font_size
            ancestor = text.parent
            while ancestor:
                try:
                    fs = int(ancestor.attrs['fontsize'])
                    break
                except (AttributeError, KeyError):
                    pass
                try:
                    fs = int(ancestor.textSettings['fontsize'])
                    break
                except (AttributeError, KeyError):
                    pass
                try:
                    fs = int(ancestor.textStyle.attrs['fontsize'])
                    break
                except (AttributeError, KeyError):
                    pass
                ancestor = ancestor.parent
            length = len(text.text)
            fonts[fs] = fonts.get(fs, 0) + length
        if not fonts:
            print('WARNING: LRF seems to have no textual content. Cannot rationalize font sizes.')
            return
        old_base_font_size = float(max(fonts.items(), key=operator.itemgetter(1))[0])
        factor = base_font_size / old_base_font_size

        def rescale(old):
            if False:
                for i in range(10):
                    print('nop')
            return str(int(int(old) * factor))
        text_blocks = list(main.get_all(lambda x: isinstance(x, TextBlock)))
        for tb in text_blocks:
            if 'fontsize' in tb.textSettings:
                tb.textSettings['fontsize'] = rescale(tb.textSettings['fontsize'])
            for span in tb.get_all(lambda x: isinstance(x, Span)):
                if 'fontsize' in span.attrs:
                    span.attrs['fontsize'] = rescale(span.attrs['fontsize'])
                if 'baselineskip' in span.attrs:
                    span.attrs['baselineskip'] = rescale(span.attrs['baselineskip'])
        text_styles = (tb.textStyle for tb in text_blocks)
        for ts in text_styles:
            ts.attrs['fontsize'] = rescale(ts.attrs['fontsize'])
            ts.attrs['baselineskip'] = rescale(ts.attrs['baselineskip'])

    def renderLrs(self, lrsFile, encoding='UTF-8'):
        if False:
            while True:
                i = 10
        if isinstance(lrsFile, string_or_bytes):
            lrsFile = codecs.open(lrsFile, 'wb', encoding=encoding)
        self.render(lrsFile, outputEncodingName=encoding)
        lrsFile.close()

    def renderLrf(self, lrfFile):
        if False:
            for i in range(10):
                print('nop')
        self.appendReferencedObjects(self)
        if isinstance(lrfFile, string_or_bytes):
            lrfFile = open(lrfFile, 'wb')
        lrfWriter = LrfWriter(self.sourceencoding)
        lrfWriter.optimizeTags = self.optimizeTags
        lrfWriter.optimizeCompression = self.optimizeCompression
        self.toLrf(lrfWriter)
        lrfWriter.writeFile(lrfFile)
        lrfFile.close()

    def toElement(self, se):
        if False:
            return 10
        root = Element('BBeBXylog', version='1.0')
        root.append(Element('Property'))
        self.appendDelegates(root, self.sourceencoding)
        return root

    def render(self, f, outputEncodingName='UTF-8'):
        if False:
            while True:
                i = 10
        ' Write the book as an LRS to file f. '
        self.appendReferencedObjects(self)
        root = self.toElement(self.sourceencoding)
        _formatXml(root)
        tree = ElementTree(element=root)
        tree.write(f, encoding=native_string_type(outputEncodingName), xml_declaration=True)

class BookInformation(Delegator):
    """ Just a container for the Info and TableOfContents elements. """

    def __init__(self):
        if False:
            print('Hello World!')
        Delegator.__init__(self, [Info(), TableOfContents()])

    def toElement(self, se):
        if False:
            print('Hello World!')
        bi = Element('BookInformation')
        self.appendDelegates(bi, se)
        return bi

class Info(Delegator):
    """ Just a container for the BookInfo and DocInfo elements. """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.genreading = DEFAULT_GENREADING
        Delegator.__init__(self, [BookInfo(), DocInfo()])

    def getSettings(self):
        if False:
            for i in range(10):
                print('nop')
        return ['genreading']

    def toElement(self, se):
        if False:
            while True:
                i = 10
        info = Element('Info', version='1.1')
        info.append(self.delegates[0].toElement(se, reading='s' in self.genreading))
        info.append(self.delegates[1].toElement(se))
        return info

    def toLrf(self, lrfWriter):
        if False:
            while True:
                i = 10
        info = Element('Info', version='1.1')
        info.append(self.delegates[0].toElement(lrfWriter.getSourceEncoding(), reading='f' in self.genreading))
        info.append(self.delegates[1].toElement(lrfWriter.getSourceEncoding()))
        tnail = info.find('DocInfo/CThumbnail')
        if tnail is not None:
            lrfWriter.setThumbnailFile(tnail.get('file'))
        _formatXml(info)
        tree = ElementTree(element=info)
        f = io.BytesIO()
        tree.write(f, encoding=native_string_type('utf-8'), xml_declaration=True)
        xmlInfo = f.getvalue().decode('utf-8')
        xmlInfo = re.sub('<CThumbnail.*?>\\n', '', xmlInfo)
        xmlInfo = xmlInfo.replace('SumPage>', 'Page>')
        lrfWriter.docInfoXml = xmlInfo

class TableOfContents:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.tocEntries = []

    def appendReferencedObjects(self, parent):
        if False:
            print('Hello World!')
        pass

    def getMethods(self):
        if False:
            for i in range(10):
                print('nop')
        return ['addTocEntry']

    def getSettings(self):
        if False:
            print('Hello World!')
        return []

    def addTocEntry(self, tocLabel, textBlock):
        if False:
            i = 10
            return i + 15
        if not isinstance(textBlock, (Canvas, TextBlock, ImageBlock, RuledLine)):
            raise LrsError('TOC destination must be a Canvas, TextBlock, ImageBlock or RuledLine' + ' not a ' + str(type(textBlock)))
        if textBlock.parent is None:
            raise LrsError('TOC text block must be already appended to a page')
        if False and textBlock.parent.parent is None:
            raise LrsError('TOC destination page must be already appended to a book')
        if not hasattr(textBlock.parent, 'objId'):
            raise LrsError('TOC destination must be appended to a container with an objID')
        for tl in self.tocEntries:
            if tl.label == tocLabel and tl.textBlock == textBlock:
                return
        self.tocEntries.append(TocLabel(tocLabel, textBlock))
        textBlock.tocLabel = tocLabel

    def toElement(self, se):
        if False:
            print('Hello World!')
        if len(self.tocEntries) == 0:
            return None
        toc = Element('TOC')
        for t in self.tocEntries:
            toc.append(t.toElement(se))
        return toc

    def toLrf(self, lrfWriter):
        if False:
            return 10
        if len(self.tocEntries) == 0:
            return
        toc = []
        for t in self.tocEntries:
            toc.append((t.textBlock.parent.objId, t.textBlock.objId, t.label))
        lrfToc = LrfToc(LrsObject.getNextObjId(), toc, lrfWriter.getSourceEncoding())
        lrfWriter.append(lrfToc)
        lrfWriter.setTocObject(lrfToc)

class TocLabel:

    def __init__(self, label, textBlock):
        if False:
            for i in range(10):
                print('nop')
        self.label = escape(re.sub('&(\\S+?);', entity_to_unicode, label))
        self.textBlock = textBlock

    def toElement(self, se):
        if False:
            i = 10
            return i + 15
        return ElementWithText('TocLabel', self.label, refobj=str(self.textBlock.objId), refpage=str(self.textBlock.parent.objId))

class BookInfo:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.title = 'Untitled'
        self.author = 'Anonymous'
        self.bookid = None
        self.pi = None
        self.isbn = None
        self.publisher = None
        self.freetext = '\n\n'
        self.label = None
        self.category = None
        self.classification = None

    def appendReferencedObjects(self, parent):
        if False:
            print('Hello World!')
        pass

    def getMethods(self):
        if False:
            i = 10
            return i + 15
        return []

    def getSettings(self):
        if False:
            return 10
        return ['author', 'title', 'bookid', 'isbn', 'publisher', 'freetext', 'label', 'category', 'classification']

    def _appendISBN(self, bi):
        if False:
            print('Hello World!')
        pi = Element('ProductIdentifier')
        isbnElement = ElementWithText('ISBNPrintable', self.isbn)
        isbnValueElement = ElementWithText('ISBNValue', self.isbn.replace('-', ''))
        pi.append(isbnElement)
        pi.append(isbnValueElement)
        bi.append(pi)

    def toElement(self, se, reading=True):
        if False:
            i = 10
            return i + 15
        bi = Element('BookInfo')
        bi.append(ElementWithReading('Title', self.title, reading=reading))
        bi.append(ElementWithReading('Author', self.author, reading=reading))
        bi.append(ElementWithText('BookID', self.bookid))
        if self.isbn is not None:
            self._appendISBN(bi)
        if self.publisher is not None:
            bi.append(ElementWithReading('Publisher', self.publisher))
        bi.append(ElementWithReading('Label', self.label, reading=reading))
        bi.append(ElementWithText('Category', self.category))
        bi.append(ElementWithText('Classification', self.classification))
        bi.append(ElementWithText('FreeText', self.freetext))
        return bi

class DocInfo:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.thumbnail = None
        self.language = 'en'
        self.creator = None
        self.creationdate = str(isoformat(date.today()))
        self.producer = '%s v%s' % (__appname__, __version__)
        self.numberofpages = '0'

    def appendReferencedObjects(self, parent):
        if False:
            i = 10
            return i + 15
        pass

    def getMethods(self):
        if False:
            print('Hello World!')
        return []

    def getSettings(self):
        if False:
            for i in range(10):
                print('nop')
        return ['thumbnail', 'language', 'creator', 'creationdate', 'producer', 'numberofpages']

    def toElement(self, se):
        if False:
            return 10
        docInfo = Element('DocInfo')
        if self.thumbnail is not None:
            docInfo.append(Element('CThumbnail', file=self.thumbnail))
        docInfo.append(ElementWithText('Language', self.language))
        docInfo.append(ElementWithText('Creator', self.creator))
        docInfo.append(ElementWithText('CreationDate', self.creationdate))
        docInfo.append(ElementWithText('Producer', self.producer))
        docInfo.append(ElementWithText('SumPage', str(self.numberofpages)))
        return docInfo

class Main(LrsContainer):

    def __init__(self):
        if False:
            print('Hello World!')
        LrsContainer.__init__(self, [Page])

    def getMethods(self):
        if False:
            return 10
        return ['appendPage', 'Page']

    def getSettings(self):
        if False:
            while True:
                i = 10
        return []

    def Page(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        p = Page(*args, **kwargs)
        self.append(p)
        return p

    def appendPage(self, page):
        if False:
            while True:
                i = 10
        self.append(page)

    def toElement(self, sourceEncoding):
        if False:
            return 10
        main = Element(self.__class__.__name__)
        for page in self.contents:
            main.append(page.toElement(sourceEncoding))
        return main

    def toLrf(self, lrfWriter):
        if False:
            for i in range(10):
                print('nop')
        pageIds = []
        pageTreeId = LrsObject.getNextObjId()
        lrfWriter.setPageTreeId(pageTreeId)
        for p in self.contents:
            pageIds.append(p.objId)
            p.toLrf(lrfWriter)
        pageTree = LrfObject('PageTree', pageTreeId)
        pageTree.appendLrfTag(LrfTag('PageList', pageIds))
        lrfWriter.append(pageTree)

class Solos(LrsContainer):

    def __init__(self):
        if False:
            while True:
                i = 10
        LrsContainer.__init__(self, [Solo])

    def getMethods(self):
        if False:
            i = 10
            return i + 15
        return ['appendSolo', 'Solo']

    def getSettings(self):
        if False:
            for i in range(10):
                print('nop')
        return []

    def Solo(self, *args, **kwargs):
        if False:
            print('Hello World!')
        p = Solo(*args, **kwargs)
        self.append(p)
        return p

    def appendSolo(self, solo):
        if False:
            return 10
        self.append(solo)

    def toLrf(self, lrfWriter):
        if False:
            for i in range(10):
                print('nop')
        for s in self.contents:
            s.toLrf(lrfWriter)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        solos = []
        for s in self.contents:
            solos.append(s.toElement(se))
        if len(solos) == 0:
            return None
        return solos

class Solo(Main):
    pass

class Template:
    """ Does nothing that I know of. """

    def appendReferencedObjects(self, parent):
        if False:
            while True:
                i = 10
        pass

    def getMethods(self):
        if False:
            return 10
        return []

    def getSettings(self):
        if False:
            return 10
        return []

    def toElement(self, se):
        if False:
            return 10
        t = Element('Template')
        t.attrib['version'] = '1.0'
        return t

    def toLrf(self, lrfWriter):
        if False:
            i = 10
            return i + 15
        pass

class StyleDefault(LrsAttributes):
    """
        Supply some defaults for all TextBlocks.
        The legal values are a subset of what is allowed on a
        TextBlock -- ruby, emphasis, and waitprop settings.
    """
    defaults = dict(rubyalign='start', rubyadjust='none', rubyoverhang='none', empdotsposition='before', empdotsfontname='Dutch801 Rm BT Roman', empdotscode='0x002e', emplineposition='after', emplinetype='solid', setwaitprop='noreplay')
    alsoAllow = ['refempdotsfont', 'rubyAlignAndAdjust']

    def __init__(self, **settings):
        if False:
            i = 10
            return i + 15
        LrsAttributes.__init__(self, self.defaults, alsoAllow=self.alsoAllow, **settings)

    def toElement(self, se):
        if False:
            i = 10
            return i + 15
        return Element('SetDefault', self.attrs)

class Style(LrsContainer, Delegator):

    def __init__(self, styledefault=StyleDefault()):
        if False:
            print('Hello World!')
        LrsContainer.__init__(self, [PageStyle, TextStyle, BlockStyle])
        Delegator.__init__(self, [BookStyle(styledefault=styledefault)])
        self.bookStyle = self.delegates[0]
        self.appendPageStyle = self.appendTextStyle = self.appendBlockStyle = self.append

    def appendReferencedObjects(self, parent):
        if False:
            for i in range(10):
                print('nop')
        LrsContainer.appendReferencedObjects(self, parent)

    def getMethods(self):
        if False:
            i = 10
            return i + 15
        return ['PageStyle', 'TextStyle', 'BlockStyle', 'appendPageStyle', 'appendTextStyle', 'appendBlockStyle'] + self.delegatedMethods

    def getSettings(self):
        if False:
            i = 10
            return i + 15
        return [(self.bookStyle, x) for x in self.bookStyle.getSettings()]

    def PageStyle(self, *args, **kwargs):
        if False:
            return 10
        ps = PageStyle(*args, **kwargs)
        self.append(ps)
        return ps

    def TextStyle(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ts = TextStyle(*args, **kwargs)
        self.append(ts)
        return ts

    def BlockStyle(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        bs = BlockStyle(*args, **kwargs)
        self.append(bs)
        return bs

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        style = Element('Style')
        style.append(self.bookStyle.toElement(se))
        for content in self.contents:
            style.append(content.toElement(se))
        return style

    def toLrf(self, lrfWriter):
        if False:
            return 10
        self.bookStyle.toLrf(lrfWriter)
        for s in self.contents:
            s.toLrf(lrfWriter)

class BookStyle(LrsObject, LrsContainer):

    def __init__(self, styledefault=StyleDefault()):
        if False:
            for i in range(10):
                print('nop')
        LrsObject.__init__(self, assignId=True)
        LrsContainer.__init__(self, [Font])
        self.styledefault = styledefault
        self.booksetting = BookSetting()
        self.appendFont = self.append

    def getSettings(self):
        if False:
            print('Hello World!')
        return ['styledefault', 'booksetting']

    def getMethods(self):
        if False:
            i = 10
            return i + 15
        return ['Font', 'appendFont']

    def Font(self, *args, **kwargs):
        if False:
            print('Hello World!')
        f = Font(*args, **kwargs)
        self.append(f)
        return

    def toElement(self, se):
        if False:
            return 10
        bookStyle = self.lrsObjectElement('BookStyle', objlabel='stylelabel', labelDecorate=False)
        bookStyle.append(self.styledefault.toElement(se))
        bookStyle.append(self.booksetting.toElement(se))
        for font in self.contents:
            bookStyle.append(font.toElement(se))
        return bookStyle

    def toLrf(self, lrfWriter):
        if False:
            print('Hello World!')
        bookAtr = LrfObject('BookAtr', self.objId)
        bookAtr.appendLrfTag(LrfTag('ChildPageTree', lrfWriter.getPageTreeId()))
        bookAtr.appendTagDict(self.styledefault.attrs)
        self.booksetting.toLrf(lrfWriter)
        lrfWriter.append(bookAtr)
        lrfWriter.setRootObject(bookAtr)
        for font in self.contents:
            font.toLrf(lrfWriter)

class BookSetting(LrsAttributes):

    def __init__(self, **settings):
        if False:
            while True:
                i = 10
        defaults = dict(bindingdirection='Lr', dpi='1660', screenheight='800', screenwidth='600', colordepth='24')
        LrsAttributes.__init__(self, defaults, **settings)

    def toLrf(self, lrfWriter):
        if False:
            while True:
                i = 10
        a = self.attrs
        lrfWriter.dpi = int(a['dpi'])
        lrfWriter.bindingdirection = BINDING_DIRECTION_ENCODING[a['bindingdirection']]
        lrfWriter.height = int(a['screenheight'])
        lrfWriter.width = int(a['screenwidth'])
        lrfWriter.colorDepth = int(a['colordepth'])

    def toElement(self, se):
        if False:
            while True:
                i = 10
        return Element('BookSetting', self.attrs)

class LrsStyle(LrsObject, LrsAttributes, LrsContainer):
    """ A mixin class for styles. """

    def __init__(self, elementName, defaults=None, alsoAllow=None, **overrides):
        if False:
            return 10
        if defaults is None:
            defaults = {}
        LrsObject.__init__(self)
        LrsAttributes.__init__(self, defaults, alsoAllow=alsoAllow, **overrides)
        LrsContainer.__init__(self, [])
        self.elementName = elementName
        self.objectsAppended = False

    def update(self, settings):
        if False:
            print('Hello World!')
        for (name, value) in settings.items():
            if name not in self.__class__.validSettings:
                raise LrsError(f'{name} not a valid setting for {self.__class__.__name__}')
            self.attrs[name] = value

    def getLabel(self):
        if False:
            print('Hello World!')
        return str(self.objId)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        element = Element(self.elementName, stylelabel=self.getLabel(), objid=str(self.objId))
        element.attrib.update(self.attrs)
        return element

    def toLrf(self, lrfWriter):
        if False:
            i = 10
            return i + 15
        obj = LrfObject(self.elementName, self.objId)
        obj.appendTagDict(self.attrs, self.__class__.__name__)
        lrfWriter.append(obj)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if hasattr(other, 'attrs'):
            return self.__class__ == other.__class__ and self.attrs == other.attrs
        return False

class TextStyle(LrsStyle):
    """
        The text style of a TextBlock.  Default is 10 pt. Times Roman.

        Setting         Value                   Default
        --------        -----                   -------
        align           "head","center","foot"  "head" (left aligned)
        baselineskip    points * 10             120 (12 pt. distance between
                                                  bottoms of lines)
        fontsize        points * 10             100 (10 pt.)
        fontweight      1 to 1000               400 (normal, 800 is bold)
        fontwidth       points * 10 or -10      -10 (use values from font)
        linespace       points * 10             10 (min space btw. lines?)
        wordspace       points * 10             25 (min space btw. each word)

    """
    baseDefaults = dict(columnsep='0', charspace='0', textlinewidth='2', align='head', linecolor='0x00000000', column='1', fontsize='100', fontwidth='-10', fontescapement='0', fontorientation='0', fontweight='400', fontfacename='Dutch801 Rm BT Roman', textcolor='0x00000000', wordspace='25', letterspace='0', baselineskip='120', linespace='10', parindent='0', parskip='0', textbgcolor='0xFF000000')
    alsoAllow = ['empdotscode', 'empdotsfontname', 'refempdotsfont', 'rubyadjust', 'rubyalign', 'rubyoverhang', 'empdotsposition', 'emplinetype', 'emplineposition']
    validSettings = list(baseDefaults) + alsoAllow
    defaults = baseDefaults.copy()

    def __init__(self, **overrides):
        if False:
            return 10
        LrsStyle.__init__(self, 'TextStyle', self.defaults, alsoAllow=self.alsoAllow, **overrides)

    def copy(self):
        if False:
            return 10
        tb = TextStyle()
        tb.attrs = self.attrs.copy()
        return tb

class BlockStyle(LrsStyle):
    """
        The block style of a TextBlock.  Default is an expandable 560 pixel
        wide area with no space for headers or footers.

        Setting      Value                  Default
        --------     -----                  -------
        blockwidth   pixels                 560
        sidemargin   pixels                 0
    """
    baseDefaults = dict(bgimagemode='fix', framemode='square', blockwidth='560', blockheight='100', blockrule='horz-adjustable', layout='LrTb', framewidth='0', framecolor='0x00000000', topskip='0', sidemargin='0', footskip='0', bgcolor='0xFF000000')
    validSettings = baseDefaults.keys()
    defaults = baseDefaults.copy()

    def __init__(self, **overrides):
        if False:
            print('Hello World!')
        LrsStyle.__init__(self, 'BlockStyle', self.defaults, **overrides)

    def copy(self):
        if False:
            while True:
                i = 10
        tb = BlockStyle()
        tb.attrs = self.attrs.copy()
        return tb

class PageStyle(LrsStyle):
    """
        Setting         Value                   Default
        --------        -----                   -------
        evensidemargin  pixels                  20
        oddsidemargin   pixels                  20
        topmargin       pixels                  20
    """
    baseDefaults = dict(topmargin='20', headheight='0', headsep='0', oddsidemargin='20', textheight='747', textwidth='575', footspace='0', evensidemargin='20', footheight='0', layout='LrTb', bgimagemode='fix', pageposition='any', setwaitprop='noreplay', setemptyview='show')
    alsoAllow = ['header', 'evenheader', 'oddheader', 'footer', 'evenfooter', 'oddfooter']
    validSettings = list(baseDefaults) + alsoAllow
    defaults = baseDefaults.copy()

    @classmethod
    def translateHeaderAndFooter(selfClass, parent, settings):
        if False:
            i = 10
            return i + 15
        selfClass._fixup(parent, 'header', settings)
        selfClass._fixup(parent, 'footer', settings)

    @classmethod
    def _fixup(selfClass, parent, basename, settings):
        if False:
            for i in range(10):
                print('nop')
        evenbase = 'even' + basename
        oddbase = 'odd' + basename
        if basename in settings:
            baseObj = settings[basename]
            del settings[basename]
            settings[evenbase] = settings[oddbase] = baseObj
        if evenbase in settings:
            evenObj = settings[evenbase]
            del settings[evenbase]
            if evenObj.parent is None:
                parent.append(evenObj)
            settings[evenbase + 'id'] = str(evenObj.objId)
        if oddbase in settings:
            oddObj = settings[oddbase]
            del settings[oddbase]
            if oddObj.parent is None:
                parent.append(oddObj)
            settings[oddbase + 'id'] = str(oddObj.objId)

    def appendReferencedObjects(self, parent):
        if False:
            print('Hello World!')
        if self.objectsAppended:
            return
        PageStyle.translateHeaderAndFooter(parent, self.attrs)
        self.objectsAppended = True

    def __init__(self, **settings):
        if False:
            return 10
        LrsStyle.__init__(self, 'PageStyle', self.defaults, alsoAllow=self.alsoAllow, **settings)

class Page(LrsObject, LrsContainer):
    """
        Pages are added to Books.  Pages can be supplied a PageStyle.
        If they are not, Page.defaultPageStyle will be used.
    """
    defaultPageStyle = PageStyle()

    def __init__(self, pageStyle=defaultPageStyle, **settings):
        if False:
            for i in range(10):
                print('nop')
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [TextBlock, BlockSpace, RuledLine, ImageBlock, Canvas])
        self.pageStyle = pageStyle
        for settingName in settings.keys():
            if settingName not in PageStyle.defaults and settingName not in PageStyle.alsoAllow:
                raise LrsError('setting %s not allowed on Page' % settingName)
        self.settings = settings.copy()

    def appendReferencedObjects(self, parent):
        if False:
            i = 10
            return i + 15
        PageStyle.translateHeaderAndFooter(parent, self.settings)
        self.pageStyle.appendReferencedObjects(parent)
        if self.pageStyle.parent is None:
            parent.append(self.pageStyle)
        LrsContainer.appendReferencedObjects(self, parent)

    def RuledLine(self, *args, **kwargs):
        if False:
            return 10
        rl = RuledLine(*args, **kwargs)
        self.append(rl)
        return rl

    def BlockSpace(self, *args, **kwargs):
        if False:
            print('Hello World!')
        bs = BlockSpace(*args, **kwargs)
        self.append(bs)
        return bs

    def TextBlock(self, *args, **kwargs):
        if False:
            return 10
        ' Create and append a new text block (shortcut). '
        tb = TextBlock(*args, **kwargs)
        self.append(tb)
        return tb

    def ImageBlock(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ' Create and append and new Image block (shorthand). '
        ib = ImageBlock(*args, **kwargs)
        self.append(ib)
        return ib

    def addLrfObject(self, objId):
        if False:
            while True:
                i = 10
        self.stream.appendLrfTag(LrfTag('Link', objId))

    def appendLrfTag(self, lrfTag):
        if False:
            while True:
                i = 10
        self.stream.appendLrfTag(lrfTag)

    def toLrf(self, lrfWriter):
        if False:
            return 10
        p = LrfObject('Page', self.objId)
        lrfWriter.append(p)
        pageContent = set()
        self.stream = LrfTagStream(0)
        for content in self.contents:
            content.toLrfContainer(lrfWriter, self)
            if hasattr(content, 'getReferencedObjIds'):
                pageContent.update(content.getReferencedObjIds())
        p.appendLrfTag(LrfTag('Link', self.pageStyle.objId))
        p.appendLrfTag(LrfTag('ParentPageTree', lrfWriter.getPageTreeId()))
        p.appendTagDict(self.settings)
        p.appendLrfTags(self.stream.getStreamTags(lrfWriter.getSourceEncoding()))

    def toElement(self, sourceEncoding):
        if False:
            i = 10
            return i + 15
        page = self.lrsObjectElement('Page')
        page.set('pagestyle', self.pageStyle.getLabel())
        page.attrib.update(self.settings)
        for content in self.contents:
            page.append(content.toElement(sourceEncoding))
        return page

class TextBlock(LrsObject, LrsContainer):
    """
        TextBlocks are added to Pages.  They hold Paragraphs or CRs.

        If a TextBlock is used in a header, it should be appended to
        the Book, not to a specific Page.
    """
    defaultTextStyle = TextStyle()
    defaultBlockStyle = BlockStyle()

    def __init__(self, textStyle=defaultTextStyle, blockStyle=defaultBlockStyle, **settings):
        if False:
            i = 10
            return i + 15
        '\n        Create TextBlock.\n        @param textStyle: The L{TextStyle} for this block.\n        @param blockStyle: The L{BlockStyle} for this block.\n        @param settings: C{dict} of extra settings to apply to this block.\n        '
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [Paragraph, CR])
        self.textSettings = {}
        self.blockSettings = {}
        for (name, value) in settings.items():
            if name in TextStyle.validSettings:
                self.textSettings[name] = value
            elif name in BlockStyle.validSettings:
                self.blockSettings[name] = value
            elif name == 'toclabel':
                self.tocLabel = value
            else:
                raise LrsError('%s not a valid setting for TextBlock' % name)
        self.textStyle = textStyle
        self.blockStyle = blockStyle
        self.currentTextStyle = textStyle.copy() if self.textSettings else textStyle
        self.currentTextStyle.attrs.update(self.textSettings)

    def appendReferencedObjects(self, parent):
        if False:
            return 10
        if self.textStyle.parent is None:
            parent.append(self.textStyle)
        if self.blockStyle.parent is None:
            parent.append(self.blockStyle)
        LrsContainer.appendReferencedObjects(self, parent)

    def Paragraph(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n            Create and append a Paragraph to this TextBlock.  A CR is\n            automatically inserted after the Paragraph.  To avoid this\n            behavior, create the Paragraph and append it to the TextBlock\n            in a separate call.\n        '
        p = Paragraph(*args, **kwargs)
        self.append(p)
        self.append(CR())
        return p

    def toElement(self, sourceEncoding):
        if False:
            return 10
        tb = self.lrsObjectElement('TextBlock', labelName='Block')
        tb.attrib.update(self.textSettings)
        tb.attrib.update(self.blockSettings)
        tb.set('textstyle', self.textStyle.getLabel())
        tb.set('blockstyle', self.blockStyle.getLabel())
        if hasattr(self, 'tocLabel'):
            tb.set('toclabel', self.tocLabel)
        for content in self.contents:
            tb.append(content.toElement(sourceEncoding))
        return tb

    def getReferencedObjIds(self):
        if False:
            print('Hello World!')
        ids = [self.objId, self.extraId, self.blockStyle.objId, self.textStyle.objId]
        for content in self.contents:
            if hasattr(content, 'getReferencedObjIds'):
                ids.extend(content.getReferencedObjIds())
        return ids

    def toLrf(self, lrfWriter):
        if False:
            i = 10
            return i + 15
        self.toLrfContainer(lrfWriter, lrfWriter)

    def toLrfContainer(self, lrfWriter, container):
        if False:
            for i in range(10):
                print('nop')
        extraId = LrsObject.getNextObjId()
        b = LrfObject('Block', self.objId)
        b.appendLrfTag(LrfTag('Link', self.blockStyle.objId))
        b.appendLrfTags(LrfTagStream(0, [LrfTag('Link', extraId)]).getStreamTags(lrfWriter.getSourceEncoding()))
        b.appendTagDict(self.blockSettings)
        container.addLrfObject(b.objId)
        lrfWriter.append(b)
        tb = LrfObject('TextBlock', extraId)
        tb.appendLrfTag(LrfTag('Link', self.textStyle.objId))
        tb.appendTagDict(self.textSettings)
        stream = LrfTagStream(STREAM_COMPRESSED)
        for content in self.contents:
            content.toLrfContainer(lrfWriter, stream)
        if lrfWriter.saveStreamTags:
            tb.saveStreamTags = stream.tags
        tb.appendLrfTags(stream.getStreamTags(lrfWriter.getSourceEncoding(), optimizeTags=lrfWriter.optimizeTags, optimizeCompression=lrfWriter.optimizeCompression))
        lrfWriter.append(tb)
        self.extraId = extraId

class Paragraph(LrsContainer):
    """
        Note: <P> alone does not make a paragraph.  Only a CR inserted
        into a text block right after a <P> makes a real paragraph.
        Two Paragraphs appended in a row act like a single Paragraph.

        Also note that there are few autoappenders for Paragraph (and
        the things that can go in it.)  It's less confusing (to me) to use
        explicit .append methods to build up the text stream.
    """

    def __init__(self, text=None):
        if False:
            for i in range(10):
                print('nop')
        LrsContainer.__init__(self, [Text, CR, DropCaps, CharButton, LrsSimpleChar1, bytes, str])
        if text is not None:
            if isinstance(text, string_or_bytes):
                text = Text(text)
            self.append(text)

    def CR(self):
        if False:
            print('Hello World!')
        cr = CR()
        self.append(cr)
        return cr

    def getReferencedObjIds(self):
        if False:
            i = 10
            return i + 15
        ids = []
        for content in self.contents:
            if hasattr(content, 'getReferencedObjIds'):
                ids.extend(content.getReferencedObjIds())
        return ids

    def toLrfContainer(self, lrfWriter, parent):
        if False:
            for i in range(10):
                print('nop')
        parent.appendLrfTag(LrfTag('pstart', 0))
        for content in self.contents:
            content.toLrfContainer(lrfWriter, parent)
        parent.appendLrfTag(LrfTag('pend'))

    def toElement(self, sourceEncoding):
        if False:
            return 10
        p = Element('P')
        appendTextElements(p, self.contents, sourceEncoding)
        return p

class LrsTextTag(LrsContainer):

    def __init__(self, text, validContents):
        if False:
            return 10
        LrsContainer.__init__(self, [Text, bytes, str] + validContents)
        if text is not None:
            self.append(text)

    def toLrfContainer(self, lrfWriter, parent):
        if False:
            print('Hello World!')
        if hasattr(self, 'tagName'):
            tagName = self.tagName
        else:
            tagName = self.__class__.__name__
        parent.appendLrfTag(LrfTag(tagName))
        for content in self.contents:
            content.toLrfContainer(lrfWriter, parent)
        parent.appendLrfTag(LrfTag(tagName + 'End'))

    def toElement(self, se):
        if False:
            i = 10
            return i + 15
        if hasattr(self, 'tagName'):
            tagName = self.tagName
        else:
            tagName = self.__class__.__name__
        p = Element(tagName)
        appendTextElements(p, self.contents, se)
        return p

class LrsSimpleChar1:

    def isEmpty(self):
        if False:
            while True:
                i = 10
        for content in self.contents:
            if not content.isEmpty():
                return False
        return True

    def hasFollowingContent(self):
        if False:
            for i in range(10):
                print('nop')
        foundSelf = False
        for content in self.parent.contents:
            if content == self:
                foundSelf = True
            elif foundSelf:
                if not content.isEmpty():
                    return True
        return False

class DropCaps(LrsTextTag):

    def __init__(self, line=1):
        if False:
            while True:
                i = 10
        LrsTextTag.__init__(self, None, [LrsSimpleChar1])
        if int(line) <= 0:
            raise LrsError('A DrawChar must span at least one line.')
        self.line = int(line)

    def isEmpty(self):
        if False:
            return 10
        return self.text is None or not self.text.strip()

    def toElement(self, se):
        if False:
            while True:
                i = 10
        elem = Element('DrawChar', line=str(self.line))
        appendTextElements(elem, self.contents, se)
        return elem

    def toLrfContainer(self, lrfWriter, parent):
        if False:
            i = 10
            return i + 15
        parent.appendLrfTag(LrfTag('DrawChar', (int(self.line),)))
        for content in self.contents:
            content.toLrfContainer(lrfWriter, parent)
        parent.appendLrfTag(LrfTag('DrawCharEnd'))

class Button(LrsObject, LrsContainer):

    def __init__(self, **settings):
        if False:
            return 10
        LrsObject.__init__(self, **settings)
        LrsContainer.__init__(self, [PushButton])

    def findJumpToRefs(self):
        if False:
            for i in range(10):
                print('nop')
        for sub1 in self.contents:
            if isinstance(sub1, PushButton):
                for sub2 in sub1.contents:
                    if isinstance(sub2, JumpTo):
                        return (sub2.textBlock.objId, sub2.textBlock.parent.objId)
        raise LrsError('%s has no PushButton or JumpTo subs' % self.__class__.__name__)

    def toLrf(self, lrfWriter):
        if False:
            print('Hello World!')
        (refobj, refpage) = self.findJumpToRefs()
        button = LrfObject('Button', self.objId)
        button.appendLrfTag(LrfTag('buttonflags', 16))
        button.appendLrfTag(LrfTag('PushButtonStart'))
        button.appendLrfTag(LrfTag('buttonactions'))
        button.appendLrfTag(LrfTag('jumpto', (int(refpage), int(refobj))))
        button.append(LrfTag('endbuttonactions'))
        button.appendLrfTag(LrfTag('PushButtonEnd'))
        lrfWriter.append(button)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        b = self.lrsObjectElement('Button')
        for content in self.contents:
            b.append(content.toElement(se))
        return b

class ButtonBlock(Button):
    pass

class PushButton(LrsContainer):

    def __init__(self, **settings):
        if False:
            return 10
        LrsContainer.__init__(self, [JumpTo])

    def toElement(self, se):
        if False:
            while True:
                i = 10
        b = Element('PushButton')
        for content in self.contents:
            b.append(content.toElement(se))
        return b

class JumpTo(LrsContainer):

    def __init__(self, textBlock):
        if False:
            return 10
        LrsContainer.__init__(self, [])
        self.textBlock = textBlock

    def setTextBlock(self, textBlock):
        if False:
            while True:
                i = 10
        self.textBlock = textBlock

    def toElement(self, se):
        if False:
            i = 10
            return i + 15
        return Element('JumpTo', refpage=str(self.textBlock.parent.objId), refobj=str(self.textBlock.objId))

class Plot(LrsSimpleChar1, LrsContainer):
    ADJUSTMENT_VALUES = {'center': 1, 'baseline': 2, 'top': 3, 'bottom': 4}

    def __init__(self, obj, xsize=0, ysize=0, adjustment=None):
        if False:
            i = 10
            return i + 15
        LrsContainer.__init__(self, [])
        if obj is not None:
            self.setObj(obj)
        if xsize < 0 or ysize < 0:
            raise LrsError('Sizes must be positive semi-definite')
        self.xsize = int(xsize)
        self.ysize = int(ysize)
        if adjustment and adjustment not in Plot.ADJUSTMENT_VALUES.keys():
            raise LrsError('adjustment must be one of' + Plot.ADJUSTMENT_VALUES.keys())
        self.adjustment = adjustment

    def setObj(self, obj):
        if False:
            return 10
        if not isinstance(obj, (Image, Button)):
            raise LrsError('Plot elements can only refer to Image or Button elements')
        self.obj = obj

    def getReferencedObjIds(self):
        if False:
            for i in range(10):
                print('nop')
        return [self.obj.objId]

    def appendReferencedObjects(self, parent):
        if False:
            for i in range(10):
                print('nop')
        if self.obj.parent is None:
            parent.append(self.obj)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        elem = Element('Plot', xsize=str(self.xsize), ysize=str(self.ysize), refobj=str(self.obj.objId))
        if self.adjustment:
            elem.set('adjustment', self.adjustment)
        return elem

    def toLrfContainer(self, lrfWriter, parent):
        if False:
            while True:
                i = 10
        adj = self.adjustment if self.adjustment else 'bottom'
        params = (int(self.xsize), int(self.ysize), int(self.obj.objId), Plot.ADJUSTMENT_VALUES[adj])
        parent.appendLrfTag(LrfTag('Plot', params))

class Text(LrsContainer):
    """ A object that represents raw text.  Does not have a toElement. """

    def __init__(self, text):
        if False:
            for i in range(10):
                print('nop')
        LrsContainer.__init__(self, [])
        self.text = text

    def isEmpty(self):
        if False:
            i = 10
            return i + 15
        return not self.text or not self.text.strip()

    def toLrfContainer(self, lrfWriter, parent):
        if False:
            for i in range(10):
                print('nop')
        if self.text:
            if isinstance(self.text, bytes):
                parent.appendLrfTag(LrfTag('rawtext', self.text))
            else:
                parent.appendLrfTag(LrfTag('textstring', self.text))

class CR(LrsSimpleChar1, LrsContainer):
    """
        A line break (when appended to a Paragraph) or a paragraph break
        (when appended to a TextBlock).
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        LrsContainer.__init__(self, [])

    def toElement(self, se):
        if False:
            while True:
                i = 10
        return Element('CR')

    def toLrfContainer(self, lrfWriter, parent):
        if False:
            for i in range(10):
                print('nop')
        parent.appendLrfTag(LrfTag('CR'))

class Italic(LrsSimpleChar1, LrsTextTag):

    def __init__(self, text=None):
        if False:
            return 10
        LrsTextTag.__init__(self, text, [LrsSimpleChar1])

class Sub(LrsSimpleChar1, LrsTextTag):

    def __init__(self, text=None):
        if False:
            print('Hello World!')
        LrsTextTag.__init__(self, text, [])

class Sup(LrsSimpleChar1, LrsTextTag):

    def __init__(self, text=None):
        if False:
            i = 10
            return i + 15
        LrsTextTag.__init__(self, text, [])

class NoBR(LrsSimpleChar1, LrsTextTag):

    def __init__(self, text=None):
        if False:
            i = 10
            return i + 15
        LrsTextTag.__init__(self, text, [LrsSimpleChar1])

class Space(LrsSimpleChar1, LrsContainer):

    def __init__(self, xsize=0, x=0):
        if False:
            i = 10
            return i + 15
        LrsContainer.__init__(self, [])
        if xsize == 0 and x != 0:
            xsize = x
        self.xsize = xsize

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        if self.xsize == 0:
            return
        return Element('Space', xsize=str(self.xsize))

    def toLrfContainer(self, lrfWriter, container):
        if False:
            return 10
        if self.xsize != 0:
            container.appendLrfTag(LrfTag('Space', self.xsize))

class Box(LrsSimpleChar1, LrsContainer):
    """
        Draw a box around text.  Unfortunately, does not seem to do
        anything on the PRS-500.
    """

    def __init__(self, linetype='solid'):
        if False:
            return 10
        LrsContainer.__init__(self, [Text, bytes, str])
        if linetype not in LINE_TYPE_ENCODING:
            raise LrsError(linetype + ' is not a valid line type')
        self.linetype = linetype

    def toElement(self, se):
        if False:
            print('Hello World!')
        e = Element('Box', linetype=self.linetype)
        appendTextElements(e, self.contents, se)
        return e

    def toLrfContainer(self, lrfWriter, container):
        if False:
            print('Hello World!')
        container.appendLrfTag(LrfTag('Box', self.linetype))
        for content in self.contents:
            content.toLrfContainer(lrfWriter, container)
        container.appendLrfTag(LrfTag('BoxEnd'))

class Span(LrsSimpleChar1, LrsContainer):

    def __init__(self, text=None, **attrs):
        if False:
            while True:
                i = 10
        LrsContainer.__init__(self, [LrsSimpleChar1, Text, bytes, str])
        if text is not None:
            if isinstance(text, string_or_bytes):
                text = Text(text)
            self.append(text)
        for attrname in attrs.keys():
            if attrname not in TextStyle.defaults and attrname not in TextStyle.alsoAllow:
                raise LrsError('setting %s not allowed on Span' % attrname)
        self.attrs = attrs

    def findCurrentTextStyle(self):
        if False:
            while True:
                i = 10
        parent = self.parent
        while 1:
            if parent is None or hasattr(parent, 'currentTextStyle'):
                break
            parent = parent.parent
        if parent is None:
            raise LrsError('no enclosing current TextStyle found')
        return parent.currentTextStyle

    def toLrfContainer(self, lrfWriter, container):
        if False:
            print('Hello World!')
        oldTextStyle = self.findCurrentTextStyle()
        for (name, value) in tuple(iteritems(self.attrs)):
            if name in oldTextStyle.attrs and oldTextStyle.attrs[name] == self.attrs[name]:
                self.attrs.pop(name)
            else:
                container.appendLrfTag(LrfTag(name, value))
        oldTextStyle = self.findCurrentTextStyle()
        self.currentTextStyle = oldTextStyle.copy()
        self.currentTextStyle.attrs.update(self.attrs)
        for content in self.contents:
            content.toLrfContainer(lrfWriter, container)
        for name in self.attrs.keys():
            container.appendLrfTag(LrfTag(name, oldTextStyle.attrs[name]))

    def toElement(self, se):
        if False:
            print('Hello World!')
        element = Element('Span')
        for (key, value) in self.attrs.items():
            element.set(key, str(value))
        appendTextElements(element, self.contents, se)
        return element

class EmpLine(LrsTextTag, LrsSimpleChar1):
    emplinetypes = ['none', 'solid', 'dotted', 'dashed', 'double']
    emplinepositions = ['before', 'after']

    def __init__(self, text=None, emplineposition='before', emplinetype='solid'):
        if False:
            print('Hello World!')
        LrsTextTag.__init__(self, text, [LrsSimpleChar1])
        if emplineposition not in self.__class__.emplinepositions:
            raise LrsError('emplineposition for an EmpLine must be one of: ' + str(self.__class__.emplinepositions))
        if emplinetype not in self.__class__.emplinetypes:
            raise LrsError('emplinetype for an EmpLine must be one of: ' + str(self.__class__.emplinetypes))
        self.emplinetype = emplinetype
        self.emplineposition = emplineposition

    def toLrfContainer(self, lrfWriter, parent):
        if False:
            i = 10
            return i + 15
        parent.appendLrfTag(LrfTag(self.__class__.__name__, (self.emplineposition, self.emplinetype)))
        parent.appendLrfTag(LrfTag('emplineposition', self.emplineposition))
        parent.appendLrfTag(LrfTag('emplinetype', self.emplinetype))
        for content in self.contents:
            content.toLrfContainer(lrfWriter, parent)
        parent.appendLrfTag(LrfTag(self.__class__.__name__ + 'End'))

    def toElement(self, se):
        if False:
            return 10
        element = Element(self.__class__.__name__)
        element.set('emplineposition', self.emplineposition)
        element.set('emplinetype', self.emplinetype)
        appendTextElements(element, self.contents, se)
        return element

class Bold(Span):
    """
        There is no known "bold" lrf tag. Use Span with a fontweight in LRF,
        but use the word Bold in the LRS.
    """

    def __init__(self, text=None):
        if False:
            i = 10
            return i + 15
        Span.__init__(self, text, fontweight=800)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        e = Element('Bold')
        appendTextElements(e, self.contents, se)
        return e

class BlockSpace(LrsContainer):
    """ Can be appended to a page to move the text point. """

    def __init__(self, xspace=0, yspace=0, x=0, y=0):
        if False:
            return 10
        LrsContainer.__init__(self, [])
        if xspace == 0 and x != 0:
            xspace = x
        if yspace == 0 and y != 0:
            yspace = y
        self.xspace = xspace
        self.yspace = yspace

    def toLrfContainer(self, lrfWriter, container):
        if False:
            return 10
        if self.xspace != 0:
            container.appendLrfTag(LrfTag('xspace', self.xspace))
        if self.yspace != 0:
            container.appendLrfTag(LrfTag('yspace', self.yspace))

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        element = Element('BlockSpace')
        if self.xspace != 0:
            element.attrib['xspace'] = str(self.xspace)
        if self.yspace != 0:
            element.attrib['yspace'] = str(self.yspace)
        return element

class CharButton(LrsSimpleChar1, LrsContainer):
    """
        Define the text and target of a CharButton.  Must be passed a
        JumpButton that is the destination of the CharButton.

        Only text or SimpleChars can be appended to the CharButton.
    """

    def __init__(self, button, text=None):
        if False:
            return 10
        LrsContainer.__init__(self, [bytes, str, Text, LrsSimpleChar1])
        self.button = None
        if button is not None:
            self.setButton(button)
        if text is not None:
            self.append(text)

    def setButton(self, button):
        if False:
            i = 10
            return i + 15
        if not isinstance(button, (JumpButton, Button)):
            raise LrsError('CharButton button must be a JumpButton or Button')
        self.button = button

    def appendReferencedObjects(self, parent):
        if False:
            print('Hello World!')
        if self.button.parent is None:
            parent.append(self.button)

    def getReferencedObjIds(self):
        if False:
            i = 10
            return i + 15
        return [self.button.objId]

    def toLrfContainer(self, lrfWriter, container):
        if False:
            i = 10
            return i + 15
        container.appendLrfTag(LrfTag('CharButton', self.button.objId))
        for content in self.contents:
            content.toLrfContainer(lrfWriter, container)
        container.appendLrfTag(LrfTag('CharButtonEnd'))

    def toElement(self, se):
        if False:
            return 10
        cb = Element('CharButton', refobj=str(self.button.objId))
        appendTextElements(cb, self.contents, se)
        return cb

class Objects(LrsContainer):

    def __init__(self):
        if False:
            return 10
        LrsContainer.__init__(self, [JumpButton, TextBlock, HeaderOrFooter, ImageStream, Image, ImageBlock, Button, ButtonBlock])
        self.appendJumpButton = self.appendTextBlock = self.appendHeader = self.appendFooter = self.appendImageStream = self.appendImage = self.appendImageBlock = self.append

    def getMethods(self):
        if False:
            while True:
                i = 10
        return ['JumpButton', 'appendJumpButton', 'TextBlock', 'appendTextBlock', 'Header', 'appendHeader', 'Footer', 'appendFooter', 'ImageBlock', 'ImageStream', 'appendImageStream', 'Image', 'appendImage', 'appendImageBlock']

    def getSettings(self):
        if False:
            while True:
                i = 10
        return []

    def ImageBlock(self, *args, **kwargs):
        if False:
            return 10
        ib = ImageBlock(*args, **kwargs)
        self.append(ib)
        return ib

    def JumpButton(self, textBlock):
        if False:
            i = 10
            return i + 15
        b = JumpButton(textBlock)
        self.append(b)
        return b

    def TextBlock(self, *args, **kwargs):
        if False:
            print('Hello World!')
        tb = TextBlock(*args, **kwargs)
        self.append(tb)
        return tb

    def Header(self, *args, **kwargs):
        if False:
            return 10
        h = Header(*args, **kwargs)
        self.append(h)
        return h

    def Footer(self, *args, **kwargs):
        if False:
            return 10
        h = Footer(*args, **kwargs)
        self.append(h)
        return h

    def ImageStream(self, *args, **kwargs):
        if False:
            return 10
        i = ImageStream(*args, **kwargs)
        self.append(i)
        return i

    def Image(self, *args, **kwargs):
        if False:
            return 10
        i = Image(*args, **kwargs)
        self.append(i)
        return i

    def toElement(self, se):
        if False:
            i = 10
            return i + 15
        o = Element('Objects')
        for content in self.contents:
            o.append(content.toElement(se))
        return o

    def toLrf(self, lrfWriter):
        if False:
            for i in range(10):
                print('nop')
        for content in self.contents:
            content.toLrf(lrfWriter)

class JumpButton(LrsObject, LrsContainer):
    """
        The target of a CharButton.  Needs a parented TextBlock to jump to.
        Actually creates several elements in the XML.  JumpButtons must
        be eventually appended to a Book (actually, an Object.)
    """

    def __init__(self, textBlock):
        if False:
            while True:
                i = 10
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [])
        self.textBlock = textBlock

    def setTextBlock(self, textBlock):
        if False:
            print('Hello World!')
        self.textBlock = textBlock

    def toLrf(self, lrfWriter):
        if False:
            i = 10
            return i + 15
        button = LrfObject('Button', self.objId)
        button.appendLrfTag(LrfTag('buttonflags', 16))
        button.appendLrfTag(LrfTag('PushButtonStart'))
        button.appendLrfTag(LrfTag('buttonactions'))
        button.appendLrfTag(LrfTag('jumpto', (self.textBlock.parent.objId, self.textBlock.objId)))
        button.append(LrfTag('endbuttonactions'))
        button.appendLrfTag(LrfTag('PushButtonEnd'))
        lrfWriter.append(button)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        b = self.lrsObjectElement('Button')
        pb = SubElement(b, 'PushButton')
        SubElement(pb, 'JumpTo', refpage=str(self.textBlock.parent.objId), refobj=str(self.textBlock.objId))
        return b

class RuledLine(LrsContainer, LrsAttributes, LrsObject):
    """ A line.  Default is 500 pixels long, 2 pixels wide. """
    defaults = dict(linelength='500', linetype='solid', linewidth='2', linecolor='0x00000000')

    def __init__(self, **settings):
        if False:
            i = 10
            return i + 15
        LrsContainer.__init__(self, [])
        LrsAttributes.__init__(self, self.defaults, **settings)
        LrsObject.__init__(self)

    def toLrfContainer(self, lrfWriter, container):
        if False:
            i = 10
            return i + 15
        a = self.attrs
        container.appendLrfTag(LrfTag('RuledLine', (a['linelength'], a['linetype'], a['linewidth'], a['linecolor'])))

    def toElement(self, se):
        if False:
            while True:
                i = 10
        return Element('RuledLine', self.attrs)

class HeaderOrFooter(LrsObject, LrsContainer, LrsAttributes):
    """
        Creates empty header or footer objects.  Append PutObj objects to
        the header or footer to create the text.

        Note: it seems that adding multiple PutObjs to a header or footer
              only shows the last one.
    """
    defaults = dict(framemode='square', layout='LrTb', framewidth='0', framecolor='0x00000000', bgcolor='0xFF000000')

    def __init__(self, **settings):
        if False:
            i = 10
            return i + 15
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [PutObj])
        LrsAttributes.__init__(self, self.defaults, **settings)

    def put_object(self, obj, x1, y1):
        if False:
            while True:
                i = 10
        self.append(PutObj(obj, x1, y1))

    def PutObj(self, *args, **kwargs):
        if False:
            return 10
        p = PutObj(*args, **kwargs)
        self.append(p)
        return p

    def toLrf(self, lrfWriter):
        if False:
            while True:
                i = 10
        hd = LrfObject(self.__class__.__name__, self.objId)
        hd.appendTagDict(self.attrs)
        stream = LrfTagStream(0)
        for content in self.contents:
            content.toLrfContainer(lrfWriter, stream)
        hd.appendLrfTags(stream.getStreamTags(lrfWriter.getSourceEncoding()))
        lrfWriter.append(hd)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        name = self.__class__.__name__
        labelName = name.lower() + 'label'
        hd = self.lrsObjectElement(name, objlabel=labelName)
        hd.attrib.update(self.attrs)
        for content in self.contents:
            hd.append(content.toElement(se))
        return hd

class Header(HeaderOrFooter):
    pass

class Footer(HeaderOrFooter):
    pass

class Canvas(LrsObject, LrsContainer, LrsAttributes):
    defaults = dict(framemode='square', layout='LrTb', framewidth='0', framecolor='0x00000000', bgcolor='0xFF000000', canvasheight=0, canvaswidth=0, blockrule='block-adjustable')

    def __init__(self, width, height, **settings):
        if False:
            i = 10
            return i + 15
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [PutObj])
        LrsAttributes.__init__(self, self.defaults, **settings)
        self.settings = self.defaults.copy()
        self.settings.update(settings)
        self.settings['canvasheight'] = int(height)
        self.settings['canvaswidth'] = int(width)

    def put_object(self, obj, x1, y1):
        if False:
            i = 10
            return i + 15
        self.append(PutObj(obj, x1, y1))

    def toElement(self, source_encoding):
        if False:
            print('Hello World!')
        el = self.lrsObjectElement('Canvas', **self.settings)
        for po in self.contents:
            el.append(po.toElement(source_encoding))
        return el

    def toLrf(self, lrfWriter):
        if False:
            for i in range(10):
                print('nop')
        self.toLrfContainer(lrfWriter, lrfWriter)

    def toLrfContainer(self, lrfWriter, container):
        if False:
            return 10
        c = LrfObject('Canvas', self.objId)
        c.appendTagDict(self.settings)
        stream = LrfTagStream(STREAM_COMPRESSED)
        for content in self.contents:
            content.toLrfContainer(lrfWriter, stream)
        if lrfWriter.saveStreamTags:
            c.saveStreamTags = stream.tags
        c.appendLrfTags(stream.getStreamTags(lrfWriter.getSourceEncoding(), optimizeTags=lrfWriter.optimizeTags, optimizeCompression=lrfWriter.optimizeCompression))
        container.addLrfObject(c.objId)
        lrfWriter.append(c)

    def has_text(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.contents)

class PutObj(LrsContainer):
    """ PutObj holds other objects that are drawn on a Canvas or Header. """

    def __init__(self, content, x1=0, y1=0):
        if False:
            for i in range(10):
                print('nop')
        LrsContainer.__init__(self, [TextBlock, ImageBlock])
        self.content = content
        self.x1 = int(x1)
        self.y1 = int(y1)

    def setContent(self, content):
        if False:
            while True:
                i = 10
        self.content = content

    def appendReferencedObjects(self, parent):
        if False:
            for i in range(10):
                print('nop')
        if self.content.parent is None:
            parent.append(self.content)

    def toLrfContainer(self, lrfWriter, container):
        if False:
            for i in range(10):
                print('nop')
        container.appendLrfTag(LrfTag('PutObj', (self.x1, self.y1, self.content.objId)))

    def toElement(self, se):
        if False:
            print('Hello World!')
        el = Element('PutObj', x1=str(self.x1), y1=str(self.y1), refobj=str(self.content.objId))
        return el

class ImageStream(LrsObject, LrsContainer):
    """
        Embed an image file into an Lrf.
    """
    VALID_ENCODINGS = ['JPEG', 'GIF', 'BMP', 'PNG']

    def __init__(self, file=None, encoding=None, comment=None):
        if False:
            for i in range(10):
                print('nop')
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [])
        _checkExists(file)
        self.filename = file
        self.comment = comment
        if encoding is None:
            extension = os.path.splitext(file)[1]
            if not extension:
                raise LrsError('file must have extension if encoding is not specified')
            extension = extension[1:].upper()
            if extension == 'JPG':
                extension = 'JPEG'
            encoding = extension
        else:
            encoding = encoding.upper()
        if encoding not in self.VALID_ENCODINGS:
            raise LrsError('encoding or file extension not JPEG, GIF, BMP, or PNG')
        self.encoding = encoding

    def toLrf(self, lrfWriter):
        if False:
            print('Hello World!')
        with open(self.filename, 'rb') as f:
            imageData = f.read()
        isObj = LrfObject('ImageStream', self.objId)
        if self.comment is not None:
            isObj.appendLrfTag(LrfTag('comment', self.comment))
        streamFlags = IMAGE_TYPE_ENCODING[self.encoding]
        stream = LrfStreamBase(streamFlags, imageData)
        isObj.appendLrfTags(stream.getStreamTags())
        lrfWriter.append(isObj)

    def toElement(self, se):
        if False:
            return 10
        element = self.lrsObjectElement('ImageStream', objlabel='imagestreamlabel', encoding=self.encoding, file=self.filename)
        element.text = self.comment
        return element

class Image(LrsObject, LrsContainer, LrsAttributes):
    defaults = dict()

    def __init__(self, refstream, x0=0, x1=0, y0=0, y1=0, xsize=0, ysize=0, **settings):
        if False:
            return 10
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [])
        LrsAttributes.__init__(self, self.defaults, settings)
        (self.x0, self.y0, self.x1, self.y1) = (int(x0), int(y0), int(x1), int(y1))
        (self.xsize, self.ysize) = (int(xsize), int(ysize))
        self.setRefstream(refstream)

    def setRefstream(self, refstream):
        if False:
            return 10
        self.refstream = refstream

    def appendReferencedObjects(self, parent):
        if False:
            return 10
        if self.refstream.parent is None:
            parent.append(self.refstream)

    def getReferencedObjIds(self):
        if False:
            while True:
                i = 10
        return [self.objId, self.refstream.objId]

    def toElement(self, se):
        if False:
            while True:
                i = 10
        element = self.lrsObjectElement('Image', **self.attrs)
        element.set('refstream', str(self.refstream.objId))
        for name in ['x0', 'y0', 'x1', 'y1', 'xsize', 'ysize']:
            element.set(name, str(getattr(self, name)))
        return element

    def toLrf(self, lrfWriter):
        if False:
            print('Hello World!')
        ib = LrfObject('Image', self.objId)
        ib.appendLrfTag(LrfTag('ImageRect', (self.x0, self.y0, self.x1, self.y1)))
        ib.appendLrfTag(LrfTag('ImageSize', (self.xsize, self.ysize)))
        ib.appendLrfTag(LrfTag('RefObjId', self.refstream.objId))
        lrfWriter.append(ib)

class ImageBlock(LrsObject, LrsContainer, LrsAttributes):
    """ Create an image on a page. """
    defaults = BlockStyle.baseDefaults.copy()

    def __init__(self, refstream, x0='0', y0='0', x1='600', y1='800', xsize='600', ysize='800', blockStyle=BlockStyle(blockrule='block-fixed'), alttext=None, **settings):
        if False:
            return 10
        LrsObject.__init__(self)
        LrsContainer.__init__(self, [Text, Image])
        LrsAttributes.__init__(self, self.defaults, **settings)
        (self.x0, self.y0, self.x1, self.y1) = (int(x0), int(y0), int(x1), int(y1))
        (self.xsize, self.ysize) = (int(xsize), int(ysize))
        self.setRefstream(refstream)
        self.blockStyle = blockStyle
        self.alttext = alttext

    def setRefstream(self, refstream):
        if False:
            print('Hello World!')
        self.refstream = refstream

    def appendReferencedObjects(self, parent):
        if False:
            print('Hello World!')
        if self.refstream.parent is None:
            parent.append(self.refstream)
        if self.blockStyle is not None and self.blockStyle.parent is None:
            parent.append(self.blockStyle)

    def getReferencedObjIds(self):
        if False:
            i = 10
            return i + 15
        objects = [self.objId, self.extraId, self.refstream.objId]
        if self.blockStyle is not None:
            objects.append(self.blockStyle.objId)
        return objects

    def toLrf(self, lrfWriter):
        if False:
            for i in range(10):
                print('nop')
        self.toLrfContainer(lrfWriter, lrfWriter)

    def toLrfContainer(self, lrfWriter, container):
        if False:
            i = 10
            return i + 15
        extraId = LrsObject.getNextObjId()
        b = LrfObject('Block', self.objId)
        if self.blockStyle is not None:
            b.appendLrfTag(LrfTag('Link', self.blockStyle.objId))
        b.appendTagDict(self.attrs)
        b.appendLrfTags(LrfTagStream(0, [LrfTag('Link', extraId)]).getStreamTags(lrfWriter.getSourceEncoding()))
        container.addLrfObject(b.objId)
        lrfWriter.append(b)
        ib = LrfObject('Image', extraId)
        ib.appendLrfTag(LrfTag('ImageRect', (self.x0, self.y0, self.x1, self.y1)))
        ib.appendLrfTag(LrfTag('ImageSize', (self.xsize, self.ysize)))
        ib.appendLrfTag(LrfTag('RefObjId', self.refstream.objId))
        if self.alttext:
            ib.appendLrfTag('Comment', self.alttext)
        lrfWriter.append(ib)
        self.extraId = extraId

    def toElement(self, se):
        if False:
            while True:
                i = 10
        element = self.lrsObjectElement('ImageBlock', **self.attrs)
        element.set('refstream', str(self.refstream.objId))
        for name in ['x0', 'y0', 'x1', 'y1', 'xsize', 'ysize']:
            element.set(name, str(getattr(self, name)))
        element.text = self.alttext
        return element

class Font(LrsContainer):
    """ Allows a TrueType file to be embedded in an Lrf. """

    def __init__(self, file=None, fontname=None, fontfilename=None, encoding=None):
        if False:
            return 10
        LrsContainer.__init__(self, [])
        try:
            _checkExists(fontfilename)
            self.truefile = fontfilename
        except:
            try:
                _checkExists(file)
                self.truefile = file
            except:
                raise LrsError("neither '%s' nor '%s' exists" % (fontfilename, file))
        self.file = file
        self.fontname = fontname
        self.fontfilename = fontfilename
        self.encoding = encoding

    def toLrf(self, lrfWriter):
        if False:
            while True:
                i = 10
        font = LrfObject('Font', LrsObject.getNextObjId())
        lrfWriter.registerFontId(font.objId)
        font.appendLrfTag(LrfTag('FontFilename', lrfWriter.toUnicode(self.truefile)))
        font.appendLrfTag(LrfTag('FontFacename', lrfWriter.toUnicode(self.fontname)))
        stream = LrfFileStream(STREAM_FORCE_COMPRESSED, self.truefile)
        font.appendLrfTags(stream.getStreamTags())
        lrfWriter.append(font)

    def toElement(self, se):
        if False:
            for i in range(10):
                print('nop')
        element = Element('RegistFont', encoding='TTF', fontname=self.fontname, file=self.file, fontfilename=self.file)
        return element