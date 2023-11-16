__doc__ = 'Use OpenDocument to generate your documents.'
import mimetypes
import sys
import time
import zipfile
from io import BytesIO
from xml.sax.xmlreader import InputSource
from polyglot.io import PolyglotBytesIO, PolyglotStringIO
from polyglot.builtins import unicode_type
from . import element, manifest, meta
from .attrconverters import make_NCName
from .namespaces import CHARTNS, DRAWNS, METANS, OFFICENS, PRESENTATIONNS, STYLENS, TABLENS, TEXTNS, TOOLSVERSION
from .odfmanifest import manifestlist
from .office import AutomaticStyles, Body, Chart, Document, DocumentContent, DocumentMeta, DocumentSettings, DocumentStyles, Drawing, FontFaceDecls, Image, MasterStyles, Meta, Presentation, Scripts, Settings, Spreadsheet, Styles, Text
__version__ = TOOLSVERSION
_XMLPROLOGUE = "<?xml version='1.0' encoding='UTF-8'?>\n"
UNIXPERMS = 33188 << 16
IS_FILENAME = 0
IS_IMAGE = 1
assert sys.version_info[0] >= 2 and sys.version_info[1] >= 2
odmimetypes = {'application/vnd.oasis.opendocument.text': '.odt', 'application/vnd.oasis.opendocument.text-template': '.ott', 'application/vnd.oasis.opendocument.graphics': '.odg', 'application/vnd.oasis.opendocument.graphics-template': '.otg', 'application/vnd.oasis.opendocument.presentation': '.odp', 'application/vnd.oasis.opendocument.presentation-template': '.otp', 'application/vnd.oasis.opendocument.spreadsheet': '.ods', 'application/vnd.oasis.opendocument.spreadsheet-template': '.ots', 'application/vnd.oasis.opendocument.chart': '.odc', 'application/vnd.oasis.opendocument.chart-template': '.otc', 'application/vnd.oasis.opendocument.image': '.odi', 'application/vnd.oasis.opendocument.image-template': '.oti', 'application/vnd.oasis.opendocument.formula': '.odf', 'application/vnd.oasis.opendocument.formula-template': '.otf', 'application/vnd.oasis.opendocument.text-master': '.odm', 'application/vnd.oasis.opendocument.text-web': '.oth'}

class OpaqueObject:

    def __init__(self, filename, mediatype, content=None):
        if False:
            while True:
                i = 10
        self.mediatype = mediatype
        self.filename = filename
        self.content = content

class OpenDocument:
    """ A class to hold the content of an OpenDocument document
        Use the xml method to write the XML
        source to the screen or to a file
        d = OpenDocument(mimetype)
        fd.write(d.xml())
    """
    thumbnail = None

    def __init__(self, mimetype, add_generator=True):
        if False:
            while True:
                i = 10
        self.mimetype = mimetype
        self.childobjects = []
        self._extra = []
        self.folder = ''
        self.topnode = Document(mimetype=self.mimetype)
        self.topnode.ownerDocument = self
        self.clear_caches()
        self.Pictures = {}
        self.meta = Meta()
        self.topnode.addElement(self.meta)
        if add_generator:
            self.meta.addElement(meta.Generator(text=TOOLSVERSION))
        self.scripts = Scripts()
        self.topnode.addElement(self.scripts)
        self.fontfacedecls = FontFaceDecls()
        self.topnode.addElement(self.fontfacedecls)
        self.settings = Settings()
        self.topnode.addElement(self.settings)
        self.styles = Styles()
        self.topnode.addElement(self.styles)
        self.automaticstyles = AutomaticStyles()
        self.topnode.addElement(self.automaticstyles)
        self.masterstyles = MasterStyles()
        self.topnode.addElement(self.masterstyles)
        self.body = Body()
        self.topnode.addElement(self.body)

    def rebuild_caches(self, node=None):
        if False:
            while True:
                i = 10
        if node is None:
            node = self.topnode
        self.build_caches(node)
        for e in node.childNodes:
            if e.nodeType == element.Node.ELEMENT_NODE:
                self.rebuild_caches(e)

    def clear_caches(self):
        if False:
            print('Hello World!')
        self.element_dict = {}
        self._styles_dict = {}
        self._styles_ooo_fix = {}

    def build_caches(self, element):
        if False:
            while True:
                i = 10
        ' Called from element.py\n        '
        if element.qname not in self.element_dict:
            self.element_dict[element.qname] = []
        self.element_dict[element.qname].append(element)
        if element.qname == (STYLENS, 'style'):
            self.__register_stylename(element)
        styleref = element.getAttrNS(TEXTNS, 'style-name')
        if styleref is not None and styleref in self._styles_ooo_fix:
            element.setAttrNS(TEXTNS, 'style-name', self._styles_ooo_fix[styleref])

    def __register_stylename(self, element):
        if False:
            i = 10
            return i + 15
        ' Register a style. But there are three style dictionaries:\n            office:styles, office:automatic-styles and office:master-styles\n            Chapter 14\n        '
        name = element.getAttrNS(STYLENS, 'name')
        if name is None:
            return
        if element.parentNode.qname in ((OFFICENS, 'styles'), (OFFICENS, 'automatic-styles')):
            if name in self._styles_dict:
                newname = 'M' + name
                self._styles_ooo_fix[name] = newname
                name = newname
                element.setAttrNS(STYLENS, 'name', name)
            self._styles_dict[name] = element

    def toXml(self, filename=''):
        if False:
            for i in range(10):
                print('nop')
        xml = PolyglotBytesIO()
        xml.write(_XMLPROLOGUE)
        self.body.toXml(0, xml)
        if not filename:
            return xml.getvalue()
        else:
            f = open(filename, 'wb')
            f.write(xml.getvalue())
            f.close()

    def xml(self):
        if False:
            print('Hello World!')
        ' Generates the full document as an XML file\n            Always written as a bytestream in UTF-8 encoding\n        '
        self.__replaceGenerator()
        xml = PolyglotBytesIO()
        xml.write(_XMLPROLOGUE)
        self.topnode.toXml(0, xml)
        return xml.getvalue()

    def contentxml(self):
        if False:
            i = 10
            return i + 15
        ' Generates the content.xml file\n            Always written as a bytestream in UTF-8 encoding\n        '
        xml = PolyglotBytesIO()
        xml.write(_XMLPROLOGUE)
        x = DocumentContent()
        x.write_open_tag(0, xml)
        if self.scripts.hasChildNodes():
            self.scripts.toXml(1, xml)
        if self.fontfacedecls.hasChildNodes():
            self.fontfacedecls.toXml(1, xml)
        a = AutomaticStyles()
        stylelist = self._used_auto_styles([self.styles, self.automaticstyles, self.body])
        if len(stylelist) > 0:
            a.write_open_tag(1, xml)
            for s in stylelist:
                s.toXml(2, xml)
            a.write_close_tag(1, xml)
        else:
            a.toXml(1, xml)
        self.body.toXml(1, xml)
        x.write_close_tag(0, xml)
        return xml.getvalue()

    def __manifestxml(self):
        if False:
            return 10
        " Generates the manifest.xml file\n            The self.manifest isn't available unless the document is being saved\n        "
        xml = PolyglotBytesIO()
        xml.write(_XMLPROLOGUE)
        self.manifest.toXml(0, xml)
        return xml.getvalue()

    def metaxml(self):
        if False:
            return 10
        ' Generates the meta.xml file '
        self.__replaceGenerator()
        x = DocumentMeta()
        x.addElement(self.meta)
        xml = PolyglotStringIO()
        xml.write(_XMLPROLOGUE)
        x.toXml(0, xml)
        return xml.getvalue()

    def settingsxml(self):
        if False:
            for i in range(10):
                print('nop')
        ' Generates the settings.xml file '
        x = DocumentSettings()
        x.addElement(self.settings)
        xml = PolyglotStringIO()
        xml.write(_XMLPROLOGUE)
        x.toXml(0, xml)
        return xml.getvalue()

    def _parseoneelement(self, top, stylenamelist):
        if False:
            i = 10
            return i + 15
        ' Finds references to style objects in master-styles\n            and add the style name to the style list if not already there.\n            Recursive\n        '
        for e in top.childNodes:
            if e.nodeType == element.Node.ELEMENT_NODE:
                for styleref in ((CHARTNS, 'style-name'), (DRAWNS, 'style-name'), (DRAWNS, 'text-style-name'), (PRESENTATIONNS, 'style-name'), (STYLENS, 'data-style-name'), (STYLENS, 'list-style-name'), (STYLENS, 'page-layout-name'), (STYLENS, 'style-name'), (TABLENS, 'default-cell-style-name'), (TABLENS, 'style-name'), (TEXTNS, 'style-name')):
                    if e.getAttrNS(styleref[0], styleref[1]):
                        stylename = e.getAttrNS(styleref[0], styleref[1])
                        if stylename not in stylenamelist:
                            stylenamelist.append(stylename)
                stylenamelist = self._parseoneelement(e, stylenamelist)
        return stylenamelist

    def _used_auto_styles(self, segments):
        if False:
            print('Hello World!')
        ' Loop through the masterstyles elements, and find the automatic\n            styles that are used. These will be added to the automatic-styles\n            element in styles.xml\n        '
        stylenamelist = []
        for top in segments:
            stylenamelist = self._parseoneelement(top, stylenamelist)
        stylelist = []
        for e in self.automaticstyles.childNodes:
            if e.getAttrNS(STYLENS, 'name') in stylenamelist:
                stylelist.append(e)
        return stylelist

    def stylesxml(self):
        if False:
            return 10
        ' Generates the styles.xml file '
        xml = PolyglotStringIO()
        xml.write(_XMLPROLOGUE)
        x = DocumentStyles()
        x.write_open_tag(0, xml)
        if self.fontfacedecls.hasChildNodes():
            self.fontfacedecls.toXml(1, xml)
        self.styles.toXml(1, xml)
        a = AutomaticStyles()
        a.write_open_tag(1, xml)
        for s in self._used_auto_styles([self.masterstyles]):
            s.toXml(2, xml)
        a.write_close_tag(1, xml)
        if self.masterstyles.hasChildNodes():
            self.masterstyles.toXml(1, xml)
        x.write_close_tag(0, xml)
        return xml.getvalue()

    def addPicture(self, filename, mediatype=None, content=None):
        if False:
            while True:
                i = 10
        " Add a picture\n            It uses the same convention as OOo, in that it saves the picture in\n            the zipfile in the subdirectory 'Pictures'\n            If passed a file ptr, mediatype must be set\n        "
        if content is None:
            if mediatype is None:
                (mediatype, encoding) = mimetypes.guess_type(filename)
            if mediatype is None:
                mediatype = ''
                try:
                    ext = filename[filename.rindex('.'):]
                except:
                    ext = ''
            else:
                ext = mimetypes.guess_extension(mediatype)
            manifestfn = f'Pictures/{time.time() * 10000000000:0.0f}{ext}'
            self.Pictures[manifestfn] = (IS_FILENAME, filename, mediatype)
        else:
            manifestfn = filename
            self.Pictures[manifestfn] = (IS_IMAGE, content, mediatype)
        return manifestfn

    def addPictureFromFile(self, filename, mediatype=None):
        if False:
            i = 10
            return i + 15
        " Add a picture\n            It uses the same convention as OOo, in that it saves the picture in\n            the zipfile in the subdirectory 'Pictures'.\n            If mediatype is not given, it will be guessed from the filename\n            extension.\n        "
        if mediatype is None:
            (mediatype, encoding) = mimetypes.guess_type(filename)
        if mediatype is None:
            mediatype = ''
            try:
                ext = filename[filename.rindex('.'):]
            except ValueError:
                ext = ''
        else:
            ext = mimetypes.guess_extension(mediatype)
        manifestfn = f'Pictures/{time.time() * 10000000000:0.0f}{ext}'
        self.Pictures[manifestfn] = (IS_FILENAME, filename, mediatype)
        return manifestfn

    def addPictureFromString(self, content, mediatype):
        if False:
            for i in range(10):
                print('nop')
        " Add a picture\n            It uses the same convention as OOo, in that it saves the picture in\n            the zipfile in the subdirectory 'Pictures'. The content variable\n            is a string that contains the binary image data. The mediatype\n            indicates the image format.\n        "
        ext = mimetypes.guess_extension(mediatype)
        manifestfn = f'Pictures/{time.time() * 10000000000:0.0f}{ext}'
        self.Pictures[manifestfn] = (IS_IMAGE, content, mediatype)
        return manifestfn

    def addThumbnail(self, filecontent=None):
        if False:
            return 10
        ' Add a fixed thumbnail\n            The thumbnail in the library is big, so this is pretty useless.\n        '
        if filecontent is None:
            import thumbnail
            self.thumbnail = thumbnail.thumbnail()
        else:
            self.thumbnail = filecontent

    def addObject(self, document, objectname=None):
        if False:
            print('Hello World!')
        ' Adds an object (subdocument). The object must be an OpenDocument class\n            The return value will be the folder in the zipfile the object is stored in\n        '
        self.childobjects.append(document)
        if objectname is None:
            document.folder = '%s/Object %d' % (self.folder, len(self.childobjects))
        else:
            document.folder = objectname
        return '.%s' % document.folder

    def _savePictures(self, object, folder):
        if False:
            for i in range(10):
                print('nop')
        for (arcname, picturerec) in object.Pictures.items():
            (what_it_is, fileobj, mediatype) = picturerec
            self.manifest.addElement(manifest.FileEntry(fullpath=f'{folder}{arcname}', mediatype=mediatype))
            if what_it_is == IS_FILENAME:
                self._z.write(fileobj, arcname, zipfile.ZIP_STORED)
            else:
                zi = zipfile.ZipInfo(unicode_type(arcname), self._now)
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = UNIXPERMS
                self._z.writestr(zi, fileobj)
        subobjectnum = 1
        for subobject in object.childobjects:
            self._savePictures(subobject, '%sObject %d/' % (folder, subobjectnum))
            subobjectnum += 1

    def __replaceGenerator(self):
        if False:
            while True:
                i = 10
        ' Section 3.1.1: The application MUST NOT export the original identifier\n            belonging to the application that created the document.\n        '
        for m in self.meta.childNodes[:]:
            if m.qname == (METANS, 'generator'):
                self.meta.removeChild(m)
        self.meta.addElement(meta.Generator(text=TOOLSVERSION))

    def save(self, outputfile, addsuffix=False):
        if False:
            while True:
                i = 10
        " Save the document under the filename.\n            If the filename is '-' then save to stdout\n        "
        if outputfile == '-':
            outputfp = zipfile.ZipFile(sys.stdout, 'w')
        else:
            if addsuffix:
                outputfile = outputfile + odmimetypes.get(self.mimetype, '.xxx')
            outputfp = zipfile.ZipFile(outputfile, 'w')
        self.__zipwrite(outputfp)
        outputfp.close()

    def write(self, outputfp):
        if False:
            print('Hello World!')
        ' User API to write the ODF file to an open file descriptor\n            Writes the ZIP format\n        '
        zipoutputfp = zipfile.ZipFile(outputfp, 'w')
        self.__zipwrite(zipoutputfp)

    def __zipwrite(self, outputfp):
        if False:
            while True:
                i = 10
        ' Write the document to an open file pointer\n            This is where the real work is done\n        '
        self._z = outputfp
        self._now = time.localtime()[:6]
        self.manifest = manifest.Manifest()
        zi = zipfile.ZipInfo('mimetype', self._now)
        zi.compress_type = zipfile.ZIP_STORED
        zi.external_attr = UNIXPERMS
        self._z.writestr(zi, self.mimetype)
        self._saveXmlObjects(self, '')
        self._savePictures(self, '')
        if self.thumbnail is not None:
            self.manifest.addElement(manifest.FileEntry(fullpath='Thumbnails/', mediatype=''))
            self.manifest.addElement(manifest.FileEntry(fullpath='Thumbnails/thumbnail.png', mediatype=''))
            zi = zipfile.ZipInfo('Thumbnails/thumbnail.png', self._now)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = UNIXPERMS
            self._z.writestr(zi, self.thumbnail)
        for op in self._extra:
            if op.filename == 'META-INF/documentsignatures.xml':
                continue
            self.manifest.addElement(manifest.FileEntry(fullpath=op.filename, mediatype=op.mediatype))
            zi = zipfile.ZipInfo(op.filename.encode('utf-8'), self._now)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = UNIXPERMS
            if op.content is not None:
                self._z.writestr(zi, op.content)
        zi = zipfile.ZipInfo('META-INF/manifest.xml', self._now)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = UNIXPERMS
        self._z.writestr(zi, self.__manifestxml())
        del self._z
        del self._now
        del self.manifest

    def _saveXmlObjects(self, object, folder):
        if False:
            for i in range(10):
                print('nop')
        if self == object:
            self.manifest.addElement(manifest.FileEntry(fullpath='/', mediatype=object.mimetype))
        else:
            self.manifest.addElement(manifest.FileEntry(fullpath=folder, mediatype=object.mimetype))
        self.manifest.addElement(manifest.FileEntry(fullpath='%sstyles.xml' % folder, mediatype='text/xml'))
        zi = zipfile.ZipInfo('%sstyles.xml' % folder, self._now)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = UNIXPERMS
        self._z.writestr(zi, object.stylesxml())
        self.manifest.addElement(manifest.FileEntry(fullpath='%scontent.xml' % folder, mediatype='text/xml'))
        zi = zipfile.ZipInfo('%scontent.xml' % folder, self._now)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.external_attr = UNIXPERMS
        self._z.writestr(zi, object.contentxml())
        if object.settings.hasChildNodes():
            self.manifest.addElement(manifest.FileEntry(fullpath='%ssettings.xml' % folder, mediatype='text/xml'))
            zi = zipfile.ZipInfo('%ssettings.xml' % folder, self._now)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = UNIXPERMS
            self._z.writestr(zi, object.settingsxml())
        if self == object:
            self.manifest.addElement(manifest.FileEntry(fullpath='meta.xml', mediatype='text/xml'))
            zi = zipfile.ZipInfo('meta.xml', self._now)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = UNIXPERMS
            self._z.writestr(zi, object.metaxml())
        subobjectnum = 1
        for subobject in object.childobjects:
            self._saveXmlObjects(subobject, '%sObject %d/' % (folder, subobjectnum))
            subobjectnum += 1

    def createElement(self, element):
        if False:
            print('Hello World!')
        " Inconvenient interface to create an element, but follows XML-DOM.\n            Does not allow attributes as argument, therefore can't check grammar.\n        "
        return element(check_grammar=False)

    def createTextNode(self, data):
        if False:
            while True:
                i = 10
        ' Method to create a text node '
        return element.Text(data)

    def createCDATASection(self, data):
        if False:
            print('Hello World!')
        ' Method to create a CDATA section '
        return element.CDATASection(data)

    def getMediaType(self):
        if False:
            return 10
        ' Returns the media type '
        return self.mimetype

    def getStyleByName(self, name):
        if False:
            for i in range(10):
                print('nop')
        ' Finds a style object based on the name '
        ncname = make_NCName(name)
        if self._styles_dict == {}:
            self.rebuild_caches()
        return self._styles_dict.get(ncname, None)

    def getElementsByType(self, element):
        if False:
            i = 10
            return i + 15
        ' Gets elements based on the type, which is function from text.py, draw.py etc. '
        obj = element(check_grammar=False)
        if self.element_dict == {}:
            self.rebuild_caches()
        return self.element_dict.get(obj.qname, [])

def OpenDocumentChart():
    if False:
        for i in range(10):
            print('nop')
    ' Creates a chart document '
    doc = OpenDocument('application/vnd.oasis.opendocument.chart')
    doc.chart = Chart()
    doc.body.addElement(doc.chart)
    return doc

def OpenDocumentDrawing():
    if False:
        return 10
    ' Creates a drawing document '
    doc = OpenDocument('application/vnd.oasis.opendocument.graphics')
    doc.drawing = Drawing()
    doc.body.addElement(doc.drawing)
    return doc

def OpenDocumentImage():
    if False:
        while True:
            i = 10
    ' Creates an image document '
    doc = OpenDocument('application/vnd.oasis.opendocument.image')
    doc.image = Image()
    doc.body.addElement(doc.image)
    return doc

def OpenDocumentPresentation():
    if False:
        for i in range(10):
            print('nop')
    ' Creates a presentation document '
    doc = OpenDocument('application/vnd.oasis.opendocument.presentation')
    doc.presentation = Presentation()
    doc.body.addElement(doc.presentation)
    return doc

def OpenDocumentSpreadsheet():
    if False:
        while True:
            i = 10
    ' Creates a spreadsheet document '
    doc = OpenDocument('application/vnd.oasis.opendocument.spreadsheet')
    doc.spreadsheet = Spreadsheet()
    doc.body.addElement(doc.spreadsheet)
    return doc

def OpenDocumentText():
    if False:
        return 10
    ' Creates a text document '
    doc = OpenDocument('application/vnd.oasis.opendocument.text')
    doc.text = Text()
    doc.body.addElement(doc.text)
    return doc

def OpenDocumentTextMaster():
    if False:
        for i in range(10):
            print('nop')
    ' Creates a text master document '
    doc = OpenDocument('application/vnd.oasis.opendocument.text-master')
    doc.text = Text()
    doc.body.addElement(doc.text)
    return doc

def __loadxmlparts(z, manifest, doc, objectpath):
    if False:
        i = 10
        return i + 15
    from .load import LoadParser
    from xml.sax import make_parser, handler
    for xmlfile in (objectpath + 'settings.xml', objectpath + 'meta.xml', objectpath + 'content.xml', objectpath + 'styles.xml'):
        if xmlfile not in manifest:
            continue
        try:
            xmlpart = z.read(xmlfile)
            doc._parsing = xmlfile
            parser = make_parser()
            parser.setFeature(handler.feature_namespaces, 1)
            parser.setContentHandler(LoadParser(doc))
            parser.setErrorHandler(handler.ErrorHandler())
            inpsrc = InputSource()
            inpsrc.setByteStream(BytesIO(xmlpart))
            parser.setFeature(handler.feature_external_ges, False)
            parser.parse(inpsrc)
            del doc._parsing
        except KeyError:
            pass

def load(odffile):
    if False:
        i = 10
        return i + 15
    ' Load an ODF file into memory\n        Returns a reference to the structure\n    '
    z = zipfile.ZipFile(odffile)
    try:
        mimetype = z.read('mimetype')
    except KeyError:
        mimetype = 'application/vnd.oasis.opendocument.text'
    doc = OpenDocument(mimetype, add_generator=False)
    manifestpart = z.read('META-INF/manifest.xml')
    manifest = manifestlist(manifestpart)
    __loadxmlparts(z, manifest, doc, '')
    for (mentry, mvalue) in manifest.items():
        if mentry[:9] == 'Pictures/' and len(mentry) > 9:
            doc.addPicture(mvalue['full-path'], mvalue['media-type'], z.read(mentry))
        elif mentry == 'Thumbnails/thumbnail.png':
            doc.addThumbnail(z.read(mentry))
        elif mentry in ('settings.xml', 'meta.xml', 'content.xml', 'styles.xml'):
            pass
        elif mentry[:7] == 'Object ' and len(mentry) < 11 and (mentry[-1] == '/'):
            subdoc = OpenDocument(mvalue['media-type'], add_generator=False)
            doc.addObject(subdoc, '/' + mentry[:-1])
            __loadxmlparts(z, manifest, subdoc, mentry)
        elif mentry[:7] == 'Object ':
            pass
        elif mvalue['full-path'][-1] == '/':
            doc._extra.append(OpaqueObject(mvalue['full-path'], mvalue['media-type'], None))
        else:
            doc._extra.append(OpaqueObject(mvalue['full-path'], mvalue['media-type'], z.read(mentry)))
    z.close()
    b = doc.getElementsByType(Body)
    if mimetype[:39] == 'application/vnd.oasis.opendocument.text':
        doc.text = b[0].firstChild
    elif mimetype[:43] == 'application/vnd.oasis.opendocument.graphics':
        doc.graphics = b[0].firstChild
    elif mimetype[:47] == 'application/vnd.oasis.opendocument.presentation':
        doc.presentation = b[0].firstChild
    elif mimetype[:46] == 'application/vnd.oasis.opendocument.spreadsheet':
        doc.spreadsheet = b[0].firstChild
    elif mimetype[:40] == 'application/vnd.oasis.opendocument.chart':
        doc.chart = b[0].firstChild
    elif mimetype[:40] == 'application/vnd.oasis.opendocument.image':
        doc.image = b[0].firstChild
    elif mimetype[:42] == 'application/vnd.oasis.opendocument.formula':
        doc.formula = b[0].firstChild
    return doc