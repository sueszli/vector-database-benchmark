"""
    pylrf.py -- very low level interface to create lrf files.  See pylrs for
    higher level interface that can use this module to render books to lrf.
"""
import struct
import zlib
import io
import codecs
import os
from .pylrfopt import tagListOptimizer
from polyglot.builtins import iteritems, string_or_bytes
PYLRF_VERSION = '1.0'

class LrfError(Exception):
    pass

def writeByte(f, byte):
    if False:
        i = 10
        return i + 15
    f.write(struct.pack('<B', byte))

def writeWord(f, word):
    if False:
        while True:
            i = 10
    if int(word) > 65535:
        raise LrfError('Cannot encode a number greater than 65535 in a word.')
    if int(word) < 0:
        raise LrfError('Cannot encode a number < 0 in a word: ' + str(word))
    f.write(struct.pack('<H', int(word)))

def writeSignedWord(f, sword):
    if False:
        return 10
    f.write(struct.pack('<h', int(float(sword))))

def writeWords(f, *words):
    if False:
        for i in range(10):
            print('nop')
    f.write(struct.pack('<%dH' % len(words), *words))

def writeDWord(f, dword):
    if False:
        i = 10
        return i + 15
    f.write(struct.pack('<I', int(dword)))

def writeDWords(f, *dwords):
    if False:
        return 10
    f.write(struct.pack('<%dI' % len(dwords), *dwords))

def writeQWord(f, qword):
    if False:
        while True:
            i = 10
    f.write(struct.pack('<Q', qword))

def writeZeros(f, nZeros):
    if False:
        while True:
            i = 10
    f.write(b'\x00' * nZeros)

def writeString(f, s):
    if False:
        print('Hello World!')
    f.write(s)

def writeIdList(f, idList):
    if False:
        for i in range(10):
            print('nop')
    writeWord(f, len(idList))
    writeDWords(f, *idList)

def writeColor(f, color):
    if False:
        print('Hello World!')
    f.write(struct.pack('>I', int(color, 0)))

def writeLineWidth(f, width):
    if False:
        while True:
            i = 10
    writeWord(f, int(width))

def writeUnicode(f, string, encoding):
    if False:
        i = 10
        return i + 15
    if isinstance(string, bytes):
        string = string.decode(encoding)
    string = string.encode('utf-16-le')
    length = len(string)
    if length > 65535:
        raise LrfError('Cannot write strings longer than 65535 characters.')
    writeWord(f, length)
    writeString(f, string)

def writeRaw(f, string, encoding):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(string, bytes):
        string = string.decode(encoding)
    string = string.encode('utf-16-le')
    writeString(f, string)

def writeRubyAA(f, rubyAA):
    if False:
        i = 10
        return i + 15
    (ralign, radjust) = rubyAA
    radjust = {'line-edge': 16, 'none': 0}[radjust]
    ralign = {'start': 1, 'center': 2}[ralign]
    writeWord(f, ralign | radjust)

def writeBgImage(f, bgInfo):
    if False:
        return 10
    (imode, iid) = bgInfo
    imode = {'pfix': 0, 'fix': 1, 'tile': 2, 'centering': 3}[imode]
    writeWord(f, imode)
    writeDWord(f, iid)

def writeEmpDots(f, dotsInfo, encoding):
    if False:
        return 10
    (refDotsFont, dotsFontName, dotsCode) = dotsInfo
    writeDWord(f, refDotsFont)
    LrfTag('fontfacename', dotsFontName).write(f, encoding)
    writeWord(f, int(dotsCode, 0))

def writeRuledLine(f, lineInfo):
    if False:
        print('Hello World!')
    (lineLength, lineType, lineWidth, lineColor) = lineInfo
    writeWord(f, lineLength)
    writeWord(f, LINE_TYPE_ENCODING[lineType])
    writeWord(f, lineWidth)
    writeColor(f, lineColor)
LRF_SIGNATURE = b'L\x00R\x00F\x00\x00\x00'
XOR_KEY = 65024
LRF_VERSION = 1000
IMAGE_TYPE_ENCODING = dict(GIF=20, PNG=18, BMP=19, JPEG=17, JPG=17)
OBJECT_TYPE_ENCODING = dict(PageTree=1, Page=2, Header=3, Footer=4, PageAtr=5, PageStyle=5, Block=6, BlockAtr=7, BlockStyle=7, MiniPage=8, TextBlock=10, Text=10, TextAtr=11, TextStyle=11, ImageBlock=12, Image=12, Canvas=13, ESound=14, ImageStream=17, Import=18, Button=19, Window=20, PopUpWindow=21, Sound=22, SoundStream=23, Font=25, ObjectInfo=26, BookAtr=28, BookStyle=28, SimpleTextBlock=29, TOC=30)
LINE_TYPE_ENCODING = {'none': 0, 'solid': 16, 'dashed': 32, 'double': 48, 'dotted': 64}
BINDING_DIRECTION_ENCODING = dict(Lr=1, Rl=16)
TAG_INFO = dict(rawtext=(0, writeRaw), ObjectStart=(62720, '<IH'), ObjectEnd=(62721,), Link=(62723, '<I'), StreamSize=(62724, writeDWord), StreamData=(62725, writeString), StreamEnd=(62726,), oddheaderid=(62727, writeDWord), evenheaderid=(62728, writeDWord), oddfooterid=(62729, writeDWord), evenfooterid=(62730, writeDWord), ObjectList=(62731, writeIdList), fontsize=(62737, writeSignedWord), fontwidth=(62738, writeSignedWord), fontescapement=(62739, writeSignedWord), fontorientation=(62740, writeSignedWord), fontweight=(62741, writeWord), fontfacename=(62742, writeUnicode), textcolor=(62743, writeColor), textbgcolor=(62744, writeColor), wordspace=(62745, writeSignedWord), letterspace=(62746, writeSignedWord), baselineskip=(62747, writeSignedWord), linespace=(62748, writeSignedWord), parindent=(62749, writeSignedWord), parskip=(62750, writeSignedWord), topmargin=(62753, writeWord), headheight=(62754, writeWord), headsep=(62755, writeWord), oddsidemargin=(62756, writeWord), textheight=(62757, writeWord), textwidth=(62758, writeWord), canvaswidth=(62801, writeWord), canvasheight=(62802, writeWord), footspace=(62759, writeWord), footheight=(62760, writeWord), bgimage=(62761, writeBgImage), setemptyview=(62762, {'show': 1, 'empty': 0}, writeWord), pageposition=(62763, {'any': 0, 'upper': 1, 'lower': 2}, writeWord), evensidemargin=(62764, writeWord), framemode=(62766, {'None': 0, 'curve': 2, 'square': 1}, writeWord), blockwidth=(62769, writeWord), blockheight=(62770, writeWord), blockrule=(62771, {'horz-fixed': 20, 'horz-adjustable': 18, 'vert-fixed': 65, 'vert-adjustable': 33, 'block-fixed': 68, 'block-adjustable': 34}, writeWord), bgcolor=(62772, writeColor), layout=(62773, {'TbRl': 65, 'LrTb': 52}, writeWord), framewidth=(62774, writeWord), framecolor=(62775, writeColor), topskip=(62776, writeWord), sidemargin=(62777, writeWord), footskip=(62778, writeWord), align=(62780, {'head': 1, 'center': 4, 'foot': 8}, writeWord), column=(62781, writeWord), columnsep=(62782, writeSignedWord), minipagewidth=(62785, writeWord), minipageheight=(62786, writeWord), yspace=(62790, writeWord), xspace=(62791, writeWord), PutObj=(62793, '<HHI'), ImageRect=(62794, '<HHHH'), ImageSize=(62795, '<HH'), RefObjId=(62796, '<I'), PageDiv=(62798, '<HIHI'), StreamFlags=(62804, writeWord), Comment=(62805, writeUnicode), FontFilename=(62809, writeUnicode), PageList=(62812, writeIdList), FontFacename=(62813, writeUnicode), buttonflags=(62817, writeWord), PushButtonStart=(62822,), PushButtonEnd=(62823,), buttonactions=(62826,), endbuttonactions=(62827,), jumpto=(62828, '<II'), RuledLine=(62835, writeRuledLine), rubyaa=(62837, writeRubyAA), rubyoverhang=(62838, {'none': 0, 'auto': 1}, writeWord), empdotsposition=(62839, {'before': 1, 'after': 2}, writeWord), empdots=(62840, writeEmpDots), emplineposition=(62841, {'before': 1, 'after': 2}, writeWord), emplinetype=(62842, LINE_TYPE_ENCODING, writeWord), ChildPageTree=(62843, '<I'), ParentPageTree=(62844, '<I'), Italic=(62849,), ItalicEnd=(62850,), pstart=(62881, writeDWord), pend=(62882,), CharButton=(62887, writeDWord), CharButtonEnd=(62888,), Rubi=(62889,), RubiEnd=(62890,), Oyamoji=(62891,), OyamojiEnd=(62892,), Rubimoji=(62893,), RubimojiEnd=(62894,), Yoko=(62897,), YokoEnd=(62898,), Tate=(62899,), TateEnd=(62900,), Nekase=(62901,), NekaseEnd=(62902,), Sup=(62903,), SupEnd=(62904,), Sub=(62905,), SubEnd=(62906,), NoBR=(62907,), NoBREnd=(62908,), EmpDots=(62909,), EmpDotsEnd=(62910,), EmpLine=(62913,), EmpLineEnd=(62914,), DrawChar=(62915, '<H'), DrawCharEnd=(62916,), Box=(62918, LINE_TYPE_ENCODING, writeWord), BoxEnd=(62919,), Space=(62922, writeSignedWord), textstring=(62924, writeUnicode), Plot=(62929, '<HHII'), CR=(62930,), RegisterFont=(62936, writeDWord), setwaitprop=(62938, {'replay': 1, 'noreplay': 2}, writeWord), charspace=(62941, writeSignedWord), textlinewidth=(62961, writeLineWidth), linecolor=(62962, writeColor))

class ObjectTableEntry:

    def __init__(self, objId, offset, size):
        if False:
            print('Hello World!')
        self.objId = objId
        self.offset = offset
        self.size = size

    def write(self, f):
        if False:
            i = 10
            return i + 15
        writeDWords(f, self.objId, self.offset, self.size, 0)

class LrfTag:

    def __init__(self, name, *parameters):
        if False:
            return 10
        try:
            tagInfo = TAG_INFO[name]
        except KeyError:
            raise LrfError('tag name %s not recognized' % name)
        self.name = name
        self.type = tagInfo[0]
        self.format = tagInfo[1:]
        if len(parameters) > 1:
            raise LrfError('only one parameter allowed on tag %s' % name)
        if len(parameters) == 0:
            self.parameter = None
        else:
            self.parameter = parameters[0]

    def write(self, lrf, encoding=None):
        if False:
            print('Hello World!')
        if self.type != 0:
            writeWord(lrf, self.type)
        p = self.parameter
        if p is None:
            return
        for f in self.format:
            if isinstance(f, dict):
                p = f[p]
            elif isinstance(f, string_or_bytes):
                if isinstance(p, tuple):
                    writeString(lrf, struct.pack(f, *p))
                else:
                    writeString(lrf, struct.pack(f, p))
            elif f in [writeUnicode, writeRaw, writeEmpDots]:
                if encoding is None:
                    raise LrfError('Tag requires encoding')
                f(lrf, p, encoding)
            else:
                f(lrf, p)
STREAM_SCRAMBLED = 512
STREAM_COMPRESSED = 256
STREAM_FORCE_COMPRESSED = 33024
STREAM_TOC = 81

class LrfStreamBase:

    def __init__(self, streamFlags, streamData=None):
        if False:
            for i in range(10):
                print('nop')
        self.streamFlags = streamFlags
        self.streamData = streamData

    def setStreamData(self, streamData):
        if False:
            return 10
        self.streamData = streamData

    def getStreamTags(self, optimize=False):
        if False:
            while True:
                i = 10
        flags = self.streamFlags
        streamBuffer = self.streamData
        if flags & STREAM_FORCE_COMPRESSED == STREAM_FORCE_COMPRESSED:
            optimize = False
        if flags & STREAM_COMPRESSED == STREAM_COMPRESSED:
            uncompLen = len(streamBuffer)
            compStreamBuffer = zlib.compress(streamBuffer)
            if optimize and uncompLen <= len(compStreamBuffer) + 4:
                flags &= ~STREAM_COMPRESSED
            else:
                streamBuffer = struct.pack('<I', uncompLen) + compStreamBuffer
        return [LrfTag('StreamFlags', flags & 511), LrfTag('StreamSize', len(streamBuffer)), LrfTag('StreamData', streamBuffer), LrfTag('StreamEnd')]

class LrfTagStream(LrfStreamBase):

    def __init__(self, streamFlags, streamTags=None):
        if False:
            print('Hello World!')
        LrfStreamBase.__init__(self, streamFlags)
        if streamTags is None:
            self.tags = []
        else:
            self.tags = streamTags[:]

    def appendLrfTag(self, tag):
        if False:
            while True:
                i = 10
        self.tags.append(tag)

    def getStreamTags(self, encoding, optimizeTags=False, optimizeCompression=False):
        if False:
            return 10
        stream = io.BytesIO()
        if optimizeTags:
            tagListOptimizer(self.tags)
        for tag in self.tags:
            tag.write(stream, encoding)
        self.streamData = stream.getvalue()
        stream.close()
        return LrfStreamBase.getStreamTags(self, optimize=optimizeCompression)

class LrfFileStream(LrfStreamBase):

    def __init__(self, streamFlags, filename):
        if False:
            while True:
                i = 10
        LrfStreamBase.__init__(self, streamFlags)
        with open(filename, 'rb') as f:
            self.streamData = f.read()

class LrfObject:

    def __init__(self, name, objId):
        if False:
            return 10
        if objId <= 0:
            raise LrfError('invalid objId for ' + name)
        self.name = name
        self.objId = objId
        self.tags = []
        try:
            self.type = OBJECT_TYPE_ENCODING[name]
        except KeyError:
            raise LrfError('object name %s not recognized' % name)

    def __str__(self):
        if False:
            return 10
        return 'LRFObject: ' + self.name + ', ' + str(self.objId)

    def appendLrfTag(self, tag):
        if False:
            for i in range(10):
                print('nop')
        self.tags.append(tag)

    def appendLrfTags(self, tagList):
        if False:
            while True:
                i = 10
        self.tags.extend(tagList)
    append = appendLrfTag

    def appendTagDict(self, tagDict, genClass=None):
        if False:
            i = 10
            return i + 15
        composites = {}
        for (name, value) in iteritems(tagDict):
            if name == 'rubyAlignAndAdjust':
                continue
            if name in {'bgimagemode', 'bgimageid', 'rubyalign', 'rubyadjust', 'empdotscode', 'empdotsfontname', 'refempdotsfont'}:
                composites[name] = value
            else:
                self.append(LrfTag(name, value))
        if 'rubyalign' in composites or 'rubyadjust' in composites:
            ralign = composites.get('rubyalign', 'none')
            radjust = composites.get('rubyadjust', 'start')
            self.append(LrfTag('rubyaa', (ralign, radjust)))
        if 'bgimagemode' in composites or 'bgimageid' in composites:
            imode = composites.get('bgimagemode', 'fix')
            iid = composites.get('bgimageid', 0)
            if genClass == 'PageStyle' and imode == 'fix':
                imode = 'pfix'
            self.append(LrfTag('bgimage', (imode, iid)))
        if 'empdotscode' in composites or 'empdotsfontname' in composites or 'refempdotsfont' in composites:
            dotscode = composites.get('empdotscode', '0x002E')
            dotsfontname = composites.get('empdotsfontname', 'Dutch801 Rm BT Roman')
            refdotsfont = composites.get('refempdotsfont', 0)
            self.append(LrfTag('empdots', (refdotsfont, dotsfontname, dotscode)))

    def write(self, lrf, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        LrfTag('ObjectStart', (self.objId, self.type)).write(lrf)
        for tag in self.tags:
            tag.write(lrf, encoding)
        LrfTag('ObjectEnd').write(lrf)

class LrfToc(LrfObject):
    """
        Table of contents.  Format of toc is:
        [ (pageid, objid, string)...]
    """

    def __init__(self, objId, toc, se):
        if False:
            print('Hello World!')
        LrfObject.__init__(self, 'TOC', objId)
        streamData = self._makeTocStream(toc, se)
        self._makeStreamTags(streamData)

    def _makeStreamTags(self, streamData):
        if False:
            i = 10
            return i + 15
        stream = LrfStreamBase(STREAM_TOC, streamData)
        self.tags.extend(stream.getStreamTags())

    def _makeTocStream(self, toc, se):
        if False:
            i = 10
            return i + 15
        stream = io.BytesIO()
        nEntries = len(toc)
        writeDWord(stream, nEntries)
        lastOffset = 0
        writeDWord(stream, lastOffset)
        for i in range(nEntries - 1):
            (pageId, objId, label) = toc[i]
            entryLen = 4 + 4 + 2 + len(label) * 2
            lastOffset += entryLen
            writeDWord(stream, lastOffset)
        for entry in toc:
            (pageId, objId, label) = entry
            if pageId <= 0:
                raise LrfError('page id invalid in toc: ' + label)
            if objId <= 0:
                raise LrfError('textblock id invalid in toc: ' + label)
            writeDWord(stream, pageId)
            writeDWord(stream, objId)
            writeUnicode(stream, label, se)
        streamData = stream.getvalue()
        stream.close()
        return streamData

class LrfWriter:

    def __init__(self, sourceEncoding):
        if False:
            for i in range(10):
                print('nop')
        self.sourceEncoding = sourceEncoding
        self.saveStreamTags = False
        self.optimizeTags = False
        self.optimizeCompression = False
        self.rootObjId = 0
        self.rootObj = None
        self.binding = 1
        self.dpi = 1600
        self.width = 600
        self.height = 800
        self.colorDepth = 24
        self.tocObjId = 0
        self.docInfoXml = ''
        self.thumbnailEncoding = 'JPEG'
        self.thumbnailData = b''
        self.objects = []
        self.objectTable = []

    def getSourceEncoding(self):
        if False:
            i = 10
            return i + 15
        return self.sourceEncoding

    def toUnicode(self, string):
        if False:
            i = 10
            return i + 15
        if isinstance(string, bytes):
            string = string.decode(self.sourceEncoding)
        return string

    def getDocInfoXml(self):
        if False:
            for i in range(10):
                print('nop')
        return self.docInfoXml

    def setPageTreeId(self, objId):
        if False:
            for i in range(10):
                print('nop')
        self.pageTreeId = objId

    def getPageTreeId(self):
        if False:
            while True:
                i = 10
        return self.pageTreeId

    def setRootObject(self, obj):
        if False:
            for i in range(10):
                print('nop')
        if self.rootObjId != 0:
            raise LrfError('root object already set')
        self.rootObjId = obj.objId
        self.rootObj = obj

    def registerFontId(self, id):
        if False:
            for i in range(10):
                print('nop')
        if self.rootObj is None:
            raise LrfError("can't register font -- no root object")
        self.rootObj.append(LrfTag('RegisterFont', id))

    def setTocObject(self, obj):
        if False:
            return 10
        if self.tocObjId != 0:
            raise LrfError('toc object already set')
        self.tocObjId = obj.objId

    def setThumbnailFile(self, filename, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        with open(filename, 'rb') as f:
            self.thumbnailData = f.read()
        if encoding is None:
            encoding = os.path.splitext(filename)[1][1:]
        encoding = encoding.upper()
        if encoding not in IMAGE_TYPE_ENCODING:
            raise LrfError('unknown image type: ' + encoding)
        self.thumbnailEncoding = encoding

    def append(self, obj):
        if False:
            for i in range(10):
                print('nop')
        self.objects.append(obj)

    def addLrfObject(self, objId):
        if False:
            for i in range(10):
                print('nop')
        pass

    def writeFile(self, lrf):
        if False:
            for i in range(10):
                print('nop')
        if self.rootObjId == 0:
            raise LrfError('no root object has been set')
        self.writeHeader(lrf)
        self.writeObjects(lrf)
        self.updateObjectTableOffset(lrf)
        self.updateTocObjectOffset(lrf)
        self.writeObjectTable(lrf)

    def writeHeader(self, lrf):
        if False:
            while True:
                i = 10
        writeString(lrf, LRF_SIGNATURE)
        writeWord(lrf, LRF_VERSION)
        writeWord(lrf, XOR_KEY)
        writeDWord(lrf, self.rootObjId)
        writeQWord(lrf, len(self.objects))
        writeQWord(lrf, 0)
        writeZeros(lrf, 4)
        writeWord(lrf, self.binding)
        writeDWord(lrf, self.dpi)
        writeWords(lrf, self.width, self.height, self.colorDepth)
        writeZeros(lrf, 20)
        writeDWord(lrf, self.tocObjId)
        writeDWord(lrf, 0)
        docInfoXml = codecs.BOM_UTF8 + self.docInfoXml.encode('utf-8')
        compDocInfo = zlib.compress(docInfoXml)
        writeWord(lrf, len(compDocInfo) + 4)
        writeWord(lrf, IMAGE_TYPE_ENCODING[self.thumbnailEncoding])
        writeDWord(lrf, len(self.thumbnailData))
        writeDWord(lrf, len(docInfoXml))
        writeString(lrf, compDocInfo)
        writeString(lrf, self.thumbnailData)

    def writeObjects(self, lrf):
        if False:
            i = 10
            return i + 15
        self.objectTable = []
        for obj in self.objects:
            objStart = lrf.tell()
            obj.write(lrf, self.sourceEncoding)
            objEnd = lrf.tell()
            self.objectTable.append(ObjectTableEntry(obj.objId, objStart, objEnd - objStart))

    def updateObjectTableOffset(self, lrf):
        if False:
            print('Hello World!')
        tableOffset = lrf.tell()
        lrf.seek(24, 0)
        writeQWord(lrf, tableOffset)
        lrf.seek(0, 2)

    def updateTocObjectOffset(self, lrf):
        if False:
            i = 10
            return i + 15
        if self.tocObjId == 0:
            return
        for entry in self.objectTable:
            if entry.objId == self.tocObjId:
                lrf.seek(72, 0)
                writeDWord(lrf, entry.offset)
                lrf.seek(0, 2)
                break
        else:
            raise LrfError('toc object not in object table')

    def writeObjectTable(self, lrf):
        if False:
            for i in range(10):
                print('nop')
        for tableEntry in self.objectTable:
            tableEntry.write(lrf)