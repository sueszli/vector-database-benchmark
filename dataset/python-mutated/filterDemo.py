import pythoncom
import pywintypes
from win32com import storagecon
from win32com.ifilter import ifilter
from win32com.ifilter.ifiltercon import *

class FileParser:
    PIDS_BODY = 19
    PIDH_DESCRIPTION = 'DESCRIPTION'
    PIDH_HREF = 'A.HREF'
    PIDH_IMGSRC = 'IMG.SRC'
    propertyToName = {PSGUID_STORAGE: {PIDS_BODY: 'body'}, PSGUID_SUMMARYINFORMATION: {PIDSI_TITLE: 'title', PIDSI_SUBJECT: 'description', PIDSI_AUTHOR: 'author', PIDSI_KEYWORDS: 'keywords', PIDSI_COMMENTS: 'comments'}, PSGUID_HTMLINFORMATION: {PIDH_DESCRIPTION: 'description'}, PSGUID_HTML2_INFORMATION: {PIDH_HREF: 'href', PIDH_IMGSRC: 'img'}}

    def __init__(self, verbose=False):
        if False:
            while True:
                i = 10
        self.f = None
        self.stg = None
        self.verbose = verbose

    def Close(self):
        if False:
            return 10
        self.f = None
        self.stg = None

    def Parse(self, fileName, maxErrors=10):
        if False:
            for i in range(10):
                print('nop')
        properties = {}
        try:
            self._bind_to_filter(fileName)
            try:
                flags = self.f.Init(IFILTER_INIT_APPLY_INDEX_ATTRIBUTES | IFILTER_INIT_APPLY_OTHER_ATTRIBUTES)
                if flags == IFILTER_FLAGS_OLE_PROPERTIES and self.stg is not None:
                    self._trace('filter requires to get properities via ole')
                    self._get_properties(properties)
                errCnt = 0
                while True:
                    try:
                        (idChunk, breakType, flags, locale, attr, idChunkSource, startSource, lenSource) = self.f.GetChunk()
                        self._trace('Chunk details:', idChunk, breakType, flags, locale, attr, idChunkSource, startSource, lenSource)
                        propSet = self.propertyToName.get(attr[0])
                        if propSet:
                            propName = propSet.get(attr[1], '{}:{}'.format(*attr))
                        else:
                            propName = '{}:{}'.format(*attr)
                    except pythoncom.com_error as e:
                        if e[0] == FILTER_E_END_OF_CHUNKS:
                            break
                        elif e[0] in [FILTER_E_EMBEDDING_UNAVAILABLE, FILTER_E_LINK_UNAVAILABLE]:
                            errCnt += 1
                            if errCnt > maxErrors:
                                raise
                            else:
                                continue
                        elif e[0] == FILTER_E_ACCESS:
                            self._trace('Access denied')
                            raise
                        elif e[0] == FILTER_E_PASSWORD:
                            self._trace('Password required')
                            raise
                        else:
                            raise
                    errCnt = 0
                    if flags == CHUNK_TEXT:
                        body_chunks = properties.setdefault(propName, [])
                        self._get_text(body_chunks)
                    elif flags == CHUNK_VALUE:
                        properties[propName] = self.f.GetValue()
                    else:
                        self._trace('Unknown flag returned by GetChunk:', flags)
            finally:
                self.Close()
        except pythoncom.com_error as e:
            self._trace('ERROR processing file', e)
            raise
        return properties

    def _bind_to_filter(self, fileName):
        if False:
            print('Hello World!')
        '\n        See if the file is a structured storage file or a normal file\n        and then return an ifilter interface by calling the appropriate bind/load function\n        '
        if pythoncom.StgIsStorageFile(fileName):
            self.stg = pythoncom.StgOpenStorage(fileName, None, storagecon.STGM_READ | storagecon.STGM_SHARE_DENY_WRITE)
            try:
                self.f = ifilter.BindIFilterFromStorage(self.stg)
            except pythoncom.com_error as e:
                if e[0] == -2147467262:
                    self.f = ifilter.LoadIFilter(fileName)
                else:
                    raise
        else:
            self.f = ifilter.LoadIFilter(fileName)
            self.stg = None

    def _get_text(self, body_chunks):
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets all the text for a particular chunk. We need to keep calling get text till all the\n        segments for this chunk are retrieved\n        '
        while True:
            try:
                body_chunks.append(self.f.GetText())
            except pythoncom.com_error as e:
                if e[0] in [FILTER_E_NO_MORE_TEXT, FILTER_E_NO_MORE_TEXT, FILTER_E_NO_TEXT]:
                    break
                else:
                    raise

    def _get_properties(self, properties):
        if False:
            while True:
                i = 10
        '\n        Use OLE property sets to get base properties\n        '
        try:
            pss = self.stg.QueryInterface(pythoncom.IID_IPropertySetStorage)
        except pythoncom.com_error as e:
            self._trace('No Property information could be retrieved', e)
            return
        ps = pss.Open(PSGUID_SUMMARYINFORMATION)
        props = (PIDSI_TITLE, PIDSI_SUBJECT, PIDSI_AUTHOR, PIDSI_KEYWORDS, PIDSI_COMMENTS)
        (title, subject, author, keywords, comments) = ps.ReadMultiple(props)
        if title is not None:
            properties['title'] = title
        if subject is not None:
            properties['description'] = subject
        if author is not None:
            properties['author'] = author
        if keywords is not None:
            properties['keywords'] = keywords
        if comments is not None:
            properties['comments'] = comments

    def _trace(self, *args):
        if False:
            print('Hello World!')
        if self.verbose:
            ret = ' '.join([str(arg) for arg in args])
            try:
                print(ret)
            except OSError:
                pass

def _usage():
    if False:
        print('Hello World!')
    import os
    print(f'Usage: {os.path.basename(sys.argv[0])} filename [verbose [dumpbody]]')
    print()
    print('Where:-')
    print('filename = name of the file to extract text & properties from')
    print('verbose = 1=debug output, 0=no debug output (default=0)')
    print("dumpbody = 1=print text content, 0=don't print content (default=1)")
    print()
    print('e.g. to dump a word file called spam.doc go:- filterDemo.py spam.doc')
    print()
    print('by default .htm, .txt, .doc, .dot, .xls, .xlt, .ppt are supported')
    print("you can filter .pdf's by downloading adobes ifilter component. ")
    print('(currently found at http://download.adobe.com/pub/adobe/acrobat/win/all/ifilter50.exe).')
    print('ifilters for other filetypes are also available.')
    print()
    print('This extension is only supported on win2000 & winXP - because thats the only')
    print('place the ifilter stuff is supported. For more info on the API check out ')
    print('MSDN under ifilters')
if __name__ == '__main__':
    import operator
    import sys
    fName = ''
    verbose = False
    bDumpBody = True
    if len(sys.argv) < 2:
        _usage()
        sys.exit(1)
    try:
        fName = sys.argv[1]
        verbose = sys.argv[2] != '0'
        bDumpBody = sys.argv[3] != '0'
    except:
        pass
    p = FileParser(verbose)
    propMap = p.Parse(fName)
    if bDumpBody:
        print('Body')
        ch = ' '.join(propMap.get('body', []))
        try:
            print(ch)
        except UnicodeError:
            print(ch.encode('iso8859-1', 'ignore'))
    print('Properties')
    for (propName, propValue) in propMap.items():
        print(propName, ':', end=' ')
        if propName == 'body':
            print('<%s length: %d>' % (propName, reduce(operator.add, [len(p) for p in propValue])))
        elif isinstance(propValue, list):
            print()
            for pv in propValue:
                print(pv)
        else:
            print(propValue)
        print()