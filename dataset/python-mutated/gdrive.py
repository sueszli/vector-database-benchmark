import visidata
from visidata import vd, VisiData, Sheet, IndexSheet, SequenceSheet, ColumnItem, Path, AttrDict, ColumnAttr, asyncthread, Progress, ColumnExpr, date
from .gsheets import GSheetsIndex

@VisiData.api
def open_gdrive(vd, p):
    if False:
        i = 10
        return i + 15
    return GDriveSheet(p.name)
FILES_FIELDS_VISIBLE = 'name size modifiedTime mimeType description'.split()
FILES_FIELDS = '\nid name size modifiedTime mimeType description owners\nstarred properties spaces version webContentLink webViewLink sharingUser lastModifyingUser shared\nownedByMe originalFilename md5Checksum size quotaBytesUsed headRevisionId imageMediaMetadata videoMediaMetadata parents\nexportLinks contentRestrictions contentHints trashed\n'.split()

@VisiData.cached_property
def _drivebuild(vd):
    if False:
        return 10
    return vd.google_discovery.build('drive', 'v3', credentials=vd.google_auth('drive.readonly'))

@VisiData.cached_property
def _gdrive(self):
    if False:
        return 10
    return vd.google_discovery.build('drive', 'v3', credentials=vd.google_auth('drive.readonly')).files()

@VisiData.cached_property
def _gdrive_rw(self):
    if False:
        print('Hello World!')
    return vd.google_discovery.build('drive', 'v3', credentials=vd.google_auth('drive')).files()

class GDriveSheet(Sheet):
    rowtype = 'files'
    defer = True
    columns = [ColumnItem('name'), ColumnItem('size', type=int), ColumnItem('modifiedTime', type=date), ColumnItem('mimeType'), ColumnItem('name'), ColumnExpr('owner', expr='owners[0]["displayName"]')] + [ColumnItem(x, width=0) for x in FILES_FIELDS if x not in FILES_FIELDS_VISIBLE]

    def iterload(self):
        if False:
            return 10
        self.results = []
        page_token = None
        while True:
            ret = vd._gdrive.list(pageSize=1000, pageToken=page_token, fields='nextPageToken, files(%s)' % ','.join(FILES_FIELDS)).execute()
            self.results.append(ret)
            for r in ret.get('files', []):
                yield AttrDict(r)
            page_token = ret.get('nextPageToken', None)
            if not page_token:
                break

    def openRow(self, r):
        if False:
            for i in range(10):
                print('nop')
        if r.mimeType == 'application/vnd.google-apps.spreadsheet':
            return GSheetsIndex(r.name, source=r.id)
        if r.mimeType.startswith('image'):
            return vd.launchBrowser(r.webViewLink)
        return vd.openSource(r.webContentLink)

    @asyncthread
    def deleteFile(self, **kwargs):
        if False:
            print('Hello World!')
        with Progress(total=1) as prog:
            vd._gdrive_rw.delete(**kwargs).execute()
            prog.addProgress(1)

    @asyncthread
    def putChanges(self):
        if False:
            return 10
        (adds, mods, dels) = self.getDeferredChanges()
        for row in Progress(dels.values()):
            self.deleteFile(fileId=row.id)
        vd.sync()
        self.preloadHook()
        self.reload()