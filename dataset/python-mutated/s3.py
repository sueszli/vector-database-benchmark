"""Allow VisiData to work directly with Amazon S3 paths.

Functionality is more limited than local paths, but supports:

* Navigating among directories (S3 prefixes)
* Opening supported filetypes, including compressed files
* Versioned buckets
"""
import textwrap
from visidata import ENTER, Column, ItemColumn, Path, Sheet, VisiData, asyncthread, date, vd
vd.option('s3_endpoint', '', 'alternate S3 endpoint, used for local testing or alternative S3-compatible services', replay=True)
vd.option('s3_glob', True, 'enable glob-matching for S3 paths', replay=True)
vd.option('s3_version_aware', False, 'show all object versions in a versioned bucket', replay=True)

class S3Path(Path):
    """A Path-like object representing an S3 file (object) or directory (prefix)."""
    _fs = None

    def __init__(self, path, version_aware=None, version_id=None):
        if False:
            return 10
        super().__init__(path)
        self.given = path
        self.version_aware = version_aware or vd.options.s3_version_aware
        self.version_id = self.version_aware and version_id or None

    @property
    def fs(self):
        if False:
            print('Hello World!')
        if self._fs is None:
            s3fs_core = vd.importExternal('s3fs.core', 's3fs')
            self._fs = s3fs_core.S3FileSystem(client_kwargs={'endpoint_url': vd.options.s3_endpoint or None}, version_aware=self.version_aware)
        return self._fs

    @fs.setter
    def fs(self, val):
        if False:
            print('Hello World!')
        self._fs = val

    def open(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Open the current S3 path, decompressing along the way if needed.'
        mode = 'rb' if self.compression else 'r'
        fp = self.fs.open(self.given, mode=mode, version_id=self.version_id)
        if hasattr(fp, 'cache') and fp.cache.size != fp.size:
            vd.debug(f'updating cache size from {fp.cache.size} to {fp.size} to match object size')
            fp.cache.size = fp.size
        if self.compression == 'gz':
            import gzip
            return gzip.open(fp, *args, **kwargs)
        if self.compression == 'bz2':
            import bz2
            return bz2.open(fp, *args, **kwargs)
        if self.compression == 'xz':
            import lzma
            return lzma.open(fp, *args, **kwargs)
        return fp

class S3DirSheet(Sheet):
    """Display a listing of files and directories (objects and prefixes) in an S3 path.

    Allow single or multiple entries to be opened in separate sheets.
    """
    columns = [Column('name', getter=lambda col, row: col.sheet.object_display_name(row)), ItemColumn('type'), ItemColumn('size', type=int), ItemColumn('modtime', 'LastModified', type=date), ItemColumn('latest', 'IsLatest', type=bool), ItemColumn('version_id', 'VersionId', type=str, width=0)]

    def __init__(self, name, source, version_aware=None):
        if False:
            print('Hello World!')
        import re
        super().__init__(name=name, source=source)
        self.rowtype = 'files'
        self.nKeys = 1
        self.use_glob_matching = vd.options.s3_glob and re.search('[*?\\[\\]]', self.source.given)
        self.version_aware = vd.options.s3_version_aware if version_aware is None else version_aware
        self.fs = source.fs

    def object_display_name(self, row):
        if False:
            i = 10
            return i + 15
        'Provide a friendly display name for an S3 path.\n\n        When listing the contents of a single S3 prefix, the name can chop off\n        prefix bits to imitate a directory browser. When glob matching,\n        include the full key name for each entry.\n        '
        return row.get('name') if self.use_glob_matching else row.get('name').rpartition('/')[2]

    def iterload(self):
        if False:
            i = 10
            return i + 15
        'Delegate to the underlying filesystem to fetch S3 entries.'
        list_func = self.fs.glob if self.use_glob_matching else self.fs.ls
        if not (self.use_glob_matching or self.fs.exists(self.source.given) or self.fs.isdir(self.source.given)):
            vd.fail(f'unable to open S3 path: {self.source.given}')
        if self.version_aware:
            self.column('latest').hide(False)
        else:
            self.column('latest').hide(True)
        for key in list_func(str(self.source)):
            if self.version_aware and self.fs.isfile(key):
                yield from ({**obj_version, 'name': key, 'type': 'file'} for obj_version in self.fs.object_version_info(key) if key.partition('/')[2] == obj_version['Key'])
            else:
                yield self.fs.stat(key)

    @asyncthread
    def download(self, rows, savepath):
        if False:
            print('Hello World!')
        'Download files and directories to a local path.\n\n        Recurse through through subdirectories.\n        '
        remote_files = [row['name'] for row in rows]
        self.fs.download(remote_files, str(savepath), recursive=True)

    def open_rows(self, rows):
        if False:
            while True:
                i = 10
        'Open new sheets for the target rows.'
        return (vd.openSource(S3Path('s3://{}'.format(row['name']), version_aware=self.version_aware, version_id=row.get('VersionId'))) for row in rows)

    def join_rows(self, rows):
        if False:
            print('Hello World!')
        'Open new sheets for the target rows and concatenate their contents.'
        sheets = list(self.open_rows(rows))
        for sheet in vd.Progress(sheets):
            sheet.reload()
        vd.sync()
        return sheets[0].openJoin(sheets[1:], jointype='append')

    def refresh_path(self, path=None):
        if False:
            print('Hello World!')
        'Clear the s3fs cache for the given path and reload.\n\n        By default, clear the entire cache.\n        '
        self.fs.invalidate_cache(path)
        self.reload()

    def toggle_versioning(self):
        if False:
            while True:
                i = 10
        'Enable or disable support for S3 versioning.'
        self.version_aware = not self.version_aware
        self.fs.version_aware = self.version_aware
        vd.status(f"s3 versioning {('enabled' if self.version_aware else 'disabled')}")
        if self.currentThreads:
            vd.debug('cancelling threads before reloading')
            vd.cancelThread(*self.currentThreads)
        self.reload()

@VisiData.api
def openurl_s3(vd, p, filetype):
    if False:
        i = 10
        return i + 15
    'Open a sheet for an S3 path.\n\n    S3 directories (prefixes) require special handling, but files (objects)\n    can use standard VisiData "open" functions.\n    '
    endpoint = vd.options.s3_endpoint or None
    p = S3Path(str(p.given), version_aware=getattr(p, 'version_aware', vd.options.s3_version_aware), version_id=getattr(p, 'version_id', None))
    p.fs.version_aware = p.version_aware
    if p.fs.client_kwargs.get('endpoint_url', '') != endpoint:
        p.fs.client_kwargs = {'endpoint_url': endpoint}
        p.fs.connect()
    if not p.fs.isfile(str(p.given)):
        return S3DirSheet(p.name, source=p, version_aware=p.version_aware)
    if not filetype:
        filetype = p.ext or 'txt'
    openfunc = getattr(vd, f'open_{filetype.lower()}')
    if not openfunc:
        vd.warning(f'no loader found for {filetype} files, falling back to txt')
        filetype = 'txt'
        openfunc = vd.open_txt
    assert callable(openfunc), f'no function/method available to open {p.given}'
    vs = openfunc(p)
    vd.status(f"opening {p.given} as {filetype} (version id: {p.version_id or 'latest'})")
    return vs
S3DirSheet.addCommand(ENTER, 's3-open-row', 'vd.push(next(sheet.open_rows([cursorRow])))', 'open the current S3 entry')
S3DirSheet.addCommand('g' + ENTER, 's3-open-rows', 'for vs in sheet.open_rows(selectedRows): vd.push(vs)', 'open all selected S3 entries')
S3DirSheet.addCommand('z^R', 's3-refresh-sheet', 'sheet.refresh_path(str(sheet.source))', 'clear the s3fs cache for this path, then reload')
S3DirSheet.addCommand('gz^R', 's3-refresh-sheet-all', 'sheet.refresh_path()', 'clear the entire s3fs cache, then reload')
S3DirSheet.addCommand('^V', 's3-toggle-versioning', 'sheet.toggle_versioning()', 'enable/disable support for S3 versioning')
S3DirSheet.addCommand('&', 's3-join-rows', 'vd.push(sheet.join_rows(selectedRows))', 'open and join sheets for selected S3 entries')
S3DirSheet.addCommand('gx', 's3-download-rows', textwrap.dedent('\n        savepath = inputPath("download selected rows to: ", value=".")\n        sheet.download(selectedRows, savepath)\n    '), 'download selected files and directories')
S3DirSheet.addCommand('x', 's3-download-row', textwrap.dedent('\n        savepath = inputPath("download to: ", value=Path(cursorRow["name"])._path.name)\n        sheet.download([cursorRow], savepath)\n    '), 'download the file or directory in the cursor row')
vd.addMenuItems('\n    File > Toggle versioning > s3-toggle-versioning\n    File > Refresh > Current path > s3-refresh-sheet\n    File > Refresh > All > s3-refresh-sheet-all\n    Row > Download > Current row > s3-download-row\n    Row > Download > Selected rows > s3-download-rows\n    Data > Join > Selected rows > s3-join-rows\n')
vd.addGlobals(S3DirSheet=S3DirSheet)