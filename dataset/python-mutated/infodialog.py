from collections import defaultdict, namedtuple
from html import escape
import os.path
import re
import traceback
from PyQt6 import QtCore, QtGui, QtWidgets
from picard import log
from picard.album import Album
from picard.coverart.image import CoverArtImageIOError
from picard.file import File
from picard.track import Track
from picard.util import bytes2human, encode_filename, format_time, open_local_path, union_sorted_lists
from picard.ui import PicardDialog
from picard.ui.colors import interface_colors
from picard.ui.ui_infodialog import Ui_InfoDialog
from picard.ui.util import StandardButton

class ArtworkCoverWidget(QtWidgets.QWidget):
    """A QWidget that can be added to artwork column cell of ArtworkTable."""
    SIZE = 170

    def __init__(self, pixmap=None, text=None, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        layout = QtWidgets.QVBoxLayout()
        if pixmap is not None:
            image_label = QtWidgets.QLabel()
            image_label.setPixmap(pixmap.scaled(self.SIZE, self.SIZE, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation))
            image_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(image_label)
        if text is not None:
            text_label = QtWidgets.QLabel()
            text_label.setText(text)
            text_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            text_label.setWordWrap(True)
            layout.addWidget(text_label)
        self.setLayout(layout)

class ArtworkTable(QtWidgets.QTableWidget):

    def __init__(self, display_existing_art):
        if False:
            i = 10
            return i + 15
        super().__init__(0, 2)
        self.display_existing_art = display_existing_art
        h_header = self.horizontalHeader()
        v_header = self.verticalHeader()
        h_header.setDefaultSectionSize(200)
        v_header.setDefaultSectionSize(230)
        if self.display_existing_art:
            self._existing_cover_col = 0
            self._type_col = 1
            self._new_cover_col = 2
            self.insertColumn(2)
            self.setHorizontalHeaderLabels([_('Existing Cover'), _('Type'), _('New Cover')])
        else:
            self._type_col = 0
            self._new_cover_col = 1
            self.setHorizontalHeaderLabels([_('Type'), _('Cover')])
            self.setColumnWidth(self._type_col, 140)

class InfoDialog(PicardDialog):

    def __init__(self, obj, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.obj = obj
        self.images = []
        self.existing_images = []
        self.ui = Ui_InfoDialog()
        self.display_existing_artwork = False
        if isinstance(obj, File) and isinstance(obj.parent, Track) or isinstance(obj, Track) or (isinstance(obj, Album) and obj.get_num_total_files() > 0):
            if getattr(obj, 'orig_metadata', None) is not None and obj.orig_metadata.images and (obj.orig_metadata.images != obj.metadata.images):
                self.display_existing_artwork = True
                self.existing_images = obj.orig_metadata.images
        if obj.metadata.images:
            self.images = obj.metadata.images
        if not self.images and self.existing_images:
            self.images = self.existing_images
            self.existing_images = []
            self.display_existing_artwork = False
        self.ui.setupUi(self)
        self.ui.buttonBox.addButton(StandardButton(StandardButton.CLOSE), QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.artwork_table = ArtworkTable(self.display_existing_artwork)
        self.ui.artwork_table.setObjectName('artwork_table')
        self.ui.artwork_tab.layout().addWidget(self.ui.artwork_table)
        self.setTabOrder(self.ui.tabWidget, self.ui.artwork_table)
        self.setTabOrder(self.ui.artwork_table, self.ui.buttonBox)
        self.setWindowTitle(_('Info'))
        self.artwork_table = self.ui.artwork_table
        self._display_tabs()

    def _display_tabs(self):
        if False:
            i = 10
            return i + 15
        self._display_info_tab()
        self._display_error_tab()
        self._display_artwork_tab()

    def _display_error_tab(self):
        if False:
            return 10
        if hasattr(self.obj, 'errors') and self.obj.errors:
            self._show_errors(self.obj.errors)
        else:
            self.tab_hide(self.ui.error_tab)

    def _show_errors(self, errors):
        if False:
            return 10
        if errors:
            color = interface_colors.get_color('log_error')
            text = '<br />'.join(map(lambda s: '<font color="%s">%s</font>' % (color, text_as_html(s)), errors))
            self.ui.error.setText(text + '<hr />')

    def _display_artwork(self, images, col):
        if False:
            i = 10
            return i + 15
        'Draw artwork in corresponding cell if image type matches type in Type column.\n\n        Arguments:\n        images -- The images to be drawn.\n        col -- Column in which images are to be drawn. Can be _new_cover_col or _existing_cover_col.\n        '
        row = 0
        row_count = self.artwork_table.rowCount()
        missing_pixmap = QtGui.QPixmap(':/images/image-missing.png')
        for image in images:
            while row != row_count:
                image_type = self.artwork_table.item(row, self.artwork_table._type_col)
                if image_type and image_type.data(QtCore.Qt.ItemDataRole.UserRole) == image.types_as_string():
                    break
                row += 1
            if row == row_count:
                continue
            data = None
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, image)
            pixmap = QtGui.QPixmap()
            try:
                if image.thumbnail:
                    try:
                        data = image.thumbnail.data
                    except CoverArtImageIOError as e:
                        log.warning(e)
                else:
                    data = image.data
                if data:
                    pixmap.loadFromData(data)
                    item.setToolTip(_('Double-click to open in external viewer\nTemporary file: %(tempfile)s\nSource: %(sourcefile)s') % {'tempfile': image.tempfile_filename, 'sourcefile': image.source})
            except CoverArtImageIOError:
                log.error(traceback.format_exc())
                pixmap = missing_pixmap
                item.setToolTip(_('Missing temporary file: %(tempfile)s\nSource: %(sourcefile)s') % {'tempfile': image.tempfile_filename, 'sourcefile': image.source})
            infos = []
            if image.comment:
                infos.append(image.comment)
            infos.append('%s (%s)' % (bytes2human.decimal(image.datalength), bytes2human.binary(image.datalength)))
            if image.width and image.height:
                infos.append('%d x %d' % (image.width, image.height))
            infos.append(image.mimetype)
            img_wgt = ArtworkCoverWidget(pixmap=pixmap, text='\n'.join(infos))
            self.artwork_table.setCellWidget(row, col, img_wgt)
            self.artwork_table.setItem(row, col, item)
            row += 1

    def _display_artwork_type(self):
        if False:
            while True:
                i = 10
        'Display image type in Type column.\n        If both existing covers and new covers are to be displayed, take union of both cover types list.\n        '
        types = [image.types_as_string() for image in self.images]
        if self.display_existing_artwork:
            existing_types = [image.types_as_string() for image in self.existing_images]
            types = union_sorted_lists(types, existing_types)
            pixmap_arrow = QtGui.QPixmap(':/images/arrow.png')
        else:
            pixmap_arrow = None
        for (row, artwork_type) in enumerate(types):
            self.artwork_table.insertRow(row)
            item = QtWidgets.QTableWidgetItem()
            item.setData(QtCore.Qt.ItemDataRole.UserRole, artwork_type)
            type_wgt = ArtworkCoverWidget(pixmap=pixmap_arrow, text=artwork_type)
            self.artwork_table.setCellWidget(row, self.artwork_table._type_col, type_wgt)
            self.artwork_table.setItem(row, self.artwork_table._type_col, item)

    def _display_artwork_tab(self):
        if False:
            while True:
                i = 10
        if not self.images:
            self.tab_hide(self.ui.artwork_tab)
        self._display_artwork_type()
        self._display_artwork(self.images, self.artwork_table._new_cover_col)
        if self.existing_images:
            self._display_artwork(self.existing_images, self.artwork_table._existing_cover_col)
        self.artwork_table.itemDoubleClicked.connect(self.show_item)
        self.artwork_table.verticalHeader().resizeSections(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)

    def tab_hide(self, widget):
        if False:
            return 10
        tab = self.ui.tabWidget
        index = tab.indexOf(widget)
        tab.removeTab(index)

    def show_item(self, item):
        if False:
            i = 10
            return i + 15
        data = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if isinstance(data, str):
            return
        filename = data.tempfile_filename
        if filename:
            open_local_path(filename)

def format_file_info(file_):
    if False:
        while True:
            i = 10
    info = []
    info.append((_('Filename:'), file_.filename))
    if '~format' in file_.orig_metadata:
        info.append((_('Format:'), file_.orig_metadata['~format']))
    try:
        size = os.path.getsize(encode_filename(file_.filename))
        sizestr = '%s (%s)' % (bytes2human.decimal(size), bytes2human.binary(size))
        info.append((_('Size:'), sizestr))
    except BaseException:
        pass
    if file_.orig_metadata.length:
        info.append((_('Length:'), format_time(file_.orig_metadata.length)))
    if '~bitrate' in file_.orig_metadata:
        info.append((_('Bitrate:'), '%s kbps' % file_.orig_metadata['~bitrate']))
    if '~sample_rate' in file_.orig_metadata:
        info.append((_('Sample rate:'), '%s Hz' % file_.orig_metadata['~sample_rate']))
    if '~bits_per_sample' in file_.orig_metadata:
        info.append((_('Bits per sample:'), str(file_.orig_metadata['~bits_per_sample'])))
    if '~channels' in file_.orig_metadata:
        ch = file_.orig_metadata['~channels']
        if ch == '1':
            ch = _('Mono')
        elif ch == '2':
            ch = _('Stereo')
        info.append((_('Channels:'), ch))
    return '<br/>'.join(map(lambda i: '<b>%s</b> %s' % (escape(i[0]), escape(i[1])), info))

def format_tracklist(cluster):
    if False:
        return 10
    info = []
    info.append('<b>%s</b> %s' % (_('Album:'), escape(cluster.metadata['album'])))
    info.append('<b>%s</b> %s' % (_('Artist:'), escape(cluster.metadata['albumartist'])))
    info.append('')
    TrackListItem = namedtuple('TrackListItem', 'number, title, artist, length')
    tracklists = defaultdict(list)
    if isinstance(cluster, Album):
        objlist = cluster.tracks
    else:
        objlist = cluster.iterfiles(False)
    for obj_ in objlist:
        m = obj_.metadata
        artist = m['artist'] or m['albumartist'] or cluster.metadata['albumartist']
        track = TrackListItem(m['tracknumber'], m['title'], artist, m['~length'])
        tracklists[obj_.discnumber].append(track)

    def sorttracknum(track):
        if False:
            while True:
                i = 10
        try:
            return int(track.number)
        except ValueError:
            try:
                m = re.search('^\\d+', track.number)
                return int(m.group(0))
            except AttributeError:
                return 0
    ndiscs = len(tracklists)
    for discnumber in sorted(tracklists):
        tracklist = tracklists[discnumber]
        if ndiscs > 1:
            info.append('<b>%s</b>' % (_('Disc %d') % discnumber))
        lines = ['%s %s - %s (%s)' % item for item in sorted(tracklist, key=sorttracknum)]
        info.append('<b>%s</b><br />%s<br />' % (_('Tracklist:'), '<br />'.join((escape(s).replace(' ', '&nbsp;') for s in lines))))
    return '<br/>'.join(info)

def text_as_html(text):
    if False:
        return 10
    return '<br />'.join(escape(str(text)).replace('\t', ' ').replace(' ', '&nbsp;').splitlines())

class FileInfoDialog(InfoDialog):

    def __init__(self, file_, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(file_, parent)
        self.setWindowTitle(_('Info') + ' - ' + file_.base_filename)

    def _display_info_tab(self):
        if False:
            i = 10
            return i + 15
        file_ = self.obj
        text = format_file_info(file_)
        self.ui.info.setText(text)

class AlbumInfoDialog(InfoDialog):

    def __init__(self, album, parent=None):
        if False:
            print('Hello World!')
        super().__init__(album, parent)
        self.setWindowTitle(_('Album Info'))

    def _display_info_tab(self):
        if False:
            while True:
                i = 10
        album = self.obj
        if album._tracks_loaded:
            self.ui.info.setText(format_tracklist(album))
        else:
            self.tab_hide(self.ui.info_tab)

class TrackInfoDialog(InfoDialog):

    def __init__(self, track, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(track, parent)
        self.setWindowTitle(_('Track Info'))

    def _display_info_tab(self):
        if False:
            i = 10
            return i + 15
        track = self.obj
        tab = self.ui.info_tab
        tabWidget = self.ui.tabWidget
        tab_index = tabWidget.indexOf(tab)
        if track.num_linked_files == 0:
            tabWidget.setTabText(tab_index, _('&Info'))
            self.tab_hide(tab)
            return
        tabWidget.setTabText(tab_index, _('&Info'))
        text = ngettext('%i file in this track', '%i files in this track', track.num_linked_files) % track.num_linked_files
        info_files = [format_file_info(file_) for file_ in track.files]
        text += '<hr />' + '<hr />'.join(info_files)
        self.ui.info.setText(text)

class ClusterInfoDialog(InfoDialog):

    def __init__(self, cluster, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(cluster, parent)
        self.setWindowTitle(_('Cluster Info'))

    def _display_info_tab(self):
        if False:
            i = 10
            return i + 15
        tab = self.ui.info_tab
        tabWidget = self.ui.tabWidget
        tab_index = tabWidget.indexOf(tab)
        tabWidget.setTabText(tab_index, _('&Info'))
        self.ui.info.setText(format_tracklist(self.obj))