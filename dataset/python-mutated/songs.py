import logging
from enum import IntEnum, Enum
from functools import partial
from PyQt5.QtCore import pyqtSignal, Qt, QVariant, QEvent, QAbstractTableModel, QAbstractListModel, QModelIndex, QSize, QRect, QPoint, QPointF, QSortFilterProxyModel
from PyQt5.QtGui import QPainter, QPalette, QMouseEvent, QPolygonF
from PyQt5.QtWidgets import QAction, QFrame, QHBoxLayout, QAbstractItemView, QHeaderView, QPushButton, QTableView, QWidget, QMenu, QListView, QStyle, QSizePolicy, QStyledItemDelegate
from feeluown.utils import aio
from feeluown.utils.dispatch import Signal
from feeluown.library import ModelState, ModelFlags
from feeluown.models import ModelExistence
from feeluown.gui.mimedata import ModelMimeData
from feeluown.gui.helpers import ItemViewNoScrollMixin, ReaderFetchMoreMixin
logger = logging.getLogger(__name__)

class ColumnsMode(Enum):
    """
    Different mode show different columns.
    """
    normal = 'normal'
    album = 'album'
    artist = 'artist'
    playlist = 'playlist'

class Column(IntEnum):
    index = 0
    song = 1
    source = 5
    duration = 4
    artist = 2
    album = 3

class ColumnsConfig:
    """
    TableView use sizeHint to control the width of each row. In order to make
    size hint taking effects, resizeMode should be se to ResizeToContents.
    """

    def __init__(self, widths):
        if False:
            while True:
                i = 10
        self._widths = widths

    def set_width_ratio(self, column, ratio):
        if False:
            print('Hello World!')
        self._widths[column] = ratio

    def get_width(self, column, table_width):
        if False:
            return 10
        width_index = 36
        if column == Column.index:
            return width_index
        width = table_width - width_index
        ratio = self._widths[column]
        return int(width * ratio)

    @classmethod
    def default(cls):
        if False:
            while True:
                i = 10
        widths = {Column.song: 0.4, Column.artist: 0.15, Column.album: 0.25, Column.duration: 0.1, Column.source: 0.15}
        return cls(widths=widths)

def get_column_name(column):
    if False:
        i = 10
        return i + 15
    return {Column.index: '', Column.song: '歌曲标题', Column.artist: '歌手', Column.album: '专辑', Column.duration: '时长', Column.source: '来源'}[column]

class SongListModel(QAbstractListModel, ReaderFetchMoreMixin):

    def __init__(self, reader, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._reader = reader
        self._fetch_more_step = 10
        self._items = []
        self._is_fetching = False

    def rowCount(self, _=QModelIndex()):
        if False:
            while True:
                i = 10
        return len(self._items)

    def flags(self, index):
        if False:
            print('Hello World!')
        if not index.isValid():
            return 0
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        return flags

    def data(self, index, role=Qt.DisplayRole):
        if False:
            i = 10
            return i + 15
        row = index.row()
        if role == Qt.DisplayRole:
            return self._items[row].title_display
        elif role == Qt.UserRole:
            return self._items[row]
        return None

class SongListDelegate(QStyledItemDelegate):

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        super().__init__(parent=parent)
        self.number_rect_x = 20
        self.play_btn_pressed = False

    def paint(self, painter, option, index):
        if False:
            while True:
                i = 10
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        song = index.data(Qt.UserRole)
        top = option.rect.top()
        bottom = option.rect.bottom()
        no_x = self.number_rect_x
        duration_width = 100
        artists_name_width = 150
        duration_x = option.rect.topRight().x() - duration_width
        duration_rect = QRect(QPoint(duration_x, top), option.rect.bottomRight())
        painter.drawText(duration_rect, Qt.AlignRight | Qt.AlignVCenter, song.duration_ms_display)
        artists_name_x = option.rect.topRight().x() - duration_width - artists_name_width
        artists_name_rect = QRect(QPoint(artists_name_x, top), QPoint(duration_x, bottom))
        painter.drawText(artists_name_rect, Qt.AlignRight | Qt.AlignVCenter, song.artists_name_display)
        no_bottom_right = QPoint(no_x, bottom)
        no_rect = QRect(option.rect.topLeft(), no_bottom_right)
        if option.state & QStyle.State_MouseOver:
            painter.drawText(no_rect, Qt.AlignLeft | Qt.AlignVCenter, '►')
        else:
            painter.drawText(no_rect, Qt.AlignLeft | Qt.AlignVCenter, str(index.row() + 1))
        title_rect = QRect(QPoint(no_x, top), QPoint(artists_name_x, bottom))
        painter.drawText(title_rect, Qt.AlignVCenter, song.title_display)
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if False:
            return 10
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            no_bottom_right = QPoint(self.number_rect_x, option.rect.bottom())
            no_rect = QRect(option.rect.topLeft(), no_bottom_right)
            mouse_event = QMouseEvent(event)
            if no_rect.contains(mouse_event.pos()):
                if event.type() == QEvent.MouseButtonPress:
                    self.play_btn_pressed = True
                if event.type() == QEvent.MouseButtonRelease:
                    if self.play_btn_pressed is True:
                        parent = self.parent()
                        assert isinstance(parent, SongListView)
                        parent.play_song_needed.emit(index.data(Qt.UserRole))
            if event.type() == QEvent.MouseButtonRelease:
                self.play_btn_pressed = False
        return super().editorEvent(event, model, option, index)

    def sizeHint(self, option, index):
        if False:
            for i in range(10):
                print('nop')
        size = super().sizeHint(option, index)
        if index.isValid():
            return QSize(size.width(), 36)
        return size

class SongListView(ItemViewNoScrollMixin, QListView):
    play_song_needed = pyqtSignal([object])

    def __init__(self, parent=None, **kwargs):
        if False:
            return 10
        super().__init__(parent=parent, **kwargs)
        self.delegate = SongListDelegate(self)
        self.setItemDelegate(self.delegate)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMouseTracking(True)
        self.setFrameShape(QFrame.NoFrame)
        self.activated.connect(self._on_activated)

    def _on_activated(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.play_song_needed.emit(index.data(Qt.UserRole))

class BaseSongsTableModel(QAbstractTableModel):

    def __init__(self, source_name_map=None, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.columns_config = ColumnsConfig.default()
        self._items = []
        self._source_name_map = source_name_map or {}

    def update_columns_config(self, columns_config):
        if False:
            print('Hello World!')
        '\n        :param columns: see `create_columns` result.\n        '
        self.columns_config = columns_config

    def removeRows(self, row, count, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        self.beginRemoveRows(parent, row, row + count - 1)
        while count > 0:
            self._items.pop(row)
            count -= 1
        self.endRemoveRows()
        return True

    def flags(self, index):
        if False:
            print('Hello World!')
        no_item_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if index.column() in (Column.index, Column.source, Column.duration):
            return no_item_flags
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsDragEnabled
        song = index.data(Qt.UserRole)
        incomplete = False
        if ModelFlags.v2 & song.meta.flags:
            if song.state in (ModelState.not_exists, ModelState.cant_upgrade):
                incomplete = True
        elif song and song.exists == ModelExistence.no:
            incomplete = True
        if incomplete:
            if index.column() != Column.song:
                flags = no_item_flags
        elif index.column() == Column.album:
            flags |= Qt.ItemIsDragEnabled
        elif index.column() == Column.artist:
            flags |= Qt.ItemIsEditable
        return flags

    def rowCount(self, parent=QModelIndex()):
        if False:
            i = 10
            return i + 15
        return len(self._items)

    def columnCount(self, _=QModelIndex()):
        if False:
            print('Hello World!')
        return 6

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if False:
            for i in range(10):
                print('nop')
        if orientation == Qt.Horizontal:
            if role == Qt.DisplayRole:
                return get_column_name(section)
            elif role == Qt.SizeHintRole and self.parent() is not None:
                height = 25
                parent = self.parent()
                assert isinstance(parent, QWidget)
                w = self.columns_config.get_width(section, parent.width())
                return QSize(w, height)
        elif role == Qt.DisplayRole:
            return section
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignRight
        return QVariant()

    def data(self, index, role=Qt.DisplayRole):
        if False:
            return 10
        if not index.isValid():
            return QVariant()
        if index.row() >= len(self._items) or index.row() < 0:
            return QVariant()
        song = self._items[index.row()]
        if role in (Qt.DisplayRole, Qt.ToolTipRole):
            if role == Qt.ToolTipRole and index.column() not in (Column.song, Column.artist, Column.album, Column.duration):
                return QVariant()
            if index.column() == Column.index:
                return index.row() + 1
            elif index.column() == Column.source:
                name = source = song.source
                return self._source_name_map.get(source, name).strip()
            elif index.column() == Column.song:
                return song.title_display
            elif index.column() == Column.duration:
                return song.duration_ms_display
            elif index.column() == Column.artist:
                return song.artists_name_display
            elif index.column() == Column.album:
                return song.album_name_display
        elif role == Qt.TextAlignmentRole:
            if index.column() == Column.index:
                return Qt.AlignCenter | Qt.AlignVCenter
            elif index.column() == Column.source:
                return Qt.AlignLeft | Qt.AlignBaseline | Qt.AlignVCenter
        elif role == Qt.EditRole:
            return 1
        elif role == Qt.UserRole:
            return song
        return QVariant()

    def mimeData(self, indexes):
        if False:
            while True:
                i = 10
        indexes = list(indexes)
        if len(indexes) > 1:
            index = indexes[0]
            song = index.data(Qt.UserRole)
            return ModelMimeData(song)

class SongsTableModel(BaseSongsTableModel, ReaderFetchMoreMixin):

    def __init__(self, reader, **kwargs):
        if False:
            while True:
                i = 10
        '\n\n        :param songs: 歌曲列表\n        :param songs_g: 歌曲列表生成器（当歌曲列表生成器不为 None 时，忽略 songs 参数）\n        '
        super().__init__(**kwargs)
        self._reader = reader
        self._fetch_more_step = 30
        self._is_fetching = False

    @property
    def reader(self):
        if False:
            return 10
        return self._reader

class SongFilterProxyModel(QSortFilterProxyModel):

    def __init__(self, parent=None, text=''):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.text = text

    def filter_by_text(self, text):
        if False:
            print('Hello World!')
        self.text = text or ''
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if False:
            while True:
                i = 10
        if not self.text:
            return super().filterAcceptsRow(source_row, source_parent)
        source_model = self.sourceModel()
        index = source_model.index(source_row, Column.song, parent=source_parent)
        song = index.data(Qt.UserRole)
        text = self.text.lower()
        ctx = song.title_display.lower() + song.album_name_display.lower() + song.artists_name_display.lower()
        return text in ctx

class ArtistsModel(QAbstractListModel):

    def __init__(self, artists):
        if False:
            return 10
        super().__init__()
        self.artists = artists

    def rowCount(self, parent=QModelIndex()):
        if False:
            for i in range(10):
                print('nop')
        return len(self.artists)

    def data(self, index, role):
        if False:
            return 10
        artist = self.artists[index.row()]
        if role == Qt.DisplayRole:
            return artist.name
        elif role == Qt.UserRole:
            return artist
        elif role == Qt.SizeHintRole:
            return QSize(100, 30)
        return QVariant()

class SongOpsEditor(QWidget):
    """song editor for playlist table view"""

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.download_btn = QPushButton('↧', self)
        self.play_btn = QPushButton('☊', self)
        self._layout = QHBoxLayout(self)
        self._layout.addWidget(self.play_btn)
        self._layout.addWidget(self.download_btn)
        self._layout.setSpacing(0)
        self._layout.setContentsMargins(0, 0, 0, 0)

class ArtistsSelectionView(QListView):

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Dialog | Qt.FramelessWindowHint)
        self.setObjectName('artists_selection_view')

class SongsTableDelegate(QStyledItemDelegate):

    def __init__(self, app, parent):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._app = app
        self.view = parent
        self.row_hovered = None
        self.pressed_cell = None

    def on_row_hovered(self, row):
        if False:
            return 10
        self.row_hovered = row

    def createEditor(self, parent, option, index):
        if False:
            return 10
        if index.column() == Column.artist:
            editor = ArtistsSelectionView(parent)
            editor.clicked.connect(partial(self.commitData.emit, editor))
            editor.move(parent.mapToGlobal(option.rect.bottomLeft()))
            editor.setFixedWidth(option.rect.width())
            return editor

    def setEditorData(self, editor, index):
        if False:
            print('Hello World!')
        super().setEditorData(editor, index)

        def cb(future):
            if False:
                return 10
            try:
                song = future.result()
                artists = song.artists
            except:
                logger.exception('get song.artists failed')
            else:
                model = ArtistsModel(artists)
                editor.setModel(model)
                editor.setCurrentIndex(QModelIndex())
        if index.column() == Column.artist:
            song = index.data(role=Qt.UserRole)
            future = aio.run_fn(self._app.library.song_upgrade, song)
            future.add_done_callback(cb)

    def setModelData(self, editor, model, index):
        if False:
            return 10
        if index.column() == Column.artist:
            index = editor.currentIndex()
            if index.isValid():
                artist = index.data(Qt.UserRole)
                self.view.show_artist_needed.emit(artist)
        super().setModelData(editor, model, index)

    def paint(self, painter, option, index):
        if False:
            return 10
        super().paint(painter, option, index)
        painter.setRenderHint(QPainter.Antialiasing)
        hovered = index.row() == self.row_hovered
        if hovered and index.column() == Column.index:
            painter.save()
            painter.setPen(Qt.NoPen)
            if index.row() % 2 == 0:
                painter.setBrush(option.palette.color(QPalette.Base))
            else:
                painter.setBrush(option.palette.color(QPalette.AlternateBase))
            painter.drawRect(option.rect)
            painter.setBrush(option.palette.color(QPalette.Text))
            triangle_edge = 12
            triangle_height = 10
            painter.translate(2 + option.rect.x() + (option.rect.width() - triangle_height) // 2, option.rect.y() + (option.rect.height() - triangle_edge) // 2)
            triangle = QPolygonF([QPointF(0, 0), QPointF(triangle_height, triangle_edge // 2), QPointF(0, triangle_edge)])
            painter.drawPolygon(triangle)
            painter.restore()
        if hovered:
            painter.save()
            mask_color = option.palette.color(QPalette.Active, QPalette.Text)
            mask_color.setAlpha(20)
            painter.setPen(Qt.NoPen)
            painter.setBrush(mask_color)
            painter.drawRect(option.rect)
            painter.restore()

    def sizeHint(self, option, index):
        if False:
            while True:
                i = 10
        "set proper width for each column\n\n        HELP: If we do not set width here, the column width\n        can be uncertain. I don't know why this would happen,\n        since we have set width for the header.\n        "
        if index.isValid() and self.parent() is not None:
            parent = self.parent()
            assert isinstance(parent, QWidget)
            w = index.model().sourceModel().columns_config.get_width(index.column(), parent.width())
            h = option.rect.height()
            return QSize(w, h)
        return super().sizeHint(option, index)

    def editorEvent(self, event, model, option, index):
        if False:
            return 10
        etype = event.type()
        if etype in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease):
            cell = (index.row(), index.column())
            if etype == QEvent.MouseButtonPress:
                self.pressed_cell = cell
            elif etype == QEvent.MouseButtonRelease:
                if cell == self.pressed_cell and cell[1] == Column.index:
                    parent = self.parent()
                    assert isinstance(parent, SongsTableView)
                    parent.play_song_needed.emit(index.data(Qt.UserRole))
                self.pressed_cell = None
        return super().editorEvent(event, model, option, index)

    def updateEditorGeometry(self, editor, option, index):
        if False:
            print('Hello World!')
        if index.column() != Column.artist:
            super().updateEditorGeometry(editor, option, index)

class SongsTableView(ItemViewNoScrollMixin, QTableView):
    show_artist_needed = pyqtSignal([object])
    show_album_needed = pyqtSignal([object])
    play_song_needed = pyqtSignal([object])
    add_to_playlist_needed = pyqtSignal(list)
    row_hovered = pyqtSignal([object])

    def __init__(self, app, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self._app = app
        self._least_row_count = 6
        self.remove_song_func = None
        self.delegate = SongsTableDelegate(app, self)
        self.setItemDelegate(self.delegate)
        self.about_to_show_menu = Signal()
        self._setup_ui()
        self.row_hovered.connect(self.delegate.on_row_hovered)
        self.entered.connect(lambda index: self.row_hovered.emit(index.row()))

    def _setup_ui(self):
        if False:
            print('Hello World!')
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setAlternatingRowColors(True)
        self.verticalHeader().hide()
        self.horizontalHeader().hide()
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.setMouseTracking(True)
        self.setEditTriggers(QAbstractItemView.SelectedClicked)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setShowGrid(False)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragOnly)

    def setModel(self, model):
        if False:
            i = 10
            return i + 15
        super().setModel(model)
        self.horizontalHeader().setSectionResizeMode(Column.song, QHeaderView.Stretch)

    def set_columns_mode(self, mode):
        if False:
            while True:
                i = 10
        mode = ColumnsMode(mode)
        columns_config = ColumnsConfig.default()
        if mode is ColumnsMode.normal:
            hide_columns = []
        elif mode is ColumnsMode.album:
            hide_columns = [Column.album, Column.source]
            columns_config.set_width_ratio(Column.artist, 0.25)
            columns_config.set_width_ratio(Column.duration, 0.1)
        else:
            hide_columns = [Column.source]
            columns_config.set_width_ratio(Column.artist, 0.2)
            columns_config.set_width_ratio(Column.album, 0.3)
        model = self.model()
        assert isinstance(model, SongFilterProxyModel)
        source = model.sourceModel()
        assert isinstance(source, SongsTableModel)
        source.update_columns_config(columns_config)
        for i in range(0, model.columnCount()):
            if i in hide_columns:
                self.hideColumn(i)
            else:
                self.showColumn(i)

    def show_artists_by_index(self, index):
        if False:
            print('Hello World!')
        self.edit(index)

    def contextMenuEvent(self, event):
        if False:
            return 10
        indexes = self.selectionModel().selectedIndexes()
        if len(indexes) <= 0:
            return
        menu = QMenu()
        add_to_playlist_action = QAction('添加到播放队列', menu)
        add_to_playlist_action.triggered.connect(lambda : self._add_to_playlist(indexes))
        menu.addAction(add_to_playlist_action)
        if self.remove_song_func is not None:
            remove_song_action = QAction('移除歌曲', menu)
            remove_song_action.triggered.connect(lambda : self._remove_by_indexes(indexes))
            menu.addSeparator()
            menu.addAction(remove_song_action)
        model = self.model()
        models = [model.data(index, Qt.UserRole) for index in indexes]

        def add_action(text, callback):
            if False:
                i = 10
                return i + 15
            action = QAction(text, menu)
            menu.addSeparator()
            menu.addAction(action)
            action.triggered.connect(lambda : callback(models))
        self.about_to_show_menu.emit({'add_action': add_action, 'menu': menu, 'models': models})
        menu.exec(event.globalPos())

    def _add_to_playlist(self, indexes):
        if False:
            for i in range(10):
                print('nop')
        model = self.model()
        songs = []
        for index in indexes:
            song = model.data(index, Qt.UserRole)
            songs.append(song)
        self.add_to_playlist_needed.emit(songs)

    def _remove_by_indexes(self, indexes):
        if False:
            for i in range(10):
                print('nop')
        model = self.model()
        songs_to_remove = []
        for index in indexes:
            song = model.data(index, Qt.UserRole)
            if song not in songs_to_remove:
                songs_to_remove.append(song)
        for song in songs_to_remove:
            assert callable(self.remove_song_func)
            self.remove_song_func(song)

    def viewportEvent(self, event):
        if False:
            return 10
        res = super().viewportEvent(event)
        if event.type() == QEvent.Leave:
            self.row_hovered.emit(None)
        return res

    def mouseMoveEvent(self, event):
        if False:
            return 10
        super().mouseMoveEvent(event)
        if not self.indexAt(event.pos()).isValid():
            self.row_hovered.emit(None)