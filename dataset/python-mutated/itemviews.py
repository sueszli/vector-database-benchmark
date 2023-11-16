from collections import defaultdict
from functools import partial
from heapq import heappop, heappush
from PyQt6 import QtCore, QtGui, QtWidgets
from picard import log
from picard.album import Album, NatAlbum
from picard.cluster import Cluster, ClusterList, UnclusteredFiles
from picard.config import BoolOption, Option, get_config
from picard.file import File, FileErrorType
from picard.plugin import ExtensionPoint
from picard.track import NonAlbumTrack, Track
from picard.util import icontheme, iter_files_from_objects, natsort, normpath, restore_method, strxfrm
from picard.ui.collectionmenu import CollectionMenu
from picard.ui.colors import interface_colors
from picard.ui.ratingwidget import RatingWidget
from picard.ui.scriptsmenu import ScriptsMenu
from picard.ui.widgets.tristatesortheaderview import TristateSortHeaderView
COLUMN_ICON_SIZE = 16
COLUMN_ICON_BORDER = 2
ICON_SIZE = QtCore.QSize(COLUMN_ICON_SIZE + COLUMN_ICON_BORDER, COLUMN_ICON_SIZE + COLUMN_ICON_BORDER)

class BaseAction(QtGui.QAction):
    NAME = 'Unknown'
    MENU = []

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__(self.NAME, None)
        self.triggered.connect(self.__callback)

    def __callback(self):
        if False:
            i = 10
            return i + 15
        objs = self.tagger.window.selected_objects
        self.callback(objs)

    def callback(self, objs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError
_album_actions = ExtensionPoint(label='album_actions')
_cluster_actions = ExtensionPoint(label='cluster_actions')
_clusterlist_actions = ExtensionPoint(label='clusterlist_actions')
_track_actions = ExtensionPoint(label='track_actions')
_file_actions = ExtensionPoint(label='file_actions')

def register_album_action(action):
    if False:
        print('Hello World!')
    _album_actions.register(action.__module__, action)

def register_cluster_action(action):
    if False:
        while True:
            i = 10
    _cluster_actions.register(action.__module__, action)

def register_clusterlist_action(action):
    if False:
        while True:
            i = 10
    _clusterlist_actions.register(action.__module__, action)

def register_track_action(action):
    if False:
        i = 10
        return i + 15
    _track_actions.register(action.__module__, action)

def register_file_action(action):
    if False:
        for i in range(10):
            print('nop')
    _file_actions.register(action.__module__, action)

def get_match_color(similarity, basecolor):
    if False:
        while True:
            i = 10
    c1 = (basecolor.red(), basecolor.green(), basecolor.blue())
    c2 = (223, 125, 125)
    return QtGui.QColor(int(c2[0] + (c1[0] - c2[0]) * similarity), int(c2[1] + (c1[1] - c2[1]) * similarity), int(c2[2] + (c1[2] - c2[2]) * similarity))

class MainPanel(QtWidgets.QSplitter):
    options = []
    columns = [(N_('Title'), 'title'), (N_('Length'), '~length'), (N_('Artist'), 'artist'), (N_('Album Artist'), 'albumartist'), (N_('Composer'), 'composer'), (N_('Album'), 'album'), (N_('Disc Subtitle'), 'discsubtitle'), (N_('Track No.'), 'tracknumber'), (N_('Disc No.'), 'discnumber'), (N_('Catalog No.'), 'catalognumber'), (N_('Barcode'), 'barcode'), (N_('Media'), 'media'), (N_('Genre'), 'genre'), (N_('Fingerprint status'), '~fingerprint'), (N_('Date'), 'date'), (N_('Original Release Date'), 'originaldate'), (N_('Release Date'), 'releasedate'), (N_('Cover'), 'covercount')]
    _column_indexes = {column[1]: i for (i, column) in enumerate(columns)}
    TITLE_COLUMN = _column_indexes['title']
    TRACKNUMBER_COLUMN = _column_indexes['tracknumber']
    DISCNUMBER_COLUMN = _column_indexes['discnumber']
    LENGTH_COLUMN = _column_indexes['~length']
    FINGERPRINT_COLUMN = _column_indexes['~fingerprint']
    NAT_SORT_COLUMNS = [_column_indexes['title'], _column_indexes['album'], _column_indexes['discsubtitle'], _column_indexes['tracknumber'], _column_indexes['discnumber'], _column_indexes['catalognumber']]

    def __init__(self, window, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.setChildrenCollapsible(False)
        self.window = window
        self.create_icons()
        self._views = [FileTreeView(window, self), AlbumTreeView(window, self)]
        self._selected_view = self._views[0]
        self._ignore_selection_changes = False

        def _view_update_selection(view):
            if False:
                return 10
            if not self._ignore_selection_changes:
                self._ignore_selection_changes = True
                self._update_selection(view)
                self._ignore_selection_changes = False
        for view in self._views:
            view.itemSelectionChanged.connect(partial(_view_update_selection, view))
        TreeItem.window = window
        TreeItem.base_color = self.palette().base().color()
        TreeItem.text_color = self.palette().text().color()
        TreeItem.text_color_secondary = self.palette().brush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text).color()
        TrackItem.track_colors = defaultdict(lambda : TreeItem.text_color, {File.NORMAL: interface_colors.get_qcolor('entity_saved'), File.CHANGED: TreeItem.text_color, File.PENDING: interface_colors.get_qcolor('entity_pending'), File.ERROR: interface_colors.get_qcolor('entity_error')})
        FileItem.file_colors = defaultdict(lambda : TreeItem.text_color, {File.NORMAL: TreeItem.text_color, File.CHANGED: TreeItem.text_color, File.PENDING: interface_colors.get_qcolor('entity_pending'), File.ERROR: interface_colors.get_qcolor('entity_error')})

    def set_processing(self, processing=True):
        if False:
            for i in range(10):
                print('nop')
        self._ignore_selection_changes = processing

    def tab_order(self, tab_order, before, after):
        if False:
            while True:
                i = 10
        prev = before
        for view in self._views:
            tab_order(prev, view)
            prev = view
        tab_order(prev, after)

    def save_state(self):
        if False:
            while True:
                i = 10
        for view in self._views:
            view.save_state()

    def create_icons(self):
        if False:
            i = 10
            return i + 15
        if hasattr(QtWidgets.QStyle, 'SP_DirIcon'):
            ClusterItem.icon_dir = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DirIcon)
        else:
            ClusterItem.icon_dir = icontheme.lookup('folder', icontheme.ICON_SIZE_MENU)
        AlbumItem.icon_cd = icontheme.lookup('media-optical', icontheme.ICON_SIZE_MENU)
        AlbumItem.icon_cd_modified = icontheme.lookup('media-optical-modified', icontheme.ICON_SIZE_MENU)
        AlbumItem.icon_cd_saved = icontheme.lookup('media-optical-saved', icontheme.ICON_SIZE_MENU)
        AlbumItem.icon_cd_saved_modified = icontheme.lookup('media-optical-saved-modified', icontheme.ICON_SIZE_MENU)
        AlbumItem.icon_error = icontheme.lookup('media-optical-error', icontheme.ICON_SIZE_MENU)
        TrackItem.icon_audio = QtGui.QIcon(':/images/track-audio.png')
        TrackItem.icon_video = QtGui.QIcon(':/images/track-video.png')
        TrackItem.icon_data = QtGui.QIcon(':/images/track-data.png')
        TrackItem.icon_error = icontheme.lookup('dialog-error', icontheme.ICON_SIZE_MENU)
        FileItem.icon_file = QtGui.QIcon(':/images/file.png')
        FileItem.icon_file_pending = QtGui.QIcon(':/images/file-pending.png')
        FileItem.icon_error = icontheme.lookup('dialog-error', icontheme.ICON_SIZE_MENU)
        FileItem.icon_error_not_found = icontheme.lookup('error-not-found', icontheme.ICON_SIZE_MENU)
        FileItem.icon_error_no_access = icontheme.lookup('error-no-access', icontheme.ICON_SIZE_MENU)
        FileItem.icon_saved = QtGui.QIcon(':/images/track-saved.png')
        FileItem.icon_fingerprint = icontheme.lookup('fingerprint', icontheme.ICON_SIZE_MENU)
        FileItem.icon_fingerprint_gray = icontheme.lookup('fingerprint-gray', icontheme.ICON_SIZE_MENU)
        FileItem.match_icons = [QtGui.QIcon(':/images/match-50.png'), QtGui.QIcon(':/images/match-60.png'), QtGui.QIcon(':/images/match-70.png'), QtGui.QIcon(':/images/match-80.png'), QtGui.QIcon(':/images/match-90.png'), QtGui.QIcon(':/images/match-100.png')]
        FileItem.match_icons_info = [N_('Bad match'), N_('Poor match'), N_('Ok match'), N_('Good match'), N_('Great match'), N_('Excellent match')]
        FileItem.match_pending_icons = [QtGui.QIcon(':/images/match-pending-50.png'), QtGui.QIcon(':/images/match-pending-60.png'), QtGui.QIcon(':/images/match-pending-70.png'), QtGui.QIcon(':/images/match-pending-80.png'), QtGui.QIcon(':/images/match-pending-90.png'), QtGui.QIcon(':/images/match-pending-100.png')]
        self.icon_plugins = icontheme.lookup('applications-system', icontheme.ICON_SIZE_MENU)

    def _update_selection(self, selected_view):
        if False:
            for i in range(10):
                print('nop')
        for view in self._views:
            if view != selected_view:
                view.clearSelection()
            else:
                self._selected_view = view
                self.window.update_selection([item.obj for item in view.selectedItems()])

    def update_current_view(self):
        if False:
            return 10
        self._update_selection(self._selected_view)

    def remove(self, objects):
        if False:
            for i in range(10):
                print('nop')
        self._ignore_selection_changes = True
        self.tagger.remove(objects)
        self._ignore_selection_changes = False
        view = self._selected_view
        index = view.currentIndex()
        if index.isValid():
            view.setCurrentIndex(index)
        else:
            self.update_current_view()

    def set_sorting(self, sort=True):
        if False:
            for i in range(10):
                print('nop')
        for view in self._views:
            view.setSortingEnabled(sort)

    def select_object(self, obj):
        if False:
            return 10
        item = obj.item
        for view in self._views:
            if view.indexFromItem(item).isValid():
                view.setCurrentItem(item)
                self._update_selection(view)
                break

def paint_column_icon(painter, rect, icon):
    if False:
        return 10
    if not icon:
        return
    size = COLUMN_ICON_SIZE
    padding_h = COLUMN_ICON_BORDER
    padding_v = (rect.height() - size) // 2
    target_rect = QtCore.QRect(rect.x() + padding_h, rect.y() + padding_v, size, size)
    painter.drawPixmap(target_rect, icon.pixmap(size, size))

class ConfigurableColumnsHeader(TristateSortHeaderView):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self._visible_columns = set([0])
        self.setSectionsMovable(True)
        self.setStretchLastSection(True)
        self.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setSectionsClickable(False)
        self.sortIndicatorChanged.connect(self.on_sort_indicator_changed)
        self.setSortIndicator(-1, QtCore.Qt.SortOrder.AscendingOrder)
        self.setDefaultSectionSize(100)

    def show_column(self, column, show):
        if False:
            while True:
                i = 10
        if column == 0:
            return
        self.parent().setColumnHidden(column, not show)
        if show:
            if self.sectionSize(column) == 0:
                self.resizeSection(column, self.defaultSectionSize())
            self._visible_columns.add(column)
            if column == MainPanel.FINGERPRINT_COLUMN:
                self.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Fixed)
                self.resizeSection(column, COLUMN_ICON_SIZE)
            else:
                self.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeMode.Interactive)
        elif column in self._visible_columns:
            self._visible_columns.remove(column)

    def update_visible_columns(self, columns):
        if False:
            return 10
        for (i, column) in enumerate(MainPanel.columns):
            self.show_column(i, i in columns)

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        menu = QtWidgets.QMenu(self)
        parent = self.parent()
        for (i, column) in enumerate(MainPanel.columns):
            if i == 0:
                continue
            action = QtGui.QAction(_(column[0]), parent)
            action.setCheckable(True)
            action.setChecked(i in self._visible_columns)
            action.setEnabled(not self.is_locked)
            action.triggered.connect(partial(self.show_column, i))
            menu.addAction(action)
        menu.addSeparator()
        restore_action = QtGui.QAction(_('Restore default columns'), parent)
        restore_action.setEnabled(not self.is_locked)
        restore_action.triggered.connect(self.restore_defaults)
        menu.addAction(restore_action)
        lock_action = QtGui.QAction(_('Lock columns'), parent)
        lock_action.setCheckable(True)
        lock_action.setChecked(self.is_locked)
        lock_action.toggled.connect(self.lock)
        menu.addAction(lock_action)
        menu.exec(event.globalPos())
        event.accept()

    def restore_defaults(self):
        if False:
            while True:
                i = 10
        self.parent().restore_default_columns()

    def paintSection(self, painter, rect, index):
        if False:
            print('Hello World!')
        if index == MainPanel.FINGERPRINT_COLUMN:
            painter.save()
            super().paintSection(painter, rect, index)
            painter.restore()
            paint_column_icon(painter, rect, FileItem.icon_fingerprint_gray)
        else:
            super().paintSection(painter, rect, index)

    def on_sort_indicator_changed(self, index, order):
        if False:
            for i in range(10):
                print('nop')
        if index == MainPanel.FINGERPRINT_COLUMN:
            self.setSortIndicator(-1, QtCore.Qt.SortOrder.AscendingOrder)

    def lock(self, is_locked):
        if False:
            return 10
        super().lock(is_locked)
        column_index = MainPanel.FINGERPRINT_COLUMN
        if not self.is_locked and self.count() > column_index:
            self.setSectionResizeMode(column_index, QtWidgets.QHeaderView.ResizeMode.Fixed)

class BaseTreeView(QtWidgets.QTreeWidget):

    def __init__(self, window, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.setHeader(ConfigurableColumnsHeader(self))
        self.window = window
        self.panel = parent
        self._move_to_multi_tracks = True
        self.setHeaderLabels([_(h) if n != '~fingerprint' else '' for (h, n) in MainPanel.columns])
        self.restore_state()
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setSortingEnabled(True)
        self.expand_all_action = QtGui.QAction(_('&Expand all'), self)
        self.expand_all_action.triggered.connect(self.expandAll)
        self.collapse_all_action = QtGui.QAction(_('&Collapse all'), self)
        self.collapse_all_action.triggered.connect(self.collapseAll)
        self.select_all_action = QtGui.QAction(_('Select &all'), self)
        self.select_all_action.triggered.connect(self.selectAll)
        self.select_all_action.setShortcut(QtGui.QKeySequence(_('Ctrl+A')))
        self.doubleClicked.connect(self.activate_item)
        self.setUniformRowHeights(True)

    def contextMenuEvent(self, event):
        if False:
            i = 10
            return i + 15
        item = self.itemAt(event.pos())
        if not item:
            return
        config = get_config()
        obj = item.obj
        plugin_actions = None
        can_view_info = self.window.view_info_action.isEnabled()
        menu = QtWidgets.QMenu(self)
        if isinstance(obj, Track):
            if can_view_info:
                menu.addAction(self.window.view_info_action)
            plugin_actions = list(_track_actions)
            if obj.num_linked_files == 1:
                menu.addAction(self.window.play_file_action)
                menu.addAction(self.window.open_folder_action)
                menu.addAction(self.window.track_search_action)
                plugin_actions.extend(_file_actions)
            menu.addAction(self.window.browser_lookup_action)
            if obj.num_linked_files > 0:
                menu.addAction(self.window.generate_fingerprints_action)
            menu.addSeparator()
            if isinstance(obj, NonAlbumTrack):
                menu.addAction(self.window.refresh_action)
        elif isinstance(obj, Cluster):
            if can_view_info:
                menu.addAction(self.window.view_info_action)
            menu.addAction(self.window.browser_lookup_action)
            if self.window.submit_cluster_action:
                menu.addAction(self.window.submit_cluster_action)
            menu.addSeparator()
            menu.addAction(self.window.autotag_action)
            menu.addAction(self.window.analyze_action)
            if isinstance(obj, UnclusteredFiles):
                menu.addAction(self.window.cluster_action)
            else:
                menu.addAction(self.window.album_search_action)
            menu.addAction(self.window.generate_fingerprints_action)
            plugin_actions = list(_cluster_actions)
        elif isinstance(obj, ClusterList):
            menu.addAction(self.window.autotag_action)
            menu.addAction(self.window.analyze_action)
            menu.addAction(self.window.generate_fingerprints_action)
            plugin_actions = list(_clusterlist_actions)
        elif isinstance(obj, File):
            if can_view_info:
                menu.addAction(self.window.view_info_action)
            menu.addAction(self.window.play_file_action)
            menu.addAction(self.window.open_folder_action)
            menu.addAction(self.window.browser_lookup_action)
            if self.window.submit_file_as_recording_action:
                menu.addAction(self.window.submit_file_as_recording_action)
            if self.window.submit_file_as_release_action:
                menu.addAction(self.window.submit_file_as_release_action)
            menu.addSeparator()
            menu.addAction(self.window.autotag_action)
            menu.addAction(self.window.analyze_action)
            menu.addAction(self.window.track_search_action)
            menu.addAction(self.window.generate_fingerprints_action)
            plugin_actions = list(_file_actions)
        elif isinstance(obj, Album):
            if can_view_info:
                menu.addAction(self.window.view_info_action)
            menu.addAction(self.window.browser_lookup_action)
            if obj.get_num_total_files() > 0:
                menu.addAction(self.window.generate_fingerprints_action)
            menu.addSeparator()
            menu.addAction(self.window.refresh_action)
            plugin_actions = list(_album_actions)
        menu.addAction(self.window.save_action)
        menu.addAction(self.window.remove_action)
        bottom_separator = False
        if isinstance(obj, Album) and (not isinstance(obj, NatAlbum)) and obj.loaded:
            releases_menu = QtWidgets.QMenu(_('&Other versions'), menu)
            menu.addSeparator()
            menu.addMenu(releases_menu)
            loading = releases_menu.addAction(_('Loading…'))
            loading.setDisabled(True)
            action_more = releases_menu.addAction(_('Show &more details…'))
            action_more.triggered.connect(self.window.album_other_versions_action.trigger)
            bottom_separator = True
            if len(self.selectedItems()) == 1 and obj.release_group:

                def _add_other_versions():
                    if False:
                        i = 10
                        return i + 15
                    releases_menu.removeAction(loading)
                    releases_menu.removeAction(action_more)
                    heading = releases_menu.addAction(obj.release_group.version_headings)
                    heading.setDisabled(True)
                    font = heading.font()
                    font.setBold(True)
                    heading.setFont(font)
                    versions = obj.release_group.versions
                    album_tracks_count = obj.get_num_total_files() or len(obj.tracks)
                    preferred_countries = set(config.setting['preferred_release_countries'])
                    preferred_formats = set(config.setting['preferred_release_formats'])
                    (ORDER_BEFORE, ORDER_AFTER) = (0, 1)
                    alternatives = []
                    for version in versions:
                        trackmatch = countrymatch = formatmatch = ORDER_BEFORE
                        if version['totaltracks'] != album_tracks_count:
                            trackmatch = ORDER_AFTER
                        if preferred_countries:
                            countries = set(version['countries'])
                            if not countries or not countries.intersection(preferred_countries):
                                countrymatch = ORDER_AFTER
                        if preferred_formats:
                            formats = set(version['formats'])
                            if not formats or not formats.intersection(preferred_formats):
                                formatmatch = ORDER_AFTER
                        group = (trackmatch, countrymatch, formatmatch)
                        heappush(alternatives, (group, version['name'], version['id']))
                    prev_group = None
                    while alternatives:
                        (group, action_text, release_id) = heappop(alternatives)
                        if group != prev_group:
                            if prev_group is not None:
                                releases_menu.addSeparator()
                            prev_group = group
                        action = releases_menu.addAction(action_text)
                        action.setCheckable(True)
                        if obj.id == release_id:
                            action.setChecked(True)
                        action.triggered.connect(partial(obj.switch_release_version, release_id))
                    versions_count = len(versions)
                    if versions_count > 1:
                        releases_menu.setTitle(_('&Other versions (%d)') % versions_count)
                    releases_menu.addSeparator()
                    action = releases_menu.addAction(action_more)
                if obj.release_group.loaded:
                    _add_other_versions()
                else:
                    obj.release_group.load_versions(_add_other_versions)
                releases_menu.setEnabled(True)
            else:
                releases_menu.setEnabled(False)
        if config.setting['enable_ratings'] and len(self.window.selected_objects) == 1 and isinstance(obj, Track):
            menu.addSeparator()
            action = QtWidgets.QWidgetAction(menu)
            action.setDefaultWidget(RatingWidget(menu, obj))
            menu.addAction(action)
            menu.addSeparator()
        selected_albums = [a for a in self.window.selected_objects if type(a) == Album]
        if selected_albums:
            if not bottom_separator:
                menu.addSeparator()
            menu.addMenu(CollectionMenu(selected_albums, _('Collections'), menu))
        scripts = config.setting['list_of_scripts']
        if plugin_actions or scripts:
            menu.addSeparator()
        if plugin_actions:
            plugin_menu = QtWidgets.QMenu(_('P&lugins'), menu)
            plugin_menu.setIcon(self.panel.icon_plugins)
            menu.addMenu(plugin_menu)
            plugin_menus = {}
            for action in plugin_actions:
                action_menu = plugin_menu
                for index in range(1, len(action.MENU) + 1):
                    key = tuple(action.MENU[:index])
                    if key in plugin_menus:
                        action_menu = plugin_menus[key]
                    else:
                        action_menu = plugin_menus[key] = action_menu.addMenu(key[-1])
                action_menu.addAction(action)
        if scripts:
            scripts_menu = ScriptsMenu(scripts, _('&Run scripts'), menu)
            scripts_menu.setIcon(self.panel.icon_plugins)
            menu.addMenu(scripts_menu)
        if isinstance(obj, Cluster) or isinstance(obj, ClusterList) or isinstance(obj, Album):
            menu.addSeparator()
            menu.addAction(self.expand_all_action)
            menu.addAction(self.collapse_all_action)
        menu.addAction(self.select_all_action)
        menu.exec(event.globalPos())
        event.accept()

    @restore_method
    def restore_state(self):
        if False:
            while True:
                i = 10
        config = get_config()
        self._restore_state(config.persist[self.header_state.name])
        self.header().lock(config.persist[self.header_locked.name])

    def save_state(self):
        if False:
            while True:
                i = 10
        config = get_config()
        config.persist[self.header_state.name] = self.header().saveState()
        config.persist[self.header_locked.name] = self.header().is_locked

    def restore_default_columns(self):
        if False:
            for i in range(10):
                print('nop')
        self._restore_state(None)

    def _restore_state(self, header_state):
        if False:
            while True:
                i = 10
        header = self.header()
        if header_state:
            header.restoreState(header_state)
            for i in range(0, self.columnCount()):
                header.show_column(i, not self.isColumnHidden(i))
        else:
            header.update_visible_columns([0, 1, 2])
            for (i, size) in enumerate([250, 50, 100]):
                header.resizeSection(i, size)
            self.sortByColumn(-1, QtCore.Qt.SortOrder.AscendingOrder)

    def supportedDropActions(self):
        if False:
            for i in range(10):
                print('nop')
        return QtCore.Qt.DropAction.CopyAction | QtCore.Qt.DropAction.MoveAction

    def mimeTypes(self):
        if False:
            i = 10
            return i + 15
        'List of MIME types accepted by this view.'
        return ['text/uri-list', 'application/picard.album-list']

    def dragEnterEvent(self, event):
        if False:
            while True:
                i = 10
        super().dragEnterEvent(event)
        self._handle_external_drag(event)

    def dragMoveEvent(self, event):
        if False:
            while True:
                i = 10
        super().dragMoveEvent(event)
        self._handle_external_drag(event)

    def _handle_external_drag(self, event):
        if False:
            for i in range(10):
                print('nop')
        if event.isAccepted() and (not event.source() or event.mimeData().hasUrls()):
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()

    def startDrag(self, supportedActions):
        if False:
            print('Hello World!')
        'Start drag, *without* using pixmap.'
        items = self.selectedItems()
        if items:
            drag = QtGui.QDrag(self)
            drag.setMimeData(self.mimeData(items))
            item = self.currentItem()
            rectangle = self.visualItemRect(item)
            pixmap = QtGui.QPixmap(rectangle.width(), rectangle.height())
            self.viewport().render(pixmap, QtCore.QPoint(), QtGui.QRegion(rectangle))
            drag.setPixmap(pixmap)
            drag.exec(QtCore.Qt.DropAction.MoveAction)

    def mimeData(self, items):
        if False:
            for i in range(10):
                print('nop')
        'Return MIME data for specified items.'
        album_ids = []
        files = []
        url = QtCore.QUrl.fromLocalFile
        for item in items:
            obj = item.obj
            if isinstance(obj, Album):
                album_ids.append(obj.id)
            elif obj.iterfiles:
                files.extend([url(f.filename) for f in obj.iterfiles()])
        mimeData = QtCore.QMimeData()
        mimeData.setData('application/picard.album-list', '\n'.join(album_ids).encode())
        if files:
            mimeData.setUrls(files)
        return mimeData

    def scrollTo(self, index, scrolltype=QtWidgets.QAbstractItemView.ScrollHint.EnsureVisible):
        if False:
            while True:
                i = 10
        hscrollbar = self.horizontalScrollBar()
        xpos = hscrollbar.value()
        super().scrollTo(index, scrolltype)
        hscrollbar.setValue(xpos)

    @staticmethod
    def drop_urls(urls, target, move_to_multi_tracks=True):
        if False:
            i = 10
            return i + 15
        files = []
        new_paths = []
        tagger = QtCore.QObject.tagger
        for url in urls:
            log.debug('Dropped the URL: %r', url.toString(QtCore.QUrl.UrlFormattingOption.RemoveUserInfo))
            if url.scheme() == 'file' or not url.scheme():
                filename = normpath(url.toLocalFile().rstrip('\x00'))
                file = tagger.files.get(filename)
                if file:
                    files.append(file)
                else:
                    new_paths.append(filename)
            elif url.scheme() in {'http', 'https'}:
                file_lookup = tagger.get_file_lookup()
                file_lookup.mbid_lookup(url.path(), browser_fallback=False)
        if files:
            tagger.move_files(files, target, move_to_multi_tracks)
        if new_paths:
            tagger.add_paths(new_paths, target=target)

    def dropEvent(self, event):
        if False:
            return 10
        if event.proposedAction() == QtCore.Qt.DropAction.IgnoreAction:
            event.acceptProposedAction()
            return
        if event.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier:
            self._move_to_multi_tracks = False
        QtWidgets.QTreeView.dropEvent(self, event)
        if event.isAccepted() and (not event.source() or event.mimeData().hasUrls()):
            event.setDropAction(QtCore.Qt.DropAction.CopyAction)
            event.accept()

    def dropMimeData(self, parent, index, data, action):
        if False:
            i = 10
            return i + 15
        target = None
        if parent:
            if index == parent.childCount():
                item = parent
            else:
                item = parent.child(index)
            if item is not None:
                target = item.obj
        if isinstance(self, FileTreeView) and target is None:
            target = self.tagger.unclustered_files
        log.debug('Drop target = %r', target)
        handled = False
        urls = data.urls()
        if urls:
            QtCore.QTimer.singleShot(0, partial(self.drop_urls, urls, target, self._move_to_multi_tracks))
            handled = True
        albums = data.data('application/picard.album-list')
        if albums:
            album_ids = bytes(albums).decode().split('\n')
            log.debug('Dropped albums = %r', album_ids)
            files = iter_files_from_objects((self.tagger.load_album(id) for id in album_ids))
            move_files = partial(self.tagger.move_files, list(files), target)
            QtCore.QTimer.singleShot(0, move_files)
            handled = True
        self._move_to_multi_tracks = True
        return handled

    def activate_item(self, index):
        if False:
            while True:
                i = 10
        obj = self.itemFromIndex(index).obj
        if not isinstance(obj, (Album, Cluster)) and obj.can_view_info():
            self.window.view_info()

    def add_cluster(self, cluster, parent_item=None):
        if False:
            for i in range(10):
                print('nop')
        if parent_item is None:
            parent_item = self.clusters
        cluster_item = ClusterItem(cluster, not cluster.special, parent_item)
        if cluster.hide_if_empty and (not cluster.files):
            cluster_item.update()
            cluster_item.setHidden(True)
        else:
            cluster_item.add_files(cluster.files)

    def moveCursor(self, action, modifiers):
        if False:
            for i in range(10):
                print('nop')
        if action in {QtWidgets.QAbstractItemView.CursorAction.MoveUp, QtWidgets.QAbstractItemView.CursorAction.MoveDown}:
            item = self.currentItem()
            if item and (not item.isSelected()):
                self.setCurrentItem(item)
        return QtWidgets.QTreeWidget.moveCursor(self, action, modifiers)

class FileTreeView(BaseTreeView):
    header_state = Option('persist', 'file_view_header_state', QtCore.QByteArray())
    header_locked = BoolOption('persist', 'file_view_header_locked', False)

    def __init__(self, window, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(window, parent)
        self.setAccessibleName(_('file view'))
        self.setAccessibleDescription(_('Contains unmatched files and clusters'))
        self.unmatched_files = ClusterItem(self.tagger.unclustered_files, False, self)
        self.unmatched_files.update()
        self.unmatched_files.setExpanded(True)
        self.clusters = ClusterItem(self.tagger.clusters, False, self)
        self.set_clusters_text()
        self.clusters.setExpanded(True)
        self.tagger.cluster_added.connect(self.add_file_cluster)
        self.tagger.cluster_removed.connect(self.remove_file_cluster)

    def add_file_cluster(self, cluster, parent_item=None):
        if False:
            return 10
        self.add_cluster(cluster, parent_item)
        self.set_clusters_text()

    def remove_file_cluster(self, cluster):
        if False:
            while True:
                i = 10
        cluster.item.setSelected(False)
        self.clusters.removeChild(cluster.item)
        self.set_clusters_text()

    def set_clusters_text(self):
        if False:
            return 10
        self.clusters.setText(MainPanel.TITLE_COLUMN, '%s (%d)' % (_('Clusters'), len(self.tagger.clusters)))

class AlbumTreeView(BaseTreeView):
    header_state = Option('persist', 'album_view_header_state', QtCore.QByteArray())
    header_locked = BoolOption('persist', 'album_view_header_locked', False)

    def __init__(self, window, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(window, parent)
        self.setAccessibleName(_('album view'))
        self.setAccessibleDescription(_('Contains albums and matched files'))
        self.tagger.album_added.connect(self.add_album)
        self.tagger.album_removed.connect(self.remove_album)

    def add_album(self, album):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(album, NatAlbum):
            item = NatAlbumItem(album, True)
            self.insertTopLevelItem(0, item)
        else:
            item = AlbumItem(album, True, self)
        item.setIcon(MainPanel.TITLE_COLUMN, AlbumItem.icon_cd)
        for (i, column) in enumerate(MainPanel.columns):
            font = item.font(i)
            font.setBold(True)
            item.setFont(i, font)
            item.setText(i, album.column(column[1]))
        self.add_cluster(album.unmatched_files, item)

    def remove_album(self, album):
        if False:
            while True:
                i = 10
        album.item.setSelected(False)
        self.takeTopLevelItem(self.indexOfTopLevelItem(album.item))

class TreeItem(QtWidgets.QTreeWidgetItem):

    def __init__(self, obj, sortable, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self.obj = obj
        if obj is not None:
            obj.item = self
        self.sortable = sortable
        self._sortkeys = {}
        for column in (MainPanel.LENGTH_COLUMN, MainPanel.TRACKNUMBER_COLUMN, MainPanel.DISCNUMBER_COLUMN):
            self.setTextAlignment(column, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setSizeHint(MainPanel.FINGERPRINT_COLUMN, ICON_SIZE)

    def setText(self, column, text):
        if False:
            return 10
        self._sortkeys[column] = None
        return super().setText(column, text)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        tree_widget = self.treeWidget()
        if not self.sortable or not tree_widget:
            return False
        column = tree_widget.sortColumn()
        return self.sortkey(column) < other.sortkey(column)

    def sortkey(self, column):
        if False:
            print('Hello World!')
        sortkey = self._sortkeys.get(column)
        if sortkey is not None:
            return sortkey
        if column == MainPanel.LENGTH_COLUMN:
            sortkey = self.obj.metadata.length or 0
        elif column in MainPanel.NAT_SORT_COLUMNS:
            sortkey = natsort.natkey(self.text(column))
        else:
            sortkey = strxfrm(self.text(column))
        self._sortkeys[column] = sortkey
        return sortkey

class ClusterItem(TreeItem):

    def __init__(self, *args):
        if False:
            print('Hello World!')
        super().__init__(*args)
        self.setIcon(MainPanel.TITLE_COLUMN, ClusterItem.icon_dir)

    def update(self, update_selection=True):
        if False:
            print('Hello World!')
        for (i, column) in enumerate(MainPanel.columns):
            self.setText(i, self.obj.column(column[1]))
        album = self.obj.related_album
        if self.obj.special and album and album.loaded:
            album.item.update(update_tracks=False)
        if update_selection and self.isSelected():
            TreeItem.window.update_selection(new_selection=False)

    def add_file(self, file):
        if False:
            return 10
        self.add_files([file])

    def add_files(self, files):
        if False:
            for i in range(10):
                print('nop')
        if self.obj.hide_if_empty and self.obj.files:
            self.setHidden(False)
        self.update()
        for file in files:
            item = FileItem(file, True)
            self.addChild(item)
            item.update()

    def remove_file(self, file):
        if False:
            return 10
        file.item.setSelected(False)
        self.removeChild(file.item)
        self.update()
        if self.obj.hide_if_empty and (not self.obj.files):
            self.setHidden(True)

class AlbumItem(TreeItem):

    def update(self, update_tracks=True, update_selection=True):
        if False:
            print('Hello World!')
        album = self.obj
        selection_changed = self.isSelected()
        if update_tracks:
            oldnum = self.childCount() - 1
            newnum = len(album.tracks)
            if oldnum > newnum:
                for i in range(oldnum - newnum):
                    item = self.child(newnum)
                    selection_changed |= item.isSelected()
                    self.takeChild(newnum)
                oldnum = newnum
            for i in range(oldnum):
                item = self.child(i)
                track = album.tracks[i]
                selection_changed |= item.isSelected() and item.obj != track
                item.obj = track
                track.item = item
                item.update(update_album=False)
            if newnum > oldnum:
                items = []
                for i in range(oldnum, newnum):
                    item = TrackItem(album.tracks[i], False)
                    item.setHidden(False)
                    items.append(item)
                tree_widget = self.treeWidget()
                if tree_widget:
                    sorting_enabled = tree_widget.isSortingEnabled()
                    tree_widget.setSortingEnabled(False)
                self.insertChildren(oldnum, items)
                if tree_widget:
                    tree_widget.setSortingEnabled(sorting_enabled)
                for item in items:
                    item.update(update_album=False)
        if album.errors:
            self.setIcon(MainPanel.TITLE_COLUMN, AlbumItem.icon_error)
            self.setToolTip(MainPanel.TITLE_COLUMN, _('Processing error(s): See the Errors tab in the Album Info dialog'))
        elif album.is_complete():
            if album.is_modified():
                self.setIcon(MainPanel.TITLE_COLUMN, AlbumItem.icon_cd_saved_modified)
                self.setToolTip(MainPanel.TITLE_COLUMN, _('Album modified and complete'))
            else:
                self.setIcon(MainPanel.TITLE_COLUMN, AlbumItem.icon_cd_saved)
                self.setToolTip(MainPanel.TITLE_COLUMN, _('Album unchanged and complete'))
        elif album.is_modified():
            self.setIcon(MainPanel.TITLE_COLUMN, AlbumItem.icon_cd_modified)
            self.setToolTip(MainPanel.TITLE_COLUMN, _('Album modified'))
        else:
            self.setIcon(MainPanel.TITLE_COLUMN, AlbumItem.icon_cd)
            self.setToolTip(MainPanel.TITLE_COLUMN, _('Album unchanged'))
        for (i, column) in enumerate(MainPanel.columns):
            self.setText(i, album.column(column[1]))
        if selection_changed and update_selection:
            TreeItem.window.update_selection(new_selection=False)
        self.emitDataChanged()

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, NatAlbumItem):
            return not other.__lt__(self)
        return super().__lt__(other)

class NatAlbumItem(AlbumItem):

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        tree_widget = self.treeWidget()
        if not tree_widget:
            return True
        order = tree_widget.header().sortIndicatorOrder()
        return order == QtCore.Qt.SortOrder.AscendingOrder

class TrackItem(TreeItem):

    def update(self, update_album=True, update_files=True, update_selection=True):
        if False:
            i = 10
            return i + 15
        track = self.obj
        if track.num_linked_files == 1:
            file = track.files[0]
            file.item = self
            color = TrackItem.track_colors[file.state]
            bgcolor = get_match_color(file.similarity, TreeItem.base_color)
            icon = FileItem.decide_file_icon(file)
            self.setToolTip(MainPanel.TITLE_COLUMN, _(FileItem.decide_file_icon_info(file)))
            self.takeChildren()
            self.setExpanded(False)
            (fingerprint_icon, fingerprint_tooltip) = FileItem.decide_fingerprint_icon_info(file)
            self.setToolTip(MainPanel.FINGERPRINT_COLUMN, fingerprint_tooltip)
            self.setIcon(MainPanel.FINGERPRINT_COLUMN, fingerprint_icon)
        else:
            self.setToolTip(MainPanel.TITLE_COLUMN, '')
            self.setToolTip(MainPanel.FINGERPRINT_COLUMN, '')
            self.setIcon(MainPanel.FINGERPRINT_COLUMN, QtGui.QIcon())
            if track.ignored_for_completeness():
                color = TreeItem.text_color_secondary
            else:
                color = TreeItem.text_color
            bgcolor = get_match_color(1, TreeItem.base_color)
            if track.is_video():
                icon = TrackItem.icon_video
            elif track.is_data():
                icon = TrackItem.icon_data
            else:
                icon = TrackItem.icon_audio
            if update_files:
                oldnum = self.childCount()
                newnum = track.num_linked_files
                if oldnum > newnum:
                    for i in range(oldnum - newnum):
                        self.takeChild(newnum - 1).obj.item = None
                    oldnum = newnum
                for i in range(oldnum):
                    item = self.child(i)
                    file = track.files[i]
                    item.obj = file
                    file.item = item
                    item.update(update_track=False)
                if newnum > oldnum:
                    items = []
                    for i in range(newnum - 1, oldnum - 1, -1):
                        item = FileItem(track.files[i], False)
                        item.update(update_track=False, update_selection=update_selection)
                        items.append(item)
                    self.addChildren(items)
            self.setExpanded(True)
        if track.errors:
            self.setIcon(MainPanel.TITLE_COLUMN, TrackItem.icon_error)
            self.setToolTip(MainPanel.TITLE_COLUMN, _('Processing error(s): See the Errors tab in the Track Info dialog'))
        else:
            self.setIcon(MainPanel.TITLE_COLUMN, icon)
        for (i, column) in enumerate(MainPanel.columns):
            self.setText(i, track.column(column[1]))
            self.setForeground(i, color)
            self.setBackground(i, bgcolor)
        if update_selection and self.isSelected():
            TreeItem.window.update_selection(new_selection=False)
        if update_album:
            self.parent().update(update_tracks=False, update_selection=update_selection)

class FileItem(TreeItem):

    def update(self, update_track=True, update_selection=True):
        if False:
            for i in range(10):
                print('nop')
        file = self.obj
        self.setIcon(MainPanel.TITLE_COLUMN, FileItem.decide_file_icon(file))
        (fingerprint_icon, fingerprint_tooltip) = FileItem.decide_fingerprint_icon_info(file)
        self.setToolTip(MainPanel.FINGERPRINT_COLUMN, fingerprint_tooltip)
        self.setIcon(MainPanel.FINGERPRINT_COLUMN, fingerprint_icon)
        color = FileItem.file_colors[file.state]
        bgcolor = get_match_color(file.similarity, TreeItem.base_color)
        for (i, column) in enumerate(MainPanel.columns):
            self.setText(i, file.column(column[1]))
            self.setForeground(i, color)
            self.setBackground(i, bgcolor)
        if file.errors:
            self.setToolTip(MainPanel.TITLE_COLUMN, _('Processing error(s): See the Errors tab in the File Info dialog'))
        if update_selection and self.isSelected():
            TreeItem.window.update_selection(new_selection=False)
        parent = self.parent()
        if isinstance(parent, TrackItem) and update_track:
            parent.update(update_files=False, update_selection=update_selection)

    @staticmethod
    def decide_file_icon(file):
        if False:
            while True:
                i = 10
        if file.state == File.ERROR:
            if file.error_type == FileErrorType.NOTFOUND:
                return FileItem.icon_error_not_found
            elif file.error_type == FileErrorType.NOACCESS:
                return FileItem.icon_error_no_access
            else:
                return FileItem.icon_error
        elif isinstance(file.parent, Track):
            if file.state == File.NORMAL:
                return FileItem.icon_saved
            elif file.state == File.PENDING:
                return FileItem.match_pending_icons[int(file.similarity * 5 + 0.5)]
            else:
                return FileItem.match_icons[int(file.similarity * 5 + 0.5)]
        elif file.state == File.PENDING:
            return FileItem.icon_file_pending
        else:
            return FileItem.icon_file

    @staticmethod
    def decide_file_icon_info(file):
        if False:
            print('Hello World!')
        if isinstance(file.parent, Track):
            if file.state == File.NORMAL:
                return N_('Track saved')
            elif file.state == File.PENDING:
                return N_('Pending')
            else:
                return FileItem.match_icons_info[int(file.similarity * 5 + 0.5)]
        elif file.state == File.PENDING:
            return N_('Pending')

    @staticmethod
    def decide_fingerprint_icon_info(file):
        if False:
            while True:
                i = 10
        if getattr(file, 'acoustid_fingerprint', None):
            if QtCore.QObject.tagger.acoustidmanager.is_submitted(file):
                icon = FileItem.icon_fingerprint_gray
                tooltip = _('Fingerprint has already been submitted')
            else:
                icon = FileItem.icon_fingerprint
                tooltip = _('Unsubmitted fingerprint')
        else:
            icon = QtGui.QIcon()
            tooltip = _('No fingerprint was calculated for this file, use "Scan" or "Generate AcoustID Fingerprints" to calculate the fingerprint.')
        return (icon, tooltip)