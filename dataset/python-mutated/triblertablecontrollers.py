"""
This file contains various controllers for table views.
The responsibility of the controller is to populate the table view with some data, contained in a specific model.
"""
import json
import logging
from PyQt5.QtCore import QObject, QTimer, Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtNetwork import QNetworkRequest
from PyQt5.QtWidgets import QAction
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT, COLLECTION_NODE, REGULAR_TORRENT
from tribler.core.utilities.simpledefs import CHANNEL_STATE
from tribler.gui.defs import HEALTH_CHECKING, HEALTH_UNCHECKED
from tribler.gui.network.request_manager import request_manager
from tribler.gui.tribler_action_menu import TriblerActionMenu
from tribler.gui.utilities import connect, dict_item_is_any_of, tr
from tribler.gui.widgets.tablecontentmodel import Column
HEALTHCHECK_DELAY_MS = 500

class TriblerTableViewController(QObject):
    """
    Base controller for a table view that displays some data.
    """

    def __init__(self, table_view, *args, filter_input=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.model = None
        self.table_view = table_view
        connect(self.table_view.verticalScrollBar().valueChanged, self._on_list_scroll)
        connect(self.table_view.delegate.subscribe_control.clicked, self.table_view.on_subscribe_control_clicked)
        connect(self.table_view.delegate.download_button.clicked, self.table_view.start_download_from_index)
        connect(self.table_view.torrent_doubleclicked, self.table_view.start_download_from_dataitem)
        self.filter_input = filter_input
        if self.filter_input:
            connect(self.filter_input.returnPressed, self.on_filter_input_return_pressed)

    def set_model(self, model):
        if False:
            return 10
        self.model = model
        self.table_view.setModel(self.model)
        if self.model.saved_header_state:
            self.table_view.horizontalHeader().restoreState(self.model.saved_header_state)
        if self.model.saved_scroll_state is not None:
            self.table_view.scrollTo(self.model.index(self.model.saved_scroll_state, 0), 1)
            self.table_view.scrollTo(self.model.index(self.model.saved_scroll_state, 0), 1)

    def _on_list_scroll(self, event):
        if False:
            while True:
                i = 10
        if self.table_view.verticalScrollBar().value() == self.table_view.verticalScrollBar().maximum() and self.model.data_items and (not self.model.all_local_entries_loaded):
            self.model.perform_query()

    def _get_sort_parameters(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a tuple (column_name, sort_desc) that indicates the sorting column/order of the table view.\n        '
        sort_column_number = self.table_view.horizontalHeader().sortIndicatorSection()
        sort_by = self.model.columns[sort_column_number].dict_key if sort_column_number >= 0 else None
        sort_asc = self.table_view.horizontalHeader().sortIndicatorOrder()
        return (sort_by, sort_asc)

    def on_filter_input_return_pressed(self):
        if False:
            for i in range(10):
                print('nop')
        self.model.text_filter = self.filter_input.text().lower()
        self.model.reset()

    def brain_dead_refresh(self):
        if False:
            return 10
        '\n        ACHTUNG! Brain-dead refresh is back!\n        It shows the rows eaten by a closed channel description widget.\n        Note that none of the more civilized ways to fix it work:\n        various updateGeometry, viewport().update, adjustSize - nothing works!\n        '
        window = self.table_view.window()
        window.resize(window.geometry().width() + 1, window.geometry().height())
        window.resize(window.geometry().width() - 1, window.geometry().height())

    def unset_model(self):
        if False:
            return 10
        self.model = None

class TableLoadingAnimationMixin:

    def set_model(self, model):
        if False:
            print('Hello World!')
        if not model.loaded:
            self.table_view.show_loading_animation_delayed()
        connect(model.query_complete, self.table_view.hide_loading_animation)
        connect(model.query_started, self.table_view.show_loading_animation_delayed)
        super().set_model(model)

    def unset_model(self):
        if False:
            return 10
        if self.table_view.model:
            self.model.query_complete.disconnect()
            self.model.query_started.disconnect()
        self.table_view.hide_loading_animation()
        super().unset_model()

class TableSelectionMixin:

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.healthcheck_cooldown = QTimer()
        self.healthcheck_cooldown.setSingleShot(True)
        connect(self.healthcheck_cooldown.timeout, lambda : self._on_selection_changed(None, None))

    def set_model(self, model):
        if False:
            while True:
                i = 10
        super().set_model(model)
        connect(self.table_view.selectionModel().selectionChanged, self._on_selection_changed)

    def unset_model(self):
        if False:
            while True:
                i = 10
        if self.table_view.model:
            self.table_view.selectionModel().selectionChanged.disconnect()
        super().unset_model()

    def _on_selection_changed(self, selected, deselected):
        if False:
            for i in range(10):
                print('nop')
        selected_indices = self.table_view.selectedIndexes()
        if not selected_indices:
            self.table_view.clearSelection()
            return
        data_item = selected_indices[-1].model().data_items[selected_indices[-1].row()]
        if not dict_item_is_any_of(data_item, 'type', [REGULAR_TORRENT]):
            return
        if issubclass(type(self), HealthCheckerMixin):
            if data_item['last_tracker_check'] == 0 and data_item.get('health') != HEALTH_CHECKING:
                if self.healthcheck_cooldown.isActive():
                    self.healthcheck_cooldown.stop()
                else:
                    self.check_torrent_health(data_item)
                self.healthcheck_cooldown.start(HEALTHCHECK_DELAY_MS)

class HealthCheckerMixin:

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.health_checker_logger = logging.getLogger('HealthCheckerMixin')
        connect(self.table_view.delegate.health_status_widget.clicked, lambda index: self.check_torrent_health(index.model().data_items[index.row()], forced=True))
        connect(self.table_view.torrent_clicked, self.check_torrent_health)

    def check_torrent_health(self, data_item, forced=False):
        if False:
            while True:
                i = 10
        if not dict_item_is_any_of(data_item, 'type', [REGULAR_TORRENT]):
            return
        if Column.HEALTH not in self.model.column_position:
            return
        infohash = data_item['infohash']
        row = self.model.item_uid_map.get(infohash)
        if row is None:
            return
        if not forced and data_item.get('health', HEALTH_UNCHECKED) != HEALTH_UNCHECKED:
            return
        data_item['health'] = HEALTH_CHECKING
        health_cell_index = self.model.index(row, self.model.column_position[Column.HEALTH])
        self.model.dataChanged.emit(health_cell_index, health_cell_index, [])
        request_manager.get(f'metadata/torrents/{infohash}/health', capture_errors=False, priority=QNetworkRequest.LowPriority)

class ContextMenuMixin:

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.enable_context_menu(self.table_view)

    def enable_context_menu(self, widget):
        if False:
            print('Hello World!')
        self.table_view = widget
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        connect(self.table_view.customContextMenuRequested, self._show_context_menu)

    def _trigger_name_editor(self, index):
        if False:
            return 10
        model = index.model()
        title_index = model.index(index.row(), model.columns_shown.index(Column.NAME))
        self.table_view.edit(title_index)

    def _trigger_category_editor(self, index):
        if False:
            print('Hello World!')
        model = index.model()
        title_index = model.index(index.row(), model.columns_shown.index(Column.CATEGORY))
        self.table_view.edit(title_index)

    def _show_context_menu(self, pos):
        if False:
            print('Hello World!')
        if not self.table_view or not self.model:
            return
        item_index = self.table_view.indexAt(pos)
        if not item_index or item_index.row() < 0:
            return
        menu = TriblerActionMenu(self.table_view)
        num_selected = len(self.table_view.selectionModel().selectedRows())
        if num_selected == 1 and item_index.model().data_items[item_index.row()]['type'] == REGULAR_TORRENT:
            self.add_menu_item(menu, tr(' Download '), item_index, self.table_view.start_download_from_index)
            if issubclass(type(self), HealthCheckerMixin):
                self.add_menu_item(menu, tr(' Recheck health'), item_index.model().data_items[item_index.row()], lambda x: self.check_torrent_health(x, forced=True))
        if num_selected == 1 and item_index.model().column_position.get(Column.SUBSCRIBED) is not None:
            data_item = item_index.model().data_items[item_index.row()]
            if data_item['type'] == CHANNEL_TORRENT and data_item['state'] != CHANNEL_STATE.PERSONAL.value:
                self.add_menu_item(menu, tr('Unsubscribe channel') if data_item['subscribed'] else tr('Subscribe channel'), item_index.model().index(item_index.row(), item_index.model().column_position[Column.SUBSCRIBED]), self.table_view.delegate.subscribe_control.clicked.emit)
        menu.addSeparator()
        entries = [self.model.data_items[index.row()] for index in self.table_view.selectionModel().selectedRows()]

        def on_add_to_channel(_):
            if False:
                return 10

            def on_confirm_clicked(channel_id):
                if False:
                    for i in range(10):
                        print('nop')
                request_manager.post(f'channels/mychannel/{channel_id}/copy', on_success=lambda _: self.table_view.window().tray_show_message(tr('Channel update'), tr('Torrent(s) added to your channel')), data=json.dumps(entries))
            self.table_view.window().add_to_channel_dialog.show_dialog(on_confirm_clicked, confirm_button_text=tr('Copy'))

        def on_move(_):
            if False:
                while True:
                    i = 10

            def on_confirm_clicked(channel_id):
                if False:
                    while True:
                        i = 10
                changes_list = [{'public_key': entry['public_key'], 'id': entry['id'], 'origin_id': channel_id} for entry in entries]
                self.model.remove_items(entries)
                request_manager.patch('metadata', data=changes_list)
            self.table_view.window().add_to_channel_dialog.show_dialog(on_confirm_clicked, confirm_button_text=tr('Move'))
        if not self.model.edit_enabled:
            if self.selection_can_be_added_to_channel():
                self.add_menu_item(menu, tr(' Copy into personal channel'), item_index, on_add_to_channel)
        else:
            self.add_menu_item(menu, tr(' Move '), item_index, on_move)
            self.add_menu_item(menu, tr(' Rename '), item_index, self._trigger_name_editor)
            self.add_menu_item(menu, tr(' Change category '), item_index, self._trigger_category_editor)
            menu.addSeparator()
            self.add_menu_item(menu, tr(' Remove from channel'), item_index, self.table_view.on_delete_button_clicked)
        menu.exec_(QCursor.pos())

    def add_menu_item(self, menu, name, item_index, callback):
        if False:
            for i in range(10):
                print('nop')
        action = QAction(name, self.table_view)
        connect(action.triggered, lambda _: callback(item_index))
        menu.addAction(action)

    def selection_can_be_added_to_channel(self):
        if False:
            for i in range(10):
                print('nop')
        for row in self.table_view.selectionModel().selectedRows():
            data_item = row.model().data_items[row.row()]
            if dict_item_is_any_of(data_item, 'type', [REGULAR_TORRENT, CHANNEL_TORRENT, COLLECTION_NODE]):
                return True
        return False

class PopularContentTableViewController(TableSelectionMixin, ContextMenuMixin, TableLoadingAnimationMixin, TriblerTableViewController):
    pass

class ContentTableViewController(TableSelectionMixin, ContextMenuMixin, HealthCheckerMixin, TableLoadingAnimationMixin, TriblerTableViewController):
    pass