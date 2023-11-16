from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QBrush, QColor, QIcon, QPixmap
from PyQt5.QtWidgets import QAbstractItemView, QAbstractScrollArea, QAction, QListWidget, QListWidgetItem
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT
from tribler.core.utilities.simpledefs import CHANNEL_STATE
from tribler.gui.network.request_manager import request_manager
from tribler.gui.tribler_action_menu import TriblerActionMenu
from tribler.gui.utilities import connect, get_image_path, tr

def entry_to_tuple(entry):
    if False:
        while True:
            i = 10
    return (entry['public_key'], entry['id'], entry.get('subscribed', False), entry.get('state'), entry.get('progress'))

class ChannelListItem(QListWidgetItem):
    loading_brush = QBrush(Qt.darkGray)

    def __init__(self, parent=None, channel_info=None):
        if False:
            while True:
                i = 10
        self.channel_info = channel_info
        title = channel_info.get('name')
        QListWidgetItem.__init__(self, title, parent=parent)
        self.setSizeHint(QSize(50, 25))
        if channel_info.get('state') not in (CHANNEL_STATE.COMPLETE.value, CHANNEL_STATE.PERSONAL.value):
            self.setForeground(self.loading_brush)

    def setData(self, role, new_value):
        if False:
            return 10
        if role == Qt.EditRole:
            item = self.channel_info
            if item['name'] != new_value:
                request_manager.patch(f"metadata/{item['public_key']}/{item['id']}", data={'title': new_value})
        return super().setData(role, new_value)

class ChannelsMenuListWidget(QListWidget):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        QListWidget.__init__(self, parent=parent)
        self.base_url = 'channels'
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.items_set = frozenset()
        self.personal_channel_icon = QIcon(get_image_path('share.png'))
        empty_transparent_image = QPixmap(15, 15)
        empty_transparent_image.fill(QColor(0, 0, 0, 0))
        self.empty_image = QIcon(empty_transparent_image)
        self.foreign_channel_menu = self.create_foreign_menu()
        self.personal_channel_menu = self.create_personal_menu()
        self.setSelectionMode(QAbstractItemView.NoSelection)

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        count = self.count()
        height = self.sizeHintForRow(0) * count if count else 0
        self.setMaximumHeight(height)
        return QSize(self.width(), height)

    def contextMenuEvent(self, event):
        if False:
            i = 10
            return i + 15
        item = self.itemAt(event.pos())
        if item is None:
            return
        if item.channel_info['state'] == CHANNEL_STATE.PERSONAL.value:
            self.personal_channel_menu.exec_(self.mapToGlobal(event.pos()))
        else:
            self.foreign_channel_menu.exec_(self.mapToGlobal(event.pos()))

    def create_foreign_menu(self):
        if False:
            i = 10
            return i + 15
        menu = TriblerActionMenu(self)
        unsubscribe_action = QAction(tr('Unsubscribe'), self)
        connect(unsubscribe_action.triggered, self._on_unsubscribe_action)
        menu.addAction(unsubscribe_action)
        return menu

    def create_personal_menu(self):
        if False:
            while True:
                i = 10
        menu = TriblerActionMenu(self)
        delete_action = QAction(tr('Delete channel'), self)
        connect(delete_action.triggered, self._on_delete_action)
        menu.addAction(delete_action)
        rename_action = QAction(tr('Rename channel'), self)
        connect(rename_action.triggered, self._trigger_name_editor)
        menu.addAction(rename_action)
        return menu

    def _trigger_name_editor(self, checked):
        if False:
            return 10
        self.editItem(self.currentItem())

    def _on_unsubscribe_action(self, checked):
        if False:
            print('Hello World!')
        self.window().on_channel_unsubscribe(self.currentItem().channel_info)

    def _on_delete_action(self, checked):
        if False:
            i = 10
            return i + 15
        self.window().on_channel_delete(self.currentItem().channel_info)

    def on_query_results(self, response):
        if False:
            while True:
                i = 10
        channels = response.get('results')
        if channels is None:
            return
        self.clear()
        for channel_info in sorted(channels, key=lambda x: x.get('state') != 'Personal'):
            item = ChannelListItem(channel_info=channel_info)
            self.addItem(item)
            if channel_info.get('state') == CHANNEL_STATE.PERSONAL.value:
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable)
                item.setIcon(self.personal_channel_icon)
            else:
                item.setIcon(self.empty_image)
            tooltip_text = channel_info['name'] + '\n' + channel_info['state']
            if channel_info.get('progress'):
                tooltip_text += f" {int(float(channel_info['progress']) * 100)}%"
            item.setToolTip(tooltip_text)
        self.items_set = frozenset((entry_to_tuple(channel_info) for channel_info in channels))

    def load_channels(self):
        if False:
            i = 10
            return i + 15
        request_manager.get(self.base_url, self.on_query_results, url_params={'subscribed': True, 'last': 1000})

    def reload_if_necessary(self, changed_entries):
        if False:
            i = 10
            return i + 15
        changeset = frozenset((entry_to_tuple(entry) for entry in changed_entries if entry.get('state') == 'Deleted' or entry.get('type') == CHANNEL_TORRENT))
        need_update = not self.items_set.issuperset(changeset)
        if need_update:
            self.load_channels()