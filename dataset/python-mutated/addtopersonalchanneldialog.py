import json
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import pyqtSignal
from tribler.core.components.metadata_store.db.serialization import CHANNEL_TORRENT, COLLECTION_NODE
from tribler.gui.dialogs.dialogcontainer import DialogContainer
from tribler.gui.dialogs.new_channel_dialog import NewChannelDialog
from tribler.gui.network.request_manager import request_manager
from tribler.gui.utilities import connect, get_ui_file_path

class ChannelQTreeWidgetItem(QtWidgets.QTreeWidgetItem):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.id_ = kwargs.pop('id_') if 'id_' in kwargs else 0
        QtWidgets.QTreeWidgetItem.__init__(self, *args, **kwargs)

class AddToChannelDialog(DialogContainer):
    create_torrent_notification = pyqtSignal(dict)

    def __init__(self, parent):
        if False:
            return 10
        DialogContainer.__init__(self, parent)
        uic.loadUi(get_ui_file_path('addtochanneldialog.ui'), self.dialog_widget)
        connect(self.dialog_widget.btn_cancel.clicked, self.close_dialog)
        connect(self.dialog_widget.btn_confirm.clicked, self.on_confirm_clicked)
        connect(self.dialog_widget.btn_new_channel.clicked, self.on_create_new_channel_clicked)
        connect(self.dialog_widget.btn_new_folder.clicked, self.on_create_new_folder_clicked)
        self.confirm_clicked_callback = None
        self.root_requests_list = []
        self.channels_tree = {}
        self.id2wt_mapping = {0: self.dialog_widget.channels_tree_wt}
        connect(self.dialog_widget.channels_tree_wt.itemExpanded, self.on_item_expanded)
        self.dialog_widget.channels_tree_wt.setHeaderLabels(['Name'])
        self.on_main_window_resize()

    def on_new_channel_response(self, response):
        if False:
            for i in range(10):
                print('nop')
        if not response or not response.get('results', None):
            return
        self.window().channels_menu_list.reload_if_necessary(response['results'])
        self.load_channel(response['results'][0]['origin_id'])

    def on_create_new_channel_clicked(self, checked):
        if False:
            return 10

        def create_channel_callback(channel_name=None):
            if False:
                while True:
                    i = 10
            request_manager.post('channels/mychannel/0/channels', self.on_new_channel_response, data=json.dumps({'name': channel_name}) if channel_name else None)
        NewChannelDialog(self, create_channel_callback)

    def on_create_new_folder_clicked(self, checked):
        if False:
            i = 10
            return i + 15
        selected = self.dialog_widget.channels_tree_wt.selectedItems()
        if not selected:
            return
        channel_id = selected[0].id_
        postfix = 'channels' if not channel_id else 'collections'
        endpoint = f'channels/mychannel/{channel_id}/{postfix}'

        def create_channel_callback(channel_name=None):
            if False:
                print('Hello World!')
            request_manager.post(endpoint, self.on_new_channel_response, data=json.dumps({'name': channel_name}) if channel_name else None)
        NewChannelDialog(self, create_channel_callback)

    def clear_channels_tree(self):
        if False:
            while True:
                i = 10
        for rq in self.root_requests_list:
            rq.cancel()
        self.dialog_widget.channels_tree_wt.clear()
        self.id2wt_mapping = {0: self.dialog_widget.channels_tree_wt}
        self.load_channel(0)

    def show_dialog(self, on_confirm, confirm_button_text='CONFIRM_BUTTON'):
        if False:
            print('Hello World!')
        self.dialog_widget.btn_confirm.setText(confirm_button_text)
        self.show()
        self.confirm_clicked_callback = on_confirm

    def on_item_expanded(self, item):
        if False:
            while True:
                i = 10
        for channel_id in self.channels_tree.get(item.id_, None):
            subchannels_set = self.channels_tree.get(channel_id, set())
            if subchannels_set is None or subchannels_set:
                continue
            self.load_channel(channel_id)

    def load_channel(self, channel_id):
        if False:
            i = 10
            return i + 15
        request = request_manager.get(f'channels/mychannel/{channel_id}', on_success=lambda result: self.on_channel_contents(result, channel_id), url_params={'metadata_type': [CHANNEL_TORRENT, COLLECTION_NODE], 'first': 1, 'last': 1000, 'exclude_deleted': True})
        if request:
            self.root_requests_list.append(request)

    def get_selected_channel_id(self):
        if False:
            print('Hello World!')
        selected = self.dialog_widget.channels_tree_wt.selectedItems()
        return None if not selected else selected[0].id_

    def on_confirm_clicked(self, checked):
        if False:
            for i in range(10):
                print('nop')
        channel_id = self.get_selected_channel_id()
        if channel_id is None:
            return
        if self.confirm_clicked_callback:
            self.confirm_clicked_callback(channel_id)
        self.close_dialog()

    def on_channel_contents(self, response, channel_id):
        if False:
            while True:
                i = 10
        if not response:
            return
        self.channels_tree[channel_id] = set() if response.get('results') else None
        for subchannel in response.get('results', []):
            subchannel_id = subchannel['id']
            if subchannel_id in self.id2wt_mapping:
                continue
            wt = ChannelQTreeWidgetItem(self.id2wt_mapping[channel_id], [subchannel['name']], id_=subchannel_id)
            self.id2wt_mapping[subchannel_id] = wt
            self.channels_tree[channel_id].add(subchannel_id)
            if channel_id == 0:
                self.load_channel(subchannel_id)

    def close_dialog(self, checked=False):
        if False:
            for i in range(10):
                print('nop')
        self.hide()