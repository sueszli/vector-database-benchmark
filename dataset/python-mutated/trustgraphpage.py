import math
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtNetwork import QNetworkRequest
from PyQt5.QtWidgets import QWidget
from tribler.gui.defs import COLOR_DEFAULT, COLOR_GREEN, COLOR_NEUTRAL, COLOR_RED, COLOR_ROOT, COLOR_SELECTED, HTML_SPACE, TB, TRUST_GRAPH_PEER_LEGENDS
from tribler.gui.network.request_manager import request_manager
from tribler.gui.sentry_mixin import AddBreadcrumbOnShowMixin
from tribler.gui.utilities import connect, format_size, html_label, tr

class TrustGraph(pg.GraphItem):

    def __init__(self):
        if False:
            print('Hello World!')
        pg.GraphItem.__init__(self)
        self.data = None
        self.dragPoint = None
        self.dragOffset = None

    def set_node_selection_listener(self, listener):
        if False:
            print('Hello World!')
        connect(self.scatter.sigClicked, listener)

    def setData(self, **data):
        if False:
            i = 10
            return i + 15
        self.data = data
        if 'pos' in self.data:
            num_nodes = self.data['pos'].shape[0]
            self.data['data'] = np.empty(num_nodes, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(num_nodes)
            pg.GraphItem.setData(self, **self.data)

    def mouseDragEvent(self, event):
        if False:
            i = 10
            return i + 15
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        if event.isStart():
            clicked_position = event.buttonDownPos()
            clicked_nodes = self.scatter.pointsAt(clicked_position)
            if not clicked_nodes:
                event.ignore()
                return
            self.dragPoint = clicked_nodes[0]
            clicked_index = clicked_nodes[0].data()[0]
            self.dragOffset = self.data['pos'][clicked_index] - clicked_position
        elif event.isFinish():
            self.dragPoint = None
            return
        elif self.dragPoint is None:
            event.ignore()
            return
        clicked_index = self.dragPoint.data()[0]
        if clicked_index == 0:
            event.ignore()
            return
        self.data['pos'][clicked_index] = event.pos() + self.dragOffset
        pg.GraphItem.setData(self, **self.data)
        event.accept()

class TrustGraphPage(AddBreadcrumbOnShowMixin, QWidget):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        QWidget.__init__(self)
        self.trust_graph = None
        self.graph_view = None
        self.selected_node = dict()
        self.root_public_key = None
        self.graph_data = None
        self.rest_request = None

    def showEvent(self, QShowEvent):
        if False:
            for i in range(10):
                print('nop')
        super().showEvent(QShowEvent)
        self.fetch_graph_data()

    def hideEvent(self, QHideEvent):
        if False:
            i = 10
            return i + 15
        super().hideEvent(QHideEvent)

    def initialize_trust_graph(self):
        if False:
            i = 10
            return i + 15
        pg.setConfigOption('background', '#222222')
        pg.setConfigOption('foreground', '#555')
        pg.setConfigOption('antialias', True)
        graph_layout = pg.GraphicsLayoutWidget()
        self.graph_view = graph_layout.addViewBox()
        self.graph_view.setAspectLocked()
        self.graph_view.setMenuEnabled(False)
        self.reset_graph()
        self.graph_view.wheelEvent = lambda evt: None
        self.trust_graph = TrustGraph()
        self.trust_graph.set_node_selection_listener(self.on_node_clicked)
        self.graph_view.addItem(self.trust_graph)
        self.graph_view.addItem(pg.TextItem(text='YOU'))
        self.window().trust_graph_plot_widget.layout().addWidget(graph_layout)
        connect(self.window().tr_control_refresh_btn.clicked, self.fetch_graph_data)
        self.window().tr_selected_node_pub_key.setHidden(True)
        self.window().tr_selected_node_stats.setHidden(True)
        self.window().trust_graph_progress_bar.setHidden(True)

    def on_node_clicked(self, top_point, *_other_overlapping_points):
        if False:
            return 10
        clicked_node_data = top_point.ptsClicked[0].data()
        clicked_node = self.graph_data['node'][clicked_node_data[0]]
        if not self.selected_node:
            self.selected_node = dict()
        elif 'spot' in self.selected_node and self.selected_node['spot']:
            self.selected_node['spot'].setBrush(self.selected_node['color'])
        self.selected_node['public_key'] = clicked_node['key']
        self.selected_node['total_up'] = clicked_node.get('total_up', 0)
        self.selected_node['total_down'] = clicked_node.get('total_down', 0)
        self.selected_node['color'] = self.get_node_color(clicked_node)
        self.selected_node['spot'] = top_point.ptsClicked[0]
        spot = top_point.ptsClicked[0]
        spot.setBrush(COLOR_SELECTED)
        self.update_status_bar(self.selected_node)

    def update_status_bar(self, selected_node):
        if False:
            for i in range(10):
                print('nop')
        if not selected_node:
            return
        peer_message = f"<b>User</b> {HTML_SPACE * 16}{selected_node.get('public_key', '')[:74]}..."
        self.window().tr_selected_node_pub_key.setHidden(False)
        self.window().tr_selected_node_pub_key.setText(peer_message)
        diff = selected_node.get('total_up', 0) - selected_node.get('total_down', 0)
        color = COLOR_GREEN if diff > 0 else COLOR_RED if diff < 0 else COLOR_DEFAULT
        bandwidth_message = '<b>Bandwidth</b> ' + HTML_SPACE * 2 + ' Given ' + HTML_SPACE + html_label(format_size(selected_node.get('total_up', 0))) + ' Taken ' + HTML_SPACE + html_label(format_size(selected_node.get('total_down', 0))) + ' Balance ' + HTML_SPACE + html_label(format_size(diff), color=color)
        self.window().tr_selected_node_stats.setHidden(False)
        self.window().tr_selected_node_stats.setText(bandwidth_message)

    def reset_graph(self):
        if False:
            print('Hello World!')
        self.graph_view.setXRange(-1, 1)
        self.graph_view.setYRange(-1, 1)

    def fetch_graph_data(self, checked=False):
        if False:
            i = 10
            return i + 15
        if self.rest_request:
            self.rest_request.cancel()
        request_manager.get('trustview', self.on_received_data, url_params={'refresh': 1}, priority=QNetworkRequest.LowPriority)

    def on_received_data(self, data):
        if False:
            print('Hello World!')
        if data is None or not isinstance(data, dict) or 'graph' not in data:
            return
        self.update_gui_labels(data)
        self.root_public_key = data['root_public_key']
        self.graph_data = data['graph']
        plot_data = dict()
        plot_data['pxMode'] = False
        plot_data['pen'] = (100, 100, 100, 150)
        plot_data['brush'] = (255, 0, 0, 255)
        plot_data['pos'] = np.array([node['pos'] for node in data['graph']['node']])
        plot_data['size'] = np.array([self.get_node_size(node) for node in data['graph']['node']])
        plot_data['symbolBrush'] = np.array([self.get_node_color(node) for node in data['graph']['node']])
        if data['graph']['edge']:
            plot_data['adj'] = np.array(data['graph']['edge'])
        self.trust_graph.setData(**plot_data)

    def get_node_color(self, node, selected=False):
        if False:
            return 10
        if not selected and self.root_public_key == node['key']:
            return COLOR_ROOT
        if selected and self.selected_node and (self.selected_node.get('public_key', None) == node['key']):
            return COLOR_SELECTED
        diff = node.get('total_up', 0) - node.get('total_down', 0)
        return COLOR_GREEN if diff > 0 else COLOR_NEUTRAL if diff == 0 else COLOR_RED

    def get_node_size(self, node):
        if False:
            while True:
                i = 10
        min_size = 0.01 if node['key'] != self.root_public_key else 0.1
        diff = abs(node.get('total_up', 0) - node.get('total_down', 0))
        if diff == 0:
            return min_size
        elif diff > 10 * TB:
            return 0.1
        elif diff > TB:
            return 0.05 + 0.005 * diff / TB
        return math.log(diff, 2) / 512 + min_size

    def update_gui_labels(self, data):
        if False:
            print('Hello World!')
        header_message = tr('The graph below is based on your historical interactions with other users in the network. It shows <strong>%(num_interactions)s</strong> interactions made by <strong>%(num_users)s</strong> users.<br/>') % {'num_interactions': data['num_tx'], 'num_users': len(data['graph']['node'])}
        self.window().trust_graph_explanation_label.setText(header_message)
        self.window().trust_graph_status_bar.setText(TRUST_GRAPH_PEER_LEGENDS)