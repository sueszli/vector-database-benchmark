import logging
from datetime import datetime
from typing import Dict, Optional
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressBar, QTreeWidgetItem, QVBoxLayout, QWidget
from tribler.core.utilities.simpledefs import DownloadStatus
from tribler.gui.defs import STATUS_STRING
from tribler.gui.utilities import duration_to_string, format_size, format_speed

class LoadingDownloadWidgetItem(QTreeWidgetItem):
    """
    This class is used for the placeholder "Loading" item for the downloads list
    """

    def __init__(self):
        if False:
            print('Hello World!')
        QTreeWidgetItem.__init__(self)
        self.setFlags(Qt.NoItemFlags)

    def get_raw_download_status(self):
        if False:
            return 10
        return 'PLACEHOLDER'

def create_progress_bar_widget() -> (QWidget, QProgressBar):
    if False:
        for i in range(10):
            print('nop')
    progress_slider = QProgressBar()
    bar_container = QWidget()
    bar_container.setLayout(QVBoxLayout())
    bar_container.setStyleSheet('background-color: transparent;')
    progress_slider.setStyleSheet('\n    QProgressBar {\n        background-color: white;\n        color: black;\n        font-size: 12px;\n        text-align: center;\n        border: 0px solid transparent;\n    }\n\n    QProgressBar::chunk {\n        background-color: #e67300;\n    }\n    ')
    progress_slider.setAutoFillBackground(True)
    bar_container.layout().addWidget(progress_slider)
    bar_container.layout().setContentsMargins(4, 4, 8, 4)
    return (bar_container, progress_slider)

class DownloadWidgetItem(QTreeWidgetItem):
    """
    This class is responsible for managing the item in the downloads list and fills the item with the relevant data.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        QTreeWidgetItem.__init__(self)
        self.download_info: Optional[Dict] = None
        self.infohash: Optional[str] = None
        self._logger = logging.getLogger('TriblerGUI')
        (self.bar_container, self.progress_slider) = create_progress_bar_widget()

    def update_with_download(self, download: Dict):
        if False:
            for i in range(10):
                print('nop')
        self.download_info = download
        self.infohash = download['infohash']
        self.update_item()

    def get_status(self) -> DownloadStatus:
        if False:
            i = 10
            return i + 15
        return DownloadStatus(self.download_info['status_code'])

    def update_item(self):
        if False:
            return 10
        self.setText(0, self.download_info['name'])
        if self.download_info['size'] == 0 and self.get_status() == DownloadStatus.METADATA:
            self.setText(1, 'unknown')
        else:
            self.setText(1, format_size(float(self.download_info['size'])))
        try:
            self.progress_slider.setValue(int(self.download_info['progress'] * 100))
        except RuntimeError:
            self._logger.error('The underlying GUI widget has already been removed.')
        if self.download_info['vod_mode']:
            self.setText(3, 'Streaming')
        else:
            status = DownloadStatus(self.download_info['status_code'])
            status_string = STATUS_STRING[status]
            self.setText(3, status_string)
        self.setText(4, f"{self.download_info['num_connected_seeds']} ({self.download_info['num_seeds']})")
        self.setText(5, f"{self.download_info['num_connected_peers']} ({self.download_info['num_peers']})")
        self.setText(6, format_speed(self.download_info['speed_down']))
        self.setText(7, format_speed(self.download_info['speed_up']))
        self.setText(8, f"{float(self.download_info['ratio']):.3f}")
        self.setText(9, 'yes' if self.download_info['anon_download'] else 'no')
        self.setText(10, str(self.download_info['hops']) if self.download_info['anon_download'] else '-')
        self.setText(12, datetime.fromtimestamp(int(self.download_info['time_added'])).strftime('%Y-%m-%d %H:%M'))
        eta_text = '-'
        if self.get_status() == DownloadStatus.DOWNLOADING:
            eta_text = duration_to_string(self.download_info['eta'])
        self.setText(11, eta_text)

    def __lt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if not self.download_info or not isinstance(other, DownloadWidgetItem):
            return True
        elif not other.download_info:
            return False
        column = self.treeWidget().sortColumn()
        if column == 1:
            return float(self.download_info['size']) > float(other.download_info['size'])
        elif column == 2:
            return int(self.download_info['progress'] * 100) > int(other.download_info['progress'] * 100)
        elif column == 4:
            return self.download_info['num_seeds'] > other.download_info['num_seeds']
        elif column == 5:
            return self.download_info['num_peers'] > other.download_info['num_peers']
        elif column == 6:
            return float(self.download_info['speed_down']) > float(other.download_info['speed_down'])
        elif column == 7:
            return float(self.download_info['speed_up']) > float(other.download_info['speed_up'])
        elif column == 8:
            return float(self.download_info['ratio']) > float(other.download_info['ratio'])
        elif column == 11:
            return (float(self.download_info['eta']) or float('inf')) > (float(other.download_info['eta']) or float('inf'))
        elif column == 12:
            return int(self.download_info['time_added']) > int(other.download_info['time_added'])
        return self.text(column) > other.text(column)