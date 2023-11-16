from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import pyqtgraph as pg
import requests
import requests_cache
from collections import defaultdict
from datetime import datetime, timedelta, date
from itertools import cycle
import sys
import time
import traceback
requests_cache.install_cache('http_cache')
DEFAULT_BASE_CURRENCY = 'EUR'
DEFAULT_DISPLAY_CURRENCIES = ['CAD', 'CYP', 'AUD', 'USD', 'EUR', 'GBP', 'NZD', 'SGD']
HISTORIC_DAYS_N = 180
BREWER12PAIRED = cycle(['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928'])
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
DATE_REQUEST_OFFSETS = [0]
current = [(0, HISTORIC_DAYS_N)]
while current:
    (a, b) = current.pop(0)
    n = (a + b) // 2
    DATE_REQUEST_OFFSETS.append(n)
    if abs(a - n) > 1:
        current.insert(0, (a, n))
    if abs(b - n) > 1:
        current.append((b, n))

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.
    """
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    progress = pyqtSignal(int)
    data = pyqtSignal(int, dict)
    cancel = pyqtSignal()

class UpdateWorker(QRunnable):
    """
    Worker thread for updating currency.
    """
    signals = WorkerSignals()
    is_interrupted = False

    def __init__(self, base_currency):
        if False:
            for i in range(10):
                print('nop')
        super(UpdateWorker, self).__init__()
        self.base_currency = base_currency
        self.signals.cancel.connect(self.cancel)

    @pyqtSlot()
    def run(self):
        if False:
            i = 10
            return i + 15
        try:
            today = date.today()
            total_requests = len(DATE_REQUEST_OFFSETS)
            for (n, offset) in enumerate(DATE_REQUEST_OFFSETS, 1):
                when = today - timedelta(days=offset)
                url = 'http://api.fixer.io/{}'.format(when.isoformat())
                r = requests.get(url, params={'base': self.base_currency})
                r.raise_for_status()
                data = r.json()
                rates = data['rates']
                rates[self.base_currency] = 1.0
                self.signals.data.emit(offset, rates)
                self.signals.progress.emit(int(100 * n / total_requests))
                if not r.from_cache:
                    time.sleep(1)
                if self.is_interrupted:
                    break
        except Exception as e:
            print(e)
            (exctype, value) = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
            return
        self.signals.finished.emit()

    def cancel(self):
        if False:
            i = 10
            return i + 15
        self.is_interrupted = True

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MainWindow, self).__init__(*args, **kwargs)
        layout = QHBoxLayout()
        self.ax = pg.PlotWidget()
        self.ax.showGrid(True, True)
        self.line = pg.InfiniteLine(pos=-20, pen=pg.mkPen('k', width=3), movable=False)
        self.ax.addItem(self.line)
        self.ax.setLimits(xMin=-HISTORIC_DAYS_N + 1, xMax=0)
        self.ax.getPlotItem().scene().sigMouseMoved.connect(self.mouse_move_handler)
        self.base_currency = DEFAULT_BASE_CURRENCY
        self._data_lines = dict()
        self._data_items = dict()
        self._data_colors = dict()
        self._data_visible = DEFAULT_DISPLAY_CURRENCIES
        self._last_updated = None
        self.listView = QTableView()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(['Currency', 'Rate'])
        self.model.itemChanged.connect(self.check_check_state)
        self.listView.setModel(self.model)
        self.threadpool = QThreadPool()
        self.worker = False
        layout.addWidget(self.ax)
        layout.addWidget(self.listView)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        self.listView.setFixedSize(226, 400)
        self.setFixedSize(650, 400)
        toolbar = QToolBar('Main')
        self.addToolBar(toolbar)
        self.currencyList = QComboBox()
        toolbar.addWidget(self.currencyList)
        self.update_currency_list(DEFAULT_DISPLAY_CURRENCIES)
        self.currencyList.setCurrentText(self.base_currency)
        self.currencyList.currentTextChanged.connect(self.change_base_currency)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        toolbar.addWidget(self.progress)
        self.refresh_historic_rates()
        self.setWindowTitle('Doughnut')
        self.show()

    def update_currency_list(self, currencies):
        if False:
            for i in range(10):
                print('nop')
        for currency in currencies:
            if self.currencyList.findText(currency) == -1:
                self.currencyList.addItem(currency)
        self.currencyList.model().sort(0)

    def check_check_state(self, i):
        if False:
            return 10
        if not i.isCheckable():
            return
        currency = i.text()
        checked = i.checkState() == Qt.Checked
        if currency in self._data_visible:
            if not checked:
                self._data_visible.remove(currency)
                self.redraw()
        elif checked:
            self._data_visible.append(currency)
            self.redraw()

    def get_currency_color(self, currency):
        if False:
            for i in range(10):
                print('nop')
        if currency not in self._data_colors:
            self._data_colors[currency] = next(BREWER12PAIRED)
        return self._data_colors[currency]

    def add_data_row(self, currency):
        if False:
            print('Hello World!')
        citem = QStandardItem()
        citem.setText(currency)
        citem.setForeground(QBrush(QColor(self.get_currency_color(currency))))
        citem.setColumnCount(2)
        citem.setCheckable(True)
        if currency in DEFAULT_DISPLAY_CURRENCIES:
            citem.setCheckState(Qt.Checked)
        vitem = QStandardItem()
        vitem.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.model.setColumnCount(2)
        self.model.appendRow([citem, vitem])
        self.model.sort(0)
        return (citem, vitem)

    def get_or_create_data_row(self, currency):
        if False:
            while True:
                i = 10
        if currency not in self._data_items:
            self._data_items[currency] = self.add_data_row(currency)
        return self._data_items[currency]

    def mouse_move_handler(self, pos):
        if False:
            for i in range(10):
                print('nop')
        pos = self.ax.getViewBox().mapSceneToView(pos)
        self.line.setPos(pos.x())
        self.update_data_viewer(int(pos.x()))

    def update_data_row(self, currency, value):
        if False:
            print('Hello World!')
        (citem, vitem) = self.get_or_create_data_row(currency)
        vitem.setText('%.4f' % value)

    def update_data_viewer(self, d):
        if False:
            while True:
                i = 10
        try:
            data = self.data[d]
        except IndexError:
            return
        if not data:
            return
        for (k, v) in data.items():
            self.update_data_row(k, v)

    def change_base_currency(self, currency):
        if False:
            return 10
        self.base_currency = currency
        self.refresh_historic_rates()

    def refresh_historic_rates(self):
        if False:
            i = 10
            return i + 15
        if self.worker:
            self.worker.signals.cancel.emit()
        self.data = [None] * HISTORIC_DAYS_N
        self.worker = UpdateWorker(self.base_currency)
        self.worker.signals.data.connect(self.result_data_callback)
        self.worker.signals.finished.connect(self.refresh_finished)
        self.worker.signals.progress.connect(self.progress_callback)
        self.threadpool.start(self.worker)

    def result_data_callback(self, n, rates):
        if False:
            print('Hello World!')
        self.data[n] = rates
        if self._last_updated is None or self._last_updated < datetime.now() - timedelta(seconds=1):
            self.redraw()
            self._last_updated = datetime.now()

    def progress_callback(self, progress):
        if False:
            return 10
        self.progress.setValue(progress)

    def refresh_finished(self):
        if False:
            for i in range(10):
                print('nop')
        self.worker = False
        self.redraw()
        self.update_currency_list(self._data_items.keys())

    def redraw(self):
        if False:
            i = 10
            return i + 15
        '\n        Process data from store and prefer to draw.\n        :return:\n        '
        today = date.today()
        plotd = defaultdict(list)
        x_ticks = []
        tick_step_size = HISTORIC_DAYS_N / 6
        for (n, data) in enumerate(self.data):
            if data:
                for (currency, v) in data.items():
                    plotd[currency].append((-n, v))
            when = today - timedelta(days=n)
            if (n - tick_step_size // 2) % tick_step_size == 0:
                x_ticks.append((-n, when.strftime('%d-%m')))
        keys = sorted(plotd.keys())
        (y_min, y_max) = (sys.maxsize, 0)
        for currency in keys:
            (x, y) = zip(*plotd[currency])
            if currency in self._data_visible:
                y_min = min(y_min, *y)
                y_max = max(y_max, *y)
            else:
                (x, y) = ([], [])
            if currency in self._data_lines:
                self._data_lines[currency].setData(x, y)
            else:
                self._data_lines[currency] = self.ax.plot(x, y, pen=pg.mkPen(self.get_currency_color(currency), width=2))
        self.ax.setLimits(yMin=y_min * 0.9, yMax=y_max * 1.1)
        self.ax.getAxis('bottom').setTicks([x_ticks, []])
if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    app.exec_()