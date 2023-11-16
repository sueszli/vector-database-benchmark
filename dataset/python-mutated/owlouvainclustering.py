from functools import partial
from concurrent import futures
from concurrent.futures import Future
from types import SimpleNamespace as namespace
from typing import Optional, Callable, Tuple, Any
import numpy as np
import scipy.sparse as sp
import networkx as nx
from AnyQt.QtCore import Qt, QObject, QTimer, pyqtSignal as Signal, pyqtSlot as Slot
from AnyQt.QtWidgets import QSlider, QCheckBox, QWidget, QLabel
from Orange.clustering.louvain import matrix_to_knn_graph, Louvain
from Orange.data import Table, DiscreteVariable
from Orange.data.util import get_unique_names, array_equal
from Orange import preprocess
from Orange.projection import PCA
from Orange.widgets import widget, gui, report
from Orange.widgets.settings import Setting
from Orange.widgets.utils.annotated_data import add_columns, ANNOTATED_DATA_SIGNAL_NAME
from Orange.widgets.utils.concurrent import FutureWatcher
from Orange.widgets.utils.localization import pl
from Orange.widgets.utils.signals import Input, Output
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Msg
try:
    from orangecontrib.network.network import Network
except ImportError:
    Network = None
_MAX_PCA_COMPONENTS = 50
_DEFAULT_PCA_COMPONENTS = 25
_MAX_K_NEIGBOURS = 200
_DEFAULT_K_NEIGHBORS = 30
METRICS = [('Euclidean', 'l2'), ('Manhattan', 'l1'), ('Cosine', 'cosine')]

class OWLouvainClustering(widget.OWWidget):
    name = 'Louvain Clustering'
    description = 'Detects communities in a network of nearest neighbors.'
    icon = 'icons/LouvainClustering.svg'
    priority = 2110
    settings_version = 2
    want_main_area = False
    resizing_enabled = False

    class Inputs:
        data = Input('Data', Table, default=True)

    class Outputs:
        annotated_data = Output(ANNOTATED_DATA_SIGNAL_NAME, Table, default=True)
        if Network is not None:
            graph = Output('Network', Network)
    apply_pca = Setting(True)
    pca_components = Setting(_DEFAULT_PCA_COMPONENTS)
    normalize = Setting(True)
    metric_idx = Setting(0)
    k_neighbors = Setting(_DEFAULT_K_NEIGHBORS)
    resolution = Setting(1.0)
    auto_commit = Setting(False)

    class Information(widget.OWWidget.Information):
        modified = Msg('Press commit to recompute clusters and send new data')

    class Error(widget.OWWidget.Error):
        empty_dataset = Msg('No features in data')

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.data = None
        self.preprocessed_data = None
        self.pca_projection = None
        self.graph = None
        self.partition = None
        self.__executor = futures.ThreadPoolExecutor(max_workers=1)
        self.__task = None
        self.__invalidated = False
        self.__commit_timer = QTimer(self, singleShot=True)
        self.__commit_timer.timeout.connect(self.commit)
        info_box = gui.vBox(self.controlArea, 'Info')
        self.info_label = gui.widgetLabel(info_box, 'No data on input.')
        preprocessing_box = gui.vBox(self.controlArea, 'Preprocessing')
        self.normalize_cbx = gui.checkBox(preprocessing_box, self, 'normalize', label='Normalize data', callback=self._invalidate_preprocessed_data, attribute=Qt.WA_LayoutUsesWidgetRect)
        self.apply_pca_cbx = gui.checkBox(preprocessing_box, self, 'apply_pca', label='Apply PCA preprocessing', callback=self._apply_pca_changed, attribute=Qt.WA_LayoutUsesWidgetRect)
        self.pca_components_slider = gui.hSlider(preprocessing_box, self, 'pca_components', label='PCA Components: ', minValue=2, maxValue=_MAX_PCA_COMPONENTS, callback=self._invalidate_pca_projection, tracking=False)
        graph_box = gui.vBox(self.controlArea, 'Graph parameters')
        self.metric_combo = gui.comboBox(graph_box, self, 'metric_idx', label='Distance metric', items=[m[0] for m in METRICS], callback=self._invalidate_graph, orientation=Qt.Horizontal)
        self.k_neighbors_spin = gui.spin(graph_box, self, 'k_neighbors', minv=1, maxv=_MAX_K_NEIGBOURS, label='k neighbors', controlWidth=80, alignment=Qt.AlignRight, callback=self._invalidate_graph)
        self.resolution_spin = gui.hSlider(graph_box, self, 'resolution', minValue=0, maxValue=5.0, step=0.1, label='Resolution', intOnly=False, labelFormat='%.1f', callback=self._invalidate_partition, tracking=False)
        self.resolution_spin.parent().setToolTip('The resolution parameter affects the number of clusters to find. Smaller values tend to produce more clusters and larger values retrieve less clusters.')
        self.apply_button = gui.auto_apply(self.buttonsArea, self, 'auto_commit', commit=lambda : self.commit(), callback=lambda : self._on_auto_commit_changed()).button

    def _preprocess_data(self):
        if False:
            return 10
        if self.preprocessed_data is None:
            if self.normalize:
                normalizer = preprocess.Normalize(center=False)
                self.preprocessed_data = normalizer(self.data)
            else:
                self.preprocessed_data = self.data

    def _apply_pca_changed(self):
        if False:
            print('Hello World!')
        self.controls.pca_components.setEnabled(self.apply_pca)
        self._invalidate_graph()

    def _invalidate_preprocessed_data(self):
        if False:
            while True:
                i = 10
        self.preprocessed_data = None
        self._invalidate_pca_projection()
        if not self.apply_pca:
            self._invalidate_graph()

    def _invalidate_pca_projection(self):
        if False:
            for i in range(10):
                print('nop')
        self.pca_projection = None
        if not self.apply_pca:
            return
        self._invalidate_graph()
        self._set_modified(True)

    def _invalidate_graph(self):
        if False:
            print('Hello World!')
        self.graph = None
        self._invalidate_partition()
        self._set_modified(True)

    def _invalidate_partition(self):
        if False:
            return 10
        self.partition = None
        self._invalidate_output()
        self.Information.modified()
        self._set_modified(True)

    def _invalidate_output(self):
        if False:
            return 10
        self.__invalidated = True
        if self.__task is not None:
            self.__cancel_task(wait=False)
        if self.auto_commit:
            self.__commit_timer.start()
        else:
            self.__set_state_ready()

    def _set_modified(self, state):
        if False:
            print('Hello World!')
        '\n        Mark the widget (GUI) as containing modified state.\n        '
        if self.data is None:
            state = False
        elif self.auto_commit:
            state = False
        self.apply_button.setEnabled(state)
        self.Information.modified(shown=state)

    def _on_auto_commit_changed(self):
        if False:
            while True:
                i = 10
        if self.auto_commit and self.__invalidated:
            self.commit()

    def cancel(self):
        if False:
            for i in range(10):
                print('nop')
        'Cancel any running jobs.'
        self.__cancel_task(wait=False)
        self.__set_state_ready()

    def commit(self):
        if False:
            return 10
        self.__commit_timer.stop()
        self.__invalidated = False
        self._set_modified(False)
        self.__cancel_task(wait=False)
        if self.data is None:
            self.__set_state_ready()
            return
        self.Error.clear()
        if self.partition is not None:
            self.__set_state_ready()
            self._send_data()
            return
        self._preprocess_data()
        state = TaskState(self)
        if self.apply_pca:
            if self.pca_projection is not None:
                data = self.pca_projection
                pca_components = None
            else:
                data = self.preprocessed_data
                pca_components = self.pca_components
        else:
            data = self.preprocessed_data
            pca_components = None
        if self.graph is not None:
            graph = self.graph
            k_neighbors = metric = None
        else:
            (k_neighbors, metric) = (self.k_neighbors, METRICS[self.metric_idx][1])
            graph = None
        if graph is None:
            task = partial(run_on_data, data, pca_components=pca_components, normalize=self.normalize, k_neighbors=k_neighbors, metric=metric, resolution=self.resolution, state=state)
        else:
            task = partial(run_on_graph, graph, resolution=self.resolution, state=state)
        self.info_label.setText('Running...')
        self.__set_state_busy()
        self.__start_task(task, state)

    @Slot(object)
    def __set_partial_results(self, result):
        if False:
            while True:
                i = 10
        (which, res) = result
        if which == 'pca_projection':
            assert isinstance(res, Table) and len(res) == len(self.data)
            self.pca_projection = res
        elif which == 'graph':
            assert isinstance(res, nx.Graph)
            self.graph = res
        elif which == 'partition':
            assert isinstance(res, np.ndarray)
            self.partition = res
        else:
            assert False, which

    @Slot(object)
    def __on_done(self, future):
        if False:
            print('Hello World!')
        assert future.done()
        assert self.__task is not None
        assert self.__task.future is future
        assert self.__task.watcher.future() is future
        (self.__task, task) = (None, self.__task)
        task.deleteLater()
        self.__set_state_ready()
        result = future.result()
        self.__set_results(result)

    @Slot(str)
    def setStatusMessage(self, text):
        if False:
            for i in range(10):
                print('nop')
        super().setStatusMessage(text)

    @Slot(float)
    def progressBarSet(self, value, *a, **kw):
        if False:
            print('Hello World!')
        super().progressBarSet(value, *a, **kw)

    def __set_state_ready(self):
        if False:
            return 10
        self.progressBarFinished()
        self.setInvalidated(False)
        self.setStatusMessage('')

    def __set_state_busy(self):
        if False:
            print('Hello World!')
        self.progressBarInit()
        self.setInvalidated(True)

    def __start_task(self, task, state):
        if False:
            i = 10
            return i + 15
        assert self.__task is None
        state.status_changed.connect(self.setStatusMessage)
        state.progress_changed.connect(self.progressBarSet)
        state.partial_result_ready.connect(self.__set_partial_results)
        state.watcher.done.connect(self.__on_done)
        state.start(self.__executor, task)
        state.setParent(self)
        self.__task = state

    def __cancel_task(self, wait=True):
        if False:
            for i in range(10):
                print('nop')
        if self.__task is not None:
            (state, self.__task) = (self.__task, None)
            state.cancel()
            state.partial_result_ready.disconnect(self.__set_partial_results)
            state.status_changed.disconnect(self.setStatusMessage)
            state.progress_changed.disconnect(self.progressBarSet)
            state.watcher.done.disconnect(self.__on_done)
            if wait:
                futures.wait([state.future])
                state.deleteLater()
            else:
                w = FutureWatcher(state.future, parent=state)
                w.done.connect(state.deleteLater)

    def __set_results(self, results):
        if False:
            while True:
                i = 10
        if results.pca_projection is not None:
            assert self.pca_components == results.pca_components
            assert self.pca_projection is results.pca_projection
            self.pca_projection = results.pca_projection
        if results.graph is not None:
            assert results.metric == METRICS[self.metric_idx][1]
            assert results.k_neighbors == self.k_neighbors
            assert self.graph is results.graph
            self.graph = results.graph
        if results.partition is not None:
            assert results.resolution == self.resolution
            assert self.partition is results.partition
            self.partition = results.partition
        num_clusters = len(np.unique(self.partition))
        self.info_label.setText(f"{num_clusters} {pl(num_clusters, 'cluster')} found.")
        self._send_data()

    def _send_data(self):
        if False:
            print('Hello World!')
        if self.partition is None or self.data is None:
            return
        domain = self.data.domain
        counts = np.bincount(self.partition)
        indices = np.argsort(counts)[::-1]
        index_map = {n: o for (n, o) in zip(indices, range(len(indices)))}
        new_partition = list(map(index_map.get, self.partition))
        cluster_var = DiscreteVariable(get_unique_names(domain, 'Cluster'), values=['C%d' % (i + 1) for (i, _) in enumerate(np.unique(new_partition))])
        new_domain = add_columns(domain, metas=[cluster_var])
        new_table = self.data.transform(new_domain)
        with new_table.unlocked(new_table.metas):
            new_table.set_column(cluster_var, new_partition)
        self.Outputs.annotated_data.send(new_table)
        if Network is not None:
            n_edges = self.graph.number_of_edges()
            edges = sp.coo_matrix((np.ones(n_edges), np.array(self.graph.edges()).T), shape=(n_edges, n_edges))
            graph = Network(new_table, edges)
            self.Outputs.graph.send(graph)

    @Inputs.data
    def set_data(self, data):
        if False:
            i = 10
            return i + 15
        self.Error.clear()
        (prev_data, self.data) = (self.data, data)
        self.controls.pca_components.setEnabled(self.apply_pca)
        if prev_data and self.data and array_equal(prev_data.X, self.data.X):
            if self.auto_commit and (not self.isInvalidated()):
                self._send_data()
            return
        self.cancel()
        self.Outputs.annotated_data.send(None)
        if Network is not None:
            self.Outputs.graph.send(None)
        self.clear()
        self._invalidate_pca_projection()
        if self.data is not None and len(self.data.domain.attributes) < 1:
            self.Error.empty_dataset()
            self.data = None
        if self.data is None:
            return
        n_attrs = len(data.domain.attributes)
        self.pca_components_slider.setMaximum(min(_MAX_PCA_COMPONENTS, n_attrs))
        self.k_neighbors_spin.setMaximum(min(_MAX_K_NEIGBOURS, len(data) - 1))
        self.info_label.setText('Clustering not yet run.')
        self.commit()

    def clear(self):
        if False:
            print('Hello World!')
        self.__cancel_task(wait=False)
        self.preprocessed_data = None
        self.pca_projection = None
        self.graph = None
        self.partition = None
        self.Error.clear()
        self.Information.modified.clear()
        self.info_label.setText('No data on input.')

    def onDeleteWidget(self):
        if False:
            print('Hello World!')
        self.__cancel_task(wait=True)
        self.__executor.shutdown(True)
        self.clear()
        self.data = None
        super().onDeleteWidget()

    def send_report(self):
        if False:
            for i in range(10):
                print('nop')
        pca = report.bool_str(self.apply_pca)
        if self.apply_pca:
            pca += f", {self.pca_components} {pl(self.pca_components, 'component')}"
        self.report_items((('Normalize data', report.bool_str(self.normalize)), ('PCA preprocessing', pca), ('Metric', METRICS[self.metric_idx][0]), ('k neighbors', self.k_neighbors), ('Resolution', self.resolution)))

    @classmethod
    def migrate_settings(cls, settings, version):
        if False:
            i = 10
            return i + 15
        if version < 2 and 'context_settings' in settings:
            try:
                current_context = settings['context_settings'][0]
                for n in ['apply_pca', 'k_neighbors', 'metric_idx', 'normalize', 'pca_components', 'resolution']:
                    if n in current_context.values:
                        settings[n] = current_context.values[n][0]
            except:
                pass
            finally:
                del settings['context_settings']

class TaskState(QObject):
    status_changed = Signal(str)
    _p_status_changed = Signal(str)
    progress_changed = Signal(float)
    _p_progress_changed = Signal(float)
    partial_result_ready = Signal(object)
    _p_partial_result_ready = Signal(object)

    def __init__(self, *args):
        if False:
            return 10
        super().__init__(*args)
        self.__future = None
        self.watcher = FutureWatcher()
        self.__interuption_requested = False
        self.__progress = 0
        self._p_status_changed.connect(self.status_changed, Qt.QueuedConnection)
        self._p_progress_changed.connect(self.progress_changed, Qt.QueuedConnection)
        self._p_partial_result_ready.connect(self.partial_result_ready, Qt.QueuedConnection)

    @property
    def future(self):
        if False:
            print('Hello World!')
        return self.__future

    def set_status(self, text):
        if False:
            print('Hello World!')
        self._p_status_changed.emit(text)

    def set_progress_value(self, value):
        if False:
            return 10
        if round(value, 1) > round(self.__progress, 1):
            self._p_progress_changed.emit(value)
            self.__progress = value

    def set_partial_results(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._p_partial_result_ready.emit(value)

    def is_interuption_requested(self):
        if False:
            print('Hello World!')
        return self.__interuption_requested

    def start(self, executor, func=None):
        if False:
            i = 10
            return i + 15
        assert self.future is None
        assert not self.__interuption_requested
        self.__future = executor.submit(func)
        self.watcher.setFuture(self.future)
        return self.future

    def cancel(self):
        if False:
            print('Hello World!')
        assert not self.__interuption_requested
        self.__interuption_requested = True
        if self.future is not None:
            rval = self.future.cancel()
        else:
            rval = True
        return rval

class InteruptRequested(BaseException):
    pass

class Results(namespace):
    pca_projection = None
    pca_components = None
    normalize = None
    k_neighbors = None
    metric = None
    graph = None
    resolution = None
    partition = None

def run_on_data(data, normalize, pca_components, k_neighbors, metric, resolution, state):
    if False:
        print('Hello World!')
    '\n    Run the louvain clustering on `data`.\n\n    state is used to report progress and partial results. Returns early if\n    `task.is_interuption_requested()` returns true.\n\n    Parameters\n    ----------\n    data : Table\n        Data table\n    normalize : bool\n        If `True`, the data is first normalized before computing PCA.\n    pca_components : Optional[int]\n        If not `None` then the data is first projected onto first\n        `pca_components` principal components.\n    k_neighbors : int\n        Passed to `table_to_knn_graph`\n    metric : str\n        Passed to `table_to_knn_graph`\n    resolution : float\n        Passed to `Louvain`\n    state : TaskState\n\n    Returns\n    -------\n    res : Results\n    '
    state = state
    res = Results(normalize=normalize, pca_components=pca_components, k_neighbors=k_neighbors, metric=metric, resolution=resolution)
    step = 0
    if state.is_interuption_requested():
        return res
    if pca_components is not None:
        steps = 3
        state.set_status('Computing PCA...')
        pca = PCA(n_components=pca_components, random_state=0)
        data = res.pca_projection = pca(data)(data)
        assert isinstance(data, Table)
        state.set_partial_results(('pca_projection', res.pca_projection))
        step += 1
    else:
        steps = 2
    if state.is_interuption_requested():
        return res
    state.set_progress_value(100.0 * step / steps)
    state.set_status('Building graph...')
    louvain = Louvain(resolution=resolution, random_state=0)
    data = louvain.preprocess(data)
    if state.is_interuption_requested():
        return res

    def pcallback(val):
        if False:
            for i in range(10):
                print('nop')
        state.set_progress_value((100.0 * step + 100 * val) / steps)
        if state.is_interuption_requested():
            raise InteruptRequested()
    try:
        res.graph = graph = matrix_to_knn_graph(data.X, k_neighbors=k_neighbors, metric=metric, progress_callback=pcallback)
    except InteruptRequested:
        return res
    state.set_partial_results(('graph', res.graph))
    step += 1
    state.set_progress_value(100 * step / steps)
    state.set_status('Detecting communities...')
    if state.is_interuption_requested():
        return res
    res.partition = louvain(graph)
    state.set_partial_results(('partition', res.partition))
    return res

def run_on_graph(graph, resolution, state):
    if False:
        while True:
            i = 10
    '\n    Run the louvain clustering on `graph`.\n    '
    state = state
    res = Results(resolution=resolution)
    louvain = Louvain(resolution=resolution, random_state=0)
    state.set_status('Detecting communities...')
    if state.is_interuption_requested():
        return res
    partition = louvain(graph)
    res.partition = partition
    state.set_partial_results(('partition', res.partition))
    return res
if __name__ == '__main__':
    WidgetPreview(OWLouvainClustering).run(Table('iris'))