from typing import Optional
from types import SimpleNamespace as namespace
import numpy as np
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils.concurrent import TaskState, ConcurrentWidgetMixin
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.utils.widget import OWDataProjectionWidget

class Result(namespace):
    embedding = None

def run(data: Table, embedding: Optional[np.ndarray], state: TaskState):
    if False:
        print('Hello World!')
    res = Result(embedding=embedding)
    (step, steps) = (0, 10)
    state.set_status('Calculating...')
    while step < steps:
        for _ in range(steps):
            x_data = np.array(np.mean(data.X, axis=1))
            if x_data.ndim == 2:
                x_data = x_data.ravel()
            y_data = np.random.rand(len(x_data))
            embedding = np.vstack((x_data, y_data)).T
        step += 1
        if step % (steps / 10) == 0:
            state.set_progress_value(100 * step / steps)
        if state.is_interruption_requested():
            return res
        res.embedding = embedding
        state.set_partial_result(res)
    return res

class OWConcurrentWidget(OWDataProjectionWidget, ConcurrentWidgetMixin):
    name = 'Projection'
    param = Setting(0)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        OWDataProjectionWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)
        self.embedding = None

    def _add_controls(self):
        if False:
            while True:
                i = 10
        box = gui.vBox(self.controlArea, True)
        gui.comboBox(box, self, 'param', label='Parameter:', items=['Param A', 'Param B'], labelWidth=80, callback=self.__param_combo_changed)
        self.run_button = gui.button(box, self, 'Start', self._toggle_run)
        super()._add_controls()

    def __param_combo_changed(self):
        if False:
            return 10
        self._run()

    def _toggle_run(self):
        if False:
            print('Hello World!')
        if self.task is not None:
            self.cancel()
            self.run_button.setText('Resume')
            self.commit.deferred()
        else:
            self._run()

    def _run(self):
        if False:
            for i in range(10):
                print('nop')
        if self.data is None:
            return
        self.run_button.setText('Stop')
        self.start(run, self.data, self.embedding)

    def on_partial_result(self, result: Result):
        if False:
            i = 10
            return i + 15
        assert isinstance(result.embedding, np.ndarray)
        assert len(result.embedding) == len(self.data)
        first_result = self.embedding is None
        self.embedding = result.embedding
        if first_result:
            self.setup_plot()
        else:
            self.graph.update_coordinates()
            self.graph.update_density()

    def on_done(self, result: Result):
        if False:
            print('Hello World!')
        assert isinstance(result.embedding, np.ndarray)
        assert len(result.embedding) == len(self.data)
        self.embedding = result.embedding
        self.run_button.setText('Start')
        self.commit.deferred()

    def on_exception(self, ex: Exception):
        if False:
            for i in range(10):
                print('nop')
        raise ex

    @OWDataProjectionWidget.Inputs.data
    def set_data(self, data: Table):
        if False:
            return 10
        super().set_data(data)
        if self._invalidated:
            self._run()

    def get_embedding(self):
        if False:
            print('Hello World!')
        if self.embedding is None:
            self.valid_data = None
            return None
        self.valid_data = np.all(np.isfinite(self.embedding), 1)
        return self.embedding

    def clear(self):
        if False:
            return 10
        super().clear()
        self.cancel()
        self.embedding = None

    def onDeleteWidget(self):
        if False:
            for i in range(10):
                print('nop')
        self.shutdown()
        super().onDeleteWidget()
if __name__ == '__main__':
    table = Table('iris')
    WidgetPreview(OWConcurrentWidget).run(set_data=table, set_subset_data=table[::10])