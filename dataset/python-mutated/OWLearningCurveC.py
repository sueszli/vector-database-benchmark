import logging
from collections import OrderedDict
from functools import reduce, partial
import numpy
from AnyQt.QtWidgets import QTableWidget, QTableWidgetItem
from AnyQt.QtCore import QThread, pyqtSlot
import Orange.data
import Orange.classification
import Orange.evaluation
from Orange.widgets import widget, gui, settings
from Orange.evaluation.testing import Results
import concurrent.futures
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
from Orange.widgets.utils.widgetpreview import WidgetPreview

class Task:
    """
    A class that will hold the state for an learner evaluation.
    """
    future = ...
    watcher = ...
    cancelled = False

    def cancel(self):
        if False:
            return 10
        '\n        Cancel the task.\n\n        Set the `cancelled` field to True and block until the future is done.\n        '
        self.cancelled = True
        self.future.cancel()
        concurrent.futures.wait([self.future])

class OWLearningCurveC(widget.OWWidget):
    name = 'Learning Curve (C)'
    description = 'Takes a dataset and a set of learners and shows a learning curve in a table'
    icon = 'icons/LearningCurve.svg'
    priority = 1010
    inputs = [('Data', Orange.data.Table, 'set_dataset', widget.Default), ('Test Data', Orange.data.Table, 'set_testdataset'), ('Learner', Orange.classification.Learner, 'set_learner', widget.Multiple + widget.Default)]
    folds = settings.Setting(5)
    steps = settings.Setting(10)
    scoringF = settings.Setting(0)
    commitOnChange = settings.Setting(True)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.updateCurvePoints()
        self.scoring = [('Classification Accuracy', Orange.evaluation.scoring.CA), ('AUC', Orange.evaluation.scoring.AUC), ('Precision', Orange.evaluation.scoring.Precision), ('Recall', Orange.evaluation.scoring.Recall)]
        self.data = None
        self.testdata = None
        self.learners = OrderedDict()
        self.results = OrderedDict()
        self.curves = OrderedDict()
        self._task = None
        self._executor = ThreadExecutor()
        box = gui.widgetBox(self.controlArea, 'Info')
        self.infoa = gui.widgetLabel(box, 'No data on input.')
        self.infob = gui.widgetLabel(box, 'No learners.')
        gui.separator(self.controlArea)
        box = gui.widgetBox(self.controlArea, 'Evaluation Scores')
        gui.comboBox(box, self, 'scoringF', items=[x[0] for x in self.scoring], callback=self._invalidate_curves)
        gui.separator(self.controlArea)
        box = gui.widgetBox(self.controlArea, 'Options')
        gui.spin(box, self, 'folds', 2, 100, step=1, label='Cross validation folds:  ', keyboardTracking=False, callback=lambda : self._invalidate_results() if self.commitOnChange else None)
        gui.spin(box, self, 'steps', 2, 100, step=1, label='Learning curve points:  ', keyboardTracking=False, callback=[self.updateCurvePoints, lambda : self._invalidate_results() if self.commitOnChange else None])
        gui.checkBox(box, self, 'commitOnChange', 'Apply setting on any change')
        self.commitBtn = gui.button(box, self, 'Apply Setting', callback=self._invalidate_results, disabled=True)
        gui.rubber(self.controlArea)
        self.table = gui.table(self.mainArea, selectionMode=QTableWidget.NoSelection)

    def set_dataset(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Set the input train dataset.'
        for id in list(self.results):
            self.results[id] = None
        for id in list(self.curves):
            self.curves[id] = None
        self.data = data
        if data is not None:
            self.infoa.setText('%d instances in input dataset' % len(data))
        else:
            self.infoa.setText('No data on input.')
        self.commitBtn.setEnabled(self.data is not None)

    def set_testdataset(self, testdata):
        if False:
            print('Hello World!')
        'Set a separate test dataset.'
        for id in list(self.results):
            self.results[id] = None
        for id in list(self.curves):
            self.curves[id] = None
        self.testdata = testdata

    def set_learner(self, learner, id):
        if False:
            while True:
                i = 10
        'Set the input learner for channel id.'
        if id in self.learners:
            if learner is None:
                del self.learners[id]
                del self.results[id]
                del self.curves[id]
            else:
                self.learners[id] = learner
                self.results[id] = None
                self.curves[id] = None
        elif learner is not None:
            self.learners[id] = learner
            self.results[id] = None
            self.curves[id] = None
        if len(self.learners):
            self.infob.setText('%d learners on input.' % len(self.learners))
        else:
            self.infob.setText('No learners.')
        self.commitBtn.setEnabled(len(self.learners))

    def handleNewSignals(self):
        if False:
            i = 10
            return i + 15
        self._update()

    def _invalidate_curves(self):
        if False:
            return 10
        if self.data is not None:
            self._update_curve_points()
        self._update_table()

    def _invalidate_results(self):
        if False:
            print('Hello World!')
        for id in self.learners:
            self.curves[id] = None
            self.results[id] = None
        self._update()

    def _update(self):
        if False:
            return 10
        if self._task is not None:
            self.cancel()
        assert self._task is None
        if self.data is None:
            return
        need_update = [(id, learner) for (id, learner) in self.learners.items() if self.results[id] is None]
        if not need_update:
            return
        learners = [learner for (_, learner) in need_update]
        if self.testdata is None:
            learning_curve_func = partial(learning_curve, learners, self.data, folds=self.folds, proportions=self.curvePoints)
        else:
            learning_curve_func = partial(learning_curve_with_test_data, learners, self.data, self.testdata, times=self.folds, proportions=self.curvePoints)
        self._task = task = Task()
        set_progress = methodinvoke(self, 'setProgressValue', (float,))

        def callback(finished):
            if False:
                return 10
            if task.cancelled:
                raise KeyboardInterrupt()
            set_progress(finished * 100)
        learning_curve_func = partial(learning_curve_func, callback=callback)
        self.progressBarInit()
        task.future = self._executor.submit(learning_curve_func)
        task.watcher = FutureWatcher(task.future)
        task.watcher.done.connect(self._task_finished)

    @pyqtSlot(float)
    def setProgressValue(self, value):
        if False:
            print('Hello World!')
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        f : Future\n            The future instance holding the result of learner evaluation.\n        '
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()
        self._task = None
        self.progressBarFinished()
        try:
            results = f.result()
        except Exception as ex:
            log = logging.getLogger()
            log.exception(__name__, exc_info=True)
            self.error('Exception occurred during evaluation: {!r}'.format(ex))
            for key in self.results.keys():
                self.results[key] = None
        else:
            results = [list(Results.split_by_model(p_results)) for p_results in results]
            assert all((len(r.learners) == 1 for r1 in results for r in r1))
            assert len(results) == len(self.curvePoints)
            learners = [r.learners[0] for r in results[0]]
            learner_id = {learner: id_ for (id_, learner) in self.learners.items()}
            for (i, learner) in enumerate(learners):
                id_ = learner_id[learner]
                self.results[id_] = [p_results[i] for p_results in results]
        self._update_curve_points()
        self._update_table()

    def cancel(self):
        if False:
            print('Hello World!')
        '\n        Cancel the current task (if any).\n        '
        if self._task is not None:
            self._task.cancel()
            assert self._task.future.done()
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None

    def onDeleteWidget(self):
        if False:
            return 10
        self.cancel()
        super().onDeleteWidget()

    def _update_curve_points(self):
        if False:
            i = 10
            return i + 15
        for id in self.learners:
            curve = [self.scoring[self.scoringF][1](x)[0] for x in self.results[id]]
            self.curves[id] = curve

    def _update_table(self):
        if False:
            i = 10
            return i + 15
        self.table.setRowCount(0)
        self.table.setRowCount(len(self.curvePoints))
        self.table.setColumnCount(len(self.learners))
        self.table.setHorizontalHeaderLabels([learner.name for (_, learner) in self.learners.items()])
        self.table.setVerticalHeaderLabels(['{:.2f}'.format(p) for p in self.curvePoints])
        if self.data is None:
            return
        for (column, curve) in enumerate(self.curves.values()):
            for (row, point) in enumerate(curve):
                self.table.setItem(row, column, QTableWidgetItem('{:.5f}'.format(point)))
        for i in range(len(self.learners)):
            sh = self.table.sizeHintForColumn(i)
            cwidth = self.table.columnWidth(i)
            self.table.setColumnWidth(i, max(sh, cwidth))

    def updateCurvePoints(self):
        if False:
            for i in range(10):
                print('nop')
        self.curvePoints = [(x + 1.0) / self.steps for x in range(self.steps)]

    def test_run_signals(self):
        if False:
            i = 10
            return i + 15
        data = Orange.data.Table('iris')
        indices = numpy.random.permutation(len(data))
        traindata = data[indices[:-20]]
        testdata = data[indices[-20:]]
        self.set_dataset(traindata)
        self.set_testdataset(testdata)
        l1 = Orange.classification.NaiveBayesLearner()
        l1.name = 'Naive Bayes'
        self.set_learner(l1, 1)
        l2 = Orange.classification.LogisticRegressionLearner()
        l2.name = 'Logistic Regression'
        self.set_learner(l2, 2)
        l4 = Orange.classification.SklTreeLearner()
        l4.name = 'Decision Tree'
        self.set_learner(l4, 3)

def learning_curve(learners, data, folds=10, proportions=None, random_state=None, callback=None):
    if False:
        print('Hello World!')
    if proportions is None:
        proportions = numpy.linspace(0.0, 1.0, 10 + 1, endpoint=True)[1:]

    def select_proportion_preproc(data, p, rstate=None):
        if False:
            return 10
        assert 0 < p <= 1
        rstate = numpy.random.RandomState(None) if rstate is None else rstate
        indices = rstate.permutation(len(data))
        n = int(numpy.ceil(len(data) * p))
        return data[indices[:n]]
    if callback is not None:
        parts_count = len(proportions)
        callback_wrapped = lambda part: lambda value: callback(value / parts_count + part / parts_count)
    else:
        callback_wrapped = lambda part: None
    results = [Orange.evaluation.CrossValidation(data, learners, k=folds, preprocessor=lambda data, p=p: select_proportion_preproc(data, p), callback=callback_wrapped(i)) for (i, p) in enumerate(proportions)]
    return results

def learning_curve_with_test_data(learners, traindata, testdata, times=10, proportions=None, random_state=None, callback=None):
    if False:
        print('Hello World!')
    if proportions is None:
        proportions = numpy.linspace(0.0, 1.0, 10 + 1, endpoint=True)[1:]

    def select_proportion_preproc(data, p, rstate=None):
        if False:
            for i in range(10):
                print('nop')
        assert 0 < p <= 1
        rstate = numpy.random.RandomState(None) if rstate is None else rstate
        indices = rstate.permutation(len(data))
        n = int(numpy.ceil(len(data) * p))
        return data[indices[:n]]
    if callback is not None:
        parts_count = len(proportions) * times
        callback_wrapped = lambda part: lambda value: callback(value / parts_count + part / parts_count)
    else:
        callback_wrapped = lambda part: None
    results = [[Orange.evaluation.TestOnTestData(traindata, testdata, learners, preprocessor=lambda data, p=p: select_proportion_preproc(data, p), callback=callback_wrapped(i * times + t)) for t in range(times)] for (i, p) in enumerate(proportions)]
    results = [reduce(results_add, res, Orange.evaluation.Results()) for res in results]
    return results

def results_add(x, y):
    if False:
        i = 10
        return i + 15

    def is_empty(res):
        if False:
            while True:
                i = 10
        return getattr(res, 'models', None) is None and getattr(res, 'row_indices', None) is None
    if is_empty(x):
        return y
    elif is_empty(y):
        return x
    assert x.data is y.data
    assert x.domain is y.domain
    assert x.predicted.shape[0] == y.predicted.shape[0]
    assert len(x.learners) == len(y.learners)
    assert all((xl is yl for (xl, yl) in zip(x.learners, y.learners)))
    row_indices = numpy.hstack((x.row_indices, y.row_indices))
    predicted = numpy.hstack((x.predicted, y.predicted))
    actual = numpy.hstack((x.actual, y.actual))
    xprob = getattr(x, 'probabilities', None)
    yprob = getattr(y, 'probabilities', None)
    if xprob is None and yprob is None:
        prob = None
    elif xprob is not None and yprob is not None:
        prob = numpy.concatenate((xprob, yprob), axis=1)
    else:
        raise ValueError()
    res = Orange.evaluation.Results()
    res.data = x.data
    res.domain = x.domain
    res.learners = x.learners
    res.row_indices = row_indices
    res.actual = actual
    res.predicted = predicted
    res.folds = None
    if prob is not None:
        res.probabilities = prob
    if x.models is not None and y.models is not None:
        res.models = [xm + ym for (xm, ym) in zip(x.models, y.models)]
    nmodels = predicted.shape[0]
    xfailed = getattr(x, 'failed', None) or [False] * nmodels
    yfailed = getattr(y, 'failed', None) or [False] * nmodels
    assert len(xfailed) == len(yfailed)
    res.failed = [xe or ye for (xe, ye) in zip(xfailed, yfailed)]
    return res
if __name__ == '__main__':
    WidgetPreview(OWLearningCurveC).run()