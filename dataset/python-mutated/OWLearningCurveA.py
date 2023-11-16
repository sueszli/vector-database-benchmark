from collections import OrderedDict
import numpy
from AnyQt.QtWidgets import QTableWidget, QTableWidgetItem
import Orange.data
import Orange.classification
import Orange.evaluation
from Orange.widgets import gui, settings
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import OWWidget, Input
from Orange.evaluation.testing import Results

class OWLearningCurveA(OWWidget):
    name = 'Learning Curve (A)'
    description = 'Takes a dataset and a set of learners and shows a learning curve in a table.'
    icon = 'icons/LearningCurve.svg'
    priority = 1000

    class Inputs:
        data = Input('Data', Orange.data.Table)
        learner = Input('Learner', Orange.classification.Learner, multiple=True)
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
        self.learners = OrderedDict()
        self.results = OrderedDict()
        self.curves = OrderedDict()
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

    @Inputs.data
    def set_dataset(self, data):
        if False:
            return 10
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

    @Inputs.learner
    def set_learner(self, learner, id):
        if False:
            print('Hello World!')
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
            return 10
        if self.data is not None:
            self._update()
            self._update_curve_points()
        self._update_table()

    def _invalidate_curves(self):
        if False:
            print('Hello World!')
        if self.data is not None:
            self._update_curve_points()
        self._update_table()

    def _invalidate_results(self):
        if False:
            print('Hello World!')
        for id in self.learners:
            self.curves[id] = None
            self.results[id] = None
        if self.data is not None:
            self._update()
            self._update_curve_points()
        self._update_table()

    def _update(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.data is not None
        need_update = [(id, learner) for (id, learner) in self.learners.items() if self.results[id] is None]
        if not need_update:
            return
        learners = [learner for (_, learner) in need_update]
        results = learning_curve(learners, self.data, folds=self.folds, proportions=self.curvePoints)
        results = [list(Results.split_by_model(p_results)) for p_results in results]
        for (i, (id, learner)) in enumerate(need_update):
            self.results[id] = [p_results[i] for p_results in results]

    def _update_curve_points(self):
        if False:
            i = 10
            return i + 15
        for id in self.learners:
            curve = [self.scoring[self.scoringF][1](x)[0] for x in self.results[id]]
            self.curves[id] = curve

    def _update_table(self):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        self.curvePoints = [(x + 1.0) / self.steps for x in range(self.steps)]

    def test_run_signals(self):
        if False:
            print('Hello World!')
        data = Orange.data.Table('iris')
        self.set_dataset(data)
        l1 = Orange.classification.NaiveBayesLearner()
        l1.name = 'Naive Bayes'
        self.set_learner(l1, 1)
        l2 = Orange.classification.LogisticRegressionLearner()
        l2.name = 'Logistic Regression'
        self.set_learner(l2, 2)
        l4 = Orange.classification.SklTreeLearner()
        l4.name = 'Decision Tree'
        self.set_learner(l4, 3)

    def test_run_tear_down(self):
        if False:
            return 10
        self.set_dataset(None)
        self.set_learner(None, 1)
        self.set_learner(None, 2)
        self.set_learner(None, 3)
        super().test_run_tear_down()

def learning_curve(learners, data, folds=10, proportions=None, random_state=None, callback=None):
    if False:
        return 10
    if proportions is None:
        proportions = numpy.linspace(0.0, 1.0, 10 + 1, endpoint=True)[1:]

    def select_proportion_preproc(data, p, rstate=None):
        if False:
            while True:
                i = 10
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
if __name__ == '__main__':
    WidgetPreview(OWLearningCurveA).run()