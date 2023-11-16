from Orange.classification import CalibratedLearner, ThresholdLearner, NaiveBayesLearner
from Orange.data import Table
from Orange.modelling import Learner
from Orange.widgets import gui
from Orange.widgets.widget import Input
from Orange.widgets.settings import Setting
from Orange.widgets.utils.owlearnerwidget import OWBaseLearner
from Orange.widgets.utils.widgetpreview import WidgetPreview

class OWCalibratedLearner(OWBaseLearner):
    name = 'Calibrated Learner'
    description = 'Wraps another learner with probability calibration and decision threshold optimization'
    icon = 'icons/CalibratedLearner.svg'
    priority = 20
    keywords = 'calibrated learner, calibration, threshold'
    LEARNER = CalibratedLearner
    (SigmoidCalibration, IsotonicCalibration, NoCalibration) = range(3)
    CalibrationOptions = ('Sigmoid calibration', 'Isotonic calibration', 'No calibration')
    CalibrationShort = ('Sigmoid', 'Isotonic', '')
    CalibrationMap = {SigmoidCalibration: CalibratedLearner.Sigmoid, IsotonicCalibration: CalibratedLearner.Isotonic}
    (OptimizeCA, OptimizeF1, NoThresholdOptimization) = range(3)
    ThresholdOptions = ('Optimize classification accuracy', 'Optimize F1 score', 'No threshold optimization')
    ThresholdShort = ('CA', 'F1', '')
    ThresholdMap = {OptimizeCA: ThresholdLearner.OptimizeCA, OptimizeF1: ThresholdLearner.OptimizeF1}
    learner_name = Setting('', schema_only=True)
    calibration = Setting(SigmoidCalibration)
    threshold = Setting(OptimizeCA)

    class Inputs(OWBaseLearner.Inputs):
        base_learner = Input('Base Learner', Learner)

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.base_learner = None

    def add_main_layout(self):
        if False:
            i = 10
            return i + 15
        gui.radioButtons(self.controlArea, self, 'calibration', self.CalibrationOptions, box='Probability calibration', callback=self.calibration_options_changed)
        gui.radioButtons(self.controlArea, self, 'threshold', self.ThresholdOptions, box='Decision threshold optimization', callback=self.calibration_options_changed)

    @Inputs.base_learner
    def set_learner(self, learner):
        if False:
            return 10
        self.base_learner = learner
        self._set_default_name()
        self.learner = self.model = None

    def _set_default_name(self):
        if False:
            i = 10
            return i + 15
        if self.base_learner is None:
            self.set_default_learner_name('')
        else:
            name = ' + '.join((part for part in (self.base_learner.name.title(), self.CalibrationShort[self.calibration], self.ThresholdShort[self.threshold]) if part))
            self.set_default_learner_name(name)

    def calibration_options_changed(self):
        if False:
            i = 10
            return i + 15
        self._set_default_name()
        self.apply()

    def create_learner(self):
        if False:
            for i in range(10):
                print('nop')

        class IdentityWrapper(Learner):

            def fit_storage(self, data):
                if False:
                    i = 10
                    return i + 15
                return self.base_learner.fit_storage(data)
        if self.base_learner is None:
            return None
        learner = self.base_learner
        if self.calibration != self.NoCalibration:
            learner = CalibratedLearner(learner, self.CalibrationMap[self.calibration])
        if self.threshold != self.NoThresholdOptimization:
            learner = ThresholdLearner(learner, self.ThresholdMap[self.threshold])
        if self.preprocessors:
            if learner is self.base_learner:
                learner = IdentityWrapper()
            learner.preprocessors = (self.preprocessors,)
        return learner

    def get_learner_parameters(self):
        if False:
            return 10
        return (('Calibrate probabilities', self.CalibrationOptions[self.calibration]), ('Threshold optimization', self.ThresholdOptions[self.threshold]))
if __name__ == '__main__':
    WidgetPreview(OWCalibratedLearner).run(Table('heart_disease'), set_learner=NaiveBayesLearner())