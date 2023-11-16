from copy import deepcopy
from AnyQt.QtCore import QTimer, Qt
from Orange.data import Table
from Orange.modelling import Fitter, Learner, Model
from Orange.preprocess.preprocess import Preprocess
from Orange.statistics import util as ut
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.utils import getmembers
from Orange.widgets.utils.signals import Output, Input
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, WidgetMetaClass, Msg

class OWBaseLearnerMeta(WidgetMetaClass):
    """ Meta class for learner widgets

    OWBaseLearner declares two outputs, learner and model with
    generic type (Learner and Model).

    This metaclass ensures that each of the subclasses gets
    its own Outputs class with output that match the corresponding
    learner.
    """

    def __new__(cls, name, bases, attributes, **kwargs):
        if False:
            print('Hello World!')

        def abstract_widget():
            if False:
                return 10
            return not attributes.get('name')

        def copy_outputs(template):
            if False:
                print('Hello World!')
            result = type('Outputs', (), {})
            for (name, signal) in getmembers(template, Output):
                setattr(result, name, deepcopy(signal))
            return result
        obj = super().__new__(cls, name, bases, attributes, **kwargs)
        if abstract_widget():
            return obj
        learner = attributes.get('LEARNER')
        if not learner:
            raise AttributeError("'{}' must declare attribute LEARNER".format(name))
        outputs = obj.Outputs = copy_outputs(obj.Outputs)
        outputs.learner.type = learner
        outputs.model.type = learner.__returns__
        return obj

class OWBaseLearner(OWWidget, metaclass=OWBaseLearnerMeta, openclass=True):
    """Abstract widget for classification/regression learners.

    Notes
    -----
    All learner widgets should define learner class LEARNER.
    LEARNER should have __returns__ attribute.

    Overwrite `create_learner`, `add_main_layout` and `get_learner_parameters`
    in case LEARNER has extra parameters.

    """
    LEARNER = None
    supports_sparse = True
    learner_name = Setting('', schema_only=True)
    want_main_area = False
    resizing_enabled = False
    auto_apply = Setting(True)

    class Error(OWWidget.Error):
        data_error = Msg('{}')
        fitting_failed = Msg('Fitting failed.\n{}')
        sparse_not_supported = Msg('Sparse data is not supported.')
        out_of_memory = Msg('Out of memory.')

    class Warning(OWWidget.Warning):
        outdated_learner = Msg('Press Apply to submit changes.')

    class Information(OWWidget.Information):
        ignored_preprocessors = Msg('Ignoring default preprocessing.\nDefault preprocessing, such as scaling, one-hot encoding and treatment of missing data, has been replaced with user-specified preprocessors. Problems may occur if these are inadequate for the given data.')

    class Inputs:
        data = Input('Data', Table)
        preprocessor = Input('Preprocessor', Preprocess)

    class Outputs:
        learner = Output('Learner', Learner, dynamic=False)
        model = Output('Model', Model, dynamic=False, replaces=['Classifier', 'Predictor'])
    OUTPUT_MODEL_NAME = Outputs.model.name
    (_SEND, _SOFT, _UPDATE) = range(3)

    def __init__(self, preprocessors=None):
        if False:
            while True:
                i = 10
        super().__init__()
        self.__default_learner_name = ''
        self.data = None
        self.valid_data = False
        self.learner = None
        self.model = None
        self.preprocessors = preprocessors
        self.outdated_settings = False
        self.__apply_level = []
        self.setup_layout()
        QTimer.singleShot(0, getattr(self, 'unconditional_apply', self.apply))

    def create_learner(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates a learner with current configuration.\n\n        Returns:\n            Learner: an instance of Orange.base.learner subclass.\n        '
        return self.LEARNER(preprocessors=self.preprocessors)

    def get_learner_parameters(self):
        if False:
            i = 10
            return i + 15
        'Creates an `OrderedDict` or a sequence of pairs with current model\n        configuration.\n\n        Returns:\n            OrderedDict or List: (option, value) pairs or dict\n        '
        return []

    def default_learner_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        "\n        Return the default learner name.\n\n        By default this is the same as the widget's name.\n        "
        return self.__default_learner_name or self.captionTitle

    def set_default_learner_name(self, name: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Set the default learner name if not otherwise specified by the user.\n        '
        changed = name != self.__default_learner_name
        if name:
            self.name_line_edit.setPlaceholderText(name)
        else:
            self.name_line_edit.setPlaceholderText(self.captionTitle)
        self.__default_learner_name = name
        if not self.learner_name and changed:
            self.learner_name_changed()

    @Inputs.preprocessor
    def set_preprocessor(self, preprocessor):
        if False:
            return 10
        self.preprocessors = preprocessor
        self.learner = self.model = None

    @Inputs.data
    @check_sql_input
    def set_data(self, data):
        if False:
            while True:
                i = 10
        'Set the input train dataset.'
        self.Error.data_error.clear()
        self.data = data
        if data is not None and data.domain.class_var is None:
            if data.domain.class_vars:
                self.Error.data_error('Data contains multiple target variables.\nSelect a single one with the Select Columns widget.')
            else:
                self.Error.data_error('Data has no target variable.\nSelect one with the Select Columns widget.')
            self.data = None
        self.model = None

    def apply(self):
        if False:
            while True:
                i = 10
        (level, self.__apply_level) = (max(self.__apply_level, default=self._UPDATE), [])
        'Applies learner and sends new model.'
        if level == self._SEND:
            self._send_learner()
            self._send_model()
        elif level == self._UPDATE:
            self.update_learner()
            self.update_model()
        else:
            self.learner or self.update_learner()
            self.model or self.update_model()

    def apply_as(self, level, unconditional=False):
        if False:
            while True:
                i = 10
        self.__apply_level.append(level)
        if unconditional:
            self.unconditional_apply()
        else:
            self.apply()

    def update_learner(self):
        if False:
            return 10
        self.learner = self.create_learner()
        if self.learner and issubclass(self.LEARNER, Fitter):
            self.learner.use_default_preprocessors = True
        if self.learner is not None:
            self.learner.name = self.effective_learner_name()
        self._send_learner()

    def _send_learner(self):
        if False:
            i = 10
            return i + 15
        self.Outputs.learner.send(self.learner)
        self.outdated_settings = False
        self.Warning.outdated_learner.clear()

    def handleNewSignals(self):
        if False:
            i = 10
            return i + 15
        self.apply_as(self._SOFT, True)
        self.Information.ignored_preprocessors(shown=not getattr(self.learner, 'use_default_preprocessors', False) and getattr(self.LEARNER, 'preprocessors', False) and (self.preprocessors is not None))

    def show_fitting_failed(self, exc):
        if False:
            i = 10
            return i + 15
        'Show error when fitting fails.\n            Derived widgets can override this to show more specific messages.'
        self.Error.fitting_failed(str(exc), shown=exc is not None)

    def update_model(self):
        if False:
            for i in range(10):
                print('nop')
        self.show_fitting_failed(None)
        self.model = None
        if self.check_data():
            try:
                self.model = self.learner(self.data)
            except BaseException as exc:
                self.show_fitting_failed(exc)
            else:
                self.model.name = self.learner_name or self.captionTitle
                self.model.instances = self.data
        self._send_model()

    def _send_model(self):
        if False:
            print('Hello World!')
        self.Outputs.model.send(self.model)

    def check_data(self):
        if False:
            i = 10
            return i + 15
        self.valid_data = False
        self.Error.sparse_not_supported.clear()
        if self.data is not None and self.learner is not None:
            self.Error.data_error.clear()
            reason = self.learner.incompatibility_reason(self.data.domain)
            if reason is not None:
                self.Error.data_error(reason)
            elif not len(self.data):
                self.Error.data_error('Dataset is empty.')
            elif len(ut.unique(self.data.Y)) < 2:
                self.Error.data_error('Data contains a single target value.')
            elif self.data.X.size == 0:
                self.Error.data_error('Data has no features to learn from.')
            elif self.data.is_sparse() and (not self.supports_sparse):
                self.Error.sparse_not_supported()
            else:
                self.valid_data = True
        return self.valid_data

    def settings_changed(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.outdated_settings = True
        self.Warning.outdated_learner(shown=not self.auto_apply)
        self.apply()

    def learner_name_changed(self):
        if False:
            return 10
        if self.model is not None:
            self.model.name = self.effective_learner_name()
        if self.learner is not None:
            self.learner.name = self.effective_learner_name()
        self.apply_as(self._SEND)

    def effective_learner_name(self):
        if False:
            return 10
        'Return the effective learner name.'
        return self.learner_name or self.name_line_edit.placeholderText()

    def send_report(self):
        if False:
            for i in range(10):
                print('nop')
        self.report_items((('Name', self.effective_learner_name()),))
        model_parameters = self.get_learner_parameters()
        if model_parameters:
            self.report_items('Model parameters', model_parameters)
        if self.data:
            self.report_data('Data', self.data)

    def setup_layout(self):
        if False:
            return 10
        self.add_learner_name_widget()
        self.add_main_layout()
        if issubclass(self.LEARNER, Fitter):
            if type(self).add_classification_layout is not OWBaseLearner.add_classification_layout:
                classification_box = gui.widgetBox(self.controlArea, 'Classification')
                self.add_classification_layout(classification_box)
            if type(self).add_regression_layout is not OWBaseLearner.add_regression_layout:
                regression_box = gui.widgetBox(self.controlArea, 'Regression')
                self.add_regression_layout(regression_box)
        self.add_bottom_buttons()

    def add_main_layout(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates layout with the learner configuration widgets.\n\n        Override this method for laying out any learner-specific parameter controls.\n        See setup_layout() method for execution order.\n        '

    def add_classification_layout(self, box):
        if False:
            return 10
        'Creates layout for classification specific options.\n\n        If a widget outputs a learner dispatcher, sometimes the classification\n        and regression learners require different options.\n        See `setup_layout()` method for execution order.\n        '

    def add_regression_layout(self, box):
        if False:
            for i in range(10):
                print('nop')
        'Creates layout for regression specific options.\n\n        If a widget outputs a learner dispatcher, sometimes the classification\n        and regression learners require different options.\n        See `setup_layout()` method for execution order.\n        '

    def add_learner_name_widget(self):
        if False:
            i = 10
            return i + 15
        self.name_line_edit = gui.lineEdit(self.controlArea, self, 'learner_name', box='Name', placeholderText=self.captionTitle, tooltip='The name will identify this model in other widgets', orientation=Qt.Horizontal, callback=self.learner_name_changed)

    def setCaption(self, caption):
        if False:
            i = 10
            return i + 15
        super().setCaption(caption)
        if not self.__default_learner_name:
            self.name_line_edit.setPlaceholderText(caption)
            if not self.learner_name:
                self.learner_name_changed()

    def add_bottom_buttons(self):
        if False:
            return 10
        self.apply_button = gui.auto_apply(self.buttonsArea, self, commit=self.apply)

    def send(self, signalName, value, id=None):
        if False:
            return 10
        for (_, output) in getmembers(self.Outputs, Output):
            if output.name == signalName or signalName in output.replaces:
                output.send(value, id=id)
                return
        super().send(signalName, value, id)

    @classmethod
    def get_widget_description(cls):
        if False:
            for i in range(10):
                print('nop')
        desc = super().get_widget_description()
        if cls.outputs:
            desc['outputs'].extend(cls.get_signals('outputs', True))
        if cls.inputs:
            desc['inputs'].extend(cls.get_signals('inputs', True))
        return desc