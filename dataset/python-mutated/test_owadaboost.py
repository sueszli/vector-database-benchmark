from Orange.classification import RandomForestLearner
from Orange.modelling import KNNLearner
from Orange.widgets.model.owadaboost import OWAdaBoost
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, ParameterMapping

class TestOWAdaBoost(WidgetTest, WidgetLearnerTestMixin):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget = self.create_widget(OWAdaBoost, stored_settings={'auto_apply': False})
        self.init()
        self.parameters = [ParameterMapping('algorithm', self.widget.cls_algorithm_combo, self.widget.algorithms, problem_type='classification'), ParameterMapping('loss', self.widget.reg_algorithm_combo, [x.lower() for x in self.widget.losses], problem_type='regression'), ParameterMapping('learning_rate', self.widget.learning_rate_spin), ParameterMapping('n_estimators', self.widget.n_estimators_spin), ParameterMapping.from_attribute(self.widget, 'random_seed', 'random_state')]

    def test_input_learner(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if base learner properly changes with learner on the input'
        self.assertIsNotNone(self.widget.base_estimator, 'The default base estimator should not be none')
        self.assertTrue(self.widget.base_estimator.supports_weights, 'The default base estimator should support weights')
        default_base_estimator_cls = self.widget.base_estimator
        self.send_signal(self.widget.Inputs.learner, RandomForestLearner())
        self.assertIsInstance(self.widget.base_estimator, RandomForestLearner, 'The base estimator was not updated when valid learner on input')
        self.send_signal(self.widget.Inputs.learner, None)
        self.assertIsInstance(self.widget.base_estimator, type(default_base_estimator_cls), 'The base estimator was not reset to default when None on input')

    def test_input_learner_that_does_not_support_sample_weights(self):
        if False:
            for i in range(10):
                print('nop')
        self.send_signal(self.widget.Inputs.learner, KNNLearner())
        self.assertNotIsInstance(self.widget.base_estimator, KNNLearner)
        self.assertIsNone(self.widget.base_estimator)
        self.assertTrue(self.widget.Error.no_weight_support.is_shown())

    def test_error_message_cleared_when_valid_learner_on_input(self):
        if False:
            while True:
                i = 10
        self.send_signal(self.widget.Inputs.learner, KNNLearner())
        self.send_signal(self.widget.Inputs.learner, None)
        self.assertFalse(self.widget.Error.no_weight_support.is_shown(), 'Error message was not hidden on input disconnect')
        self.send_signal(self.widget.Inputs.learner, KNNLearner())
        self.send_signal(self.widget.Inputs.learner, RandomForestLearner())
        self.assertFalse(self.widget.Error.no_weight_support.is_shown(), 'Error message was not hidden when a valid learner appeared on input')

    def test_input_learner_disconnect(self):
        if False:
            i = 10
            return i + 15
        'Check base learner after disconnecting learner on the input'
        self.send_signal(self.widget.Inputs.learner, RandomForestLearner())
        self.assertIsInstance(self.widget.base_estimator, RandomForestLearner)
        self.send_signal(self.widget.Inputs.learner, None)
        self.assertEqual(self.widget.base_estimator, self.widget.DEFAULT_BASE_ESTIMATOR)