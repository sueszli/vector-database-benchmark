import unittest
from unittest.mock import Mock, patch
import numpy as np
from Orange.data import Table, Domain
from Orange.projection import FreeViz
from Orange.projection.freeviz import FreeVizModel
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, AnchorProjectionWidgetTestMixin
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owfreeviz import OWFreeViz, Result, run_freeviz

class TestOWFreeViz(WidgetTest, AnchorProjectionWidgetTestMixin, WidgetOutputsTestMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.signal_name = OWFreeViz.Inputs.data
        cls.signal_data = cls.data
        cls.same_input_output_domain = False
        cls.heart_disease = Table('heart_disease')

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.widget = self.create_widget(OWFreeViz)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_error_msg(self):
        if False:
            while True:
                i = 10
        data = self.data[:, list(range(len(self.data.domain.attributes)))]
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.no_class_var.is_shown())
        data = self.data[:40]
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.not_enough_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.not_enough_class_vars.is_shown())

    def test_number_of_targets(self):
        if False:
            return 10
        data = self.heart_disease
        domain = data.domain
        no_target = data.transform(Domain(domain.attributes, []))
        two_targets = data.transform(Domain([domain['age']], [domain['gender'], domain['chest pain']]))
        self.send_signal(self.widget.Inputs.data, data)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.multiple_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, no_target)
        self.assertTrue(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.multiple_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, two_targets)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertTrue(self.widget.Error.multiple_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, data)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.multiple_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, two_targets)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertTrue(self.widget.Error.multiple_class_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_class_var.is_shown())
        self.assertFalse(self.widget.Error.multiple_class_vars.is_shown())

    def test_optimization(self):
        if False:
            return 10
        self.send_signal(self.widget.Inputs.data, self.heart_disease)
        self.widget.run_button.click()
        self.assertEqual(self.widget.run_button.text(), 'Stop')

    def test_optimization_cancelled(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_optimization()
        self.widget.run_button.click()
        self.assertEqual(self.widget.run_button.text(), 'Resume')

    def test_optimization_reset(self):
        if False:
            print('Hello World!')
        self.test_optimization()
        init = self.widget.controls.initialization
        simulate.combobox_activate_index(init, 0)
        self.assertEqual(self.widget.run_button.text(), 'Stop')
        simulate.combobox_activate_index(init, 1)
        self.assertEqual(self.widget.run_button.text(), 'Stop')

    def test_optimization_finish(self):
        if False:
            for i in range(10):
                print('nop')
        self.send_signal(self.widget.Inputs.data, self.data[::10].copy())
        output1 = self.get_output(self.widget.Outputs.components)
        self.widget.run_button.click()
        self.assertEqual(self.widget.run_button.text(), 'Stop')
        self.wait_until_finished()
        self.assertEqual(self.widget.run_button.text(), 'Start')
        output2 = self.get_output(self.widget.Outputs.components)
        self.assertTrue((output1.X != output2.X).any())

    def test_optimization_no_data(self):
        if False:
            print('Hello World!')
        self.widget.run_button.click()
        self.assertEqual(self.widget.run_button.text(), 'Start')

    def test_constant_data(self):
        if False:
            return 10
        data = Table('titanic')[56:59]
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.run_button.click()
        self.assertTrue(self.widget.Error.constant_data.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.constant_data.is_shown())

    def test_set_radius_no_data(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Widget should not crash when there is no data and radius slider is moved.\n        GH-2780\n        '
        w = self.widget
        self.send_signal(w.Inputs.data, None)
        self.widget.graph.controls.hide_radius.setSliderPosition(3)

    def test_output_components(self):
        if False:
            for i in range(10):
                print('nop')
        self.send_signal(self.widget.Inputs.data, self.data)
        components = self.get_output(self.widget.Outputs.components)
        domain = components.domain
        self.assertEqual(domain.attributes, self.data.domain.attributes)
        self.assertEqual(domain.class_vars, ())
        self.assertEqual([m.name for m in domain.metas], ['component'])
        X = np.array([[1, 0, -1, 0], [0, 1, 0, -1]]).astype(float)
        np.testing.assert_array_almost_equal(components.X, X)
        metas = [['freeviz-x'], ['freeviz-y']]
        np.testing.assert_array_equal(components.metas, metas)

    def test_manual_move(self):
        if False:
            i = 10
            return i + 15
        super().test_manual_move()
        array = np.array([[1, 2], [0, 1], [-1, 0], [0, -1]])
        np.testing.assert_array_almost_equal(self.get_output(self.widget.Outputs.components).X, array.T)

    def test_discrete_attributes(self):
        if False:
            i = 10
            return i + 15
        zoo = Table('zoo')
        self.send_signal(self.widget.Inputs.data, zoo)
        self.assertTrue(self.widget.Warning.removed_features.is_shown())
        self.widget.run_button.click()

    def test_gravity_slider(self):
        if False:
            i = 10
            return i + 15
        w = self.widget
        w.balance = False
        w.gravity_index = 0
        w.grav_slider.setValue(2)
        self.assertTrue(w.balance)
        self.assertEqual(w.gravity_label.text(), str(w.GravityValues[2]))
        w.grav_slider.setValue(3)
        self.assertTrue(w.balance)
        self.assertEqual(w.gravity_label.text(), str(w.GravityValues[3]))
        assert w.projector is None
        self.send_signal(self.widget.Inputs.data, Table('zoo'))
        self.wait_until_finished()
        assert w.projector is not None
        self.assertEqual(w.projector.gravity, w.GravityValues[3])
        w.grav_slider.setValue(1)
        self.assertEqual(w.projector.gravity, w.GravityValues[1])
        with patch.object(w, '_run') as run, patch.object(w, '_OWFreeViz__optimized', new=True):
            w.grav_slider.setValue(2)
            self.assertEqual(w.projector.gravity, w.GravityValues[2])
            run.assert_called_once()
            run.reset_mock()
            w.controls.balance.click()
            self.assertFalse(w.balance)
            self.assertIsNone(w.projector.gravity)
            run.assert_called_once()

class TestOWFreeVizRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.data = Table('iris')

    def setUp(self):
        if False:
            return 10
        anchors = FreeViz.init_radial(len(self.data.domain.attributes))
        self.projector = projector = FreeViz(scale=False, center=False, initial=anchors, maxiter=10)
        self.projector.domain = self.data.domain
        self.projector.components_ = anchors.T
        self.projection = FreeVizModel(projector, projector.domain, 2)
        self.projection.pre_domain = self.data.domain

    def test_Result(self):
        if False:
            while True:
                i = 10
        result = Result(projector=self.projector, projection=self.projection)
        self.assertIsInstance(result.projector, FreeViz)
        self.assertIsInstance(result.projection, FreeVizModel)

    def test_run(self):
        if False:
            for i in range(10):
                print('nop')
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)
        result = run_freeviz(self.data, self.projector, state)
        array = np.array([[0.166883742, 9.40395481e-38], [-0.0886817512, 0.996060012], [0.0667450609, -0.397675811], [-0.144947052, -0.5983842]])
        np.testing.assert_almost_equal(array.T, result.projection.components_)
        state.set_status.assert_called_once_with('Calculating...')
        self.assertGreater(state.set_partial_result.call_count, 40)
        self.assertGreater(state.set_progress_value.call_count, 40)

    def test_run_do_not_modify_model_inplace(self):
        if False:
            while True:
                i = 10
        state = Mock()
        state.is_interruption_requested.return_value = True
        result = run_freeviz(self.data, self.projector, state)
        state.set_partial_result.assert_called_once()
        self.assertIs(self.projector, result.projector)
        self.assertIsNot(self.projection.proj, result.projection.proj)
        self.assertTrue((self.projection.components_.T != result.projection.components_.T).any())
if __name__ == '__main__':
    unittest.main()