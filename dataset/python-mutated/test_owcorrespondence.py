from Orange.data import Table, Domain, DiscreteVariable, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.unsupervised.owcorrespondence import OWCorrespondenceAnalysis, select_rows

class TestOWCorrespondence(WidgetTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.widget = self.create_widget(OWCorrespondenceAnalysis)
        self.data = Table('titanic')

    def test_no_data(self):
        if False:
            while True:
                i = 10
        "Check that the widget doesn't crash on empty data"
        self.send_signal(self.widget.Inputs.data, Table.from_domain(Table('iris').domain))
        self.assertTrue(self.widget.Error.empty_data.is_shown())
        self.assertIsNone(self.widget.data)

    def test_data_values_in_column(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the widget does not crash when:\n        1) Domain has a two or more discrete variables but less than in a table\n        2) There is at least one NaN value in a column.\n        GH-2066\n        '
        table = Table.from_list(Domain([ContinuousVariable('a'), DiscreteVariable('b', values=('t', 'f')), DiscreteVariable('c', values=('y', 'n')), DiscreteVariable('d', values=('k', 'l', 'z'))]), list(zip([42.48, 16.84, 15.23, 23.8], ['t', 't', '', 'f'], 'yyyy', 'klkk')))
        self.send_signal(self.widget.Inputs.data, table)

    def test_data_one_value_zero(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the widget does not crash on discrete attributes with only\n        one value.\n        GH-2149\n        '
        table = Table.from_list(Domain([DiscreteVariable('a', values=('0',))]), [(0,), (0,), (0,)])
        self.send_signal(self.widget.Inputs.data, table)

    def test_no_discrete_variables(self):
        if False:
            i = 10
            return i + 15
        '\n        Do not crash when there are no discrete (categorical) variable(s).\n        GH-2723\n        '
        table = Table.from_list(Domain([ContinuousVariable('a')]), [(1,), (2,), (3,)])
        self.assertFalse(self.widget.Error.no_disc_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.no_disc_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_disc_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, table)
        self.assertTrue(self.widget.Error.no_disc_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, Table('iris'))
        self.assertFalse(self.widget.Error.no_disc_vars.is_shown())

    def test_outputs(self):
        if False:
            i = 10
            return i + 15
        w = self.widget
        self.assertIsNone(self.get_output(w.Outputs.coordinates), None)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTupleEqual(self.get_output(w.Outputs.coordinates).X.shape, (6, 2))
        select_rows(w.varview, [0, 1, 2])
        w.commit.now()
        self.assertTupleEqual(self.get_output(w.Outputs.coordinates).X.shape, (8, 8))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(w.Outputs.coordinates), None)