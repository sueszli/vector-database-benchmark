from Orange.widgets.model.owconstant import OWConstant
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin

class TestOWConstant(WidgetTest, WidgetLearnerTestMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.widget = self.create_widget(OWConstant, stored_settings={'auto_apply': False})
        self.init()