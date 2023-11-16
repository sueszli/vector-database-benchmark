from Orange.widgets.model.ownaivebayes import OWNaiveBayes
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin

class TestOWNaiveBayes(WidgetTest, WidgetLearnerTestMixin):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.widget = self.create_widget(OWNaiveBayes, stored_settings={'auto_apply': False})
        self.init()