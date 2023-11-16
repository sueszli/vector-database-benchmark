import pickle
from Orange.widgets.widget import Input
from Orange.base import Model
from Orange.widgets.utils.save.owsavebase import OWSaveBase
from Orange.widgets.utils.widgetpreview import WidgetPreview

class OWSaveModel(OWSaveBase):
    name = 'Save Model'
    description = 'Save a trained model to an output file.'
    icon = 'icons/SaveModel.svg'
    replaces = ['Orange.widgets.classify.owsaveclassifier.OWSaveClassifier']
    priority = 3000
    keywords = 'save model, save'

    class Inputs:
        model = Input('Model', Model)
    filters = ['Pickled model (*.pkcls)']

    @Inputs.model
    def set_model(self, model):
        if False:
            while True:
                i = 10
        self.data = model
        self.on_new_input()

    def do_save(self):
        if False:
            while True:
                i = 10
        with open(self.filename, 'wb') as f:
            pickle.dump(self.data, f)
if __name__ == '__main__':
    WidgetPreview(OWSaveModel).run()