import _plotly_utils.basevalidators

class AutosizeValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='autosize', parent_name='layout', **kwargs):
        if False:
            while True:
                i = 10
        super(AutosizeValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'none'), **kwargs)