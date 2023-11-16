import _plotly_utils.basevalidators

class UpperfenceValidator(_plotly_utils.basevalidators.DataArrayValidator):

    def __init__(self, plotly_name='upperfence', parent_name='box', **kwargs):
        if False:
            i = 10
            return i + 15
        super(UpperfenceValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)