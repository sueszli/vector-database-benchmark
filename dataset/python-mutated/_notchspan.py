import _plotly_utils.basevalidators

class NotchspanValidator(_plotly_utils.basevalidators.DataArrayValidator):

    def __init__(self, plotly_name='notchspan', parent_name='box', **kwargs):
        if False:
            i = 10
            return i + 15
        super(NotchspanValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)