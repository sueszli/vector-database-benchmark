import _plotly_utils.basevalidators

class DatarevisionValidator(_plotly_utils.basevalidators.AnyValidator):

    def __init__(self, plotly_name='datarevision', parent_name='layout', **kwargs):
        if False:
            i = 10
            return i + 15
        super(DatarevisionValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)