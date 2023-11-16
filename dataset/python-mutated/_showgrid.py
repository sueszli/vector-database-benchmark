import _plotly_utils.basevalidators

class ShowgridValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='showgrid', parent_name='carpet.baxis', **kwargs):
        if False:
            i = 10
            return i + 15
        super(ShowgridValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)