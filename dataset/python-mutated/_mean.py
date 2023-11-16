import _plotly_utils.basevalidators

class MeanValidator(_plotly_utils.basevalidators.DataArrayValidator):

    def __init__(self, plotly_name='mean', parent_name='box', **kwargs):
        if False:
            while True:
                i = 10
        super(MeanValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)