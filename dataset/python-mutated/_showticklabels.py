import _plotly_utils.basevalidators

class ShowticklabelsValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='showticklabels', parent_name='histogram2dcontour.colorbar', **kwargs):
        if False:
            while True:
                i = 10
        super(ShowticklabelsValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'colorbars'), **kwargs)