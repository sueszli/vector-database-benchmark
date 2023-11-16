import _plotly_utils.basevalidators

class ShowupperhalfValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='showupperhalf', parent_name='splom', **kwargs):
        if False:
            return 10
        super(ShowupperhalfValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)