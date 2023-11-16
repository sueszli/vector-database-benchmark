import _plotly_utils.basevalidators

class PitchValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='pitch', parent_name='layout.mapbox', **kwargs):
        if False:
            i = 10
            return i + 15
        super(PitchValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'plot'), **kwargs)