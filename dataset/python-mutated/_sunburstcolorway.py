import _plotly_utils.basevalidators

class SunburstcolorwayValidator(_plotly_utils.basevalidators.ColorlistValidator):

    def __init__(self, plotly_name='sunburstcolorway', parent_name='layout', **kwargs):
        if False:
            i = 10
            return i + 15
        super(SunburstcolorwayValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)