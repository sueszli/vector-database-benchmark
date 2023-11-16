import _plotly_utils.basevalidators

class SurfacecolorValidator(_plotly_utils.basevalidators.ColorValidator):

    def __init__(self, plotly_name='surfacecolor', parent_name='scatter3d', **kwargs):
        if False:
            i = 10
            return i + 15
        super(SurfacecolorValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)