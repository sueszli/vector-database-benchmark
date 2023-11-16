import _plotly_utils.basevalidators

class FacecolorValidator(_plotly_utils.basevalidators.DataArrayValidator):

    def __init__(self, plotly_name='facecolor', parent_name='mesh3d', **kwargs):
        if False:
            while True:
                i = 10
        super(FacecolorValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)