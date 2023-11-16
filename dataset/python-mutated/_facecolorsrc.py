import _plotly_utils.basevalidators

class FacecolorsrcValidator(_plotly_utils.basevalidators.SrcValidator):

    def __init__(self, plotly_name='facecolorsrc', parent_name='mesh3d', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(FacecolorsrcValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'none'), **kwargs)