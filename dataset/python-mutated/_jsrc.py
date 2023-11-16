import _plotly_utils.basevalidators

class JsrcValidator(_plotly_utils.basevalidators.SrcValidator):

    def __init__(self, plotly_name='jsrc', parent_name='mesh3d', **kwargs):
        if False:
            i = 10
            return i + 15
        super(JsrcValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'none'), **kwargs)