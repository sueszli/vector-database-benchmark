import _plotly_utils.basevalidators

class ArrowcolorValidator(_plotly_utils.basevalidators.ColorValidator):

    def __init__(self, plotly_name='arrowcolor', parent_name='layout.annotation', **kwargs):
        if False:
            while True:
                i = 10
        super(ArrowcolorValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'arraydraw'), **kwargs)