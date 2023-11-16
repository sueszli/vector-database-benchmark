import _plotly_utils.basevalidators

class ActivebgcolorValidator(_plotly_utils.basevalidators.ColorValidator):

    def __init__(self, plotly_name='activebgcolor', parent_name='layout.slider', **kwargs):
        if False:
            i = 10
            return i + 15
        super(ActivebgcolorValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'arraydraw'), **kwargs)