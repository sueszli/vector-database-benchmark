import _plotly_utils.basevalidators

class SizexValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='sizex', parent_name='layout.image', **kwargs):
        if False:
            i = 10
            return i + 15
        super(SizexValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'arraydraw'), **kwargs)