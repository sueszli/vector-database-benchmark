import _plotly_utils.basevalidators

class SizeyValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='sizey', parent_name='layout.image', **kwargs):
        if False:
            return 10
        super(SizeyValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'arraydraw'), **kwargs)