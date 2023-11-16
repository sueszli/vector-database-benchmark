import _plotly_utils.basevalidators

class EndlinecolorValidator(_plotly_utils.basevalidators.ColorValidator):

    def __init__(self, plotly_name='endlinecolor', parent_name='carpet.baxis', **kwargs):
        if False:
            i = 10
            return i + 15
        super(EndlinecolorValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)