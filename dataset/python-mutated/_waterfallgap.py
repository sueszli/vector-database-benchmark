import _plotly_utils.basevalidators

class WaterfallgapValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='waterfallgap', parent_name='layout', **kwargs):
        if False:
            i = 10
            return i + 15
        super(WaterfallgapValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), max=kwargs.pop('max', 1), min=kwargs.pop('min', 0), **kwargs)