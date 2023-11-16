import _plotly_utils.basevalidators

class LenValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='len', parent_name='histogram2dcontour.colorbar', **kwargs):
        if False:
            i = 10
            return i + 15
        super(LenValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'colorbars'), min=kwargs.pop('min', 0), **kwargs)