import _plotly_utils.basevalidators

class SdmultipleValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='sdmultiple', parent_name='box', **kwargs):
        if False:
            i = 10
            return i + 15
        super(SdmultipleValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), min=kwargs.pop('min', 0), **kwargs)