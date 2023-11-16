import _plotly_utils.basevalidators

class SeparatethousandsValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='separatethousands', parent_name='histogram2dcontour.colorbar', **kwargs):
        if False:
            i = 10
            return i + 15
        super(SeparatethousandsValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'colorbars'), **kwargs)