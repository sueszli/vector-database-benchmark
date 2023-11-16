import _plotly_utils.basevalidators

class AutoshiftValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='autoshift', parent_name='layout.yaxis', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(AutoshiftValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'plot'), **kwargs)