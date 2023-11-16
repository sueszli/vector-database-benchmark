import _plotly_utils.basevalidators

class ExtendpiecolorsValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='extendpiecolors', parent_name='layout', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ExtendpiecolorsValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)