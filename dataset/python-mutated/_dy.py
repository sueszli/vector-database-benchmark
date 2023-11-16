import _plotly_utils.basevalidators

class DyValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='dy', parent_name='box', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(DyValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)