import _plotly_utils.basevalidators

class StartstandoffValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='startstandoff', parent_name='layout.annotation', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(StartstandoffValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc+arraydraw'), min=kwargs.pop('min', 0), **kwargs)