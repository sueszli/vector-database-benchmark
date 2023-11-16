import _plotly_utils.basevalidators

class AxValidator(_plotly_utils.basevalidators.AnyValidator):

    def __init__(self, plotly_name='ax', parent_name='layout.annotation', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(AxValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc+arraydraw'), **kwargs)