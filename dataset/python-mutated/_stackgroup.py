import _plotly_utils.basevalidators

class StackgroupValidator(_plotly_utils.basevalidators.StringValidator):

    def __init__(self, plotly_name='stackgroup', parent_name='scatter', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(StackgroupValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)