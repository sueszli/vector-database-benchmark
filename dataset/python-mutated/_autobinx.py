import _plotly_utils.basevalidators

class AutobinxValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='autobinx', parent_name='histogram2dcontour', **kwargs):
        if False:
            i = 10
            return i + 15
        super(AutobinxValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)