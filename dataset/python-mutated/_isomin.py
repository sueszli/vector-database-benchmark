import _plotly_utils.basevalidators

class IsominValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='isomin', parent_name='isosurface', **kwargs):
        if False:
            while True:
                i = 10
        super(IsominValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)