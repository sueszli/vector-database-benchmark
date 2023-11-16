import _plotly_utils.basevalidators

class LabelangleValidator(_plotly_utils.basevalidators.AngleValidator):

    def __init__(self, plotly_name='labelangle', parent_name='parcoords', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(LabelangleValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'plot'), **kwargs)