import _plotly_utils.basevalidators

class MaxdisplayedValidator(_plotly_utils.basevalidators.IntegerValidator):

    def __init__(self, plotly_name='maxdisplayed', parent_name='streamtube', **kwargs):
        if False:
            while True:
                i = 10
        super(MaxdisplayedValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), min=kwargs.pop('min', 0), **kwargs)