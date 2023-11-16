import _plotly_utils.basevalidators

class CautoValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='cauto', parent_name='treemap.marker', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(CautoValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), implied_edits=kwargs.pop('implied_edits', {}), **kwargs)