import _plotly_utils.basevalidators

class ShowlabelsValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='showlabels', parent_name='histogram2dcontour.contours', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ShowlabelsValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'plot'), **kwargs)