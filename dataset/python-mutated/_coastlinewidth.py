import _plotly_utils.basevalidators

class CoastlinewidthValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='coastlinewidth', parent_name='layout.geo', **kwargs):
        if False:
            print('Hello World!')
        super(CoastlinewidthValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'plot'), min=kwargs.pop('min', 0), **kwargs)