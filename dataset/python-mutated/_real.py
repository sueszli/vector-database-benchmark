import _plotly_utils.basevalidators

class RealValidator(_plotly_utils.basevalidators.DataArrayValidator):

    def __init__(self, plotly_name='real', parent_name='scattersmith', **kwargs):
        if False:
            while True:
                i = 10
        super(RealValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc+clearAxisTypes'), **kwargs)