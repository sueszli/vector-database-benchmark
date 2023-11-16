import _plotly_utils.basevalidators

class ArearatioValidator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='arearatio', parent_name='pointcloud.marker.border', **kwargs):
        if False:
            i = 10
            return i + 15
        super(ArearatioValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), max=kwargs.pop('max', 1), min=kwargs.pop('min', 0), **kwargs)