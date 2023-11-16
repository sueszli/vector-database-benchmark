import _plotly_utils.basevalidators

class XbingroupValidator(_plotly_utils.basevalidators.StringValidator):

    def __init__(self, plotly_name='xbingroup', parent_name='histogram2dcontour', **kwargs):
        if False:
            while True:
                i = 10
        super(XbingroupValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)