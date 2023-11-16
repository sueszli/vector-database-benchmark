import _plotly_utils.basevalidators

class CaptureeventsValidator(_plotly_utils.basevalidators.BooleanValidator):

    def __init__(self, plotly_name='captureevents', parent_name='layout.annotation', **kwargs):
        if False:
            i = 10
            return i + 15
        super(CaptureeventsValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'arraydraw'), **kwargs)