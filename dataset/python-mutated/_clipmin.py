import _plotly_utils.basevalidators

class ClipminValidator(_plotly_utils.basevalidators.AnyValidator):

    def __init__(self, plotly_name='clipmin', parent_name='layout.polar.radialaxis.autorangeoptions', **kwargs):
        if False:
            i = 10
            return i + 15
        super(ClipminValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'plot'), implied_edits=kwargs.pop('implied_edits', {}), **kwargs)