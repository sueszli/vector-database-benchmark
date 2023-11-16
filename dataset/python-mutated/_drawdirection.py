import _plotly_utils.basevalidators

class DrawdirectionValidator(_plotly_utils.basevalidators.EnumeratedValidator):

    def __init__(self, plotly_name='drawdirection', parent_name='layout.newshape', **kwargs):
        if False:
            i = 10
            return i + 15
        super(DrawdirectionValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'none'), values=kwargs.pop('values', ['ortho', 'horizontal', 'vertical', 'diagonal']), **kwargs)