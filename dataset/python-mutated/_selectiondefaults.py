import _plotly_utils.basevalidators

class SelectiondefaultsValidator(_plotly_utils.basevalidators.CompoundValidator):

    def __init__(self, plotly_name='selectiondefaults', parent_name='layout', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(SelectiondefaultsValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=kwargs.pop('data_class_str', 'Selection'), data_docs=kwargs.pop('data_docs', '\n'), **kwargs)