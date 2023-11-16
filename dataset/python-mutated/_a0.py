import _plotly_utils.basevalidators

class A0Validator(_plotly_utils.basevalidators.NumberValidator):

    def __init__(self, plotly_name='a0', parent_name='carpet', **kwargs):
        if False:
            while True:
                i = 10
        super(A0Validator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, edit_type=kwargs.pop('edit_type', 'calc'), **kwargs)