from nni.nas.space.graph_op import TensorFlowOperation

class Conv2D(TensorFlowOperation):

    def __init__(self, type_name, parameters, _internal, attributes=None):
        if False:
            i = 10
            return i + 15
        if 'padding' not in parameters:
            parameters['padding'] = 'same'
        super().__init__(type_name, parameters, _internal)