import _plotly_utils.basevalidators

class FramesValidator(_plotly_utils.basevalidators.CompoundArrayValidator):

    def __init__(self, plotly_name='frames', parent_name='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(FramesValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, data_class_str=kwargs.pop('data_class_str', 'Frame'), data_docs=kwargs.pop('data_docs', "\n            baseframe\n                The name of the frame into which this frame's\n                properties are merged before applying. This is\n                used to unify properties and avoid needing to\n                specify the same values for the same properties\n                in multiple frames.\n            data\n                A list of traces this frame modifies. The\n                format is identical to the normal trace\n                definition.\n            group\n                An identifier that specifies the group to which\n                the frame belongs, used by animate to select a\n                subset of frames.\n            layout\n                Layout properties which this frame modifies.\n                The format is identical to the normal layout\n                definition.\n            name\n                A label by which to identify the frame\n            traces\n                A list of trace indices that identify the\n                respective traces in the data attribute\n"), **kwargs)