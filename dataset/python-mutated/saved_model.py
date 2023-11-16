"""SavedModel Splitter."""
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.tools.proto_splitter import constants
from tensorflow.tools.proto_splitter import split
from tensorflow.tools.proto_splitter import split_graph_def

class SavedModelSplitter(split.ComposableSplitter):
    """Splits a SavedModel proto into chunks of size < 2GB."""

    def build_chunks(self):
        if False:
            print('Hello World!')
        if not isinstance(self._proto, saved_model_pb2.SavedModel):
            raise TypeError(f'SavedModelSplitter can only split SavedModel protos. Got {type(self._proto)}.')
        if self._proto.ByteSize() >= constants.max_size():
            graph_def = self._proto.meta_graphs[0].graph_def
            graph_def_fields = ['meta_graphs', 0, 'graph_def']
            split_graph_def.GraphDefSplitter(self._proto.meta_graphs[0].graph_def, parent_splitter=self, fields_in_parent=graph_def_fields).build_chunks()
        if self._proto.ByteSize() >= constants.max_size():
            self.add_chunk(graph_def, graph_def_fields, index=1)
            self._proto.meta_graphs[0].ClearField('graph_def')