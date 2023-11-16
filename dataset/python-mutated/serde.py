def serialize_protobuf_struct(protobuf_struct):
    if False:
        i = 10
        return i + 15
    return protobuf_struct.SerializeToString()

def deserialize_protobuf_struct(serialized_protobuf, struct_type):
    if False:
        for i in range(10):
            print('nop')
    deser = struct_type()
    deser.ParseFromString(serialized_protobuf)
    return deser