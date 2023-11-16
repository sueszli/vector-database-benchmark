from caffe2.python import core, scope

def create_predict_net(predictor_export_meta):
    if False:
        while True:
            i = 10
    '\n    Return the input prediction net.\n    '
    net = core.Net(predictor_export_meta.predict_net.name or 'predict')
    net.Proto().op.extend(predictor_export_meta.predict_net.op)
    net.Proto().partition_info.extend(predictor_export_meta.predict_net.partition_info)
    net.Proto().external_input.extend(predictor_export_meta.inputs + predictor_export_meta.parameters)
    net.Proto().external_output.extend(predictor_export_meta.outputs)
    net.Proto().arg.extend(predictor_export_meta.predict_net.arg)
    if predictor_export_meta.net_type is not None:
        net.Proto().type = predictor_export_meta.net_type
    if predictor_export_meta.num_workers is not None:
        net.Proto().num_workers = predictor_export_meta.num_workers
    return net.Proto()

def create_predict_init_net(ws, predictor_export_meta):
    if False:
        while True:
            i = 10
    '\n    Return an initialization net that zero-fill all the input and\n    output blobs, using the shapes from the provided workspace. This is\n    necessary as there is no shape inference functionality in Caffe2.\n    '
    net = core.Net('predict-init')

    def zero_fill(blob):
        if False:
            print('Hello World!')
        shape = predictor_export_meta.shapes.get(blob)
        if shape is None:
            if blob not in ws.blobs:
                raise Exception('{} not in workspace but needed for shape: {}'.format(blob, ws.blobs))
            shape = ws.blobs[blob].fetch().shape
        with scope.EmptyDeviceScope():
            net.ConstantFill([], blob, shape=shape, value=0.0)
    external_blobs = predictor_export_meta.inputs + predictor_export_meta.outputs
    for blob in external_blobs:
        zero_fill(blob)
    net.Proto().external_input.extend(external_blobs)
    if predictor_export_meta.extra_init_net:
        net.AppendNet(predictor_export_meta.extra_init_net)
    AddModelIdArg(predictor_export_meta, net.Proto())
    return net.Proto()

def get_comp_name(string, name):
    if False:
        print('Hello World!')
    if name:
        return string + '_' + name
    return string

def to_first_match_dict(kv_list):
    if False:
        i = 10
        return i + 15
    '\n    Construct dict from kv_list\n    '
    d = {}
    for item in kv_list:
        if item.key not in d:
            d[item.key] = item.value
    return d

def _ProtoMapGet(field, key):
    if False:
        while True:
            i = 10
    "\n    Given the key, get the value of the repeated field.\n    Helper function used by protobuf since it doesn't have map construct\n    "
    for v in field:
        if v.key == key:
            return v.value
    return None

def GetPlan(meta_net_def, key):
    if False:
        return 10
    return _ProtoMapGet(meta_net_def.plans, key)

def GetPlanOriginal(meta_net_def, key):
    if False:
        print('Hello World!')
    return _ProtoMapGet(meta_net_def.plans, key)

def GetBlobs(meta_net_def, key):
    if False:
        while True:
            i = 10
    blobs = _ProtoMapGet(meta_net_def.blobs, key)
    if blobs is None:
        return []
    return blobs

def GetBlobsByTypePrefix(meta_net_def, blob_type_prefix):
    if False:
        for i in range(10):
            print('nop')
    blob_map = {}
    for b in meta_net_def.blobs:
        if b.key.startswith(blob_type_prefix):
            for blob in b.value:
                if blob not in blob_map:
                    blob_map[blob] = len(blob_map)
    return sorted(blob_map, key=lambda blob: blob_map[blob])

def GetNet(meta_net_def, key):
    if False:
        print('Hello World!')
    return _ProtoMapGet(meta_net_def.nets, key)

def GetNetOriginal(meta_net_def, key):
    if False:
        while True:
            i = 10
    return _ProtoMapGet(meta_net_def.nets, key)

def GetApplicationSpecificInfo(meta_net_def, key):
    if False:
        for i in range(10):
            print('nop')
    return _ProtoMapGet(meta_net_def.applicationSpecificInfo, key)

def GetApplicationSpecificInfoDict(meta_net_def):
    if False:
        for i in range(10):
            print('nop')
    return to_first_match_dict(meta_net_def.applicationSpecificInfo)

def AddBlobs(meta_net_def, blob_name, blob_def):
    if False:
        for i in range(10):
            print('nop')
    blobs = _ProtoMapGet(meta_net_def.blobs, blob_name)
    if blobs is None:
        blobs = meta_net_def.blobs.add()
        blobs.key = blob_name
        blobs = blobs.value
    for blob in blob_def:
        blobs.append(blob)

def ReplaceBlobs(meta_net_def, blob_name, blob_def):
    if False:
        i = 10
        return i + 15
    blobs = _ProtoMapGet(meta_net_def.blobs, blob_name)
    assert blobs is not None, 'The blob_name:{} does not exist'.format(blob_name)
    del blobs[:]
    for blob in blob_def:
        blobs.append(blob)

def AddPlan(meta_net_def, plan_name, plan_def):
    if False:
        while True:
            i = 10
    meta_net_def.plans.add(key=plan_name, value=plan_def)

def AddNet(meta_net_def, net_name, net_def):
    if False:
        return 10
    meta_net_def.nets.add(key=net_name, value=net_def)

def SetBlobsOrder(meta_net_def, blobs_order):
    if False:
        while True:
            i = 10
    for blob in blobs_order:
        meta_net_def.blobsOrder.append(blob)

def SetPreLoadBlobs(meta_net_def, pre_load_blobs):
    if False:
        for i in range(10):
            print('nop')
    for blob in pre_load_blobs:
        meta_net_def.preLoadBlobs.append(blob)

def SetRequestOnlyEmbeddings(meta_net_def, request_only_embeddings):
    if False:
        while True:
            i = 10
    for blob in request_only_embeddings:
        meta_net_def.requestOnlyEmbeddings.append(blob)

def GetBlobsOrder(meta_net_def):
    if False:
        print('Hello World!')
    return meta_net_def.blobsOrder

def SetTensorBoundShapes(meta_net_def, tensor_bound_shapes):
    if False:
        while True:
            i = 10
    meta_net_def.tensorBoundShapes.CopyFrom(tensor_bound_shapes)

def SetAOTConfig(meta_net_def, aot_config):
    if False:
        i = 10
        return i + 15
    meta_net_def.aotConfig.CopyFrom(aot_config)

def GetArgumentByName(net_def, arg_name):
    if False:
        for i in range(10):
            print('nop')
    for arg in net_def.arg:
        if arg.name == arg_name:
            return arg
    return None

def AddModelIdArg(meta_net_def, net_def):
    if False:
        while True:
            i = 10
    'Takes the model_id from the predict_net of meta_net_def (if it is\n    populated) and adds it to the net_def passed in. This is intended to be\n    called on init_nets, as their model_id is not populated by default, but\n    should be the same as that of the predict_net\n    '
    model_id = GetArgumentByName(meta_net_def.predict_net, 'model_id')
    if model_id is None:
        return
    model_id = model_id.i
    old_id = GetArgumentByName(net_def, 'model_id')
    if old_id is not None:
        old_id.i = model_id
        return
    arg = net_def.arg.add()
    arg.name = 'model_id'
    arg.i = model_id