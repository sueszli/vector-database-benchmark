import torch
import os
model = None

def entry_point_function_name(data, context):
    if False:
        i = 10
        return i + 15
    '\n    Works on data and context to create model object or process inference request.\n    Following sample demonstrates how model object can be initialized for jit mode.\n    Similarly you can do it for eager mode models.\n    :param data: Input data for prediction\n    :param context: context contains model server system properties\n    :return: prediction output\n    '
    global model
    if not data:
        manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get('model_dir')
        device = torch.device('cpu')
        serialized_file = manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError('Missing the model.pt file')
        model = torch.jit.load(model_pt_path)
    else:
        data_input = torch.Tensor(list(map(lambda x: float(x), data[0].get('body').decode().split(',')))).reshape(1, 6, 1)
        return model.forward(data_input).tolist()