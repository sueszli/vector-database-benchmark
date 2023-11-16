"""Helper functions for TF-Hub modules that handle images."""
from tensorflow_hub import image_module_info_pb2
from tensorflow_hub import native_module
IMAGE_MODULE_INFO_KEY = 'image_module_info'
ImageModuleInfo = image_module_info_pb2.ImageModuleInfo

def attach_image_module_info(image_module_info):
    if False:
        while True:
            i = 10
    'Attaches an ImageModuleInfo message from within a module_fn.\n\n  Warning: Deprecated. This belongs to the hub.Module API and TF1 Hub format.\n\n  THIS FUNCTION IS DEPRECATED.\n\n  Args:\n    image_module_info: an ImageModuleInfo message.\n  '
    native_module.attach_message(IMAGE_MODULE_INFO_KEY, image_module_info)

def get_image_module_info(module_or_spec, required=False):
    if False:
        print('Hello World!')
    "Returns the module's attached ImageModuleInfo message, or None if missing.\n\n  Warning: Deprecated. This belongs to the hub.Module API and TF1 Hub format\n\n  THIS FUNCTION IS DEPRECATED.\n\n  Args:\n    module_or_spec: a hub.Module or module_spec object.\n    required: if true, raises KeyError instead of returning None.\n  "
    return module_or_spec.get_attached_message(IMAGE_MODULE_INFO_KEY, ImageModuleInfo, required=required)

def get_expected_image_size(module_or_spec, signature=None, input_name=None):
    if False:
        while True:
            i = 10
    'Returns expected [height, width] dimensions of an image input.\n\n  TODO(b/139530454): This does not work yet with TF2.\n\n  Args:\n    module_or_spec: a Module or ModuleSpec that accepts image inputs.\n    signature: a string with the key of the signature in question.\n      If None, the default signature is used.\n    input_name: a string with the input name for images. If None, the\n      conventional input name `images` for the default signature is used.\n\n  Returns:\n    A list if integers `[height, width]`.\n\n  Raises:\n    ValueError: If the size information is missing or malformed.\n  '
    image_module_info = get_image_module_info(module_or_spec)
    if image_module_info:
        size = image_module_info.default_image_size
        if size.height and size.width:
            return [size.height, size.width]
    if input_name is None:
        input_name = 'images'
    input_info_dict = module_or_spec.get_input_info_dict(signature)
    try:
        shape = input_info_dict[input_name].get_shape()
    except KeyError:
        raise ValueError("Module is missing input '%s' in signature '%s'." % (input_name, signature or 'default'))
    try:
        (_, height, width, _) = shape.as_list()
        if not height or not width:
            raise ValueError
    except ValueError:
        raise ValueError('Shape of module input is %s, expected [batch_size, height, width, num_channels] with known height and width.' % shape)
    return [height, width]

def get_num_image_channels(module_or_spec, signature=None, input_name=None):
    if False:
        i = 10
        return i + 15
    'Returns expected num_channels dimensions of an image input.\n\n  This is for advanced users only who expect to handle modules with\n  image inputs that might not have the 3 usual RGB channels.\n\n  TODO(b/139530454): This does not work yet with TF2.\n\n  Args:\n    module_or_spec: a Module or ModuleSpec that accepts image inputs.\n    signature: a string with the key of the signature in question.\n      If None, the default signature is used.\n    input_name: a string with the input name for images. If None, the\n      conventional input name `images` for the default signature is used.\n\n  Returns:\n    An integer with the number of input channels to the module.\n\n  Raises:\n    ValueError: If the channel information is missing or malformed.\n  '
    if input_name is None:
        input_name = 'images'
    input_info_dict = module_or_spec.get_input_info_dict(signature)
    try:
        shape = input_info_dict[input_name].get_shape()
    except KeyError:
        raise ValueError("Module is missing input '%s' in signature '%s'." % (input_name, signature or 'default'))
    try:
        (_, _, _, num_channels) = shape.as_list()
        if num_channels is None:
            raise ValueError
    except ValueError:
        raise ValueError('Shape of module input is %s, expected [batch_size, height, width, num_channels] with known num_channels' % shape)
    return num_channels