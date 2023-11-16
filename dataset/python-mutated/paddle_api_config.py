import inspect
import paddle

def is_inplace_api(func):
    if False:
        while True:
            i = 10
    inplace_apis = {paddle.static.setitem}
    return func in inplace_apis

def get_tensor_methods():
    if False:
        return 10
    return [member_name for (member_name, member) in inspect.getmembers(paddle.static.Variable) if inspect.isfunction(member)]

def get_paddle_api():
    if False:
        while True:
            i = 10
    modules = [paddle, paddle.nn.functional, paddle.linalg, paddle.signal, paddle.fft, paddle.vision.ops]
    special_paddle_apis = [paddle.tensor.fill_constant]
    non_operator_related_apis = [paddle.in_dynamic_mode, paddle.save, paddle.load, paddle.get_cuda_rng_state, paddle.set_rng_state, paddle.set_cuda_rng_state, paddle.get_rng_state, paddle.set_default_dtype, paddle.check_shape, paddle.summary, paddle.finfo, paddle.iinfo, paddle.enable_static, paddle.disable_static, paddle.is_grad_enabled]
    static_apis = [paddle.static.setitem, paddle.static.accuracy]
    paddle_api_list = []
    for module in modules:
        for fn_name in getattr(module, '__all__', []):
            fn = getattr(module, fn_name)
            if inspect.isfunction(fn):
                paddle_api_list.append(fn)
    return list(set(special_paddle_apis) | set(static_apis) | set(paddle_api_list) - set(non_operator_related_apis))
paddle_tensor_methods = get_tensor_methods()
paddle_api_list = get_paddle_api()
paddle_api_module_prefix = {'paddle.nn.functional', 'paddle.nn.layer.activation'}
break_graph_set = set()
break_graph_tensor_method = {'register_hook', 'numpy', 'clear_gradient'}

def is_break_graph_tensor_methods(method_name):
    if False:
        i = 10
        return i + 15
    return method_name in break_graph_tensor_method

def add_break_graph_apis(apis: list):
    if False:
        while True:
            i = 10
    break_graph_set.update(apis)