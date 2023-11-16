"""
Functions for Auto SParsity (ASP) training and inference.
"""
import copy
import os
import numpy as np
import paddle
from paddle.base import core, global_scope, program_guard
from paddle.base.framework import dygraph_only
from paddle.incubate import asp
from .supported_layer_list import _default_pruning, supported_layers_and_prune_func_map
OpRole = core.op_proto_and_checker_maker.OpRole
OP_ROLE_KEY = core.op_proto_and_checker_maker.kOpRoleAttrName()
__all__ = []

def set_excluded_layers(param_names, main_program=None):
    if False:
        return 10
    "\n    Set parameter name of layers which would not be pruned as sparse weights.\n\n    Args:\n        param_names (list of string): A list contains names of parameters.\n        main_program (Program, optional): Program with model definition and its parameters.\n                                          If None is given, then it would be set as `paddle.static.default_main_program().\n                                          Default is None.\n    Examples:\n        .. code-block:: python\n            :name: dynamic-graph\n\n            >>> # Example1: Usage of Dynamic Graph\n            >>> import paddle\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 100)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         prediction = self.linear1(hidden)\n            ...         return prediction\n\n            >>> my_layer = MyLayer()\n            >>> optimizer = paddle.optimizer.SGD(\n            ...     learning_rate=0.01, parameters=my_layer.parameters())\n\n            >>> # Need to set excluded layers before calling decorate\n            >>> paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()])\n\n            >>> optimizer = paddle.incubate.asp.decorate(optimizer)\n\n        .. code-block:: python\n            :name: static-graph\n\n            >>> # Example2: Usage of Static Graph\n            >>> import paddle\n\n            >>> paddle.enable_static()\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 100)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         prediction = self.linear1(hidden)\n            ...         return prediction\n\n            >>> main_program = paddle.static.Program()\n            >>> startup_program = paddle.static.Program()\n\n            >>> with paddle.static.program_guard(main_program, startup_program):\n            ...     input_data = paddle.static.data(name='data', shape=[None, 3, 224, 224])\n            ...     label = paddle.static.data(name='label', shape=[None, 100])\n            ...     my_layer = MyLayer()\n            ...     prob = my_layer(input_data)\n            ...     loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))\n            ...\n            ...     # Setup exluded layers out from ASP workflow.\n            ...     # Please note, excluded_layers must be set before calling optimizer.minimize().\n            ...     paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()], main_program)\n            ...\n            ...     optimizer = paddle.optimizer.SGD(learning_rate=0.1)\n            ...     optimizer = paddle.static.amp.decorate(optimizer )\n            ...     # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which\n            ...     # will insert necessary masking operations for ASP workflow.\n            ...     optimizer = paddle.incubate.asp.decorate(optimizer)\n            ...     optimizer.minimize(loss, startup_program)\n    "
    if main_program is None:
        main_program = paddle.static.default_main_program()
    ASPHelper.set_excluded_layers(param_names=param_names, main_program=main_program)

def reset_excluded_layers(main_program=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Reset exculded layers setting corresponding to :attr:`main_program`. If :attr:`main_program`\n    is None, then all configurations of excluded_layers would be cleaned.\n\n    Args:\n        main_program (Program, optional): Program with model definition and its parameters.\n                                          If None is given, then this function would reset all excluded_layers.\n                                          Default is None.\n    Examples:\n        .. code-block:: python\n            :name: dynamic-graph\n\n            >>> # Example1: Usage of Dynamic Graph\n            >>> import paddle\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 100)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         prediction = self.linear1(hidden)\n            ...         return prediction\n\n            >>> my_layer = MyLayer()\n            >>> optimizer = paddle.optimizer.SGD(\n            ...     learning_rate=0.01, parameters=my_layer.parameters())\n\n            >>> # Need to set excluded layers before calling decorate\n            >>> paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()])\n            >>> # Reset excluded_layers, all supported layers would be included into Automatic SParsity's workflow.\n            >>> # Please note, reset_excluded_layers also must be called before calling asp.decorate().\n            >>> paddle.incubate.asp.reset_excluded_layers()\n\n            >>> optimizer = paddle.incubate.asp.decorate(optimizer)\n\n        .. code-block:: python\n            :name: static-graph\n\n            >>> # Example2: Usage of Static Graph\n            >>> import paddle\n\n            >>> paddle.enable_static()\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 100)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         prediction = self.linear1(hidden)\n            ...         return prediction\n\n            >>> main_program = paddle.static.Program()\n            >>> startup_program = paddle.static.Program()\n\n            >>> with paddle.static.program_guard(main_program, startup_program):\n            ...     input_data = paddle.static.data(name='data', shape=[None, 3, 224, 224])\n            ...     label = paddle.static.data(name='label', shape=[None, 100])\n            ...     my_layer = MyLayer()\n            ...     prob = my_layer(input_data)\n            ...     loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))\n            ...\n            ...     # Setup exluded layers out from ASP workflow.\n            ...     # Please note, excluded_layers must be set before calling optimizer.minimize().\n            ...     paddle.incubate.asp.set_excluded_layers([my_layer.linear1.full_name()], main_program)\n            ...     # Reset excluded_layers, all supported layers would be included into Automatic SParsity's workflow.\n            ...     # Please note, reset_excluded_layers also must be called before calling optimizer.minimize().\n            ...     paddle.incubate.asp.reset_excluded_layers(main_program)\n            ...\n            ...     optimizer = paddle.optimizer.SGD(learning_rate=0.1)\n            ...     optimizer = paddle.static.amp.decorate(optimizer )\n            ...     # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which\n            ...     # will insert necessary masking operations for ASP workflow.\n            ...     optimizer = paddle.incubate.asp.decorate(optimizer)\n            ...     optimizer.minimize(loss, startup_program)\n    "
    ASPHelper.reset_excluded_layers(main_program=main_program)

def decorate(optimizer):
    if False:
        i = 10
        return i + 15
    "\n    Wrap the given optimizer as a OptimizerWithSparsityGuarantee,\n    If runnig with dynamic graph mode. ASP would creates mask variables for supported parameters.\n    Else if in static graph mode, ASP would creates mask variables and inserts necessary ops\n    when calling minimize()\n\n    Args:\n        optimizer (Optimizer): A Optimizer used for training.\n    Returns:\n        OptimizerWithSparsityGuarantee: A wrapper for ASP to decorate `minimize` function of the given optimizer.\n    Examples:\n        .. code-block:: python\n            :name: dynamic-graph\n\n            >>> # Example1: Usage of Dynamic Graph\n            >>> import paddle\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 32)\n            ...         self.linear2 = paddle.nn.Linear(32, 32)\n            ...         self.linear3 = paddle.nn.Linear(32, 10)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         hidden = self.linear1(hidden)\n            ...         hidden = self.linear2(hidden)\n            ...         prediction = self.linear3(hidden)\n            ...         return prediction\n\n            >>> my_layer = MyLayer()\n            >>> optimizer = paddle.optimizer.SGD(\n            ...     learning_rate=0.01, parameters=my_layer.parameters())\n\n            >>> # Calling paddle.incubate.asp.decorate() to wrap step() in optimizer, which\n            >>> # will apply necessary masking operations for ASP workflow.\n            >>> # In dynamic graph mode, ASP would create related mask variables during decoration.\n            >>> optimizer = paddle.incubate.asp.decorate(optimizer)\n\n        .. code-block:: python\n            :name: static-graph\n\n            >>> # Example2: Usage of Static Graph\n            >>> import paddle\n\n            >>> paddle.enable_static()\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 100)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         prediction = self.linear1(hidden)\n            ...         return prediction\n\n            >>> main_program = paddle.static.Program()\n            >>> startup_program = paddle.static.Program()\n\n            >>> with paddle.static.program_guard(main_program, startup_program):\n            ...     input_data = paddle.static.data(name='data', shape=[None, 3, 224, 224])\n            ...     label = paddle.static.data(name='label', shape=[None, 100])\n            ...     my_layer = MyLayer()\n            ...     prob = my_layer(input_data)\n            ...     loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))\n            ...\n            ...     optimizer = paddle.optimizer.SGD(learning_rate=0.1)\n            ...     # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which\n            ...     # will insert necessary masking operations for ASP workflow.\n            ...     # In static graph mode, ASP creates related mask variables\n            ...     # during minimize().\n            ...     optimizer = paddle.incubate.asp.decorate(optimizer)\n            ...     optimizer.minimize(loss, startup_program)\n    "
    return ASPHelper.decorate(optimizer)

def prune_model(model, n=2, m=4, mask_algo='mask_1d', with_mask=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Pruning parameters of supported layers in :attr:`model` via\n    specified mask generation function given by :attr:`mask_algo`. This\n    function supports both training and inference controlled by :attr:`with_mask`.\n    If :attr:`with_mask` is True, it would also prune parameter related ASP mask Variables,\n    else only prunes parameters.\n\n    *Note*: (Static graph mode) If calling this function with :attr:`with_mask`, it should call `OptimizerWithSparsityGuarantee.minimize`\n    and initialization (`exe.run(startup_program`)) before (For successfully obtain mask Variable).\n    Typically set `with_mask` as true for training (have called `OptimizerWithSparsityGuarantee.minimize`) and false for\n    inference only. To obtain OptimizerWithSparsityGuarantee, please see `paddle.incubate.asp.decoreate()`.\n\n    Args:\n        model (Program|nn.Layer): Program with model definition and its parameters, or a object of `paddle.nn.Layer`.\n        n (int, optional): n of `n:m` sparse pattern. Default is 2.\n        m (int, optional): m of `n:m` sparse pattern. Default is 4.\n        mask_algo (string, optional): The function name to generate spase mask. Default is `mask_1d`.\n                                      The vaild inputs should be one of 'mask_1d', 'mask_2d_greedy' and 'mask_2d_best'.\n        with_mask (bool, optional): To prune mask Variables related to parameters or not. True is purning also, False is not. Default is True.\n    Returns:\n        dictionary: A dictionary with key: `parameter name` (string) and value: its corresponding mask Variable.\n    Examples:\n        .. code-block:: python\n            :name: dynamic-graph\n\n            >>> # Example1: Usage of Dynamic Graph\n            >>> import paddle\n            >>> import numpy as np\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 32)\n            ...         self.linear2 = paddle.nn.Linear(32, 32)\n            ...         self.linear3 = paddle.nn.Linear(32, 10)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         hidden = self.linear1(hidden)\n            ...         hidden = self.linear2(hidden)\n            ...         prediction = self.linear3(hidden)\n            ...         return prediction\n\n            >>> my_layer = MyLayer()\n            >>> loss_fn = paddle.nn.MSELoss(reduction='mean')\n\n            >>> optimizer = paddle.optimizer.SGD(\n            ...     learning_rate=0.01, parameters=my_layer.parameters())\n\n            >>> # Calling paddle.incubate.asp.decorate() to wrap step() in optimizer, which\n            >>> # will apply necessary masking operations for ASP workflow.\n            >>> # In dynamic graph mode, ASP would create related mask variables during decoration.\n            >>> optimizer = paddle.incubate.asp.decorate(optimizer)\n\n            >>> # Must call paddle.incubate.asp.decorate() first before calling paddle.incubate.asp.prune_model()\n            >>> paddle.incubate.asp.prune_model(my_layer, mask_algo='mask_2d_best')\n\n            >>> for i in range(10):\n            ...     imgs = paddle.to_tensor(\n            ...         np.random.randn(64, 3, 32, 32),\n            ...         dtype='float32', stop_gradient=False)\n            ...     labels = paddle.to_tensor(\n            ...         np.random.randint(10, size=(64, 1)),\n            ...         dtype='float32', stop_gradient=False)\n            ...     output = my_layer(imgs)\n            ...     loss = loss_fn(output, labels)\n            ...     loss.backward()\n            ...     optimizer.step()\n            ...     optimizer.clear_grad()\n\n        .. code-block:: python\n            :name: static-graph\n\n            >>> # Example2: Usage of Static Graph\n            >>> import paddle\n            >>> import numpy as np\n\n            >>> paddle.enable_static()\n\n            >>> class MyLayer(paddle.nn.Layer):\n            ...     def __init__(self):\n            ...         super().__init__()\n            ...         self.conv1 = paddle.nn.Conv2D(\n            ...             in_channels=3, out_channels=4, kernel_size=3, padding=2)\n            ...         self.linear1 = paddle.nn.Linear(4624, 32)\n            ...         self.linear2 = paddle.nn.Linear(32, 32)\n            ...         self.linear3 = paddle.nn.Linear(32, 10)\n            ...\n            ...     def forward(self, img):\n            ...         hidden = self.conv1(img)\n            ...         hidden = paddle.flatten(hidden, start_axis=1)\n            ...         hidden = self.linear1(hidden)\n            ...         hidden = self.linear2(hidden)\n            ...         prediction = self.linear3(hidden)\n            ...         return prediction\n\n            >>> main_program = paddle.static.Program()\n            >>> startup_program = paddle.static.Program()\n\n            >>> with paddle.static.program_guard(main_program, startup_program):\n            ...     input_data = paddle.static.data(name='data', shape=[None, 3, 32, 32])\n            ...     label = paddle.static.data(name='label', shape=[None, 1])\n            ...     my_layer = MyLayer()\n            ...     prob = my_layer(input_data)\n            ...     loss = paddle.mean(paddle.nn.functional.square_error_cost(prob, label))\n            ...\n            ...     optimizer = paddle.optimizer.SGD(learning_rate=0.1)\n            ...     # Calling paddle.incubate.asp.decorate() to wrap minimize() in optimizer, which\n            ...     # will insert necessary masking operations for ASP workflow.\n            ...     # In static graph mode, ASP creates related mask variables\n            ...     # during minimize().\n            ...     optimizer = paddle.incubate.asp.decorate(optimizer)\n            ...     optimizer.minimize(loss, startup_program)\n\n            >>> device = paddle.device.get_device()\n            >>> place = paddle.set_device(device)\n\n            >>> exe = paddle.static.Executor(place)\n            >>> exe.run(startup_program)\n\n            >>> # Must call exe.run(startup_program) first before calling paddle.asp.prune_model()\n            >>> paddle.incubate.asp.prune_model(my_layer, mask_algo='mask_2d_best')\n            >>> # it also be accepted to call\n            >>> # paddle.incubate.asp.prune_model(main_program, mask_algo='mask_2d_best')\n\n            >>> for i in range(10):\n            ...     imgs = np.random.randn(64, 3, 32, 32).astype('float32')\n            ...     labels = np.random.randint(10, size=(64, 1)).astype('float32')\n            ...     exe.run(main_program, feed={'data':imgs, 'label':labels})\n    "
    device = paddle.device.get_device()
    place = paddle.set_device(device)
    MaskAlgo_mapping = {'mask_1d': asp.MaskAlgo.MASK_1D, 'mask_2d_greedy': asp.MaskAlgo.MASK_2D_GREEDY, 'mask_2d_best': asp.MaskAlgo.MASK_2D_BEST}
    assert mask_algo in MaskAlgo_mapping, 'The "mask_algo" should be one of ["mask_1d", "mask_2d_greedy", "mask_2d_best"]'
    prune_func = None
    if isinstance(model, paddle.nn.Layer):
        prune_func = ASPHelper.prune_model_by_layer
    elif isinstance(model, paddle.static.Program):
        prune_func = ASPHelper.prune_model_by_program
        if hasattr(model, 'distributed_info_') and model.distributed_info_['sharding_degree'] > 1 and paddle.base.is_compiled_with_cuda():
            gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
            place = paddle.CUDAPlace(gpu_id)
    else:
        raise TypeError('model should be paddle.nn.Layer or paddle.static.Program, but got {}'.format(type(model)))
    return prune_func(place, model, n=n, m=m, mask_algo=MaskAlgo_mapping[mask_algo], with_mask=with_mask)

class ProgramASPInfo:
    """
    ProgramASPInfo is a container to keep ASP relevant information of Pragrom. It contains three inner-variables:
    1. __mask_vars (Dictionary): Key is parameter's name and vaule is its corresponding sparse mask Variable object, which is created by `ASPHelper.create_mask_variables`.
    2. __masks (Dictionary): Key is parameter's name and vaule is its corressponding sparse mask Numpy array, which is created by `ASPHelper.prune_model`.
    3. __excluded_layers (List): It stores name of layers which should not involve into ASP workflow.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__mask_vars = {}
        self.__masks = {}
        self.__excluded_layers = []

    def update_mask_vars(self, param_name, var):
        if False:
            print('Hello World!')
        self.__mask_vars[param_name] = var

    def update_masks(self, param_name, var):
        if False:
            return 10
        self.__masks[param_name] = var

    def update_excluded_layers(self, param_names):
        if False:
            for i in range(10):
                print('nop')
        self.__excluded_layers.extend(copy.deepcopy(param_names))

    def reset_excluded_layers(self):
        if False:
            i = 10
            return i + 15
        self.__excluded_layers = []

    @property
    def mask_vars(self):
        if False:
            return 10
        return self.__mask_vars

    @property
    def masks(self):
        if False:
            return 10
        return self.__masks

    @property
    def excluded_layers(self):
        if False:
            i = 10
            return i + 15
        return self.__excluded_layers

class ASPHelper:
    """
    ASPHelper is a collection of Auto SParsity (ASP) functions to enable

    1. training models with weights in 2:4 sparse pattern on FP16 or 1:2 sparse pattern on FP32 from scratch.
    2. pruning well-trained models into 2:4 sparse pattern on FP16 or 1:2 sparse pattern on FP32 for fine-tuning.
    """
    MASK_APPENDDED_NAME = 'asp_mask'
    PADDLE_WEIGHT_SUFFIX = 'w_'
    __asp_info = {}

    @classmethod
    def set_excluded_layers(cls, param_names, main_program):
        if False:
            print('Hello World!')
        '\n        This is the implementation of `asp.set_excluded_layers`, for details please see explanation in `asp.set_excluded_layers`.\n        '
        asp_info = cls._get_program_asp_info(main_program)
        asp_info.update_excluded_layers(param_names)

    @classmethod
    def reset_excluded_layers(cls, main_program=None):
        if False:
            i = 10
            return i + 15
        '\n        This is the implementation of `asp.reset_excluded_layers`, for details please see explanation in `asp.reset_excluded_layers`.\n        '
        if main_program is None:
            for prog in cls.__asp_info:
                cls.__asp_info[prog].reset_excluded_layers()
        else:
            cls._get_program_asp_info(main_program).reset_excluded_layers()

    @staticmethod
    def decorate(optimizer):
        if False:
            while True:
                i = 10
        '\n        This is the implementation of `asp.decorate`, for details please see explanation in `asp.decorate`.\n        '
        if paddle.in_dynamic_mode():
            main_prog = paddle.static.default_main_program()
            startup_prog = paddle.static.default_startup_program()
            ASPHelper._create_mask_variables(main_prog, startup_prog, optimizer._parameter_list)
        return OptimizerWithSparsityGuarantee(optimizer)

    @classmethod
    def prune_model_by_program(cls, place, main_program=None, n=2, m=4, mask_algo=asp.MaskAlgo.MASK_1D, with_mask=True):
        if False:
            i = 10
            return i + 15
        '\n        This is the implementation of `asp.prune_model`, for details please see explanation in `asp.prune_model`.\n        '
        if main_program is None:
            main_program = paddle.static.default_main_program()
        asp_info = cls._get_program_asp_info(main_program)
        for param in main_program.global_block().all_parameters():
            if ASPHelper._is_supported_layer(main_program, param.name):
                weight_tensor = global_scope().find_var(param.name).get_tensor()
                weight_nparray = np.array(weight_tensor)
                prune_func = ASPHelper._get_prune_func_by_name(param.name)
                (weight_pruned_nparray, weight_sparse_mask) = prune_func(weight_nparray, m, n, mask_algo, param.name)
                weight_pruned_nparray = weight_pruned_nparray.astype(weight_nparray.dtype)
                weight_tensor.set(weight_pruned_nparray, place)
                if with_mask:
                    weight_mask_param = global_scope().find_var(ASPHelper._get_mask_name(param.name))
                    assert weight_mask_param is not None, 'Cannot find {} variable, please call optimizer.minimize (paddle.incubate.asp.decorate(optimizer).minimize(loss) and initialization (exe.run(startup_program)) first!'.format(ASPHelper._get_mask_name(param.name))
                    weight_mask_tensor = weight_mask_param.get_tensor()
                    weight_sparse_mask = weight_sparse_mask.astype(np.array(weight_mask_tensor).dtype)
                    weight_mask_tensor.set(weight_sparse_mask, place)
                asp_info.update_masks(param.name, weight_sparse_mask)
        return asp_info.masks.copy()

    @classmethod
    def prune_model_by_layer(cls, place, layer, n=2, m=4, mask_algo=asp.MaskAlgo.MASK_1D, with_mask=True):
        if False:
            while True:
                i = 10
        '\n        This is the implementation of `asp.prune_model`, for details please see explanation in `asp.prune_model`.\n        '
        if paddle.in_dynamic_mode():
            main_program = paddle.static.default_main_program()
            asp_info = cls._get_program_asp_info(main_program)
            for param in layer.parameters():
                if ASPHelper._is_supported_layer(main_program, param.name):
                    weight_nparray = param.numpy()
                    prune_func = ASPHelper._get_prune_func_by_name(param.name)
                    (weight_pruned_nparray, weight_sparse_mask) = prune_func(weight_nparray, m, n, mask_algo, param.name)
                    weight_pruned_nparray = weight_pruned_nparray.astype(weight_nparray.dtype)
                    param.set_value(weight_pruned_nparray)
                    if with_mask:
                        weight_mask_param = asp_info.mask_vars.get(param.name, None)
                        assert weight_mask_param is not None, 'Cannot find {} variable, please call asp.decorate() to decorate your optimizer first!'.format(ASPHelper._get_mask_name(param.name))
                        weight_mask_param.set_value(weight_sparse_mask)
                    asp_info.update_masks(param.name, weight_sparse_mask)
            return asp_info.masks.copy()
        else:
            target_program = None
            for param in layer.parameters():
                target_program = param.block.program
            assert target_program is not None, 'Cannot get paddle.static.Program from Paddle.nn.Layer.'
            return ASPHelper.prune_model_by_program(place, target_program, n=n, m=m, mask_algo=mask_algo, with_mask=with_mask)

    @staticmethod
    def _get_mask_name(param_name):
        if False:
            return 10
        '\n        Return mask name by given parameter name :attr:`param_name`.\n\n        Args:\n            param_name (string): The name of parameter.\n        Returns:\n            string: The mask name of :attr:`param_name`.\n        '
        return param_name + '.' + ASPHelper.MASK_APPENDDED_NAME

    @staticmethod
    def _get_not_ASP_relevant_vars(main_program):
        if False:
            print('Hello World!')
        "\n        Get all parameters's Variables in :attr:`main_program` but excluded ASP mask Variables.\n\n        Args:\n            main_program (Program): Program with model definition and its parameters.\n        Returns:\n            list: A list of parameter Variables in :attr:`main_program` (excluded ASP mask Variables).\n        "
        var_list = []
        for param in main_program.global_block().all_parameters():
            param_name_list = param.name.split('.')
            if ASPHelper.MASK_APPENDDED_NAME not in param_name_list:
                var_list.append(param)
        return var_list

    @classmethod
    def _get_program_asp_info(cls, main_program):
        if False:
            while True:
                i = 10
        if main_program not in cls.__asp_info:
            cls.__asp_info[main_program] = ProgramASPInfo()
        return cls.__asp_info[main_program]

    @classmethod
    def _is_supported_layer(cls, main_program, param_name):
        if False:
            i = 10
            return i + 15
        "\n        Verify if given :attr:`param_name` is supported by ASP.\n\n        Args:\n            param_name (string): The name of parameter.\n        Returns:\n            bool: True if it is supported, else False.\n        Examples:\n            .. code-block:: python\n\n                >>> from paddle.incubate.asp import ASPHelper\n                >>> paddle.enable_static()\n\n                >>> main_program = paddle.static.Program()\n                >>> startup_program = paddle.static.Program()\n\n                >>> with paddle.static.program_guard(main_program, startup_program):\n                ...     input_data = paddle.static.data(name='data', shape=[None, 128])\n                ...     fc = paddle.static.nn.fc(x=input_data, num_flatten_dims=-1, size=32, activation=None)\n\n                >>> for param in main_program.global_block().all_parameters():\n                ...     print(param.name,'->',ASPHelper._is_supported_layer(main_program, param.name))\n                fc_0.w_0 -> True\n                fc_0.b_0 -> False\n        "
        param_name_list = param_name.split('.')
        if ASPHelper.MASK_APPENDDED_NAME in param_name_list:
            return False
        for layer in cls._get_program_asp_info(main_program).excluded_layers:
            if layer in param_name:
                return False
        if param_name in supported_layers_and_prune_func_map:
            return True
        if len(param_name_list) == 1:
            return False
        param_name_no_weight_suffix = param_name_list[0]
        param_type_suffix = param_name_list[1]
        layer_name = param_name_no_weight_suffix[:param_name_no_weight_suffix.rfind('_')]
        if ASPHelper.PADDLE_WEIGHT_SUFFIX not in param_type_suffix:
            return False
        if param_name_no_weight_suffix in supported_layers_and_prune_func_map or layer_name in supported_layers_and_prune_func_map:
            return True
        return False

    @classmethod
    def _get_prune_func_by_name(cls, param_name):
        if False:
            for i in range(10):
                print('nop')
        func = supported_layers_and_prune_func_map.get(param_name, None)
        param_name_no_weight_suffix = param_name.split('.')[0]
        if func is None:
            func = supported_layers_and_prune_func_map.get(param_name_no_weight_suffix, None)
        if func is None:
            layer_name = param_name_no_weight_suffix[:param_name_no_weight_suffix.rfind('_')]
            func = supported_layers_and_prune_func_map.get(layer_name, _default_pruning)
        return func

    @classmethod
    def _minimize(cls, optimizer, loss, main_program=None, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        This function is a decorator of `minimize` function in `Optimizer`.\n        There are three steps:\n\n        1. Call :attr:`optimizer`.minimize(:attr:`loss`)\n        2. Create sparse mask Tensors according to supported layers in :attr:`main_program`.\n        3. Insert masking ops in the end of parameters update.\n\n        *Note*: Please use `ASP.decorate` instead when applying distributed training with `Fleet`.\n        (Due to there is a invisiable graphs optimization in `Fleet.minimize()` which make training graph\n        cannot be modified anymore.)\n\n        Args:\n            optimizer (Optimizer): A Optimizer used for training.\n            loss (Variable): A Variable containing the value to minimize.\n            main_program (Program, optional): Program with model definition and its parameters. Default is `loss.block.program`.\n            startup_program (Program, optional): Program for initializing parameters in `parameter_list`. Default is `paddle.static.default_startup_program()`.\n            parameter_list (Iterable, optional): Iterable of `Variable` or `Variable.name` to update to minimize `loss`. The default value is None, at this time all parameters will be updated.\n            no_grad_set (set, optional): Set of `Variable  or `Variable.name` that don't need to be updated. The default value is None.\n        Returns:\n            list: operators from :attr:`optimizer`.minimize(:attr:`loss`).\n            list: pairs of parameters and their gradients.\n        "
        if main_program is None:
            main_program = loss.block.program
        if startup_program is None:
            startup_program = paddle.static.default_startup_program()
        (optimizer_ops, params_and_grads) = optimizer.minimize(loss, startup_program, parameter_list, no_grad_set=no_grad_set)
        params_only = [pg[0] for pg in params_and_grads]
        cls._create_mask_variables(main_program, startup_program, params_only)
        cls._insert_sparse_mask_ops(main_program, params_only)
        return (optimizer_ops, params_and_grads)

    @classmethod
    @dygraph_only
    def _step(cls, optimizer):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function is a decorator of `step` function in `Optimizer`.\n        There are three steps:\n\n        1. Call :attr:`optimizer`.step()\n        2. Mask parameters with sparse masks.\n\n        *Note*: Please use `ASP.decorate` instead when applying distributed training with `Fleet`.\n        (Due to there is a invisiable graphs optimization in `Fleet.minimize()` which make training graph\n        cannot be modified anymore.)\n\n        Args:\n            optimizer (Optimizer): A Optimizer used for training.\n        '
        optimizer.step()
        main_prog = paddle.static.default_main_program()
        with paddle.base.dygraph.no_grad():
            ASPHelper._insert_sparse_mask_ops(main_prog, optimizer._parameter_list)

    @classmethod
    def _create_mask_variables(cls, main_program, startup_program, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create sparse mask Tensors according to supported layers in :attr:`main_program`.\n        This function is called in second step of `ASPHelper._minimize`\n\n        Args:\n            main_program (Program): Program with model definition and its parameters.\n            startup_program (Program): Program for initializing parameters.\n            params (list): Variable parameters.\n        '
        asp_info = cls._get_program_asp_info(main_program)
        with program_guard(main_program, startup_program):
            for param in params:
                if ASPHelper._is_supported_layer(main_program, param.name):
                    if param.name not in asp_info.mask_vars:
                        mask_param = paddle.create_parameter(name=ASPHelper._get_mask_name(param.name), shape=param.shape, dtype=param.dtype, default_initializer=paddle.nn.initializer.Constant(value=1.0))
                        mask_param.stop_gradient = True
                        mask_param.trainable = False
                        asp_info.update_mask_vars(param.name, mask_param)

    @classmethod
    def _insert_sparse_mask_ops(cls, main_program, params):
        if False:
            for i in range(10):
                print('nop')
        '\n        Insert masking ops in the end of parameters update.\n        This function is called in third step of `ASPHelper._minimize`\n\n        Args:\n            main_program (Program): Program with model definition and its parameters.\n            params (list): Variable parameters.\n        '
        block = main_program.global_block()
        asp_info = cls._get_program_asp_info(main_program)
        for param in params:
            if param.name in asp_info.mask_vars:
                block.append_op(type='elementwise_mul', inputs={'X': param, 'Y': asp_info.mask_vars[param.name]}, outputs={'Out': param}, attrs={'axis': -1, 'use_mkldnn': False, OP_ROLE_KEY: int(OpRole.Optimize)})

class OptimizerWithSparsityGuarantee:
    """
    OptimizerWithSparsityGuarantee is a wrapper to decorate `minimize` function of given optimizer by `_minimize` of ASPHelper.
    The decorated `minimize` function would do three things (exactly same as `ASPHelper._minimize`):
    1. Call `minimize` function of given optimizer.
    2. Call `ASPHelper._create_mask_variables` to create mask Variables.
    3. Call `ASPHelper._insert_sparse_mask_ops` to insert weight masking ops in the end of `loss`'s Program.
    """

    def __init__(self, optimizer):
        if False:
            while True:
                i = 10
        self._optimizer = optimizer

    def __getattr__(self, item):
        if False:
            print('Hello World!')
        return getattr(self._optimizer, item)

    def minimize(self, loss, startup_program=None, parameter_list=None, no_grad_set=None):
        if False:
            print('Hello World!')
        "\n        This function is to call `ASPHelper.minimize()` and return its return\n\n        Args:\n            loss (Variable): A Variable containing the value to minimize.\n            startup_program (Program, optional): Program for initializing parameters in `parameter_list`. Default is `paddle.static.default_startup_program()`.\n            parameter_list (Iterable, optional): Iterable of `Variable` or `Variable.name` to update to minimize `loss`. The default value is None, at this time all parameters will be updated.\n            no_grad_set (set, optional): Set of `Variable  or `Variable.name` that don't need to be updated. The default value is None.\n        Returns:\n            list: operators from :attr:`optimizer`.minimize(:attr:`loss`).\n            list: pairs of parameters and their gradients.\n        "
        return ASPHelper._minimize(self._optimizer, loss, startup_program=startup_program, parameter_list=parameter_list, no_grad_set=no_grad_set)

    @dygraph_only
    def step(self):
        if False:
            i = 10
            return i + 15
        '\n        This function is a decorator of `step` function in `Optimizer`.\n        There are three steps:\n\n        1. Call :attr:`optimizer`.step()\n        2. Mask parameters with sparse masks.\n\n        *Note*: Please use `ASP.decorate` instead when applying distributed training with `Fleet`.\n        (Due to there is a invisiable graphs optimization in `Fleet.minimize()` which make training graph\n        cannot be modified anymore.)\n\n        Args:\n            optimizer (Optimizer): A Optimizer used for training.\n        '
        ASPHelper._step(self._optimizer)

    @dygraph_only
    def state_dict(self):
        if False:
            return 10
        '\n        This function is a decorator of `state_dict` function in `Optimizer`.\n\n        Returns:\n            state_dict(dict) : dict contains all the Tensor used by optimizer\n        '
        state_dict = self._optimizer.state_dict()
        asp_info = ASPHelper._get_program_asp_info(paddle.static.default_main_program())
        for (param_name, var) in asp_info.mask_vars.items():
            state_dict.update({ASPHelper._get_mask_name(param_name): var})
        return state_dict

    @dygraph_only
    def set_state_dict(self, state_dict):
        if False:
            while True:
                i = 10
        '\n        This function is a decorator of `set_state_dict` function in `Optimizer`.\n        Args:\n            state_dict(dict) : Dict contains all the Tensor needed by optimizer\n        Return:\n            None\n        '
        asp_info = ASPHelper._get_program_asp_info(paddle.static.default_main_program())
        for (param_name, var) in asp_info.mask_vars.items():
            param_mask_name = ASPHelper._get_mask_name(param_name)
            assert param_mask_name in state_dict, f'The {param_mask_name} is not found.'
            var.set_value(state_dict[param_mask_name])
            asp_info.update_masks(param_name, var.numpy())
        return self._optimizer.set_state_dict(state_dict)