import inspect
from typing import Optional, Union
import torch

class ModelInfo:

    def __init__(self, model):
        if False:
            return 10
        self.forward_fullargspec = ModelInfo.get_forward_fullargspec(model)
        "\n        This function is to get all the arguments(excepts *args and **kwargs)\n        It will return a list of arg name\n        E.g.\n        def forward(self, a, b=1, c: int = 3, *args, **kwargs):\n           pass\n        it will return ['a', 'b', 'c']\n        "
        self.forward_args = self.forward_fullargspec.args[1:]
        '\n        This function is to get all the defaults\n        It will return a list of default values\n        E.g.\n        def forward(self, a, b=1, c: int = 3, *args, **kwargs):\n            pass\n        it will return (1, 3)\n        '
        self.forward_defaults = self.forward_fullargspec.defaults
        "\n        This function is to get all the annotations\n        It will return a dict of {args: annotations}\n        E.g.\n        def forward(self, a, b=1, c: int = 3, *args, **kwargs):\n            pass\n        it will return {'c': <class 'int'>}\n        "
        self.forward_annotations = self.forward_fullargspec.annotations

    @staticmethod
    def get_forward_fullargspec(model):
        if False:
            return 10
        "\n        This function is to get all the arguments(excepts *args and **kwargs)\n        It will return a tuple of seven things is returned:\n        (args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations).\n        'args' is a list of the parameter names.\n        'varargs' and 'varkw' are the names of the * and ** parameters or None.\n        'defaults' is an n-tuple of the default values of the last n parameters.\n        'kwonlyargs' is a list of keyword-only parameter names.\n        'kwonlydefaults' is a dictionary mapping names from kwonlyargs to defaults.\n        'annotations' is a dictionary mapping parameter names to annotations.\n        "
        from bigdl.nano.pytorch.lightning import LightningModule
        from bigdl.nano.pytorch.model import AcceleratedLightningModule
        forward_fullargspec = inspect.getfullargspec(model.forward)
        if isinstance(model, LightningModule):
            if not isinstance(model, AcceleratedLightningModule):
                forward_fullargspec = ModelInfo.get_forward_fullargspec(model.model)
        return forward_fullargspec

    def get_conditional_args(self, include: Optional[Union[tuple, str]]=(torch.Tensor, torch.FloatTensor, torch.LongTensor), exclude: Optional[Union[tuple, str]]=()):
        if False:
            i = 10
            return i + 15
        '\n        This function will return all the parameters that (might) in `condition`\n        It will return a list or tensor args name\n        E.g.\n        def forward(self, a, b=1, c: int = 3, *args, **kwargs):\n            pass\n        it will return [\'a\'] if include=(torch.Tensor)\n\n        :param include: tuple of type or "all".\n        :param exclude: tuple of type or "all".\n\n        Note: "all" means all the types are allowed or disallowed, except those\n              stated in the opposite parameter.\n        Note: exclude has higher priority if conflict instruction is provided\n        '
        include_all = True if include == 'all' else False
        exclude_all = True if exclude == 'all' else False
        fitted_args = []
        if self.forward_defaults is None:
            defaults_length = 0
        else:
            defaults_length = len(self.forward_defaults)
        args_length = len(self.forward_args)
        for (i, arg) in enumerate(self.forward_args):
            flag = False
            if arg in self.forward_annotations:
                if include_all or self.forward_annotations[arg] in include:
                    flag = True
                if exclude_all or self.forward_annotations[arg] in exclude:
                    flag = False
                if flag:
                    fitted_args.append(arg)
                continue
            default_args_start_from = args_length - defaults_length
            if i >= default_args_start_from:
                flag = False
                if include_all or type(self.forward_defaults[i - default_args_start_from]) in include:
                    flag = True
                if exclude_all or type(self.forward_defaults[i - default_args_start_from]) in exclude:
                    flag = False
                if flag:
                    fitted_args.append(arg)
                continue
            fitted_args.append(arg)
        return fitted_args