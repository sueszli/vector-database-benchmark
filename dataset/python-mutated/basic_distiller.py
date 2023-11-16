from __future__ import annotations
from collections import defaultdict
import logging
from typing import Any, Callable, Dict, List, overload
import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from ..base.compressor import Compressor, Distiller, _DISTILLATION_TARGET_SPACES
from ..base.wrapper import ModuleWrapper, register_wrappers
from ..utils import Evaluator, _EVALUATOR_DOCSTRING
_logger = logging.getLogger(__name__)

class TeacherModelBasedDistiller(Distiller):
    __doc__ = 'The base class that the distiller need a teacher model.\n\n    Parameters\n    ----------\n    model\n        The student model to be distilled.\n    config_list\n        A list of dict, each dict configure which module need to be distilled, and how to distill.\n        Please refer :doc:`Compression Config Specification </compression/config_list>` for more information.\n    evaluator\n        {evaluator_docstring}\n    teacher_model\n        The distillation teacher model.\n    teacher_predict\n        A callable function with two inputs (batch, model).\n\n        Example::\n\n            def teacher_predict(batch, teacher_model):\n                return teacher_model(**batch)\n\n    origin_loss_lambda\n        A scaling factor to control the original loss scale.\n    '.format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, teacher_model: torch.nn.Module, teacher_predict: Callable[[Any, torch.nn.Module], torch.Tensor], origin_loss_lambda: float=1.0):
        if False:
            print('Hello World!')
        ...

    @overload
    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, teacher_model: torch.nn.Module, teacher_predict: Callable[[Any, torch.nn.Module], torch.Tensor], origin_loss_lambda: float=1.0, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            return 10
        ...

    def __init__(self, model: torch.nn.Module, config_list: List[Dict], evaluator: Evaluator, teacher_model: torch.nn.Module, teacher_predict: Callable[[Any, torch.nn.Module], torch.Tensor], origin_loss_lambda: float=1.0, existed_wrappers: Dict[str, ModuleWrapper] | None=None):
        if False:
            print('Hello World!')
        assert model is not teacher_model, 'Student model and teacher model should not be the same.'
        super().__init__(model=model, config_list=config_list, evaluator=evaluator, existed_wrappers=existed_wrappers)
        self.teacher_model = teacher_model
        self.teacher_predict = teacher_predict
        self.origin_loss_lambda = origin_loss_lambda
        self._set_default_link()
        self._set_default_lambda()
        (self._teacher_module_wrappers, target_spaces) = self._register_teacher_wrappers()
        self._teacher_target_spaces: _DISTILLATION_TARGET_SPACES = target_spaces
        self._teacher_is_wrapped = False
        self.wrap_teacher_model()

    @classmethod
    def from_compressor(cls, compressor: Compressor, new_config_list: List[Dict], teacher_model: torch.nn.Module, teacher_predict: Callable[[Any, torch.nn.Module], torch.Tensor], origin_loss_lambda: float=1.0, evaluator: Evaluator | None=None):
        if False:
            print('Hello World!')
        return super().from_compressor(compressor, new_config_list, evaluator=evaluator, teacher_model=teacher_model, teacher_predict=teacher_predict, origin_loss_lambda=origin_loss_lambda)

    def _set_default_link(self):
        if False:
            return 10
        for (module_name, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                link = target_space.link if target_space.link is not None else 'auto'
                link = module_name if link == 'auto' else link
                link = [link] if isinstance(link, str) else link
                target_space.link = link

    def _set_default_lambda(self):
        if False:
            for i in range(10):
                print('nop')
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.lambda_ = target_space.lambda_ if target_space.lambda_ is not None else 1.0

    def _register_teacher_wrappers(self):
        if False:
            i = 10
            return i + 15
        link2targets = defaultdict(set)
        teacher_config_list = []
        for (_, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                for link in target_space.link:
                    link2targets[link].add(target_name)
        teacher_config_list = [{'op_names': [link], 'target_names': list(target_names)} for (link, target_names) in link2targets.items()]
        return register_wrappers(self.teacher_model, teacher_config_list, mode=self.mode)

    def wrap_teacher_model(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Traverse all teacher wrappers and execute ModuleWrapper.wrap()\n        '
        if self._teacher_is_wrapped is True:
            warn_msg = 'The bound model has been wrapped, no need to wrap again.'
            _logger.warning(warn_msg)
        for (_, wrapper) in self._teacher_module_wrappers.items():
            wrapper.wrap()
        self._teacher_is_wrapped = True

    def unwrap_teacher_model(self):
        if False:
            print('Hello World!')
        '\n        Traverse all teacher wrappers and execute ModuleWrapper.unwrap()\n        '
        if self._teacher_is_wrapped is False:
            warn_msg = 'The bound model is not wrapped, can not unwrap it.'
            _logger.warning(warn_msg)
        for (_, wrapper) in self._teacher_module_wrappers.items():
            wrapper.unwrap()
        self._teacher_is_wrapped = False

    def _register_loss_patch(self, evaluator: Evaluator):
        if False:
            while True:
                i = 10

        def loss_patch(original_loss, batch):
            if False:
                return 10
            with torch.no_grad():
                self.teacher_predict(batch, self.teacher_model)
            return self.origin_loss_lambda * original_loss + self.compute_distill_loss()
        evaluator.patch_loss(loss_patch)

    def compute_distill_loss(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def _single_compress(self, max_steps: int | None, max_epochs: int | None):
        if False:
            for i in range(10):
                print('nop')
        self._fusion_compress(max_steps, max_epochs)

    def _fuse_preprocess(self, evaluator: Evaluator):
        if False:
            return 10
        self._register_loss_patch(evaluator)

    def _fuse_postprocess(self, evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')
        pass

class DynamicLayerwiseDistiller(TeacherModelBasedDistiller):
    __doc__ = "\n    Each student model distillation target (i.e., the output of a layer in the student model) will link to a list of\n    teacher model distillation targets in this distiller.\n    During distillation, a student target will compute a list of distillation losses with each of its linked teacher targets,\n    then choose the minimum loss in the loss list as current student target distillation loss.\n    The final distillation loss is the sum of each student target distillation loss multiplied by lambda.\n    The final training loss is original loss multiplied by origin_loss_lambda add final distillation loss.\n\n    Parameters\n    ----------\n    model\n        The student model to be distilled.\n    config_list\n        Config list to configure how to distill.\n        Common keys please refer :doc:`Compression Config Specification </compression/config_list>`.\n\n        Specific keys:\n\n        * 'lambda': By default, 1.\n          This is a scaling factor to control the loss scale, the final loss used during training is\n          ``(origin_loss_lambda * origin_loss + sum(lambda_i * distill_loss_i))``.\n          Here ``i`` represents the ``i-th`` distillation target.\n          The higher the value of lambda, the greater the contribution of the corresponding distillation target to the loss.\n        * 'link': By default, 'auto'.\n          'auto' or a teacher module name or a list of teacher module names,\n          the module name(s) of teacher module(s) will align with student module(s) configured in this config.\n          If 'auto' is set, will use student module name as the link,\n          usually requires the teacher model and the student model to be isomorphic.\n        * 'apply_method': By default, 'mse'.\n          'mse' and 'kl' are supported right now. 'mse' means the MSE loss, usually used to distill hidden states.\n          'kl' means the KL loss, usually used to distill logits.\n\n    evaluator\n        {evaluator_docstring}\n    teacher_model\n        The distillation teacher model.\n    teacher_predict\n        A callable function with two inputs (batch, model).\n\n        Example::\n\n            def teacher_predict(batch, teacher_model):\n                return teacher_model(**batch)\n\n    origin_loss_lambda\n        A scaling factor to control the original loss scale.\n    ".format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    def compute_distill_loss(self):
        if False:
            i = 10
            return i + 15
        distill_loss = 0
        for (_, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                stu_hs = target_space.hidden_state
                loss_list = []
                for link in target_space.link:
                    teacher_target_space = self._teacher_target_spaces[link][target_name]
                    tea_hs = teacher_target_space.hidden_state
                    if stu_hs is not None and tea_hs is not None:
                        tea_hs = tea_hs.to(stu_hs.device)
                        if target_space.apply_method == 'mse':
                            loss_list.append(target_space.lambda_ * F.mse_loss(stu_hs, tea_hs))
                        elif target_space.apply_method == 'kl':
                            loss_list.append(target_space.lambda_ * F.kl_div((stu_hs / 2).log_softmax(dim=-1), (tea_hs / 2).softmax(dim=-1), reduction='batchmean') * 2 ** 2)
                if loss_list:
                    distill_loss = distill_loss + min(loss_list)
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.clean()
        for (_, ts) in self._teacher_target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.clean()
        return distill_loss

class Adaptive1dLayerwiseDistiller(TeacherModelBasedDistiller):
    __doc__ = "\n    This distiller will adaptively align the last dimension between student distillation target and teacher distillation target\n    by adding a trainable ``torch.nn.Linear`` between them.\n    (If the last dimensions between student and teacher have already aligned, won't add a new linear layer.)\n\n    Note that this distiller need call ``Adaptive1dLayerwiseDistiller.track_forward(...)`` first to get the shape of each distillation\n    target to initialize the linear layer before call ``Adaptive1dLayerwiseDistiller.compress(...)``.\n\n    Parameters\n    ----------\n    model\n        The student model to be distilled.\n    config_list\n        Config list to configure how to distill.\n        Common keys please refer :doc:`Compression Config Specification </compression/config_list>`.\n\n        Specific keys:\n\n        * 'lambda': By default, 1.\n          This is a scaling factor to control the loss scale, the final loss used during training is\n          ``(origin_loss_lambda * origin_loss + sum(lambda_i * distill_loss_i))``.\n          Here ``i`` represents the ``i-th`` distillation target.\n          The higher the value of lambda, the greater the contribution of the corresponding distillation target to the loss.\n        * 'link': By default, 'auto'.\n          'auto' or a teacher module name or a list of teacher module names,\n          the module name(s) of teacher module(s) will align with student module(s) configured in this config.\n          If 'auto' is set, will use student module name as the link,\n          usually requires the teacher model and the student model to be isomorphic.\n        * 'apply_method': By default, 'mse'.\n          'mse' and 'kl' are supported right now. 'mse' means the MSE loss, usually used to distill hidden states.\n          'kl' means the KL loss, usually used to distill logits.\n\n    evaluator\n        {evaluator_docstring}\n    teacher_model\n        The distillation teacher model.\n    teacher_predict\n        A callable function with two inputs (batch, model).\n\n        Example::\n\n            def teacher_predict(batch, teacher_model):\n                return teacher_model(**batch)\n\n    origin_loss_lambda\n        A scaling factor to control the original loss scale.\n    ".format(evaluator_docstring=_EVALUATOR_DOCSTRING)

    def track_forward(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().track_forward(*args, **kwargs)
        with torch.no_grad():
            model_device = next(iter(self.teacher_model.parameters())).device
            args = tree_map(lambda x: x.to(model_device) if isinstance(x, torch.Tensor) else x, args)
            kwargs = tree_map(lambda x: x.to(model_device) if isinstance(x, torch.Tensor) else x, kwargs)
            self.teacher_model(*args, **kwargs)

    def _register_trans_linear(self):
        if False:
            while True:
                i = 10
        self.trans_linears = defaultdict(dict)
        for (module_name, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                assert isinstance(target_space.link, str) or len(target_space.link) == 1, f'only support set one link for target in {self.__class__.__name__}'
                stu_hs = target_space.hidden_state
                link = target_space.link if isinstance(target_space.link, str) else target_space.link[0]
                tea_hs = self._teacher_target_spaces[link][target_name].hidden_state
                assert stu_hs is not None and tea_hs is not None, 'Please run AdaptiveShapeLayerwiseDistiller.track_forward(...) first before compress.'
                if stu_hs.shape[-1] == tea_hs.shape[-1]:
                    self.trans_linears[module_name][target_name] = None
                else:
                    self.trans_linears[module_name][target_name] = torch.nn.Linear(stu_hs.shape[-1], tea_hs.shape[-1]).to(stu_hs.device)

    def _register_linears_optimization(self, evaluator: Evaluator):
        if False:
            print('Hello World!')
        linear_params = {}
        for (module_name, linears) in self.trans_linears.items():
            for (_, linear) in linears.items():
                if linear is not None:
                    linear_params[module_name] = list(linear.parameters())
        if not linear_params:
            return
        evaluator.patch_optim_param_group(linear_params)

    def compute_distill_loss(self):
        if False:
            print('Hello World!')
        distill_loss = 0
        for (module_name, ts) in self._target_spaces.items():
            for (target_name, target_space) in ts.items():
                stu_hs = target_space.hidden_state
                link = target_space.link if isinstance(target_space.link, str) else target_space.link[0]
                tea_hs = self._teacher_target_spaces[link][target_name].hidden_state
                if stu_hs is not None and tea_hs is not None:
                    if self.trans_linears[module_name][target_name] is not None:
                        self.trans_linears[module_name][target_name].to(stu_hs.device)
                        stu_hs = self.trans_linears[module_name][target_name](stu_hs)
                    tea_hs = tea_hs.to(stu_hs.device)
                    if target_space.apply_method == 'mse':
                        distill_loss += target_space.lambda_ * F.mse_loss(stu_hs, tea_hs)
                    elif target_space.apply_method == 'kl':
                        distill_loss += target_space.lambda_ * F.kl_div((stu_hs / 2).log_softmax(dim=-1), (tea_hs / 2).softmax(dim=-1), reduction='batchmean') * 2 ** 2
        for (_, ts) in self._target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.clean()
        for (_, ts) in self._teacher_target_spaces.items():
            for (_, target_space) in ts.items():
                target_space.clean()
        return distill_loss

    def _fuse_preprocess(self, evaluator: Evaluator):
        if False:
            for i in range(10):
                print('nop')
        self._register_trans_linear()
        self._register_linears_optimization(evaluator)
        self._register_loss_patch(evaluator)