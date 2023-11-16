"""Pruning methods."""
import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Tuple
import torch

class BasePruningMethod(ABC):
    """Abstract base class for creation of new pruning techniques.

    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """
    _tensor_name: str

    def __call__(self, module, inputs):
        if False:
            i = 10
            return i + 15
        "Multiply the mask into original tensor and store the result.\n\n        Multiplies the mask (stored in ``module[name + '_mask']``)\n        into the original tensor (stored in ``module[name + '_orig']``)\n        and stores the result into ``module[name]`` by using :meth:`apply_mask`.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            inputs: not used.\n        "
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t, default_mask):
        if False:
            while True:
                i = 10
        'Compute and returns a mask for the input tensor ``t``.\n\n        Starting from a base ``default_mask`` (which should be a mask of ones\n        if the tensor has not been pruned yet), generate a random mask to\n        apply on top of the ``default_mask`` according to the specific pruning\n        method recipe.\n\n        Args:\n            t (torch.Tensor): tensor representing the importance scores of the\n            parameter to prune.\n            default_mask (torch.Tensor): Base mask from previous pruning\n            iterations, that need to be respected after the new mask is\n            applied. Same dims as ``t``.\n\n        Returns:\n            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``\n        '
        pass

    def apply_mask(self, module):
        if False:
            i = 10
            return i + 15
        'Simply handles the multiplication between the parameter being pruned and the generated mask.\n\n        Fetches the mask and the original tensor from the module\n        and returns the pruned version of the tensor.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n\n        Returns:\n            pruned_tensor (torch.Tensor): pruned version of the input tensor\n        '
        assert self._tensor_name is not None, f'Module {module} has to be pruned'
        mask = getattr(module, self._tensor_name + '_mask')
        orig = getattr(module, self._tensor_name + '_orig')
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Add pruning on the fly and reparametrization of a tensor.\n\n        Adds the forward pre-hook that enables pruning on the fly and\n        the reparametrization of a tensor in terms of the original tensor\n        and the pruning mask.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            name (str): parameter name within ``module`` on which pruning\n                will act.\n            args: arguments passed on to a subclass of\n                :class:`BasePruningMethod`\n            importance_scores (torch.Tensor): tensor of importance scores (of\n                same shape as module parameter) used to compute mask for pruning.\n                The values in this tensor indicate the importance of the\n                corresponding elements in the parameter being pruned.\n                If unspecified or None, the parameter will be used in its place.\n            kwargs: keyword arguments passed on to a subclass of a\n                :class:`BasePruningMethod`\n        '

        def _get_composite_method(cls, module, name, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            old_method = None
            found = 0
            hooks_to_remove = []
            for (k, hook) in module._forward_pre_hooks.items():
                if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                    old_method = hook
                    hooks_to_remove.append(k)
                    found += 1
            assert found <= 1, f'Avoid adding multiple pruning hooks to the                same tensor {name} of module {module}. Use a PruningContainer.'
            for k in hooks_to_remove:
                del module._forward_pre_hooks[k]
            method = cls(*args, **kwargs)
            method._tensor_name = name
            if old_method is not None:
                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method
                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)
                    container.add_pruning_method(method)
                    method = container
            return method
        method = _get_composite_method(cls, module, name, *args, **kwargs)
        orig = getattr(module, name)
        if importance_scores is not None:
            assert importance_scores.shape == orig.shape, f'importance_scores should have the same shape as parameter                 {name} of {module}'
        else:
            importance_scores = orig
        if not isinstance(method, PruningContainer):
            module.register_parameter(name + '_orig', orig)
            del module._parameters[name]
            default_mask = torch.ones_like(orig)
        else:
            default_mask = getattr(module, name + '_mask').detach().clone(memory_format=torch.contiguous_format)
        try:
            mask = method.compute_mask(importance_scores, default_mask=default_mask)
            module.register_buffer(name + '_mask', mask)
            setattr(module, name, method.apply_mask(module))
            module.register_forward_pre_hook(method)
        except Exception as e:
            if not isinstance(method, PruningContainer):
                orig = getattr(module, name + '_orig')
                module.register_parameter(name, orig)
                del module._parameters[name + '_orig']
            raise e
        return method

    def prune(self, t, default_mask=None, importance_scores=None):
        if False:
            while True:
                i = 10
        'Compute and returns a pruned version of input tensor ``t``.\n\n        According to the pruning rule specified in :meth:`compute_mask`.\n\n        Args:\n            t (torch.Tensor): tensor to prune (of same dimensions as\n                ``default_mask``).\n            importance_scores (torch.Tensor): tensor of importance scores (of\n                same shape as ``t``) used to compute mask for pruning ``t``.\n                The values in this tensor indicate the importance of the\n                corresponding elements in the ``t`` that is being pruned.\n                If unspecified or None, the tensor ``t`` will be used in its place.\n            default_mask (torch.Tensor, optional): mask from previous pruning\n                iteration, if any. To be considered when determining what\n                portion of the tensor that pruning should act on. If None,\n                default to a mask of ones.\n\n        Returns:\n            pruned version of tensor ``t``.\n        '
        if importance_scores is not None:
            assert importance_scores.shape == t.shape, 'importance_scores should have the same shape as tensor t'
        else:
            importance_scores = t
        default_mask = default_mask if default_mask is not None else torch.ones_like(t)
        return t * self.compute_mask(importance_scores, default_mask=default_mask)

    def remove(self, module):
        if False:
            print('Hello World!')
        "Remove the pruning reparameterization from a module.\n\n        The pruned parameter named ``name`` remains permanently pruned,\n        and the parameter named ``name+'_orig'`` is removed from the parameter list.\n        Similarly, the buffer named ``name+'_mask'`` is removed from the buffers.\n\n        Note:\n            Pruning itself is NOT undone or reversed!\n        "
        assert self._tensor_name is not None, f'Module {module} has to be pruned            before pruning can be removed'
        weight = self.apply_mask(module)
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + '_orig']
        orig.data = weight.data
        del module._parameters[self._tensor_name + '_orig']
        del module._buffers[self._tensor_name + '_mask']
        setattr(module, self._tensor_name, orig)

class PruningContainer(BasePruningMethod):
    """Container holding a sequence of pruning methods for iterative pruning.

    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """

    def __init__(self, *args):
        if False:
            return 10
        self._pruning_methods: Tuple[BasePruningMethod, ...] = tuple()
        if not isinstance(args, Iterable):
            self._tensor_name = args._tensor_name
            self.add_pruning_method(args)
        elif len(args) == 1:
            self._tensor_name = args[0]._tensor_name
            self.add_pruning_method(args[0])
        else:
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method):
        if False:
            for i in range(10):
                print('nop')
        'Add a child pruning ``method`` to the container.\n\n        Args:\n            method (subclass of BasePruningMethod): child pruning method\n                to be added to the container.\n        '
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(f'{type(method)} is not a BasePruningMethod subclass')
        elif method is not None and self._tensor_name != method._tensor_name:
            raise ValueError(f"Can only add pruning methods acting on the parameter named '{self._tensor_name}' to PruningContainer {self}." + f" Found '{method._tensor_name}'")
        self._pruning_methods += (method,)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._pruning_methods)

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        if False:
            i = 10
            return i + 15
        return self._pruning_methods[idx]

    def compute_mask(self, t, default_mask):
        if False:
            i = 10
            return i + 15
        "Apply the latest ``method`` by computing the new partial masks and returning its combination with the ``default_mask``.\n\n        The new partial mask should be computed on the entries or channels\n        that were not zeroed out by the ``default_mask``.\n        Which portions of the tensor ``t`` the new mask will be calculated from\n        depends on the ``PRUNING_TYPE`` (handled by the type handler):\n\n        * for 'unstructured', the mask will be computed from the raveled\n          list of nonmasked entries;\n\n        * for 'structured', the mask will be computed from the nonmasked\n          channels in the tensor;\n\n        * for 'global', the mask will be computed across all entries.\n\n        Args:\n            t (torch.Tensor): tensor representing the parameter to prune\n                (of same dimensions as ``default_mask``).\n            default_mask (torch.Tensor): mask from previous pruning iteration.\n\n        Returns:\n            mask (torch.Tensor): new mask that combines the effects\n            of the ``default_mask`` and the new mask from the current\n            pruning ``method`` (of same dimensions as ``default_mask`` and\n            ``t``).\n        "

        def _combine_masks(method, t, mask):
            if False:
                print('Hello World!')
            'Combine the masks from all pruning methods and returns a new mask.\n\n            Args:\n                method (a BasePruningMethod subclass): pruning method\n                    currently being applied.\n                t (torch.Tensor): tensor representing the parameter to prune\n                    (of same dimensions as mask).\n                mask (torch.Tensor): mask from previous pruning iteration\n\n            Returns:\n                new_mask (torch.Tensor): new mask that combines the effects\n                    of the old mask and the new mask from the current\n                    pruning method (of same dimensions as mask and t).\n            '
            new_mask = mask
            new_mask = new_mask.to(dtype=t.dtype)
            if method.PRUNING_TYPE == 'unstructured':
                slc = mask == 1
            elif method.PRUNING_TYPE == 'structured':
                if not hasattr(method, 'dim'):
                    raise AttributeError('Pruning methods of PRUNING_TYPE "structured" need to have the attribute `dim` defined.')
                n_dims = t.dim()
                dim = method.dim
                if dim < 0:
                    dim = n_dims + dim
                if dim < 0:
                    raise IndexError(f'Index is out of bounds for tensor with dimensions {n_dims}')
                keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0
                slc = [slice(None)] * n_dims
                slc[dim] = keep_channel
            elif method.PRUNING_TYPE == 'global':
                n_dims = len(t.shape)
                slc = [slice(None)] * n_dims
            else:
                raise ValueError(f'Unrecognized PRUNING_TYPE {method.PRUNING_TYPE}')
            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)
            return new_mask
        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask

class Identity(BasePruningMethod):
    """Utility pruning method that does not prune any units but generates the pruning parametrization with a mask of ones."""
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        if False:
            return 10
        mask = default_mask
        return mask

    @classmethod
    def apply(cls, module, name):
        if False:
            return 10
        'Add pruning on the fly and reparametrization of a tensor.\n\n        Adds the forward pre-hook that enables pruning on the fly and\n        the reparametrization of a tensor in terms of the original tensor\n        and the pruning mask.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            name (str): parameter name within ``module`` on which pruning\n                will act.\n        '
        return super().apply(module, name)

class RandomUnstructured(BasePruningMethod):
    """Prune (currently unpruned) units in a tensor at random.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        if False:
            for i in range(10):
                print('nop')
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        if False:
            i = 10
            return i + 15
        tensor_size = t.nelement()
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        _validate_pruning_amount(nparams_toprune, tensor_size)
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if nparams_toprune != 0:
            prob = torch.rand_like(t)
            topk = torch.topk(prob.view(-1), k=nparams_toprune)
            mask.view(-1)[topk.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount):
        if False:
            for i in range(10):
                print('nop')
        'Add pruning on the fly and reparametrization of a tensor.\n\n        Adds the forward pre-hook that enables pruning on the fly and\n        the reparametrization of a tensor in terms of the original tensor\n        and the pruning mask.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            name (str): parameter name within ``module`` on which pruning\n                will act.\n            amount (int or float): quantity of parameters to prune.\n                If ``float``, should be between 0.0 and 1.0 and represent the\n                fraction of parameters to prune. If ``int``, it represents the\n                absolute number of parameters to prune.\n        '
        return super().apply(module, name, amount=amount)

class L1Unstructured(BasePruningMethod):
    """Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """
    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount):
        if False:
            while True:
                i = 10
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        if False:
            for i in range(10):
                print('nop')
        tensor_size = t.nelement()
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        _validate_pruning_amount(nparams_toprune, tensor_size)
        mask = default_mask.clone(memory_format=torch.contiguous_format)
        if nparams_toprune != 0:
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[topk.indices] = 0
        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):
        if False:
            return 10
        'Add pruning on the fly and reparametrization of a tensor.\n\n        Adds the forward pre-hook that enables pruning on the fly and\n        the reparametrization of a tensor in terms of the original tensor\n        and the pruning mask.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            name (str): parameter name within ``module`` on which pruning\n                will act.\n            amount (int or float): quantity of parameters to prune.\n                If ``float``, should be between 0.0 and 1.0 and represent the\n                fraction of parameters to prune. If ``int``, it represents the\n                absolute number of parameters to prune.\n            importance_scores (torch.Tensor): tensor of importance scores (of same\n                shape as module parameter) used to compute mask for pruning.\n                The values in this tensor indicate the importance of the corresponding\n                elements in the parameter being pruned.\n                If unspecified or None, the module parameter will be used in its place.\n        '
        return super().apply(module, name, amount=amount, importance_scores=importance_scores)

class RandomStructured(BasePruningMethod):
    """Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """
    PRUNING_TYPE = 'structured'

    def __init__(self, amount, dim=-1):
        if False:
            for i in range(10):
                print('nop')
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim

    def compute_mask(self, t, default_mask):
        if False:
            print('Hello World!')
        'Compute and returns a mask for the input tensor ``t``.\n\n        Starting from a base ``default_mask`` (which should be a mask of ones\n        if the tensor has not been pruned yet), generate a random mask to\n        apply on top of the ``default_mask`` by randomly zeroing out channels\n        along the specified dim of the tensor.\n\n        Args:\n            t (torch.Tensor): tensor representing the parameter to prune\n            default_mask (torch.Tensor): Base mask from previous pruning\n                iterations, that need to be respected after the new mask is\n                applied. Same dims as ``t``.\n\n        Returns:\n            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``\n\n        Raises:\n            IndexError: if ``self.dim >= len(t.shape)``\n        '
        _validate_structured_pruning(t)
        _validate_pruning_dim(t, self.dim)
        tensor_size = t.shape[self.dim]
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        _validate_pruning_amount(nparams_toprune, tensor_size)

        def make_mask(t, dim, nchannels, nchannels_toprune):
            if False:
                print('Hello World!')
            prob = torch.rand(nchannels)
            threshold = torch.kthvalue(prob, k=nchannels_toprune).values
            channel_mask = prob > threshold
            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = channel_mask
            mask[slc] = 1
            return mask
        if nparams_toprune == 0:
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, tensor_size, nparams_toprune)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, amount, dim=-1):
        if False:
            print('Hello World!')
        'Add pruning on the fly and reparametrization of a tensor.\n\n        Adds the forward pre-hook that enables pruning on the fly and\n        the reparametrization of a tensor in terms of the original tensor\n        and the pruning mask.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            name (str): parameter name within ``module`` on which pruning\n                will act.\n            amount (int or float): quantity of parameters to prune.\n                If ``float``, should be between 0.0 and 1.0 and represent the\n                fraction of parameters to prune. If ``int``, it represents the\n                absolute number of parameters to prune.\n            dim (int, optional): index of the dim along which we define\n                channels to prune. Default: -1.\n        '
        return super().apply(module, name, amount=amount, dim=dim)

class LnStructured(BasePruningMethod):
    """Prune entire (currently unpruned) channels in a tensor based on their L\\ ``n``-norm.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """
    PRUNING_TYPE = 'structured'

    def __init__(self, amount, n, dim=-1):
        if False:
            print('Hello World!')
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.dim = dim

    def compute_mask(self, t, default_mask):
        if False:
            i = 10
            return i + 15
        'Compute and returns a mask for the input tensor ``t``.\n\n        Starting from a base ``default_mask`` (which should be a mask of ones\n        if the tensor has not been pruned yet), generate a mask to apply on\n        top of the ``default_mask`` by zeroing out the channels along the\n        specified dim with the lowest L\\ ``n``-norm.\n\n        Args:\n            t (torch.Tensor): tensor representing the parameter to prune\n            default_mask (torch.Tensor): Base mask from previous pruning\n                iterations, that need to be respected after the new mask is\n                applied.  Same dims as ``t``.\n\n        Returns:\n            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``\n\n        Raises:\n            IndexError: if ``self.dim >= len(t.shape)``\n        '
        _validate_structured_pruning(t)
        _validate_pruning_dim(t, self.dim)
        tensor_size = t.shape[self.dim]
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        _validate_pruning_amount(nparams_toprune, tensor_size)
        norm = _compute_norm(t, self.n, self.dim)
        topk = torch.topk(norm, k=nparams_tokeep, largest=True)

        def make_mask(t, dim, indices):
            if False:
                for i in range(10):
                    print('nop')
            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = indices
            mask[slc] = 1
            return mask
        if nparams_toprune == 0:
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, amount, n, dim, importance_scores=None):
        if False:
            i = 10
            return i + 15
        "Add pruning on the fly and reparametrization of a tensor.\n\n        Adds the forward pre-hook that enables pruning on the fly and\n        the reparametrization of a tensor in terms of the original tensor\n        and the pruning mask.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            name (str): parameter name within ``module`` on which pruning\n                will act.\n            amount (int or float): quantity of parameters to prune.\n                If ``float``, should be between 0.0 and 1.0 and represent the\n                fraction of parameters to prune. If ``int``, it represents the\n                absolute number of parameters to prune.\n            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid\n                entries for argument ``p`` in :func:`torch.norm`.\n            dim (int): index of the dim along which we define channels to\n                prune.\n            importance_scores (torch.Tensor): tensor of importance scores (of same\n                shape as module parameter) used to compute mask for pruning.\n                The values in this tensor indicate the importance of the corresponding\n                elements in the parameter being pruned.\n                If unspecified or None, the module parameter will be used in its place.\n        "
        return super().apply(module, name, amount=amount, n=n, dim=dim, importance_scores=importance_scores)

class CustomFromMask(BasePruningMethod):
    PRUNING_TYPE = 'global'

    def __init__(self, mask):
        if False:
            while True:
                i = 10
        self.mask = mask

    def compute_mask(self, t, default_mask):
        if False:
            i = 10
            return i + 15
        assert default_mask.shape == self.mask.shape
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, mask):
        if False:
            return 10
        'Add pruning on the fly and reparametrization of a tensor.\n\n        Adds the forward pre-hook that enables pruning on the fly and\n        the reparametrization of a tensor in terms of the original tensor\n        and the pruning mask.\n\n        Args:\n            module (nn.Module): module containing the tensor to prune\n            name (str): parameter name within ``module`` on which pruning\n                will act.\n        '
        return super().apply(module, name, mask=mask)

def identity(module, name):
    if False:
        i = 10
        return i + 15
    "Apply pruning reparametrization without pruning any units.\n\n    Applies pruning reparametrization to the tensor corresponding to the\n    parameter called ``name`` in ``module`` without actually pruning any\n    units. Modifies module in place (and also return the modified module)\n    by:\n\n    1) adding a named buffer called ``name+'_mask'`` corresponding to the\n       binary mask applied to the parameter ``name`` by the pruning method.\n    2) replacing the parameter ``name`` by its pruned version, while the\n       original (unpruned) parameter is stored in a new parameter named\n       ``name+'_orig'``.\n\n    Note:\n        The mask is a tensor of ones.\n\n    Args:\n        module (nn.Module): module containing the tensor to prune.\n        name (str): parameter name within ``module`` on which pruning\n                will act.\n\n    Returns:\n        module (nn.Module): modified (i.e. pruned) version of the input module\n\n    Examples:\n        >>> # xdoctest: +SKIP\n        >>> m = prune.identity(nn.Linear(2, 3), 'bias')\n        >>> print(m.bias_mask)\n        tensor([1., 1., 1.])\n    "
    Identity.apply(module, name)
    return module

def random_unstructured(module, name, amount):
    if False:
        while True:
            i = 10
    "Prune tensor by removing random (currently unpruned) units.\n\n    Prunes tensor corresponding to parameter called ``name`` in ``module``\n    by removing the specified ``amount`` of (currently unpruned) units\n    selected at random.\n    Modifies module in place (and also return the modified module) by:\n\n    1) adding a named buffer called ``name+'_mask'`` corresponding to the\n       binary mask applied to the parameter ``name`` by the pruning method.\n    2) replacing the parameter ``name`` by its pruned version, while the\n       original (unpruned) parameter is stored in a new parameter named\n       ``name+'_orig'``.\n\n    Args:\n        module (nn.Module): module containing the tensor to prune\n        name (str): parameter name within ``module`` on which pruning\n                will act.\n        amount (int or float): quantity of parameters to prune.\n            If ``float``, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If ``int``, it represents the\n            absolute number of parameters to prune.\n\n    Returns:\n        module (nn.Module): modified (i.e. pruned) version of the input module\n\n    Examples:\n        >>> # xdoctest: +SKIP\n        >>> m = prune.random_unstructured(nn.Linear(2, 3), 'weight', amount=1)\n        >>> torch.sum(m.weight_mask == 0)\n        tensor(1)\n\n    "
    RandomUnstructured.apply(module, name, amount)
    return module

def l1_unstructured(module, name, amount, importance_scores=None):
    if False:
        return 10
    "Prune tensor by removing units with the lowest L1-norm.\n\n    Prunes tensor corresponding to parameter called ``name`` in ``module``\n    by removing the specified `amount` of (currently unpruned) units with the\n    lowest L1-norm.\n    Modifies module in place (and also return the modified module)\n    by:\n\n    1) adding a named buffer called ``name+'_mask'`` corresponding to the\n       binary mask applied to the parameter ``name`` by the pruning method.\n    2) replacing the parameter ``name`` by its pruned version, while the\n       original (unpruned) parameter is stored in a new parameter named\n       ``name+'_orig'``.\n\n    Args:\n        module (nn.Module): module containing the tensor to prune\n        name (str): parameter name within ``module`` on which pruning\n                will act.\n        amount (int or float): quantity of parameters to prune.\n            If ``float``, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If ``int``, it represents the\n            absolute number of parameters to prune.\n        importance_scores (torch.Tensor): tensor of importance scores (of same\n            shape as module parameter) used to compute mask for pruning.\n            The values in this tensor indicate the importance of the corresponding\n            elements in the parameter being pruned.\n            If unspecified or None, the module parameter will be used in its place.\n\n    Returns:\n        module (nn.Module): modified (i.e. pruned) version of the input module\n\n    Examples:\n        >>> # xdoctest: +SKIP\n        >>> m = prune.l1_unstructured(nn.Linear(2, 3), 'weight', amount=0.2)\n        >>> m.state_dict().keys()\n        odict_keys(['bias', 'weight_orig', 'weight_mask'])\n    "
    L1Unstructured.apply(module, name, amount=amount, importance_scores=importance_scores)
    return module

def random_structured(module, name, amount, dim):
    if False:
        for i in range(10):
            print('nop')
    "Prune tensor by removing random channels along the specified dimension.\n\n    Prunes tensor corresponding to parameter called ``name`` in ``module``\n    by removing the specified ``amount`` of (currently unpruned) channels\n    along the specified ``dim`` selected at random.\n    Modifies module in place (and also return the modified module)\n    by:\n\n    1) adding a named buffer called ``name+'_mask'`` corresponding to the\n       binary mask applied to the parameter ``name`` by the pruning method.\n    2) replacing the parameter ``name`` by its pruned version, while the\n       original (unpruned) parameter is stored in a new parameter named\n       ``name+'_orig'``.\n\n    Args:\n        module (nn.Module): module containing the tensor to prune\n        name (str): parameter name within ``module`` on which pruning\n                will act.\n        amount (int or float): quantity of parameters to prune.\n            If ``float``, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If ``int``, it represents the\n            absolute number of parameters to prune.\n        dim (int): index of the dim along which we define channels to prune.\n\n    Returns:\n        module (nn.Module): modified (i.e. pruned) version of the input module\n\n    Examples:\n        >>> # xdoctest: +SKIP\n        >>> m = prune.random_structured(\n        ...     nn.Linear(5, 3), 'weight', amount=3, dim=1\n        ... )\n        >>> columns_pruned = int(sum(torch.sum(m.weight, dim=0) == 0))\n        >>> print(columns_pruned)\n        3\n    "
    RandomStructured.apply(module, name, amount, dim)
    return module

def ln_structured(module, name, amount, n, dim, importance_scores=None):
    if False:
        print('Hello World!')
    "Prune tensor by removing channels with the lowest L\\ ``n``-norm along the specified dimension.\n\n    Prunes tensor corresponding to parameter called ``name`` in ``module``\n    by removing the specified ``amount`` of (currently unpruned) channels\n    along the specified ``dim`` with the lowest L\\ ``n``-norm.\n    Modifies module in place (and also return the modified module)\n    by:\n\n    1) adding a named buffer called ``name+'_mask'`` corresponding to the\n       binary mask applied to the parameter ``name`` by the pruning method.\n    2) replacing the parameter ``name`` by its pruned version, while the\n       original (unpruned) parameter is stored in a new parameter named\n       ``name+'_orig'``.\n\n    Args:\n        module (nn.Module): module containing the tensor to prune\n        name (str): parameter name within ``module`` on which pruning\n                will act.\n        amount (int or float): quantity of parameters to prune.\n            If ``float``, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If ``int``, it represents the\n            absolute number of parameters to prune.\n        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid\n            entries for argument ``p`` in :func:`torch.norm`.\n        dim (int): index of the dim along which we define channels to prune.\n        importance_scores (torch.Tensor): tensor of importance scores (of same\n            shape as module parameter) used to compute mask for pruning.\n            The values in this tensor indicate the importance of the corresponding\n            elements in the parameter being pruned.\n            If unspecified or None, the module parameter will be used in its place.\n\n    Returns:\n        module (nn.Module): modified (i.e. pruned) version of the input module\n\n    Examples:\n        >>> from torch.nn.utils import prune\n        >>> m = prune.ln_structured(\n        ...     nn.Conv2d(5, 3, 2), 'weight', amount=0.3, dim=1, n=float('-inf')\n        ... )\n    "
    LnStructured.apply(module, name, amount, n, dim, importance_scores=importance_scores)
    return module

def global_unstructured(parameters, pruning_method, importance_scores=None, **kwargs):
    if False:
        print('Hello World!')
    "\n    Globally prunes tensors corresponding to all parameters in ``parameters`` by applying the specified ``pruning_method``.\n\n    Modifies modules in place by:\n\n    1) adding a named buffer called ``name+'_mask'`` corresponding to the\n       binary mask applied to the parameter ``name`` by the pruning method.\n    2) replacing the parameter ``name`` by its pruned version, while the\n       original (unpruned) parameter is stored in a new parameter named\n       ``name+'_orig'``.\n\n    Args:\n        parameters (Iterable of (module, name) tuples): parameters of\n            the model to prune in a global fashion, i.e. by aggregating all\n            weights prior to deciding which ones to prune. module must be of\n            type :class:`nn.Module`, and name must be a string.\n        pruning_method (function): a valid pruning function from this module,\n            or a custom one implemented by the user that satisfies the\n            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.\n        importance_scores (dict): a dictionary mapping (module, name) tuples to\n            the corresponding parameter's importance scores tensor. The tensor\n            should be the same shape as the parameter, and is used for computing\n            mask for pruning.\n            If unspecified or None, the parameter will be used in place of its\n            importance scores.\n        kwargs: other keyword arguments such as:\n            amount (int or float): quantity of parameters to prune across the\n            specified parameters.\n            If ``float``, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If ``int``, it represents the\n            absolute number of parameters to prune.\n\n    Raises:\n        TypeError: if ``PRUNING_TYPE != 'unstructured'``\n\n    Note:\n        Since global structured pruning doesn't make much sense unless the\n        norm is normalized by the size of the parameter, we now limit the\n        scope of global pruning to unstructured methods.\n\n    Examples:\n        >>> from torch.nn.utils import prune\n        >>> from collections import OrderedDict\n        >>> net = nn.Sequential(OrderedDict([\n        ...     ('first', nn.Linear(10, 4)),\n        ...     ('second', nn.Linear(4, 1)),\n        ... ]))\n        >>> parameters_to_prune = (\n        ...     (net.first, 'weight'),\n        ...     (net.second, 'weight'),\n        ... )\n        >>> prune.global_unstructured(\n        ...     parameters_to_prune,\n        ...     pruning_method=prune.L1Unstructured,\n        ...     amount=10,\n        ... )\n        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))\n        tensor(10)\n\n    "
    if not isinstance(parameters, Iterable):
        raise TypeError('global_unstructured(): parameters is not an Iterable')
    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError('global_unstructured(): importance_scores must be of type dict')
    relevant_importance_scores = torch.nn.utils.parameters_to_vector([importance_scores.get((module, name), getattr(module, name)) for (module, name) in parameters])
    default_mask = torch.nn.utils.parameters_to_vector([getattr(module, name + '_mask', torch.ones_like(getattr(module, name))) for (module, name) in parameters])
    container = PruningContainer()
    container._tensor_name = 'temp'
    method = pruning_method(**kwargs)
    method._tensor_name = 'temp'
    if method.PRUNING_TYPE != 'unstructured':
        raise TypeError(f'Only "unstructured" PRUNING_TYPE supported for the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}')
    container.add_pruning_method(method)
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)
    pointer = 0
    for (module, name) in parameters:
        param = getattr(module, name)
        num_param = param.numel()
        param_mask = final_mask[pointer:pointer + num_param].view_as(param)
        custom_from_mask(module, name, mask=param_mask)
        pointer += num_param

def custom_from_mask(module, name, mask):
    if False:
        print('Hello World!')
    "Prune tensor corresponding to parameter called ``name`` in ``module`` by applying the pre-computed mask in ``mask``.\n\n    Modifies module in place (and also return the modified module) by:\n\n    1) adding a named buffer called ``name+'_mask'`` corresponding to the\n       binary mask applied to the parameter ``name`` by the pruning method.\n    2) replacing the parameter ``name`` by its pruned version, while the\n       original (unpruned) parameter is stored in a new parameter named\n       ``name+'_orig'``.\n\n    Args:\n        module (nn.Module): module containing the tensor to prune\n        name (str): parameter name within ``module`` on which pruning\n            will act.\n        mask (Tensor): binary mask to be applied to the parameter.\n\n    Returns:\n        module (nn.Module): modified (i.e. pruned) version of the input module\n\n    Examples:\n        >>> from torch.nn.utils import prune\n        >>> m = prune.custom_from_mask(\n        ...     nn.Linear(5, 3), name='bias', mask=torch.tensor([0, 1, 0])\n        ... )\n        >>> print(m.bias_mask)\n        tensor([0., 1., 0.])\n\n    "
    CustomFromMask.apply(module, name, mask)
    return module

def remove(module, name):
    if False:
        while True:
            i = 10
    "Remove the pruning reparameterization from a module and the pruning method from the forward hook.\n\n    The pruned parameter named ``name`` remains permanently pruned, and the parameter\n    named ``name+'_orig'`` is removed from the parameter list. Similarly,\n    the buffer named ``name+'_mask'`` is removed from the buffers.\n\n    Note:\n        Pruning itself is NOT undone or reversed!\n\n    Args:\n        module (nn.Module): module containing the tensor to prune\n        name (str): parameter name within ``module`` on which pruning\n            will act.\n\n    Examples:\n        >>> m = random_unstructured(nn.Linear(5, 7), name='weight', amount=0.2)\n        >>> m = remove(m, name='weight')\n    "
    for (k, hook) in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError(f"Parameter '{name}' of module {module} has to be pruned before pruning can be removed")

def is_pruned(module):
    if False:
        i = 10
        return i + 15
    "Check if a module is pruned by looking for pruning pre-hooks.\n\n    Check whether ``module`` is pruned by looking for\n    ``forward_pre_hooks`` in its modules that inherit from the\n    :class:`BasePruningMethod`.\n\n    Args:\n        module (nn.Module): object that is either pruned or unpruned\n\n    Returns:\n        binary answer to whether ``module`` is pruned.\n\n    Examples:\n        >>> from torch.nn.utils import prune\n        >>> m = nn.Linear(5, 7)\n        >>> print(prune.is_pruned(m))\n        False\n        >>> prune.random_unstructured(m, name='weight', amount=0.2)\n        >>> print(prune.is_pruned(m))\n        True\n    "
    for (_, submodule) in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            if isinstance(hook, BasePruningMethod):
                return True
    return False

def _validate_pruning_amount_init(amount):
    if False:
        for i in range(10):
            print('nop')
    "Validate helper to check the range of amount at init.\n\n    Args:\n        amount (int or float): quantity of parameters to prune.\n            If float, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If int, it represents the\n            absolute number of parameters to prune.\n\n    Raises:\n        ValueError: if amount is a float not in [0, 1], or if it's a negative\n            integer.\n        TypeError: if amount is neither a float nor an integer.\n\n    Note:\n        This does not take into account the number of parameters in the\n        tensor to be pruned, which is known only at prune.\n    "
    if not isinstance(amount, numbers.Real):
        raise TypeError(f'Invalid type for amount: {amount}. Must be int or float.')
    if isinstance(amount, numbers.Integral) and amount < 0 or (not isinstance(amount, numbers.Integral) and (float(amount) > 1.0 or float(amount) < 0.0)):
        raise ValueError(f'amount={amount} should either be a float in the range [0, 1] or a non-negative integer')

def _validate_pruning_amount(amount, tensor_size):
    if False:
        while True:
            i = 10
    'Validate that the pruning amount is meaningful wrt to the size of the data.\n\n    Validation helper to check that the amount of parameters to prune\n    is meaningful wrt to the size of the data (`tensor_size`).\n\n    Args:\n        amount (int or float): quantity of parameters to prune.\n            If float, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If int, it represents the\n            absolute number of parameters to prune.\n        tensor_size (int): absolute number of parameters in the tensor\n            to prune.\n    '
    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(f'amount={amount} should be smaller than the number of parameters to prune={tensor_size}')

def _validate_structured_pruning(t):
    if False:
        print('Hello World!')
    'Validate that the tensor to be pruned is at least 2-Dimensional.\n\n    Validation helper to check that the tensor to be pruned is multi-\n    dimensional, such that the concept of "channels" is well-defined.\n\n    Args:\n        t (torch.Tensor): tensor representing the parameter to prune\n\n    Raises:\n        ValueError: if the tensor `t` is not at least 2D.\n    '
    shape = t.shape
    if len(shape) <= 1:
        raise ValueError(f'Structured pruning can only be applied to multidimensional tensors. Found tensor of shape {shape} with {len(shape)} dims')

def _compute_nparams_toprune(amount, tensor_size):
    if False:
        i = 10
        return i + 15
    'Convert the pruning amount from a percentage to absolute value.\n\n    Since amount can be expressed either in absolute value or as a\n    percentage of the number of units/channels in a tensor, this utility\n    function converts the percentage to absolute value to standardize\n    the handling of pruning.\n\n    Args:\n        amount (int or float): quantity of parameters to prune.\n            If float, should be between 0.0 and 1.0 and represent the\n            fraction of parameters to prune. If int, it represents the\n            absolute number of parameters to prune.\n        tensor_size (int): absolute number of parameters in the tensor\n            to prune.\n\n    Returns:\n        int: the number of units to prune in the tensor\n    '
    if isinstance(amount, numbers.Integral):
        return amount
    else:
        return round(amount * tensor_size)

def _validate_pruning_dim(t, dim):
    if False:
        while True:
            i = 10
    'Validate that the pruning dimension is within the bounds of the tensor dimension.\n\n    Args:\n        t (torch.Tensor): tensor representing the parameter to prune\n        dim (int): index of the dim along which we define channels to prune\n    '
    if dim >= t.dim():
        raise IndexError(f'Invalid index {dim} for tensor of size {t.shape}')

def _compute_norm(t, n, dim):
    if False:
        while True:
            i = 10
    "Compute the L_n-norm of a tensor along all dimensions except for the specified dimension.\n\n    The L_n-norm will be computed across all entries in tensor `t` along all dimension\n    except for the one identified by dim.\n    Example: if `t` is of shape, say, 3x2x4 and dim=2 (the last dim),\n    then norm will have Size [4], and each entry will represent the\n    `L_n`-norm computed using the 3x2=6 entries for each of the 4 channels.\n\n    Args:\n        t (torch.Tensor): tensor representing the parameter to prune\n        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid\n            entries for argument p in torch.norm\n        dim (int): dim identifying the channels to prune\n\n    Returns:\n        norm (torch.Tensor): L_n norm computed across all dimensions except\n            for `dim`. By construction, `norm.shape = t.shape[-1]`.\n    "
    dims = list(range(t.dim()))
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)
    norm = torch.norm(t, p=n, dim=dims)
    return norm