import logging
from typing import List, Optional, Tuple, Union, Sequence
import numpy as np
import torch
import torch.autograd as autograd
from allennlp.common import Lazy
from allennlp.common.tqdm import Tqdm
from allennlp.data import DatasetReader, DatasetReaderInput, Instance
from allennlp.data.data_loaders import DataLoader, SimpleDataLoader
from allennlp.interpret.influence_interpreters.influence_interpreter import InfluenceInterpreter
from allennlp.models.model import Model
logger = logging.getLogger(__name__)

@InfluenceInterpreter.register('simple-influence')
class SimpleInfluence(InfluenceInterpreter):
    """
    Registered as an `InfluenceInterpreter` with name "simple-influence".

    This goes through every example in the train set to calculate the influence score. It uses
    [LiSSA (Linear time Stochastic Second-Order Algorithm)](https://api.semanticscholar.org/CorpusID:10569090)
    to approximate the inverse of the Hessian used for the influence score calculation.

    # Parameters

    lissa_batch_size : `int`, optional (default = `8`)
        The batch size to use for LiSSA.
        According to [Koh, P.W., & Liang, P. (2017)](https://api.semanticscholar.org/CorpusID:13193974),
        it is better to use batched samples for approximation for better stability.

    damping : `float`, optional (default = `3e-3`)
        This is a hyperparameter for LiSSA.
        A damping termed added in case the approximated Hessian (during LiSSA) has
        negative eigenvalues.

    num_samples : `int`, optional (default = `1`)
        This is a hyperparameter for LiSSA that we
        determine how many rounds of the recursion process we would like to run for approxmation.

    recursion_depth : `Union[float, int]`, optional (default = `0.25`)
        This is a hyperparameter for LiSSA that
        determines the recursion depth we would like to go through.
        If a `float`, it means X% of the training examples.
        If an `int`, it means recurse for X times.

    scale : `float`, optional, (default = `1e4`)
        This is a hyperparameter for LiSSA to tune such that the Taylor expansion converges.
        It is applied to scale down the loss during LiSSA to ensure that `H <= I`,
        where `H` is the Hessian and `I` is the identity matrix.

        See footnote 2 of [Koh, P.W., & Liang, P. (2017)](https://api.semanticscholar.org/CorpusID:13193974).

    !!! Note
        We choose the same default values for the LiSSA hyperparameters as
        [Han, Xiaochuang et al. (2020)](https://api.semanticscholar.org/CorpusID:218628619).
    """

    def __init__(self, model: Model, train_data_path: DatasetReaderInput, train_dataset_reader: DatasetReader, *, test_dataset_reader: Optional[DatasetReader]=None, train_data_loader: Lazy[DataLoader]=Lazy(SimpleDataLoader.from_dataset_reader), test_data_loader: Lazy[DataLoader]=Lazy(SimpleDataLoader.from_dataset_reader), params_to_freeze: List[str]=None, cuda_device: int=-1, lissa_batch_size: int=8, damping: float=0.003, num_samples: int=1, recursion_depth: Union[float, int]=0.25, scale: float=10000.0) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(model=model, train_data_path=train_data_path, train_dataset_reader=train_dataset_reader, test_dataset_reader=test_dataset_reader, train_data_loader=train_data_loader, test_data_loader=test_data_loader, params_to_freeze=params_to_freeze, cuda_device=cuda_device)
        self._lissa_dataloader = SimpleDataLoader(list(self._train_loader.iter_instances()), lissa_batch_size, shuffle=True, vocab=self.vocab)
        self._lissa_dataloader.set_target_device(self.device)
        if isinstance(recursion_depth, float) and recursion_depth > 0.0:
            self._lissa_dataloader.batches_per_epoch = int(len(self._lissa_dataloader) * recursion_depth)
        elif isinstance(recursion_depth, int) and recursion_depth > 0:
            self._lissa_dataloader.batches_per_epoch = recursion_depth
        else:
            raise ValueError("'recursion_depth' should be a positive int or float")
        self._damping = damping
        self._num_samples = num_samples
        self._recursion_depth = recursion_depth
        self._scale = scale

    def _calculate_influence_scores(self, test_instance: Instance, test_loss: float, test_grads: Sequence[torch.Tensor]) -> List[float]:
        if False:
            while True:
                i = 10
        inv_hvp = get_inverse_hvp_lissa(test_grads, self.model, self.used_params, self._lissa_dataloader, self._damping, self._num_samples, self._scale)
        return [torch.dot(inv_hvp, _flatten_tensors(x.grads)).item() for x in Tqdm.tqdm(self.train_instances, desc='scoring train instances')]

def get_inverse_hvp_lissa(vs: Sequence[torch.Tensor], model: Model, used_params: Sequence[torch.Tensor], lissa_data_loader: DataLoader, damping: float, num_samples: int, scale: float) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    This function approximates the product of the inverse of the Hessian and\n    the vectors `vs` using LiSSA.\n\n    Adapted from [github.com/kohpangwei/influence-release]\n    (https://github.com/kohpangwei/influence-release/blob/0f656964867da6ddcca16c14b3e4f0eef38a7472/influence/genericNeuralNet.py#L475),\n    the repo for [Koh, P.W., & Liang, P. (2017)](https://api.semanticscholar.org/CorpusID:13193974),\n    and [github.com/xhan77/influence-function-analysis]\n    (https://github.com/xhan77/influence-function-analysis/blob/78d5a967aba885f690d34e88d68da8678aee41f1/bert_util.py#L336),\n    the repo for [Han, Xiaochuang et al. (2020)](https://api.semanticscholar.org/CorpusID:218628619).\n    '
    inverse_hvps = [torch.tensor(0) for _ in vs]
    for _ in Tqdm.tqdm(range(num_samples), desc='LiSSA samples', total=num_samples):
        cur_estimates = vs
        recursion_iter = Tqdm.tqdm(lissa_data_loader, desc='LiSSA depth', total=len(lissa_data_loader))
        for (j, training_batch) in enumerate(recursion_iter):
            model.zero_grad()
            train_output_dict = model(**training_batch)
            hvps = get_hvp(train_output_dict['loss'], used_params, cur_estimates)
            cur_estimates = [v + (1 - damping) * cur_estimate - hvp / scale for (v, cur_estimate, hvp) in zip(vs, cur_estimates, hvps)]
            if j % 50 == 0 or j == len(lissa_data_loader) - 1:
                norm = np.linalg.norm(_flatten_tensors(cur_estimates).cpu().numpy())
                recursion_iter.set_description(desc=f'calculating inverse HVP, norm = {norm:.5f}')
        inverse_hvps = [inverse_hvp + cur_estimate / scale for (inverse_hvp, cur_estimate) in zip(inverse_hvps, cur_estimates)]
    return_ihvp = _flatten_tensors(inverse_hvps)
    return_ihvp /= num_samples
    return return_ihvp

def get_hvp(loss: torch.Tensor, params: Sequence[torch.Tensor], vectors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    if False:
        i = 10
        return i + 15
    '\n    Get a Hessian-Vector Product (HVP) `Hv` for each Hessian `H` of the `loss`\n    with respect to the one of the parameter tensors in `params` and the corresponding\n    vector `v` in `vectors`.\n\n    # Parameters\n\n    loss : `torch.Tensor`\n        The loss calculated from the output of the model.\n    params : `Sequence[torch.Tensor]`\n        Tunable and used parameters in the model that we will calculate the gradient and hessian\n        with respect to.\n    vectors : `Sequence[torch.Tensor]`\n        The list of vectors for calculating the HVP.\n    '
    assert len(params) == len(vectors)
    assert all((p.size() == v.size() for (p, v) in zip(params, vectors)))
    grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    hvp = autograd.grad(grads, params, grad_outputs=vectors)
    return hvp

def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    '\n    Unwraps a list of parameters gradients\n\n    # Returns\n\n    `torch.Tensor`\n        A tensor of shape `(x,)` where `x` is the total number of entires in the gradients.\n    '
    views = []
    for p in tensors:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)