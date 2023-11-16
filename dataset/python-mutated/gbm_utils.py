from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union
import lightgbm as lgb
import numpy as np
import torch
from lightgbm.callback import CallbackEnv
from numpy import typing as npt
from ludwig.constants import NUMBER
from ludwig.features.base_feature import BaseFeatureMixin, OutputFeature
from ludwig.features.category_feature import CategoryOutputFeature
from ludwig.models.base import BaseModel

def get_single_output_feature(model: BaseModel) -> BaseFeatureMixin:
    if False:
        i = 10
        return i + 15
    'Helper for getting the single output feature of a model.'
    return next(iter(model.output_features.values()))

def sigmoid(x: npt.NDArray) -> npt.NDArray:
    if False:
        i = 10
        return i + 15
    'Compute sigmoid function.\n\n    Args:\n        x: 1D array of shape (n_samples,).\n\n    Returns:\n        1D array of shape (n_samples,).\n    '
    return 1.0 / (1.0 + np.exp(-x))

def log_loss_objective(y_true: npt.NDArray, y_pred: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
    if False:
        return 10
    'Binary objective function for LightGBM. Computes the logistic loss.\n\n    Args:\n        y_true: 1D array of true labels of shape (n_samples,).\n        y_pred: 1D array of raw predictions of shape (n_samples,).\n\n    Returns:\n        1D array of gradients of shape (n_samples,) and 1D array of hessians of shape (n_samples,).\n\n    References:\n    - https://github.com/microsoft/LightGBM/issues/3312\n    - https://github.com/microsoft/LightGBM/issues/5373#issuecomment-1188595889\n    '
    y_pred = sigmoid(y_pred)
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return (grad, hess)

def softmax(x: npt.NDArray) -> npt.NDArray:
    if False:
        print('Hello World!')
    'Compute softmax values for each sets of scores in x.\n\n    Args:\n        x: 2D array of shape (n_samples, n_classes).\n\n    Returns:\n        2D array of shape (n_samples, n_classes).\n    '
    row_wise_max = np.max(x, axis=1).reshape(-1, 1)
    exp_x = np.exp(x - row_wise_max)
    return exp_x / np.sum(exp_x, axis=1).reshape(-1, 1)

def multiclass_objective(y_true: npt.NDArray, y_pred: npt.NDArray, weight: npt.NDArray=None) -> Tuple[npt.NDArray, npt.NDArray]:
    if False:
        for i in range(10):
            print('nop')
    'Multi-class objective function for LightGBM. Computes the softmax cross-entropy loss.\n\n    Args:\n        y_true: 1D array of true labels of shape (n_samples,).\n        y_pred: 1D array of raw predictions of shape (n_samples * n_classes,).\n        weight: 1D array of weights of shape (n_samples,).\n\n    Returns:\n        1D array of gradients of shape (n_samples * n_classes,) and 1D array of hessians\n        of shape (n_samples * n_classes,).\n\n    References:\n    - https://github.com/microsoft/LightGBM/blob/9afd8b/tests/python_package_test/test_sklearn.py#L1296\n    - https://github.com/microsoft/LightGBM/blob/9afd8b/tests/python_package_test/utils.py#L142\n    '
    y_pred = y_pred.reshape(y_true.shape[0], -1, order='F')
    (num_rows, num_class) = y_pred.shape
    prob = softmax(y_pred)
    grad_update = np.zeros_like(prob)
    grad_update[np.arange(num_rows), y_true.astype(np.int32)] = -1.0
    grad = prob + grad_update
    factor = num_class / (num_class - 1)
    hess = factor * prob * (1 - prob)
    if weight is not None:
        weight2d = weight.reshape(-1, 1)
        grad *= weight2d
        hess *= weight2d
    grad = grad.ravel(order='F')
    hess = hess.ravel(order='F')
    return (grad, hess)

def store_predictions(train_logits_buffer: torch.Tensor) -> Callable:
    if False:
        print('Hello World!')
    'Create a callback that records the predictions of the model on the training data in ``train_logits_buffer``.\n\n    Args:\n        train_logits_buffer: 2D tensor of shape (n_samples, n_classes) that stores the predictions of the model.\n\n    Returns:\n        A callback that records the predictions of the model in ``train_logits_buffer``.\n    '

    def _callback(env: CallbackEnv) -> None:
        if False:
            for i in range(10):
                print('nop')
        preds = env.model._Booster__inner_predict(data_idx=0).copy()
        batch_size = preds.size // env.model._Booster__num_class
        preds = preds.reshape(batch_size, env.model._Booster__num_class, order='F')
        data_view = train_logits_buffer.view(-1)
        data_view[:] = torch.from_numpy(preds).reshape(-1)
    return _callback

@dataclass
class TrainLogits:
    preds: torch.Tensor

def store_predictions_ray(boost_rounds_per_train_step: int) -> Callable:
    if False:
        i = 10
        return i + 15
    'Create a callback that records the predictions of the model on the training data in ``additional_results``\n    returned from a LightGBM on Ray model. Only the predictions of the last iteration are stored.\n\n    Args:\n        boost_rounds_per_train_step: number of boosting rounds per train step, used to compute last iteration.\n\n    Returns:\n        A callback that records the predictions of the model in ``additional_results``.\n    '

    def _callback(env: CallbackEnv) -> None:
        if False:
            while True:
                i = 10
        if env.iteration < boost_rounds_per_train_step - 1:
            return
        from xgboost_ray.session import put_queue
        preds = env.model._Booster__inner_predict(data_idx=0).copy()
        batch_size = preds.size // env.model._Booster__num_class
        if env.model._Booster__num_class > 1:
            preds = preds.reshape(batch_size, env.model._Booster__num_class, order='F')
        preds = torch.from_numpy(preds)
        put_queue(TrainLogits(preds))
    return _callback

def reshape_logits(output_feature: OutputFeature, logits: torch.Tensor) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    'Add logits for the oposite class if the output feature is category with two classes.\n\n    This is needed because LightGBM classifier only returns logits for one class.\n    '
    if isinstance(output_feature, CategoryOutputFeature) and output_feature.num_classes == 2:
        logits = logits.view(-1, 1)
        logits = torch.cat([-logits, logits], dim=1)
    return logits

def logits_to_predictions(model: BaseModel, train_logits: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
    if False:
        print('Hello World!')
    'Convert the logits of the model to Ludwig predictions.\n\n    Args:\n        model: the Ludwig model.\n        train_logits: 2D tensor of shape (n_samples, n_classes) that contains the predictions of the model.\n\n    Returns:\n        A dictionary mapping the output feature name to the predictions.\n    '
    output_feature = get_single_output_feature(model)
    train_logits = reshape_logits(output_feature, train_logits)
    return model.outputs_to_predictions({f'{output_feature.feature_name}::logits': train_logits})

def get_targets(lgb_train: Union[lgb.Dataset, 'RayDMatrix'], output_feature: BaseFeatureMixin, device: str, actor_rank: int=0) -> Dict[str, torch.Tensor]:
    if False:
        print('Hello World!')
    'Get the targets of the training data.\n\n    Args:\n        lgb_train: the training data.\n        output_feature: the output feature.\n        device: the device to store the targets on.\n        actor_rank: (optional, only used for RayDMatrix) the rank of the actor to get the targets for.\n\n    Returns:\n        A dictionary mapping the output feature name to the targets.\n    '
    is_regression = output_feature.type() == NUMBER
    if isinstance(lgb_train, lgb.Dataset):
        targets = lgb_train.get_label()
    else:
        targets = lgb_train.get_data(actor_rank, 1)['label'].to_numpy()
    targets = targets.copy() if is_regression else targets.copy().astype(int)
    return {output_feature.feature_name: torch.from_numpy(targets).cpu()}