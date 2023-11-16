from collections import OrderedDict
from enum import Enum
from functools import partial
import itertools
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple, Union
from fastai.callbacks import EarlyStoppingCallback
from fastai.metrics import accuracy
from fastai.vision import cnn_learner, get_transforms, ImageDataBunch, ImageList, imagenet_stats, Learner, models, ResizeMethod, SegmentationItemList, unet_learner
from matplotlib.axes import Axes
from matplotlib.text import Annotation
import pandas as pd
from utils_cv.common.gpu import db_num_workers
from utils_cv.segmentation.dataset import read_classes
from utils_cv.segmentation.model import get_ratio_correct_metric
Time = float
parameter_flag = 'PARAMETERS'

class TrainingSchedule(Enum):
    head_only = ('head_only',)
    body_only = ('body_only',)
    head_first_then_body = 'head_first_then_body'

class Architecture(Enum):
    resnet18 = partial(models.resnet18)
    resnet34 = partial(models.resnet34)
    resnet50 = partial(models.resnet50)
    squeezenet1_1 = partial(models.squeezenet1_1)

class DataFrameAlreadyCleaned(Exception):
    pass

def clean_sweeper_df(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        print('Hello World!')
    ' Cleans up dataframe outputed from sweeper\n\n    Cleans up experiment paramter strings in {df} by removing all experiment\n    parameters that held constant through each experiment. This method uses a\n    variable <parameter_flag> to search for strings.\n\n    Args:\n        df (pd.DataFrame): dataframe to clean up\n    Raises:\n        DataFrameAlreadyCleaned\n    Return:\n        pd.DataFrame: df with renamed experiment parameter strings\n    '
    text = df.to_html()
    if parameter_flag not in text:
        raise DataFrameAlreadyCleaned
    text = re.findall(f'>\\s{{0,1}}{parameter_flag}\\s{{0,1}}(.*?)</th>', text)
    sets = [set(t.split('|')) for t in text]
    intersection = sets[0].intersection(*sets)
    html = df.to_html()
    for i in intersection:
        html = html.replace(i, '')
    html = html.replace('PARAMETERS', 'P:')
    html = html.replace('|', ' ')
    return pd.read_html(html, index_col=[0, 1, 2])[0]

def add_value_labels(ax: Axes, spacing: int=5, percentage: bool=False) -> None:
    if False:
        return 10
    ' Add labels to the end of each bar in a bar chart.\n\n    Overwrite labels on axes if they already exist.\n\n    Args:\n        ax (Axes): The matplotlib object containing the axes of the plot to annotate.\n        spacing (int): The distance between the labels and the bars.\n        percentage (bool): if y-value is a percentage\n    '
    for child in ax.get_children():
        if isinstance(child, Annotation):
            child.remove()
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        label = '{:.2f}%'.format(y_value * 100) if percentage else '{:.1f}'.format(y_value)
        ax.annotate(label, (x_value, y_value), xytext=(0, spacing), textcoords='offset points', ha='center', va='bottom')

def plot_sweeper_df(df: pd.DataFrame, sort_by: str=None, figsize: Tuple[int, int]=(12, 8), show_cols: List[str]=None) -> None:
    if False:
        print('Hello World!')
    ' Visualize df outputed from sweeper\n\n    Visualize graph from {df}, which should contain columns "accuracy" and\n    "duration". Columns not titled "accuracy" or "duration" will also be\n    rendered.\n\n    Args:\n        df (pd.DataFrame): the dataframe to visualize.\n        sort_by (str): whether to sort visualization by accuracy or duration.\n        figsize (Tuple[int, int]): as defined in matplotlib.\n        show_cols (List[str]): a list of columns in the df to show\n    Raises:\n        ValueError: if {sort_by} is an invalid value, if elements of\n        {show_cols} is not a valid column name, or if {sort_by} is not in\n        {show_cols} if it is used.\n    '
    cols = list(df.columns.values) if show_cols is None else show_cols
    if not set(cols) <= set(list(df.columns.values)):
        raise ValueError('values of {show_cols} is not found {df}.')
    if sort_by is not None and sort_by not in cols:
        raise ValueError('{sort_by} must be in {show_cols} if {show_cols} is used.')
    if sort_by:
        df = df.sort_values(by=sort_by)
    axes = df[cols].plot.bar(rot=90, subplots=True, legend=False, figsize=figsize)
    assert len(cols) == len(axes)
    for (col, ax) in zip(cols, axes):
        top_val = df[col].max()
        min_val = df[col].min()
        ax.set_ylim(bottom=min_val / 1.01, top=top_val * 1.01)
        add_value_labels(ax)
        if col in ['accuracy']:
            add_value_labels(ax, percentage=True)
            ax.set_title('Accuracy (%)')
            ax.set_ylabel('%')
        if col in ['duration']:
            ax.set_title('Training Duration (seconds)')
            ax.set_ylabel('seconds')

class ParameterSweeper:
    """ Test different permutations of a set of parameters.

    Attributes:
        param_order <Tuple[str]>: A fixed ordering of parameters (to match the ordering of <params>)
        default_params <Dict[str, Any]>: A dict of default parameters
        params <Dict[str, List[Any]]>: The parameters to run experiments on
    """
    default_params = dict(learning_rate=0.0001, epoch=15, batch_size=16, im_size=299, architecture=Architecture.resnet18, transform=True, dropout=0.5, weight_decay=0.01, training_schedule=TrainingSchedule.head_first_then_body, discriminative_lr=False, one_cycle_policy=True)

    def __init__(self, metric_name='accuracy', **kwargs) -> None:
        if False:
            return 10
        '\n        Initialize class with default params if kwargs is empty.\n        Otherwise, initialize params with kwargs.\n        '
        self.params = OrderedDict(learning_rate=[self.default_params.get('learning_rate')], epochs=[self.default_params.get('epoch')], batch_size=[self.default_params.get('batch_size')], im_size=[self.default_params.get('im_size')], architecture=[self.default_params.get('architecture')], transform=[self.default_params.get('transform')], dropout=[self.default_params.get('dropout')], weight_decay=[self.default_params.get('weight_decay')], training_schedule=[self.default_params.get('training_schedule')], discriminative_lr=[self.default_params.get('discriminative_lr')], one_cycle_policy=[self.default_params.get('one_cycle_policy')])
        self.metric_name = metric_name
        self.param_order = tuple(self.params.keys())
        self.update_parameters(**kwargs)

    @property
    def parameters(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        ' Returns parameters to test on if run() is called. '
        return self.params

    @property
    def permutations(self) -> List[Tuple[Any]]:
        if False:
            for i in range(10):
                print('nop')
        ' Returns a list of all permutations, expressed in tuples. '
        params = tuple([self.params[k] for k in self.param_order])
        permutations = list(itertools.product(*params))
        return permutations

    @staticmethod
    def _get_data_bunch_imagelist(path: Union[Path, str], transform: bool, im_size: int, bs: int) -> ImageDataBunch:
        if False:
            print('Hello World!')
        "\n        Create ImageDataBunch and return it. TODO in future version is to allow\n        users to pass in their own image bunch or their own Transformation\n        objects (instead of using fastai's <get_transforms>)\n\n        Args:\n            path (Union[Path, str]): path to data to create databunch with\n            transform (bool): a flag to set fastai default transformations (get_transforms())\n            im_size (int): image size of databunch\n            bs (int): batch size of databunch\n        Returns:\n            ImageDataBunch\n        "
        path = path if type(path) is Path else Path(path)
        tfms = get_transforms() if transform else None
        return ImageList.from_folder(path).split_by_rand_pct(valid_pct=0.33).label_from_folder().transform(tfms=tfms, size=im_size).databunch(bs=bs, num_workers=db_num_workers()).normalize(imagenet_stats)

    @staticmethod
    def _get_data_bunch_segmentationitemlist(path: Union[Path, str], transform: bool, im_size: int, bs: int, classes: List[str]) -> ImageDataBunch:
        if False:
            return 10
        "\n        Create ImageDataBunch and return it. TODO in future version is to allow\n        users to pass in their own image bunch or their own Transformation\n        objects (instead of using fastai's <get_transforms>)\n\n        Args:\n            path (Union[Path, str]): path to data to create databunch with\n            transform (bool): a flag to set fastai default transformations (get_transforms())\n            im_size (int): image size of databunch\n            bs (int): batch size of databunch\n        Returns:\n            ImageDataBunch\n        "
        path = path if type(path) is Path else Path(path)
        tfms = get_transforms() if transform else None
        im_path = path / 'images'
        anno_path = path / 'segmentation-masks'
        get_gt_filename = lambda x: anno_path / f'{x.stem}.png'
        return SegmentationItemList.from_folder(im_path).split_by_rand_pct(valid_pct=0.33).label_from_func(get_gt_filename, classes=classes).transform(tfms=tfms, resize_method=ResizeMethod.CROP, size=im_size, tfm_y=True).databunch(bs=bs, num_workers=db_num_workers()).normalize(imagenet_stats)

    @staticmethod
    def _early_stopping_callback(metric: str='accuracy', min_delta: float=0.01, patience: int=3) -> partial:
        if False:
            print('Hello World!')
        ' Returns an early stopping callback. '
        return partial(EarlyStoppingCallback, monitor=metric, min_delta=min_delta, patience=patience)

    @staticmethod
    def _serialize_permutations(p: Tuple[Any]) -> str:
        if False:
            for i in range(10):
                print('nop')
        ' Serializes all parameters as a string that uses {parameter_flag}. '
        p = iter(p)
        return f'{parameter_flag} [learning_rate: {next(p)}]|[epochs: {next(p)}]|[batch_size: {next(p)}]|[im_size: {next(p)}]|[arch: {next(p).name}]|[transforms: {next(p)}]|[dropout: {next(p)}]|[weight_decay: {next(p)}]|[training_schedule: {next(p).name}]|[discriminative_lr: {next(p)}]|[one_cycle_policy: {next(p)}]'

    @staticmethod
    def _make_df_from_dict(results: Dict[Any, Dict[Any, Dict[Any, Dict[Any, Any]]]]) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        ' Converts a 4-times-nested dictionary into a multi-index dataframe. '
        return pd.DataFrame.from_dict({(i, j, k): results[i][j][k] for i in results.keys() for j in results[i].keys() for k in results[i][j].keys()}, orient='index')

    def _param_tuple_to_dict(self, params: Tuple[Any]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        ' Converts a tuple of parameters to a Dict. '
        return dict(learning_rate=params[self.param_order.index('learning_rate')], batch_size=params[self.param_order.index('batch_size')], transform=params[self.param_order.index('transform')], im_size=params[self.param_order.index('im_size')], epochs=params[self.param_order.index('epochs')], architecture=params[self.param_order.index('architecture')], dropout=params[self.param_order.index('dropout')], weight_decay=params[self.param_order.index('weight_decay')], discriminative_lr=params[self.param_order.index('discriminative_lr')], training_schedule=params[self.param_order.index('training_schedule')], one_cycle_policy=params[self.param_order.index('one_cycle_policy')])

    def _learn(self, data_path: Path, params: Tuple[Any], stop_early: bool, learner_type='cnn') -> Tuple[Learner, Time]:
        if False:
            i = 10
            return i + 15
        '\n        Given a set of permutations, create a learner to train and validate on\n        the dataset.\n        Args:\n            data_path (Path): The location of the data to use\n            params (Tuple[Any]): The set of parameters to train and validate on\n            stop_early (bool): Whether or not to stop early if the evaluation\n            metric does not improve\n        Returns:\n            Tuple[Learner, Time]: Learn object from Fastai and the duration in\n            seconds it took.\n        '
        start = time.time()
        params = self._param_tuple_to_dict(params)
        transform = params['transform']
        im_size = params['im_size']
        epochs = params['epochs']
        batch_size = params['batch_size']
        architecture = params['architecture']
        dropout = params['dropout']
        learning_rate = params['learning_rate']
        discriminative_lr = params['discriminative_lr']
        training_schedule = params['training_schedule']
        one_cycle_policy = params['one_cycle_policy']
        weight_decay = params['weight_decay']
        callbacks = list()
        if stop_early:
            callbacks.append(ParameterSweeper._early_stopping_callback())
        if learner_type == 'cnn':
            data = self._get_data_bunch_imagelist(data_path, transform, im_size, batch_size)
            learn = cnn_learner(data, architecture.value, metrics=accuracy, ps=dropout, callback_fns=callbacks)
        elif learner_type == 'unet':
            classes = read_classes(os.path.join(data_path, 'classes.txt'))
            data = self._get_data_bunch_segmentationitemlist(data_path, transform, im_size, batch_size, classes)
            metric = get_ratio_correct_metric(classes)
            metric.__name__ = 'ratio_correct'
            learn = unet_learner(data, architecture.value, wd=0.01, metrics=metric, callback_fns=callbacks)
        else:
            print(f'Mode learner_type={learner_type} not supported.')
        head_learning_rate = learning_rate
        body_learning_rate = slice(learning_rate, 0.003) if discriminative_lr else learning_rate

        def fit(learn: Learner, e: int, lr: Union[slice, float], wd=float) -> partial:
            if False:
                i = 10
                return i + 15
            ' Returns a partial func for either fit_one_cycle or fit\n            depending on <one_cycle_policy> '
            return partial(learn.fit_one_cycle, cyc_len=e, max_lr=lr, wd=wd) if one_cycle_policy else partial(learn.fit, epochs=e, lr=lr, wd=wd)
        if training_schedule is TrainingSchedule.head_only:
            if discriminative_lr:
                raise Exception('Cannot run discriminative_lr if training schedule is head_only.')
            else:
                fit(learn, epochs, body_learning_rate, weight_decay)()
        elif training_schedule is TrainingSchedule.body_only:
            learn.unfreeze()
            fit(learn, epochs, body_learning_rate, weight_decay)()
        elif training_schedule is TrainingSchedule.head_first_then_body:
            head_epochs = epochs // 4
            fit(learn, head_epochs, head_learning_rate, weight_decay)()
            learn.unfreeze()
            fit(learn, epochs - head_epochs, body_learning_rate, weight_decay)()
        end = time.time()
        duration = end - start
        return (learn, duration)

    def update_parameters(self, **kwargs) -> 'ParameterSweeper':
        if False:
            return 10
        " Update the class object's parameters.\n        If kwarg key is not in an existing param key, then raise exception.\n        If the kwarg value is None, pass.\n        Otherwise overwrite the corresponding self.params key.\n        "
        for (k, v) in kwargs.items():
            if k not in set(self.params.keys()):
                raise Exception(f'Parameter {k} is invalid.')
            if v is None:
                continue
            self.params[k] = v
        return self

    def run(self, datasets: List[Path], reps: int=3, early_stopping: bool=False, metric_fct=None, learner_type='cnn') -> pd.DataFrame:
        if False:
            return 10
        ' Performs the experiment.\n        Iterates through the number of specified <reps>, the list permutations\n        as defined in this class, and the <datasets> to calculate evaluation\n        metrics and duration for each run.\n\n        WARNING: this method can take a long time depending on your experiment\n        definition.\n\n        Args:\n            datasets: A list of datasets to iterate over.\n            reps: The number of runs to loop over.\n            early_stopping: Whether we want to perform early stopping.\n            metric_fct: custom metric function\n            learner_type: choose between "cnn" and "unet" learners\n        Returns:\n            pd.DataFrame: a multi-index dataframe with the results stored in it.\n        '
        count = 0
        res = dict()
        for rep in range(reps):
            res[rep] = dict()
            for (i, permutation) in enumerate(self.permutations):
                stringified_permutation = self._serialize_permutations(permutation)
                res[rep][stringified_permutation] = dict()
                for (ii, dataset) in enumerate(datasets):
                    percent_done = round(100.0 * count / (reps * len(self.permutations) * len(datasets)))
                    print(f'Percentage done: {percent_done}%. Currently processing repeat {rep + 1} of {reps}, running {i + 1} of {len(self.permutations)} permutations, dataset {ii + 1} of {len(datasets)} ({os.path.basename(dataset)}). ')
                    data_name = os.path.basename(dataset)
                    res[rep][stringified_permutation][data_name] = dict()
                    (learn, duration) = self._learn(dataset, permutation, early_stopping, learner_type)
                    if metric_fct is None and learner_type == 'cnn':
                        (_, metric) = learn.validate(learn.data.valid_dl, metrics=[accuracy])
                    elif learner_type == 'unet':
                        (_, metric) = learn.validate(learn.data.valid_dl)
                    else:
                        metric = metric_fct(learn)
                    res[rep][stringified_permutation][data_name]['duration'] = duration
                    res[rep][stringified_permutation][data_name][self.metric_name] = float(metric)
                    learn.destroy()
                    count += 1
        return self._make_df_from_dict(res)