"""
Module holds internal entities for Modin XGBoost on Ray engine.

Class ModinXGBoostActor provides interfaces to run XGBoost operations
on remote workers. Other functions create Ray actors, distribute data between them, etc.
"""
import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager
LOGGER = logging.getLogger('[modin.xgboost]')

@ray.remote(num_cpus=0)
class ModinXGBoostActor:
    """
    Ray actor-class runs training on the remote worker.

    Parameters
    ----------
    rank : int
        Rank of this actor.
    nthread : int
        Number of threads used by XGBoost in this actor.
    """

    def __init__(self, rank, nthread):
        if False:
            while True:
                i = 10
        self._evals = []
        self._rank = rank
        self._nthreads = nthread
        LOGGER.info(f'Actor <{self._rank}>, nthread = {self._nthreads} was initialized.')

    def _get_dmatrix(self, X_y, **dmatrix_kwargs):
        if False:
            print('Hello World!')
        '\n        Create xgboost.DMatrix from sequence of pandas.DataFrame objects.\n\n        First half of `X_y` should contains objects for `X`, second for `y`.\n\n        Parameters\n        ----------\n        X_y : list\n            List of pandas.DataFrame objects.\n        **dmatrix_kwargs : dict\n            Keyword parameters for ``xgb.DMatrix``.\n\n        Returns\n        -------\n        xgb.DMatrix\n            A XGBoost DMatrix.\n        '
        s = time.time()
        X = X_y[:len(X_y) // 2]
        y = X_y[len(X_y) // 2:]
        assert len(X) == len(y) and len(X) > 0, 'X and y should have the equal length more than 0'
        X = pandas.concat(X, axis=0)
        y = pandas.concat(y, axis=0)
        LOGGER.info(f'Concat time: {time.time() - s} s')
        return xgb.DMatrix(X, y, nthread=self._nthreads, **dmatrix_kwargs)

    def set_train_data(self, *X_y, add_as_eval_method=None, **dmatrix_kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Set train data for actor.\n\n        Parameters\n        ----------\n        *X_y : iterable\n            Sequence of ray.ObjectRef objects. First half of sequence is for\n            `X` data, second for `y`. When it is passed in actor, auto-materialization\n            of ray.ObjectRef -> pandas.DataFrame happens.\n        add_as_eval_method : str, optional\n            Name of eval data. Used in case when train data also used for evaluation.\n        **dmatrix_kwargs : dict\n            Keyword parameters for ``xgb.DMatrix``.\n        '
        self._dtrain = self._get_dmatrix(X_y, **dmatrix_kwargs)
        if add_as_eval_method is not None:
            self._evals.append((self._dtrain, add_as_eval_method))

    def add_eval_data(self, *X_y, eval_method, **dmatrix_kwargs):
        if False:
            print('Hello World!')
        '\n        Add evaluation data for actor.\n\n        Parameters\n        ----------\n        *X_y : iterable\n            Sequence of ray.ObjectRef objects. First half of sequence is for\n            `X` data, second for `y`. When it is passed in actor, auto-materialization\n            of ray.ObjectRef -> pandas.DataFrame happens.\n        eval_method : str\n            Name of eval data.\n        **dmatrix_kwargs : dict\n            Keyword parameters for ``xgb.DMatrix``.\n        '
        self._evals.append((self._get_dmatrix(X_y, **dmatrix_kwargs), eval_method))

    def train(self, rabit_args, params, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Run local XGBoost training.\n\n        Connects to Rabit Tracker environment to share training data between\n        actors and trains XGBoost booster using `self._dtrain`.\n\n        Parameters\n        ----------\n        rabit_args : list\n            List with environment variables for Rabit Tracker.\n        params : dict\n            Booster params.\n        *args : iterable\n            Other parameters for `xgboost.train`.\n        **kwargs : dict\n            Other parameters for `xgboost.train`.\n\n        Returns\n        -------\n        dict\n            A dictionary with trained booster and dict of\n            evaluation results\n            as {"booster": xgb.Booster, "history": dict}.\n        '
        local_params = params.copy()
        local_dtrain = self._dtrain
        local_evals = self._evals
        local_params['nthread'] = self._nthreads
        evals_result = dict()
        s = time.time()
        with RabitContext(self._rank, rabit_args):
            bst = xgb.train(local_params, local_dtrain, *args, evals=local_evals, evals_result=evals_result, **kwargs)
            LOGGER.info(f'Local training time: {time.time() - s} s')
            return {'booster': bst, 'history': evals_result}

def _get_cluster_cpus():
    if False:
        return 10
    '\n    Get number of CPUs available on Ray cluster.\n\n    Returns\n    -------\n    int\n        Number of CPUs available on cluster.\n    '
    return ray.cluster_resources().get('CPU', 1)

def _get_min_cpus_per_node():
    if False:
        i = 10
        return i + 15
    '\n    Get min number of node CPUs available on cluster nodes.\n\n    Returns\n    -------\n    int\n        Min number of CPUs per node.\n    '
    max_node_cpus = min((node.get('Resources', {}).get('CPU', 0.0) for node in ray.nodes()))
    return max_node_cpus if max_node_cpus > 0.0 else _get_cluster_cpus()

def _get_cpus_per_actor(num_actors):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get number of CPUs to use by each actor.\n\n    Parameters\n    ----------\n    num_actors : int\n        Number of Ray actors.\n\n    Returns\n    -------\n    int\n        Number of CPUs per actor.\n    '
    cluster_cpus = _get_cluster_cpus()
    cpus_per_actor = max(1, min(int(_get_min_cpus_per_node() or 1), int(cluster_cpus // num_actors)))
    return cpus_per_actor

def _get_num_actors(num_actors=None):
    if False:
        while True:
            i = 10
    '\n    Get number of actors to create.\n\n    Parameters\n    ----------\n    num_actors : int, optional\n        Desired number of actors. If is None, integer number of actors\n        will be computed by condition 2 CPUs per 1 actor.\n\n    Returns\n    -------\n    int\n        Number of actors to create.\n    '
    min_cpus_per_node = _get_min_cpus_per_node()
    if num_actors is None:
        num_actors_per_node = max(1, int(min_cpus_per_node // 2))
        return num_actors_per_node * len(ray.nodes())
    elif isinstance(num_actors, int):
        assert num_actors % len(ray.nodes()) == 0, '`num_actors` must be a multiple to number of nodes in Ray cluster.'
        return num_actors
    else:
        RuntimeError('`num_actors` must be int or None')

def create_actors(num_actors):
    if False:
        return 10
    '\n    Create ModinXGBoostActors.\n\n    Parameters\n    ----------\n    num_actors : int\n        Number of actors to create.\n\n    Returns\n    -------\n    list\n        List of pairs (ip, actor).\n    '
    num_cpus_per_actor = _get_cpus_per_actor(num_actors)
    node_ips = [key for key in ray.cluster_resources().keys() if key.startswith('node:') and '__internal_head__' not in key]
    num_actors_per_node = max(num_actors // len(node_ips), 1)
    actors_ips = [ip for ip in node_ips for _ in range(num_actors_per_node)]
    actors = [(node_ip.split('node:')[-1], ModinXGBoostActor.options(resources={node_ip: 0.01}).remote(i, nthread=num_cpus_per_actor)) for (i, node_ip) in enumerate(actors_ips)]
    return actors

def _split_data_across_actors(actors: List, set_func, X_parts, y_parts):
    if False:
        for i in range(10):
            print('nop')
    '\n    Split row partitions of data between actors.\n\n    Parameters\n    ----------\n    actors : list\n        List of used actors.\n    set_func : callable\n        The function for setting data in actor.\n    X_parts : list\n        Row partitions of X data.\n    y_parts : list\n        Row partitions of y data.\n    '
    X_parts_by_actors = _assign_row_partitions_to_actors(actors, X_parts)
    y_parts_by_actors = _assign_row_partitions_to_actors(actors, y_parts, data_for_aligning=X_parts_by_actors)
    for (rank, (_, actor)) in enumerate(actors):
        set_func(actor, *X_parts_by_actors[rank][0] + y_parts_by_actors[rank][0])

def _assign_row_partitions_to_actors(actors: List, row_partitions, data_for_aligning=None):
    if False:
        while True:
            i = 10
    "\n    Assign row_partitions to actors.\n\n    `row_partitions` will be assigned to actors according to their IPs.\n    If distribution isn't even, partitions will be moved from actor\n    with excess partitions to actor with lack of them.\n\n    Parameters\n    ----------\n    actors : list\n        List of used actors.\n    row_partitions : list\n        Row partitions of data to assign.\n    data_for_aligning : dict, optional\n        Data according to the order of which should be\n        distributed `row_partitions`. Used to align y with X.\n\n    Returns\n    -------\n    dict\n        Dictionary of assigned to actors partitions\n        as {actor_rank: (partitions, order)}.\n    "
    num_actors = len(actors)
    if data_for_aligning is None:
        (parts_ips_ref, parts_ref) = zip(*row_partitions)
        actor_ips = defaultdict(list)
        for (rank, (ip, _)) in enumerate(actors):
            actor_ips[ip].append(rank)
        init_parts_distribution = defaultdict(list)
        for (idx, (ip, part_ref)) in enumerate(zip(RayWrapper.materialize(list(parts_ips_ref)), parts_ref)):
            init_parts_distribution[ip].append((part_ref, idx))
        num_parts = len(parts_ref)
        min_parts_per_actor = math.floor(num_parts / num_actors)
        max_parts_per_actor = math.ceil(num_parts / num_actors)
        num_actors_with_max_parts = num_parts % num_actors
        row_partitions_by_actors = defaultdict(list)
        for (actor_ip, ranks) in actor_ips.items():
            for rank in ranks:
                num_parts_on_ip = len(init_parts_distribution[actor_ip])
                if num_parts_on_ip == 0:
                    break
                if num_parts_on_ip >= min_parts_per_actor:
                    if num_parts_on_ip >= max_parts_per_actor and num_actors_with_max_parts > 0:
                        pop_slice = slice(0, max_parts_per_actor)
                        num_actors_with_max_parts -= 1
                    else:
                        pop_slice = slice(0, min_parts_per_actor)
                    row_partitions_by_actors[rank].extend(init_parts_distribution[actor_ip][pop_slice])
                    del init_parts_distribution[actor_ip][pop_slice]
                else:
                    row_partitions_by_actors[rank].extend(init_parts_distribution[actor_ip])
                    init_parts_distribution[actor_ip] = []
        for ip in list(init_parts_distribution):
            if len(init_parts_distribution[ip]) == 0:
                init_parts_distribution.pop(ip)
        init_parts_distribution = [pair for pairs in init_parts_distribution.values() for pair in pairs]
        for rank in range(len(actors)):
            num_parts_on_rank = len(row_partitions_by_actors[rank])
            if num_parts_on_rank == max_parts_per_actor or (num_parts_on_rank == min_parts_per_actor and num_actors_with_max_parts == 0):
                continue
            if num_actors_with_max_parts > 0:
                pop_slice = slice(0, max_parts_per_actor - num_parts_on_rank)
                num_actors_with_max_parts -= 1
            else:
                pop_slice = slice(0, min_parts_per_actor - num_parts_on_rank)
            row_partitions_by_actors[rank].extend(init_parts_distribution[pop_slice])
            del init_parts_distribution[pop_slice]
        if len(init_parts_distribution) != 0:
            raise RuntimeError(f'Not all partitions were ditributed between actors: {len(init_parts_distribution)} left.')
        row_parts_by_ranks = dict()
        for (rank, pairs_part_pos) in dict(row_partitions_by_actors).items():
            (parts, order) = zip(*pairs_part_pos)
            row_parts_by_ranks[rank] = (list(parts), list(order))
    else:
        row_parts_by_ranks = {rank: ([], []) for rank in range(len(actors))}
        for (rank, (_, order_of_indexes)) in data_for_aligning.items():
            row_parts_by_ranks[rank][1].extend(order_of_indexes)
            for row_idx in order_of_indexes:
                row_parts_by_ranks[rank][0].append(row_partitions[row_idx])
    return row_parts_by_ranks

def _train(dtrain, params: Dict, *args, num_actors=None, evals=(), **kwargs):
    if False:
        print('Hello World!')
    '\n    Run distributed training of XGBoost model on Ray engine.\n\n    During work it evenly distributes `dtrain` between workers according\n    to IP addresses partitions (in case of not even distribution of `dtrain`\n    by nodes, part of partitions will be re-distributed between nodes),\n    runs xgb.train on each worker for subset of `dtrain` and reduces training results\n    of each worker using Rabit Context.\n\n    Parameters\n    ----------\n    dtrain : modin.experimental.DMatrix\n        Data to be trained against.\n    params : dict\n        Booster params.\n    *args : iterable\n        Other parameters for `xgboost.train`.\n    num_actors : int, optional\n        Number of actors for training. If unspecified, this value will be\n        computed automatically.\n    evals : list of pairs (modin.experimental.xgboost.DMatrix, str), default: empty\n        List of validation sets for which metrics will be evaluated during training.\n        Validation metrics will help us track the performance of the model.\n    **kwargs : dict\n        Other parameters are the same as `xgboost.train`.\n\n    Returns\n    -------\n    dict\n        A dictionary with trained booster and dict of\n        evaluation results\n        as {"booster": xgboost.Booster, "history": dict}.\n    '
    s = time.time()
    (X_row_parts, y_row_parts) = dtrain
    dmatrix_kwargs = dtrain.get_dmatrix_params()
    assert len(X_row_parts) == len(y_row_parts), 'Unaligned train data'
    num_actors = _get_num_actors(num_actors)
    if num_actors > len(X_row_parts):
        num_actors = len(X_row_parts)
    if evals:
        min_num_parts = num_actors
        for ((eval_X, _), eval_method) in evals:
            if len(eval_X) < min_num_parts:
                min_num_parts = len(eval_X)
                method_name = eval_method
        if num_actors != min_num_parts:
            num_actors = min_num_parts
            warnings.warn(f'`num_actors` is set to {num_actors}, because `evals` data with name `{method_name}` has only {num_actors} partition(s).')
    actors = create_actors(num_actors)
    add_as_eval_method = None
    if evals:
        for (eval_data, method) in evals[:]:
            if eval_data is dtrain:
                add_as_eval_method = method
                evals.remove((eval_data, method))
        for ((eval_X, eval_y), eval_method) in evals:
            _split_data_across_actors(actors, lambda actor, *X_y: actor.add_eval_data.remote(*X_y, eval_method=eval_method, **dmatrix_kwargs), eval_X, eval_y)
    _split_data_across_actors(actors, lambda actor, *X_y: actor.set_train_data.remote(*X_y, add_as_eval_method=add_as_eval_method, **dmatrix_kwargs), X_row_parts, y_row_parts)
    LOGGER.info(f'Data preparation time: {time.time() - s} s')
    s = time.time()
    with RabitContextManager(len(actors), get_node_ip_address()) as env:
        rabit_args = [('%s=%s' % item).encode() for item in env.items()]
        fut = [actor.train.remote(rabit_args, params, *args, **kwargs) for (_, actor) in actors]
        result = RayWrapper.materialize(fut[0])
        LOGGER.info(f'Training time: {time.time() - s} s')
        return result

@ray.remote
def _map_predict(booster, part, columns, dmatrix_kwargs={}, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Run prediction on a remote worker.\n\n    Parameters\n    ----------\n    booster : xgboost.Booster or ray.ObjectRef\n        A trained booster.\n    part : pandas.DataFrame or ray.ObjectRef\n        Partition of full data used for local prediction.\n    columns : list or ray.ObjectRef\n        Columns for the result.\n    dmatrix_kwargs : dict, optional\n        Keyword parameters for ``xgb.DMatrix``.\n    **kwargs : dict\n        Other parameters are the same as for ``xgboost.Booster.predict``.\n\n    Returns\n    -------\n    ray.ObjectRef\n        ``ray.ObjectRef`` with partial prediction.\n    '
    dmatrix = xgb.DMatrix(part, **dmatrix_kwargs)
    prediction = pandas.DataFrame(booster.predict(dmatrix, **kwargs), index=part.index, columns=columns)
    return prediction

def _predict(booster, data, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Run distributed prediction with a trained booster on Ray engine.\n\n    During execution it runs ``xgb.predict`` on each worker for subset of `data`\n    and creates Modin DataFrame with prediction results.\n\n    Parameters\n    ----------\n    booster : xgboost.Booster\n        A trained booster.\n    data : modin.experimental.xgboost.DMatrix\n        Input data used for prediction.\n    **kwargs : dict\n        Other parameters are the same as for ``xgboost.Booster.predict``.\n\n    Returns\n    -------\n    modin.pandas.DataFrame\n        Modin DataFrame with prediction results.\n    '
    s = time.time()
    dmatrix_kwargs = data.get_dmatrix_params()
    (input_index, input_columns, row_lengths) = data.metadata

    def _get_num_columns(booster, n_features, **kwargs):
        if False:
            return 10
        rng = np.random.RandomState(777)
        test_data = rng.randn(1, n_features)
        test_predictions = booster.predict(xgb.DMatrix(test_data), validate_features=False, **kwargs)
        num_columns = test_predictions.shape[1] if len(test_predictions.shape) > 1 else 1
        return num_columns
    result_num_columns = _get_num_columns(booster, len(input_columns), **kwargs)
    new_columns = list(range(result_num_columns))
    booster = RayWrapper.put(booster)
    new_columns_ref = RayWrapper.put(new_columns)
    prediction_refs = [_map_predict.remote(booster, part, new_columns_ref, dmatrix_kwargs, **kwargs) for (_, part) in data.data]
    predictions = from_partitions(prediction_refs, 0, index=input_index, columns=new_columns, row_lengths=row_lengths, column_widths=[len(new_columns)])
    LOGGER.info(f'Prediction time: {time.time() - s} s')
    return predictions