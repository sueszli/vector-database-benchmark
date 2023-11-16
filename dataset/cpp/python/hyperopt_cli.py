#! /usr/bin/env python
# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import logging
import sys
from typing import List, Optional, Union

from ludwig.backend import ALL_BACKENDS, Backend, initialize_backend
from ludwig.callbacks import Callback
from ludwig.contrib import add_contrib_callback_args
from ludwig.globals import LUDWIG_VERSION
from ludwig.hyperopt.run import hyperopt
from ludwig.utils.data_utils import load_config_from_str, load_yaml
from ludwig.utils.defaults import default_random_seed
from ludwig.utils.print_utils import get_logging_level_registry, print_ludwig

logger = logging.getLogger(__name__)


def hyperopt_cli(
    config: Union[str, dict],
    dataset: str = None,
    training_set: str = None,
    validation_set: str = None,
    test_set: str = None,
    training_set_metadata: str = None,
    data_format: str = None,
    experiment_name: str = "experiment",
    model_name: str = "run",
    # model_load_path=None,
    # model_resume_path=None,
    skip_save_training_description: bool = False,
    skip_save_training_statistics: bool = False,
    skip_save_model: bool = False,
    skip_save_progress: bool = False,
    skip_save_log: bool = False,
    skip_save_processed_input: bool = False,
    skip_save_unprocessed_output: bool = False,
    skip_save_predictions: bool = False,
    skip_save_eval_stats: bool = False,
    skip_save_hyperopt_statistics: bool = False,
    output_directory: str = "results",
    gpus: Union[str, int, List[int]] = None,
    gpu_memory_limit: Optional[float] = None,
    allow_parallel_threads: bool = True,
    callbacks: List[Callback] = None,
    backend: Union[Backend, str] = None,
    random_seed: int = default_random_seed,
    hyperopt_log_verbosity: int = 3,
    **kwargs,
):
    """Searches for optimal hyperparameters.

    # Inputs

    :param config: (Union[str, dict]) in-memory representation of
            config or string path to a YAML config file.
    :param dataset: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing the entire dataset to be used for training.
        If it has a split column, it will be used for splitting (0 for train,
        1 for validation, 2 for test), otherwise the dataset will be
        randomly split.
    :param training_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing training data.
    :param validation_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing validation data.
    :param test_set: (Union[str, dict, pandas.DataFrame], default: `None`)
        source containing test data.
    :param training_set_metadata: (Union[str, dict], default: `None`)
        metadata JSON file or loaded metadata.  Intermediate preprocessed
        structure containing the mappings of the input
        dataset created the first time an input file is used in the same
        directory with the same name and a '.meta.json' extension.
    :param data_format: (str, default: `None`) format to interpret data
        sources. Will be inferred automatically if not specified.  Valid
        formats are `'auto'`, `'csv'`, `'excel'`, `'feather'`,
        `'fwf'`, `'hdf5'` (cache file produced during previous training),
        `'html'` (file containing a single HTML `<table>`), `'json'`, `'jsonl'`,
        `'parquet'`, `'pickle'` (pickled Pandas DataFrame), `'sas'`, `'spss'`,
        `'stata'`, `'tsv'`.
    :param experiment_name: (str, default: `'experiment'`) name for
        the experiment.
    :param model_name: (str, default: `'run'`) name of the model that is
        being used.
    :param skip_save_training_description: (bool, default: `False`) disables
        saving the description JSON file.
    :param skip_save_training_statistics: (bool, default: `False`) disables
        saving training statistics JSON file.
    :param skip_save_model: (bool, default: `False`) disables
        saving model weights and hyperparameters each time the model
        improves. By default Ludwig saves model weights after each epoch
        the validation metric improves, but if the model is really big
        that can be time consuming. If you do not want to keep
        the weights and just find out what performance a model can get
        with a set of hyperparameters, use this parameter to skip it,
        but the model will not be loadable later on and the returned model
        will have the weights obtained at the end of training, instead of
        the weights of the epoch with the best validation performance.
    :param skip_save_progress: (bool, default: `False`) disables saving
        progress each epoch. By default Ludwig saves weights and stats
        after each epoch for enabling resuming of training, but if
        the model is really big that can be time consuming and will uses
        twice as much space, use this parameter to skip it, but training
        cannot be resumed later on.
    :param skip_save_log: (bool, default: `False`) disables saving
        TensorBoard logs. By default Ludwig saves logs for the TensorBoard,
        but if it is not needed turning it off can slightly increase the
        overall speed.
    :param skip_save_processed_input: (bool, default: `False`) if input
        dataset is provided it is preprocessed and cached by saving an HDF5
        and JSON files to avoid running the preprocessing again. If this
        parameter is `False`, the HDF5 and JSON file are not saved.
    :param skip_save_unprocessed_output: (bool, default: `False`) by default
        predictions and their probabilities are saved in both raw
        unprocessed numpy files containing tensors and as postprocessed
        CSV files (one for each output feature). If this parameter is True,
        only the CSV ones are saved and the numpy ones are skipped.
    :param skip_save_predictions: (bool, default: `False`) skips saving test
        predictions CSV files
    :param skip_save_eval_stats: (bool, default: `False`) skips saving test
        statistics JSON file
    :param skip_save_hyperopt_statistics: (bool, default: `False`) skips saving
        hyperopt stats file.
    :param output_directory: (str, default: `'results'`) the directory that
        will contain the training statistics, TensorBoard logs, the saved
        model and the training progress files.
    :param gpus: (list, default: `None`) list of GPUs that are available
        for training.
    :param gpu_memory_limit: (float: default: `None`) maximum memory fraction
        [0, 1] allowed to allocate per GPU device.
    :param allow_parallel_threads: (bool, default: `True`) allow PyTorch
        to use multithreading parallelism to improve performance at
        the cost of determinism.
    :param callbacks: (list, default: `None`) a list of
        `ludwig.callbacks.Callback` objects that provide hooks into the
        Ludwig pipeline.
    :param backend: (Union[Backend, str]) `Backend` or string name
        of backend to use to execute preprocessing / training steps.
    :param random_seed: (int: default: 42) random seed used for weights
        initialization, splits and any other random function.
    :param hyperopt_log_verbosity: (int: default: 3) Controls verbosity of ray tune log messages.  Valid values:
        0 = silent, 1 = only status updates, 2 = status and brief trial
        results, 3 = status and detailed trial results.

    # Return
    :return" (`None`)
    """
    return hyperopt(
        config=config,
        dataset=dataset,
        training_set=training_set,
        validation_set=validation_set,
        test_set=test_set,
        training_set_metadata=training_set_metadata,
        data_format=data_format,
        experiment_name=experiment_name,
        model_name=model_name,
        # model_load_path=model_load_path,
        # model_resume_path=model_resume_path,
        skip_save_training_description=skip_save_training_description,
        skip_save_training_statistics=skip_save_training_statistics,
        skip_save_model=skip_save_model,
        skip_save_progress=skip_save_progress,
        skip_save_log=skip_save_log,
        skip_save_processed_input=skip_save_processed_input,
        skip_save_unprocessed_output=skip_save_unprocessed_output,
        skip_save_predictions=skip_save_predictions,
        skip_save_eval_stats=skip_save_eval_stats,
        skip_save_hyperopt_statistics=skip_save_hyperopt_statistics,
        output_directory=output_directory,
        gpus=gpus,
        gpu_memory_limit=gpu_memory_limit,
        allow_parallel_threads=allow_parallel_threads,
        callbacks=callbacks,
        backend=backend,
        random_seed=random_seed,
        hyperopt_log_verbosity=hyperopt_log_verbosity,
        **kwargs,
    )


def cli(sys_argv):
    parser = argparse.ArgumentParser(
        description="This script searches for optimal Hyperparameters",
        prog="ludwig hyperopt",
        usage="%(prog)s [options]",
    )

    # -------------------
    # Hyperopt parameters
    # -------------------
    parser.add_argument(
        "-sshs",
        "--skip_save_hyperopt_statistics",
        help="skips saving hyperopt statistics file",
        action="store_true",
        default=False,
    )

    # ----------------------------
    # Experiment naming parameters
    # ----------------------------
    parser.add_argument(
        "--output_directory",
        type=str,
        default="results",
        help="directory that contains the results",
    )
    parser.add_argument("--experiment_name", type=str, default="hyperopt", help="experiment name")
    parser.add_argument("--model_name", type=str, default="run", help="name for the model")

    # ---------------
    # Data parameters
    # ---------------
    parser.add_argument(
        "--dataset",
        help="input data file path. "
        "If it has a split column, it will be used for splitting "
        "(0: train, 1: validation, 2: test), "
        "otherwise the dataset will be randomly split",
    )
    parser.add_argument("--training_set", help="input train data file path")
    parser.add_argument("--validation_set", help="input validation data file path")
    parser.add_argument("--test_set", help="input test data file path")

    parser.add_argument(
        "--training_set_metadata",
        help="input metadata JSON file path. An intermediate preprocessed file "
        "containing the mappings of the input file created "
        "the first time a file is used, in the same directory "
        "with the same name and a .json extension",
    )

    parser.add_argument(
        "--data_format",
        help="format of the input data",
        default="auto",
        choices=[
            "auto",
            "csv",
            "excel",
            "feather",
            "fwf",
            "hdf5",
            "html" "tables",
            "json",
            "jsonl",
            "parquet",
            "pickle",
            "sas",
            "spss",
            "stata",
            "tsv",
        ],
    )

    parser.add_argument(
        "-sspi",
        "--skip_save_processed_input",
        help="skips saving intermediate HDF5 and JSON files",
        action="store_true",
        default=False,
    )

    # ----------------
    # Model parameters
    # ----------------
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument(
        "-c",
        "--config",
        type=load_yaml,
        help="Path to the YAML file containing the model configuration",
    )
    config.add_argument(
        "-cs",
        "--config_str",
        dest="config",
        type=load_config_from_str,
        help="JSON or YAML serialized string of the model configuration",
    )

    parser.add_argument(
        "-mlp",
        "--model_load_path",
        help="path of a pretrained model to load as initialization",
    )
    parser.add_argument(
        "-mrp",
        "--model_resume_path",
        help="path of the model directory to resume training of",
    )
    parser.add_argument(
        "-sstd",
        "--skip_save_training_description",
        action="store_true",
        default=False,
        help="disables saving the description JSON file",
    )
    parser.add_argument(
        "-ssts",
        "--skip_save_training_statistics",
        action="store_true",
        default=False,
        help="disables saving training statistics JSON file",
    )
    parser.add_argument(
        "-ssm",
        "--skip_save_model",
        action="store_true",
        default=False,
        help="disables saving weights each time the model improves. "
        "By default Ludwig saves  weights after each epoch "
        "the validation metric (improves, but  if the model is really big "
        "that can be time consuming. If you do not want to keep "
        "the weights and just find out what performance a model can get "
        "with a set of hyperparameters, use this parameter to skip it",
    )
    parser.add_argument(
        "-ssp",
        "--skip_save_progress",
        action="store_true",
        default=False,
        help="disables saving weights after each epoch. By default ludwig saves "
        "weights after each epoch for enabling resuming of training, but "
        "if the model is really big that can be time consuming and will "
        "save twice as much space, use this parameter to skip it",
    )
    parser.add_argument(
        "-ssl",
        "--skip_save_log",
        action="store_true",
        default=False,
        help="disables saving TensorBoard logs. By default Ludwig saves "
        "logs for the TensorBoard, but if it is not needed turning it off "
        "can slightly increase the overall speed",
    )

    # ------------------
    # Runtime parameters
    # ------------------
    parser.add_argument(
        "-rs",
        "--random_seed",
        type=int,
        default=42,
        help="a random seed that is going to be used anywhere there is a call "
        "to a random number generator: data splitting, parameter "
        "initialization and training set shuffling",
    )
    parser.add_argument(
        "-hlv",
        "--hyperopt_log_verbosity",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Controls verbosity of ray tune log messages.  Valid values: "
        "0 = silent, 1 = only status updates, 2 = status and brief trial "
        "results, 3 = status and detailed trial results.",
    )
    parser.add_argument("-g", "--gpus", nargs="+", type=int, default=None, help="list of gpus to use")
    parser.add_argument(
        "-gml",
        "--gpu_memory_limit",
        type=float,
        default=None,
        help="maximum memory fraction [0, 1] allowed to allocate per GPU device",
    )
    parser.add_argument(
        "-b",
        "--backend",
        help="specifies backend to use for parallel / distributed execution, "
        "defaults to local execution or Horovod if called using horovodrun",
        choices=ALL_BACKENDS,
    )
    parser.add_argument(
        "-l",
        "--logging_level",
        default="info",
        help="the level of logging to use",
        choices=["critical", "error", "warning", "info", "debug", "notset"],
    )

    add_contrib_callback_args(parser)
    args = parser.parse_args(sys_argv)

    args.callbacks = args.callbacks or []
    for callback in args.callbacks:
        callback.on_cmdline("hyperopt", *sys_argv)

    args.logging_level = get_logging_level_registry()[args.logging_level]
    logging.getLogger("ludwig").setLevel(args.logging_level)
    global logger
    logger = logging.getLogger("ludwig.hyperopt")

    args.backend = initialize_backend(args.backend or args.config.get("backend"))
    if args.backend.is_coordinator():
        print_ludwig("Hyperopt", LUDWIG_VERSION)

    hyperopt_cli(**vars(args))


if __name__ == "__main__":
    cli(sys.argv[1:])
