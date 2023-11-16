#
# Copyright 2016 The BigDL Authors.
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
#

# This file is adapted from https://github.com/PyTorchLightning
# /pytorch-lightning/blob/master/pytorch_lightning/plugins/training_type/ddp_spawn.py
#
# Copyright The PyTorch Lightning team.
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

import cloudpickle
import os
import multiprocessing
import subprocess
import sys
import uuid
import json
from typing import Any, Optional, Callable
from tempfile import TemporaryDirectory

import pytorch_lightning as pl

from bigdl.nano.pytorch.strategies.ddp_spawn import DDPSpawnStrategy, _DDPSpawnLauncher
from bigdl.nano.utils.common import schedule_processors
from bigdl.nano.utils.common import invalidInputError
from bigdl.nano.pytorch.dispatcher import _get_patch_status

import logging

log = logging.getLogger(__name__)


class _DDPSubprocessLauncher(_DDPSpawnLauncher):

    @property
    def is_interactive_compatible(self) -> bool:
        return True

    def launch(self, function: Callable, *args: Any,
               trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        # pytorch_lightning 1.6 uses this method to create child processes

        # the `self._strategy.cluster_environment` should not be None in normal circumstances,
        # if you see this error message, please report an issue in BigDL.
        invalidInputError(self._strategy.cluster_environment is not None,
                          'strategy.cluster_environment cannot be None')

        os.environ["MASTER_PORT"] = str(self._strategy.cluster_environment.main_port)

        envs = schedule_processors(self._strategy.num_processes)
        cpu_procs = self._strategy.cpu_for_each_process
        if cpu_procs is not None:
            for i in range(len(cpu_procs)):
                envs[i]["KMP_AFFINITY"] = f"granularity=fine,proclist" + \
                    f"=[{','.join(map(str, cpu_procs[i]))}],explicit"
                envs[i]["OMP_NUM_THREADS"] = str(len(cpu_procs[i]))

        # the `return_queue` is necessary for recovering child process's state, we need
        # to dump it in this process and load it in subprocess, the `mp.SimpleQueue()` in
        # `ddp_spawn.py` cannot be dumped, so we use `multiprocessing.Manager().Queue()` here,
        # however, in order to be able to load it in subprocess, we must ensure the
        # `current_process().authkey` in this process and in subprocess are the same
        authkey = str(uuid.uuid1())
        multiprocessing.current_process().authkey = bytes(authkey, encoding='utf-8')
        mp = multiprocessing.Manager()

        with TemporaryDirectory() as temp_dir:
            return_queue = mp.Queue()
            error_queue = mp.Queue()
            args = (trainer, function, args, kwargs, return_queue)

            # when using trainer, if we dump `trainer` and `self._wrapping_function` at the
            # same time, then after we load them in the subprocess, the loaded `trainer` may
            # be different from the one we dumped sometimes. so now, when using trianer, we
            # don't dump the `self._wrapping_function` and access it through `trainer` in
            # subprocess, when using LightningLite, the `trainer` is None, so we must dump
            # `self._wrapping_function`.
            with open(os.path.join(temp_dir, "args.pkl"), "wb") as f:
                if trainer is not None:
                    cloudpickle.dump((None, args, error_queue), f)
                else:
                    cloudpickle.dump((self._wrapping_function, args, error_queue), f)

            # we also need to pass sys.path to subprocess
            with open(os.path.join(temp_dir, "sys_path.json"), "w") as f:
                json.dump(sys.path, f)

            with open(os.path.join(temp_dir, "patch_status.json"), "w") as f:
                json.dump(_get_patch_status(), f)

            processes = []
            cwd_path = os.path.split(os.path.realpath(__file__))[0]
            for i in range(self._strategy.num_processes):
                envs[i]["AUTHKEY"] = authkey
                processes.append(subprocess.Popen([sys.executable, f"{cwd_path}/worker.py",
                                                   temp_dir], env=envs[i]))

            for _, process in enumerate(processes):
                process.wait()
            for _, process in enumerate(processes):
                if process.returncode != 0:
                    if not error_queue.empty():
                        invalidInputError(False, f"{error_queue.get()}")
                    else:
                        invalidInputError(False, "subprocess exits incorrectly")

            # restore the state of child process
            spawn_output = return_queue.get()

            # when using pytorch lightning's trainer, the `trainer` cannot be None,
            # when using pytorch lightning's LightningLite, the `trainer` should be None
            if trainer is None:
                return spawn_output

            self._recover_results_in_main_process(spawn_output, trainer)
            return spawn_output.trainer_results


class DDPSubprocessStrategy(DDPSpawnStrategy):
    """
    Extending DDPSpawnStrategy to support launch subprocesses with optimized env variables.

    Instead of using python multiprocessing.spawn, this strategy use subprocess.Popen to start
    a new process in order to run mulit-instance training in a jupyter notebook.
    """

    strategy_name = "ddp_subprocess"

    def _configure_launcher(self):
        self._launcher = _DDPSubprocessLauncher(self)
