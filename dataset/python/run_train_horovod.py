# Copyright (c) 2020 Uber Technologies, Inc.
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
import json
import os
import shutil
import sys

import horovod.torch as hvd
import torch

import ludwig.utils.horovod_utils
from ludwig.api import LudwigModel
from ludwig.constants import BATCH_SIZE, TRAINER

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
PATH_ROOT = os.path.join(PATH_HERE, "..", "..", "..")
sys.path.insert(0, os.path.abspath(PATH_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--rel-path", required=True)
parser.add_argument("--input-features", required=True)
parser.add_argument("--output-features", required=True)
parser.add_argument("--ludwig-kwargs", required=True)


def run_api_experiment(input_features, output_features, dataset, **kwargs):
    config = {
        "input_features": input_features,
        "output_features": output_features,
        "combiner": {"type": "concat", "output_size": 14},
        TRAINER: {"epochs": 2, BATCH_SIZE: 128},
    }

    model = LudwigModel(config)
    output_dir = None

    try:
        # Training with csv
        _, _, output_dir = model.train(dataset=dataset, **kwargs)

        model.predict(dataset=dataset)

        # Attempt loading saved model, should broadcast successfully
        model_dir = os.path.join(output_dir, "model") if output_dir else None
        loaded_model = LudwigModel.load(model_dir)

        # Model loading should broadcast weights from coordinator
        loaded_state = loaded_model.model.state_dict()
        bcast_state = hvd.broadcast_object(loaded_state)
        for loaded, bcast in zip(loaded_state.values(), bcast_state.values()):
            assert torch.allclose(loaded, bcast)
    finally:
        if output_dir:
            shutil.rmtree(output_dir, ignore_errors=True)


def test_horovod_intent_classification(rel_path, input_features, output_features, **kwargs):
    run_api_experiment(input_features, output_features, dataset=rel_path, **kwargs)

    # Horovod should be initialized following training. If not, this will raise an exception.
    assert hvd.size() == 2
    assert ludwig.utils.horovod_utils._HVD.rank() == hvd.rank()


if __name__ == "__main__":
    args = parser.parse_args()
    test_horovod_intent_classification(
        args.rel_path,
        json.loads(args.input_features),
        json.loads(args.output_features),
        **json.loads(args.ludwig_kwargs)
    )
