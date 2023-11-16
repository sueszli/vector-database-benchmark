#!/usr/bin/env python
# ******************************************************************************
# Copyright 2014-2018 Intel Corporation
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
# ******************************************************************************
import os
from neon.util.argparser import NeonArgparser
from neon.optimizers import GradientDescentMomentum, Schedule
from neon.transforms import TopKMisclassification
from neon.callbacks.callbacks import Callbacks

from data import make_alexnet_train_loader, make_validation_loader
from network_allcnn import create_network


# parse the command line arguments (generates the backend)
train_config = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'train.cfg')
config_files = [train_config] if os.path.exists(train_config) else []

parser = NeonArgparser(__doc__, default_config_files=config_files,
                       default_overrides=dict(batch_size=64))
parser.add_argument('--deconv', action='store_true',
                    help='save visualization data from deconvolution')
parser.add_argument('--subset_pct', type=float, default=100,
                    help='subset of training dataset to use (percentage)')
args = parser.parse_args()

model, cost = create_network()
rseed = 0 if args.rng_seed is None else args.rng_seed

# setup data provider
assert 'train' in args.manifest, "Missing train manifest"
assert 'val' in args.manifest, "Missing validation manifest"
train = make_alexnet_train_loader(args.manifest['train'], args.manifest_root,
                                  model.be, args.subset_pct, rseed)
valid = make_validation_loader(args.manifest['val'], args.manifest_root,
                               model.be, args.subset_pct)

sched_weight = Schedule([10], change=0.1)
opt = GradientDescentMomentum(0.01, 0.9, wdecay=0.0005, schedule=sched_weight)

# configure callbacks
valmetric = TopKMisclassification(k=5)
callbacks = Callbacks(model, eval_set=valid, metric=valmetric, **args.callback_args)

if args.deconv:
    callbacks.add_deconv_callback(train, valid)

model.fit(train, optimizer=opt, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
