"""Helper code to run complete models from within python.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import sys
import tempfile
from absl import flags
from absl.testing import flagsaver
from official.utils.flags import core as flags_core

@flagsaver.flagsaver
def run_synthetic(main, tmp_root, extra_flags=None, synth=True, train_epochs=1, epochs_between_evals=1):
    if False:
        for i in range(10):
            print('nop')
    'Performs a minimal run of a model.\n\n    This function is intended to test for syntax errors throughout a model. A\n  very limited run is performed using synthetic data.\n\n  Args:\n    main: The primary function used to exercise a code path. Generally this\n      function is "<MODULE>.main(argv)".\n    tmp_root: Root path for the temp directory created by the test class.\n    extra_flags: Additional flags passed by the caller of this function.\n    synth: Use synthetic data.\n    train_epochs: Value of the --train_epochs flag.\n    epochs_between_evals: Value of the --epochs_between_evals flag.\n  '
    extra_flags = [] if extra_flags is None else extra_flags
    model_dir = tempfile.mkdtemp(dir=tmp_root)
    args = [sys.argv[0], '--model_dir', model_dir] + extra_flags
    if synth:
        args.append('--use_synthetic_data')
    if train_epochs is not None:
        args.extend(['--train_epochs', str(train_epochs)])
    if epochs_between_evals is not None:
        args.extend(['--epochs_between_evals', str(epochs_between_evals)])
    try:
        flags_core.parse_flags(argv=args)
        main(flags.FLAGS)
    finally:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)