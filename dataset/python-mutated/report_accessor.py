"""Store and retrieve adanet.IterationReport protos.

Copyright 2018 The AdaNet Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
from absl import logging
from adanet import subnetwork
import numpy as np
import six
import tensorflow.compat.v2 as tf

def _json_report_to_materialized_report(iteration_report_json):
    if False:
        i = 10
        return i + 15
    'Converts a JSON loaded iteration report to a `MaterializedReport` list.'
    subnetwork_reports = []
    for subnetwork_report_json in iteration_report_json['subnetwork_reports']:
        subnetwork_reports.append(subnetwork.MaterializedReport(iteration_number=int(iteration_report_json['iteration_number']), name=subnetwork_report_json['name'], hparams=subnetwork_report_json['hparams'], attributes=subnetwork_report_json['attributes'], metrics=subnetwork_report_json['metrics'], included_in_final_ensemble=subnetwork_report_json['included_in_final_ensemble']))
    return subnetwork_reports

def _validate_report_dict(dictionary):
    if False:
        print('Hello World!')
    'Validates that entries of a MaterializedReport dictionary field.'
    for (key, value) in dictionary.items():
        if isinstance(value, np.integer):
            dictionary[key] = int(value)
        if isinstance(value, np.float):
            dictionary[key] = float(value)
        if isinstance(value, (six.string_types, six.binary_type)):
            if six.PY2:
                if not isinstance(value, six.text_type):
                    dictionary[key] = six.u(value).encode('utf-8')
            if six.PY3:
                dictionary[key] = str(dictionary[key])
        elif not isinstance(value, (bool, six.text_type, int, float)):
            raise ValueError('Values must be a binary type (str in python 2; bytes in python 3), a text type (unicode in python 2; str in python 3), int, bool, or float, but its type is {}.'.format(type(value)))
    return dictionary

def _subnetwork_report_to_dict(subnetwork_report):
    if False:
        return 10
    'Converts a Subnetwork report to a JSON serializable dict.'
    return {'name': subnetwork_report.name, 'hparams': _validate_report_dict(subnetwork_report.hparams), 'attributes': _validate_report_dict(subnetwork_report.attributes), 'metrics': _validate_report_dict(subnetwork_report.metrics), 'included_in_final_ensemble': subnetwork_report.included_in_final_ensemble}

class _ReportAccessor(object):
    """Store and retrieve report JSON files."""

    def __init__(self, report_dir, filename='iteration_reports.json'):
        if False:
            return 10
        'Creates a `_ReportAccessor` instance.\n\n    Args:\n      report_dir: Directory to store the report.\n      filename: Name of the file.\n\n    Returns:\n      A `_ReportAccessor` instance.\n    '
        tf.io.gfile.makedirs(report_dir)
        self._full_filepath = os.path.join(report_dir, filename)

    def write_iteration_report(self, iteration_number, materialized_reports):
        if False:
            while True:
                i = 10
        "Writes an iteration's `MaterializedReports` to a JSON file.\n\n    TODO: Remove iteration_number from the argument of this method.\n\n    Note that even materialized_reports also contain iteration\n    number, those are ignored -- only the iteration_number that is passed into\n    this method would be written to the proto.\n\n    Args:\n      iteration_number: Int for the iteration number.\n      materialized_reports: A list of `adanet.subnetwork.MaterializedReport`\n        objects.\n    "
        iteration_report = {'iteration_number': int(iteration_number), 'subnetwork_reports': list(map(_subnetwork_report_to_dict, materialized_reports))}
        self._append_iteration_report_json(iteration_report)
        logging.info('Wrote IterationReport for iteration %s to %s', iteration_number, self._full_filepath)

    def _append_iteration_report_json(self, iteration_report):
        if False:
            return 10
        'Appends an iteration report dictionary object to the output file.'
        iteration_reports = []
        if os.path.exists(self._full_filepath):
            with open(self._full_filepath, 'r') as f:
                iteration_reports = json.load(f)
        iteration_reports.append(iteration_report)
        with open(self._full_filepath, 'w') as f:
            json.dump(iteration_reports, f)

    def read_iteration_reports(self):
        if False:
            return 10
        'Reads all iterations of the Report.\n\n    Each `adanet.subnetwork.MaterializedReport` list is one AdaNet iteration.\n    The first list in the sequence is iteration 0, followed by iteration 1, and\n    so on.\n\n    Returns:\n      Iterable of lists of `adanet.subnetwork.MaterializedReport`s.\n    '
        if os.path.exists(self._full_filepath):
            with open(self._full_filepath, 'r') as f:
                iteration_reports = json.load(f)
            return [_json_report_to_materialized_report(ir) for ir in iteration_reports]
        return []