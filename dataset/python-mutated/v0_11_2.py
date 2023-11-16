"""
FiftyOne v0.11.2 revision.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
logger = logging.getLogger(__name__)

def up(db, dataset_name):
    if False:
        i = 10
        return i + 15
    pass

def down(db, dataset_name):
    if False:
        print('Hello World!')
    match_d = {'name': dataset_name}
    dataset_dict = db.datasets.find_one(match_d)
    evaluations = dataset_dict.get('evaluations', {})
    for (eval_key, run_doc) in evaluations.items():
        version = run_doc.pop('version', '????')
        logger.warning("You may not be able to use evaluation results with key '%s' from v%s on dataset '%s' in older versions of FiftyOne", eval_key, version, dataset_name)
    brain_methods = dataset_dict.get('brain_methods', {})
    for (brain_key, run_doc) in brain_methods.items():
        version = run_doc.pop('version', '????')
        logger.warning("You may not be able to use brain results with key '%s' from from v%s on dataset '%s' in older versions of FiftyOne", brain_key, version, dataset_name)
    db.datasets.replace_one(match_d, dataset_dict)