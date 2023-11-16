"""
FiftyOne v0.16.3 revision.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

def up(db, dataset_name):
    if False:
        print('Hello World!')
    pass

def down(db, dataset_name):
    if False:
        print('Hello World!')
    match_d = {'name': dataset_name}
    dataset_dict = db.datasets.find_one(match_d)
    dataset_dict.pop('tags', None)
    db.datasets.replace_one(match_d, dataset_dict)