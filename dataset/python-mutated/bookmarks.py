"""
Contains the bookmarsks utilities.
"""
import os.path as osp

def _load_all_bookmarks(slots):
    if False:
        print('Hello World!')
    'Load all bookmarks from config.'
    for slot_num in list(slots.keys()):
        if not osp.isfile(slots[slot_num][0]):
            slots.pop(slot_num)
    return slots

def load_bookmarks(filename, slots):
    if False:
        i = 10
        return i + 15
    'Load all bookmarks for a specific file from config.'
    bookmarks = _load_all_bookmarks(slots)
    return {k: v for (k, v) in bookmarks.items() if v[0] == filename}

def load_bookmarks_without_file(filename, slots):
    if False:
        for i in range(10):
            print('nop')
    'Load all bookmarks but those from a specific file.'
    bookmarks = _load_all_bookmarks(slots)
    return {k: v for (k, v) in bookmarks.items() if v[0] != filename}

def update_bookmarks(filename, bookmarks, old_slots):
    if False:
        while True:
            i = 10
    '\n    Compute an updated version of all the bookmarks from a specific file.\n\n    Parameters\n    ----------\n    filename : str\n        File path that the bookmarks are related too.\n    bookmarks : dict\n        New or changed bookmarks for the file.\n    old_slots : dict\n        Base general bookmarks entries available before any changes where done.\n\n    Returns\n    -------\n    updated_slots : dict\n        Updated general bookmarks.\n\n    '
    if not osp.isfile(filename):
        return
    updated_slots = load_bookmarks_without_file(filename, old_slots)
    for (slot_num, content) in bookmarks.items():
        updated_slots[slot_num] = [filename, content[0], content[1]]
    return updated_slots