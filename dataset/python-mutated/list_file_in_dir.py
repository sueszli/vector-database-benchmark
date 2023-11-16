import os

def list_files_in_dir(path):
    if False:
        i = 10
        return i + 15
    if not os.path.isdir(path):
        return []
    return os.listdir(path)