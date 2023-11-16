import os

def remove_dir_if_exists(dir):
    if False:
        return 10
    if os.path.exists(dir):
        os.rmdir(dir)