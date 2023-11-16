from tests.support.list_file_in_subdir import list_files_in_subdir

def list_trash_dir(trash_dir_path):
    if False:
        for i in range(10):
            print('nop')
    return list_files_in_subdir(trash_dir_path, 'info') + list_files_in_subdir(trash_dir_path, 'files')