import os
import shutil
import time
from seleniumbase.config import settings
from seleniumbase.fixtures import constants
DOWNLOADS_DIR = constants.Files.DOWNLOADS_FOLDER
abs_path = os.path.abspath('.')
downloads_path = os.path.join(abs_path, DOWNLOADS_DIR)

def get_downloads_folder():
    if False:
        for i in range(10):
            print('nop')
    return downloads_path

def reset_downloads_folder():
    if False:
        return 10
    'Clears the downloads folder.\n    If settings.ARCHIVE_EXISTING_DOWNLOADS is set to True, archives it.'
    downloads_dir = constants.Files.DOWNLOADS_FOLDER
    archive_dir = constants.Files.ARCHIVED_DOWNLOADS_FOLDER
    if downloads_dir.endswith('/'):
        downloads_dir = downloads_dir[:-1]
    if downloads_dir.startswith('/'):
        downloads_dir = downloads_dir[1:]
    if archive_dir.endswith('/'):
        archive_dir = archive_dir[:-1]
    if archive_dir.startswith('/'):
        archive_dir = archive_dir[1:]
    if len(downloads_dir) < 10 or len(archive_dir) < 10:
        return
    archived_downloads_folder = os.path.join(os.getcwd(), archive_dir) + os.sep
    if os.path.exists(downloads_path) and (not os.listdir(downloads_path) == []):
        reset_downloads_folder_assistant(archived_downloads_folder)
    if os.path.exists(downloads_path) and os.listdir(downloads_path) == []:
        try:
            os.rmdir(downloads_path)
        except OSError:
            pass
    if os.path.exists(archived_downloads_folder) and os.listdir(archived_downloads_folder) == []:
        try:
            os.rmdir(archived_downloads_folder)
        except OSError:
            pass

def reset_downloads_folder_assistant(archived_downloads_folder):
    if False:
        return 10
    if not os.path.exists(archived_downloads_folder):
        try:
            os.makedirs(archived_downloads_folder, exist_ok=True)
        except Exception:
            pass
    new_archived_downloads_sub_folder = '%s/downloads_%s' % (archived_downloads_folder, int(time.time()))
    if os.path.exists(downloads_path):
        if not os.listdir(downloads_path) == []:
            try:
                shutil.move(downloads_path, new_archived_downloads_sub_folder)
                os.makedirs(downloads_path, exist_ok=True)
            except Exception:
                pass
    if not settings.ARCHIVE_EXISTING_DOWNLOADS:
        try:
            shutil.rmtree(new_archived_downloads_sub_folder)
        except OSError:
            pass