import argparse
import os
import shutil

def organize_files(directory_path):
    if False:
        return 10
    file_types = {'images': ['.png', '.jpg', '.jpeg'], 'documents': ['.pdf', '.docx', '.txt'], 'audio': ['.mp3', '.wav', '.flac']}
    for folder_name in file_types.keys():
        folder_path = os.path.join(directory_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
    for (foldername, subfolders, filenames) in os.walk(directory_path):
        for filename in filenames:
            (_, file_extension) = os.path.splitext(filename)
            for (folder_name, extensions) in file_types.items():
                if file_extension in extensions:
                    old_path = os.path.join(foldername, filename)
                    new_path = os.path.join(directory_path, folder_name, filename)
                    if old_path != new_path:
                        shutil.move(old_path, new_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Organize files in a directory based on their file types')
    parser.add_argument('--directory_path', type=str, required=True, help='The path of the directory to be organized')
    args = parser.parse_args()
    organize_files(args.directory_path)