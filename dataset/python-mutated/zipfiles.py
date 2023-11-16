import zipfile
import sys
import os

def zip_file(file_path):
    if False:
        print('Hello World!')
    compress_file = zipfile.ZipFile(file_path + '.zip', 'w')
    compress_file.write(path, compress_type=zipfile.ZIP_DEFLATED)
    compress_file.close()

def retrieve_file_paths(dir_name):
    if False:
        for i in range(10):
            print('nop')
    file_paths = []
    for (root, directories, files) in os.walk(dir_name):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

def zip_dir(dir_path, file_paths):
    if False:
        i = 10
        return i + 15
    compress_dir = zipfile.ZipFile(dir_path + '.zip', 'w')
    with compress_dir:
        for file in file_paths:
            compress_dir.write(file)
if __name__ == '__main__':
    path = sys.argv[1]
    if os.path.isdir(path):
        files_path = retrieve_file_paths(path)
        print('The following list of files will be zipped:')
        for file_name in files_path:
            print(file_name)
        zip_dir(path, files_path)
    elif os.path.isfile(path):
        print('The %s will be zipped:' % path)
        zip_file(path)
    else:
        print('a special file(socket,FIFO,device file), please input file or dir')