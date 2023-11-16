from persepolis.scripts import osCommands
import shutil
import ast

def writeList(file_path, list):
    if False:
        return 10
    dictionary = {'list': list}
    f = open(file_path, 'w')
    f.writelines(str(dictionary))
    f.close()

def readList(file_path, mode='dictionary'):
    if False:
        print('Hello World!')
    f = open(file_path, 'r')
    f_string = f.readline()
    f.close()
    dictionary = ast.literal_eval(f_string.strip())
    list = dictionary['list']
    if mode == 'string':
        list[9] = str(list[9])
    return list

def readDict(file_path):
    if False:
        for i in range(10):
            print('nop')
    f = open(file_path)
    f_lines = f.readlines()
    f.close()
    dict_str = str(f_lines[0].strip())
    return_dict = ast.literal_eval(dict_str)
    return return_dict