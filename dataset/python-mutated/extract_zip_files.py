import os
import zipfile
import sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--zippedfile', required=True, help='Zipped file')
args = vars(parser.parse_args())
zip_file = args['zippedfile']
file_name = zip_file
if os.path.exists(zip_file) == False:
    sys.exit('No such file present in the directory')

def extract(zip_file):
    if False:
        return 10
    file_name = zip_file.split('.zip')[0]
    if zip_file.endswith('.zip'):
        current_working_directory = os.getcwd()
        new_directory = current_working_directory + '/' + file_name
        with zipfile.ZipFile(zip_file, 'r') as zip_object:
            zip_object.extractall(new_directory)
        print('Extracted successfully!!!')
    else:
        print('Not a zip file')
extract(zip_file)