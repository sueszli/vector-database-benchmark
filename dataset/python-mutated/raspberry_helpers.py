import os

def clean_directory(directory):
    if False:
        return 10
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f'Error deleting file: {filepath}')