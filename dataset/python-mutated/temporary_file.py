import os
import tempfile

def cleanup_temporary_file(temp_file_name, verbose=False):
    if False:
        return 10
    '\n    clean up temporary file\n    '
    try:
        os.remove(temp_file_name)
        if verbose:
            print(f'Cleaning up temporary file {temp_file_name}')
            print('---')
    except Exception as e:
        print(f'Could not clean up temporary file.')
        print(e)
        print('')

def create_temporary_file(contents, extension=None, verbose=False):
    if False:
        while True:
            i = 10
    '\n    create a temporary file with the given contents\n    '
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'.{extension}' if extension else '') as f:
            f.write(contents)
            temp_file_name = f.name
            f.close()
        if verbose:
            print(f'Created temporary file {temp_file_name}')
            print('---')
        return temp_file_name
    except Exception as e:
        print(f'Could not create temporary file.')
        print(e)
        print('')