import Image
import subprocess
import util
import errors
tesseract_exe_name = 'tesseract'
scratch_image_name = 'temp.bmp'
scratch_text_name_root = 'temp'
cleanup_scratch_flag = False

def call_tesseract(input_filename, output_filename, bool_digits=False):
    if False:
        print('Hello World!')
    "Calls external tesseract.exe on input file (restrictions on types),\n    outputting output_filename+'txt'"
    if bool_digits:
        args = tesseract_exe_name + ' ' + input_filename + ' ' + output_filename + ' -l test_digits -psm 7 nobatch'
    else:
        args = tesseract_exe_name + ' ' + input_filename + ' ' + output_filename + ' -l eng -psm 7 nobatch eng_characters'
    proc = subprocess.Popen(args, shell=True)
    retcode = proc.wait()
    if retcode != 0:
        errors.check_for_errors()

def image_to_string(im, cleanup=cleanup_scratch_flag, bool_digits=False):
    if False:
        for i in range(10):
            print('nop')
    'Converts im to file, applies tesseract, and fetches resulting text.\n    If cleanup=True, delete scratch files after operation.'
    try:
        util.image_to_scratch(im, scratch_image_name)
        call_tesseract(scratch_image_name, scratch_text_name_root, bool_digits)
        text = util.retrieve_text(scratch_text_name_root)
    finally:
        if cleanup:
            util.perform_cleanup(scratch_image_name, scratch_text_name_root)
    return text

def image_file_to_string(filename, cleanup=cleanup_scratch_flag, graceful_errors=True, bool_digits=False):
    if False:
        return 10
    'Applies tesseract to filename; or, if image is incompatible and graceful_errors=True,\n    converts to compatible format and then applies tesseract.  Fetches resulting text.\n    If cleanup=True, delete scratch files after operation.'
    try:
        try:
            call_tesseract(filename, scratch_text_name_root, bool_digits)
            text = util.retrieve_text(scratch_text_name_root)
        except errors.Tesser_General_Exception:
            if graceful_errors:
                im = Image.open(filename)
                text = image_to_string(im, cleanup, bool_digits)
            else:
                raise
    finally:
        if cleanup:
            util.perform_cleanup(scratch_image_name, scratch_text_name_root)
    return text