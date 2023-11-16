"""TFLite model associated files copier.

A TFLite model with metadata may have 'associated files': extra data stored as a
zip file appended to the model. This script copies such files from one model to
another. This is used as part of constructing validation models.

See https://www.tensorflow.org/lite/convert/metadata for description of models
with metadata.

If there are no associated files in the provided file, the model is output
as-is.
"""
import argparse
import sys
import zipfile
parser = argparse.ArgumentParser(description='Script to generate a metrics model for mobilenet v1.')
parser.add_argument('model', help='Input model filepath')
parser.add_argument('copy_associated_files_from', help='Model with potential associated files filepath')
parser.add_argument('output', help='Output filepath')

def main(model_path, associated_files_path, output_path):
    if False:
        print('Hello World!')
    with open(model_path, 'rb') as input_file:
        with open(output_path, 'wb') as output_file:
            output_file.write(input_file.read())
    if zipfile.is_zipfile(associated_files_path):
        zip_src = zipfile.ZipFile(associated_files_path, 'r')
        zip_tgt = zipfile.ZipFile(output_path, 'a')
        for info in zip_src.infolist():
            zip_tgt.writestr(info, zip_src.read(info))
if __name__ == '__main__':
    (flags, unparsed) = parser.parse_known_args()
    if unparsed:
        parser.print_usage()
        sys.stderr.write('\nGot the following unparsed args, %r please fix.\n' % unparsed)
        exit(1)
    else:
        main(flags.model, flags.copy_associated_files_from, flags.output)
        exit(0)