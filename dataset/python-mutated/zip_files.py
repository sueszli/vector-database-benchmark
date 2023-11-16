"""Creates a zip package of the files passed in."""
import os
import zipfile
from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('export_zip_path', None, 'Path to zip file.')
flags.DEFINE_string('file_directory', None, 'Path to the files to be zipped.')

def main(_):
    if False:
        return 10
    with zipfile.ZipFile(FLAGS.export_zip_path, mode='w') as zf:
        for (root, _, files) in os.walk(FLAGS.file_directory):
            for f in files:
                if f.endswith('.java'):
                    zf.write(os.path.join(root, f))
if __name__ == '__main__':
    app.run(main)