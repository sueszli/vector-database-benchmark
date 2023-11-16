"""Generate C++ reference docs for TensorFlow.org."""
import os
import pathlib
import subprocess
from absl import app
from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, "Use this branch as the root version and don't create in version directory")
DOCS_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
TENSORFLOW_ROOT = DOCS_TOOLS_DIR.parents[2]

def build_headers(output_dir):
    if False:
        print('Hello World!')
    'Builds the headers files for TF.'
    os.makedirs(output_dir, exist_ok=True)
    yes = subprocess.Popen(['yes', ''], stdout=subprocess.PIPE)
    configure = subprocess.Popen([TENSORFLOW_ROOT / 'configure'], stdin=yes.stdout, cwd=TENSORFLOW_ROOT)
    configure.communicate()
    subprocess.check_call(['bazel', 'build', 'tensorflow/cc:cc_ops'], cwd=TENSORFLOW_ROOT)
    subprocess.check_call(['cp', '--dereference', '-r', 'bazel-bin', output_dir / 'bazel-genfiles'], cwd=TENSORFLOW_ROOT)

def main(argv):
    if False:
        print('Hello World!')
    del argv
    build_headers(pathlib.Path(FLAGS.output_dir))
if __name__ == '__main__':
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)