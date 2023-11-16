"""Generate Java reference docs for TensorFlow.org."""
import pathlib
import shutil
import subprocess
import tempfile
from absl import app
from absl import flags
from tensorflow_docs.api_generator import gen_java
FLAGS = flags.FLAGS
flags.DEFINE_string('output_dir', None, "Use this branch as the root version and don't create in version directory")
flags.DEFINE_string('site_path', 'api_docs/java', 'Path prefix in the _toc.yaml')
flags.DEFINE_string('code_url_prefix', None, '[UNUSED] The url prefix for links to code.')
flags.DEFINE_bool('search_hints', True, '[UNUSED] Include metadata search hints in the generated files')
flags.DEFINE_bool('gen_ops', True, 'enable/disable bazel-generated ops')
DOCS_TOOLS_DIR = pathlib.Path(__file__).resolve().parent
TENSORFLOW_ROOT = DOCS_TOOLS_DIR.parents[2]
SOURCE_PATH = TENSORFLOW_ROOT / 'tensorflow/java/src/main/java'
OP_SOURCE_PATH = TENSORFLOW_ROOT / 'bazel-bin/tensorflow/java/ops/src/main/java/org/tensorflow/op'

def main(unused_argv):
    if False:
        i = 10
        return i + 15
    merged_source = pathlib.Path(tempfile.mkdtemp())
    shutil.copytree(SOURCE_PATH, merged_source / 'java')
    if FLAGS.gen_ops:
        yes = subprocess.Popen(['yes', ''], stdout=subprocess.PIPE)
        configure = subprocess.Popen([TENSORFLOW_ROOT / 'configure'], stdin=yes.stdout, cwd=TENSORFLOW_ROOT)
        configure.communicate()
        subprocess.check_call(['bazel', 'build', '//tensorflow/java:java_op_gen_sources'], cwd=TENSORFLOW_ROOT)
        shutil.copytree(OP_SOURCE_PATH, merged_source / 'java/org/tensorflow/ops')
    gen_java.gen_java_docs(package='org.tensorflow', source_path=merged_source / 'java', output_dir=pathlib.Path(FLAGS.output_dir), site_path=pathlib.Path(FLAGS.site_path))
if __name__ == '__main__':
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)