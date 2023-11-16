"""Generate python docs for tf.lite.

# How to run

```
python build_docs.py --output_dir=/path/to/output
```

"""
import pathlib
from absl import app
from absl import flags
import tensorflow as tf
from tensorflow_docs.api_generator import generate_lib
flags.DEFINE_string('output_dir', '/tmp/lite_api/', 'The path to output the files to')
flags.DEFINE_string('code_url_prefix', 'https://github.com/tensorflow/tensorflow/blob/master/', 'The url prefix for links to code.')
flags.DEFINE_bool('search_hints', True, 'Include metadata search hints in the generated files')
flags.DEFINE_string('site_path', 'lite/api_docs/python', 'Path prefix in the _toc.yaml')
FLAGS = flags.FLAGS

def main(_):
    if False:
        return 10
    doc_generator = generate_lib.DocGenerator(root_title='TensorFlow Lite', py_modules=[('tf.lite', tf.lite)], base_dir=str(pathlib.Path(tf.__file__).parent), code_url_prefix=FLAGS.code_url_prefix, search_hints=FLAGS.search_hints, site_path=FLAGS.site_path, callbacks=[])
    doc_generator.build(output_dir=FLAGS.output_dir)
if __name__ == '__main__':
    app.run(main)