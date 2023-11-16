"""A tool to generate api_docs for TensorFlow2.

```
python generate2.py --output_dir=/tmp/out
```

Requires a local installation of `tensorflow_docs`:

```
pip install git+https://github.com/tensorflow/docs
```
"""
import contextlib
import pathlib
import textwrap
from typing import NamedTuple
from absl import app
from absl import flags
from packaging import version
import tensorflow as tf
from tensorflow_docs.api_generator import doc_controls
from tensorflow_docs.api_generator import doc_generator_visitor
from tensorflow_docs.api_generator import generate_lib
from tensorflow_docs.api_generator.pretty_docs import base_page
from tensorflow_docs.api_generator.pretty_docs import module_page
import yaml
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_export
from tensorflow.python.util import tf_inspect
if version.parse(tf.__version__) >= version.parse('2.14-dev'):
    from tensorflow.python.util.pywrap_xla_ops import get_gpu_kernel_names
import base_dir
try:
    from tensorflow.python.types import doc_typealias
    _EXTRA_DOCS = getattr(doc_typealias, '_EXTRA_DOCS', {})
    del doc_typealias
except ImportError:
    _EXTRA_DOCS = {}
tf.__all__ = [item_name for (item_name, value) in tf_inspect.getmembers(tf)]
tf.compat.v2 = tf
tf.losses = tf.keras.losses
tf.metrics = tf.keras.metrics
tf.optimizers = tf.keras.optimizers
tf.initializers = tf.keras.initializers
MIN_NUM_FILES_EXPECTED = 2000
FLAGS = flags.FLAGS
flags.DEFINE_string('code_url_prefix', '/code/stable/tensorflow', 'A url to prepend to code paths when creating links to defining code')
flags.DEFINE_string('output_dir', '/tmp/out', 'A directory, where the docs will be output to.')
flags.DEFINE_bool('search_hints', True, 'Include meta-data search hints at the top of each file.')
flags.DEFINE_string('site_path', '', 'The path prefix (up to `.../api_docs/python`) used in the `_toc.yaml` and `_redirects.yaml` files')
_PRIVATE_MAP = {'tf': ['python', 'core', 'compiler', 'examples', 'tools', 'contrib'], 'tf.compat.v1.compat': ['v1', 'v2'], 'tf.compat.v2.compat': ['v1', 'v2']}
tf.__doc__ = '\n  ## TensorFlow\n\n  ```\n  pip install tensorflow\n  ```\n  '

class RawOpsPageInfo(module_page.ModulePageInfo):
    """Generates a custom page for `tf.raw_ops`."""
    DEFAULT_BUILDER_CLASS = base_page.TemplatePageBuilder

    def build(self):
        if False:
            print('Hello World!')
        content = base_page.PageInfo.build(self)
        if version.parse(tf.__version__) >= version.parse('2.14-dev'):
            raw_ops_doc = self.generate_raw_ops_doc_ge_214()
        else:
            raw_ops_doc = self.generate_raw_ops_doc_lt_214()
        return '\n'.join([content, raw_ops_doc])

    def generate_raw_ops_doc_lt_214(self):
        if False:
            i = 10
            return i + 15
        'Generates docs for `tf.raw_ops`.'
        del self
        warning = textwrap.dedent('\n\n      Note: `tf.raw_ops` provides direct/low level access to all TensorFlow ops.\n      See [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20181225-tf-raw-ops.md)\n      for details. Unless you are library writer, you likely do not need to use\n      these ops directly.')
        table_header = textwrap.dedent('\n\n        | Op Name | Has Gradient |\n        |---------|:------------:|')
        parts = [warning, table_header]
        for op_name in sorted(dir(tf.raw_ops)):
            try:
                ops._gradient_registry.lookup(op_name)
                has_gradient = '✔️'
            except LookupError:
                has_gradient = '❌'
            if not op_name.startswith('_'):
                path = pathlib.Path('/') / FLAGS.site_path / 'tf/raw_ops' / op_name
                path = path.with_suffix('.md')
                link = '<a id={op_name} href="{path}">{op_name}</a>'.format(op_name=op_name, path=str(path))
                parts.append('| {link} | {has_gradient} |'.format(link=link, has_gradient=has_gradient))
        return '\n'.join(parts)

    def generate_raw_ops_doc_ge_214(self):
        if False:
            while True:
                i = 10
        'Generates docs for `tf.raw_ops`.'
        del self
        warning = textwrap.dedent('\n\n      Note: `tf.raw_ops` provides direct/low level access to all TensorFlow ops.\n      See [the RFC](https://github.com/tensorflow/community/blob/master/rfcs/20181225-tf-raw-ops.md)\n      for details. Unless you are library writer, you likely do not need to use\n      these ops directly.')
        table_header = textwrap.dedent('\n\n        | Op Name | Has Gradient | GPU XLA Support |\n        |---------|:------------:|:---------------:|')
        parts = [warning, table_header]
        xla_compiled_ops = get_gpu_kernel_names()
        for op_name in sorted(dir(tf.raw_ops)):
            try:
                ops._gradient_registry.lookup(op_name)
                has_gradient = '✔️'
            except LookupError:
                has_gradient = '❌'
            is_xla_compilable = '❌'
            if op_name in xla_compiled_ops:
                is_xla_compilable = '✔️'
            if not op_name.startswith('_'):
                path = pathlib.Path('/') / FLAGS.site_path / 'tf/raw_ops' / op_name
                path = path.with_suffix('.md')
                link = '<a id={op_name} href="{path}">{op_name}</a>'.format(op_name=op_name, path=str(path))
                parts.append('| {link} | {has_gradient} | {is_xla_compilable} |'.format(link=link, has_gradient=has_gradient, is_xla_compilable=is_xla_compilable))
        return '\n'.join(parts)

class TfExportAwareVisitor(doc_generator_visitor.DocGeneratorVisitor):
    """A `tf_export`, `keras_export` and `estimator_export` aware doc_visitor."""

    class TfNameScore(NamedTuple):
        cannonical_score: int
        name_score: doc_generator_visitor.DocGeneratorVisitor.NameScore

    def _score_name(self, path: doc_generator_visitor.ApiPath) -> TfNameScore:
        if False:
            print('Hello World!')
        name = '.'.join(path)
        all_exports = [tf_export.TENSORFLOW_API_NAME, tf_export.KERAS_API_NAME, tf_export.ESTIMATOR_API_NAME]
        for api_name in all_exports:
            try:
                canonical = tf_export.get_canonical_name_for_symbol(self._index[name], api_name=api_name)
            except AttributeError:
                canonical = None
            if canonical is not None:
                break
        canonical_score = 1
        if canonical is not None and name == 'tf.' + canonical:
            canonical_score = -1
        return self.TfNameScore(canonical_score, super()._score_name(path))

def build_docs(output_dir, code_url_prefix, search_hints):
    if False:
        while True:
            i = 10
    'Build api docs for tensorflow v2.\n\n  Args:\n    output_dir: A string path, where to put the files.\n    code_url_prefix: prefix for "Defined in" links.\n    search_hints: Bool. Include meta-data search hints at the top of each file.\n  '
    output_dir = pathlib.Path(output_dir)
    site_path = pathlib.Path('/', FLAGS.site_path)
    if version.parse(tf.__version__) >= version.parse('2.9'):
        doc_controls.set_deprecated(tf.compat.v1)
        doc_controls.set_deprecated(tf.estimator)
        doc_controls.set_deprecated(tf.feature_column)
        doc_controls.set_deprecated(tf.keras.preprocessing)
    doc_controls.set_custom_page_builder_cls(tf.raw_ops, RawOpsPageInfo)
    for (name, obj) in tf_inspect.getmembers(tf.raw_ops):
        if not name.startswith('_'):
            doc_controls.hide_from_search(obj)
    for cls in [tf.Module, tf.keras.layers.Layer, tf.keras.optimizers.Optimizer]:
        doc_controls.decorate_all_class_attributes(decorator=doc_controls.do_not_doc_in_subclasses, cls=cls, skip=['__init__'])
    do_not_document = ['tf.__internal__', 'tf.keras.__internal__', 'tf.keras.wrappers', 'tf.__operators__', 'tf.tools', 'tf.compat.v1.pywrap_tensorflow', 'tf.pywrap_tensorflow', 'tf.flags', 'tf.batch_mat_mul_v3', 'tf.sparse_segment_sum_grad']
    for path in do_not_document:
        item = tf
        for part in path.split('.')[1:]:
            item = getattr(item, part, None)
        if item is None:
            continue
        doc_controls.do_not_generate_docs(item)
    (base_dirs, code_url_prefixes) = base_dir.get_base_dirs_and_prefixes(code_url_prefix)
    doc_generator = generate_lib.DocGenerator(root_title='TensorFlow 2', py_modules=[('tf', tf)], base_dir=base_dirs, search_hints=search_hints, code_url_prefix=code_url_prefixes, site_path=site_path, visitor_cls=TfExportAwareVisitor, private_map=_PRIVATE_MAP, extra_docs=_EXTRA_DOCS, callbacks=base_dir.get_callbacks())
    doc_generator.build(output_dir)

    @contextlib.contextmanager
    def edit_yaml_file(path):
        if False:
            for i in range(10):
                print('nop')
        content = yaml.safe_load(path.read_text())
        yield content
        with path.open('w') as f:
            yaml.dump(content, f, default_flow_style=False)
    toc_path = output_dir / 'tf/_toc.yaml'
    with edit_yaml_file(toc_path) as toc:
        toc['toc'][0]['section'][0]['path'] = str(site_path / 'tf_overview')
    redirects_path = output_dir / 'tf/_redirects.yaml'
    with edit_yaml_file(redirects_path) as redirects:
        redirects['redirects'].append({'from': str(site_path / 'tf_overview'), 'to': str(site_path / 'tf')})
    expected_path_contents = {'tf/summary/audio.md': 'tensorboard/plugins/audio/summary_v2.py', 'tf/estimator/DNNClassifier.md': 'tensorflow_estimator/python/estimator/canned/dnn.py', 'tf/nn/sigmoid_cross_entropy_with_logits.md': 'python/ops/nn_impl.py', 'tf/keras/Model.md': 'engine/training.py'}
    all_passed = True
    error_msg_parts = ['Some "view source" links seem to be broken, please check:']
    for (rel_path, contents) in expected_path_contents.items():
        path = output_dir / rel_path
        if contents not in path.read_text():
            all_passed = False
            error_msg_parts.append('  ' + str(path))
    if not all_passed:
        raise ValueError('\n'.join(error_msg_parts))
    rejected_path_contents = {'tf/keras/optimizers.md': 'api/_v2/keras/optimizers/__init__.py'}
    all_passed = True
    error_msg_parts = ['Bad "view source" links in generated files, please check:']
    for (rel_path, content) in rejected_path_contents.items():
        path = output_dir / rel_path
        if content in path.read_text():
            all_passed = False
            error_msg_parts.append('  ' + str(path))
    if not all_passed:
        raise ValueError('\n'.join(error_msg_parts))
    num_files = len(list(output_dir.rglob('*')))
    if num_files < MIN_NUM_FILES_EXPECTED:
        raise ValueError(f'The TensorFlow api should be more than {MIN_NUM_FILES_EXPECTED} files(found {num_files}).')

def main(argv):
    if False:
        while True:
            i = 10
    del argv
    build_docs(output_dir=FLAGS.output_dir, code_url_prefix=FLAGS.code_url_prefix, search_hints=FLAGS.search_hints)
if __name__ == '__main__':
    app.run(main)