"""Offline dump analyzer of TensorFlow Debugger (tfdbg)."""
import argparse
import sys
from absl import app
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.lib import debug_data

def main(_):
    if False:
        i = 10
        return i + 15
    if not FLAGS.dump_dir:
        print('ERROR: dump_dir flag is empty.', file=sys.stderr)
        sys.exit(1)
    print('tfdbg offline: FLAGS.dump_dir = %s' % FLAGS.dump_dir)
    debug_dump = debug_data.DebugDumpDir(FLAGS.dump_dir, validate=FLAGS.validate_graph)
    cli = analyzer_cli.create_analyzer_ui(debug_dump, tensor_filters={'has_inf_or_nan': debug_data.has_inf_or_nan}, ui_type=FLAGS.ui_type)
    title = 'tfdbg offline @ %s' % FLAGS.dump_dir
    cli.run_ui(title=title, title_color='black_on_white', init_command='lt')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--dump_dir', type=str, default='', help='tfdbg dump directory to load')
    parser.add_argument('--log_usage', type='bool', nargs='?', const=True, default=True, help='Whether the usage of this tool is to be logged')
    parser.add_argument('--ui_type', type=str, default='readline', help='Command-line user interface type (only readline is supported)')
    parser.add_argument('--validate_graph', nargs='?', const=True, type='bool', default=True, help='      Whether the dumped tensors will be validated against the GraphDefs      ')
    (FLAGS, unparsed) = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)