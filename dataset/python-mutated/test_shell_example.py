from pyflink.testing.test_case_utils import PyFlinkTestCase

class ShellExampleTests(PyFlinkTestCase):
    """
    If these tests failed, please fix these examples code and copy them to shell.py
    """

    def test_stream_case(self):
        if False:
            return 10
        from pyflink.shell import s_env, st_env, DataTypes
        from pyflink.table.schema import Schema
        from pyflink.table.table_descriptor import TableDescriptor, FormatDescriptor
        import tempfile
        import os
        import shutil
        sink_path = tempfile.gettempdir() + '/streaming.csv'
        if os.path.exists(sink_path):
            if os.path.isfile(sink_path):
                os.remove(sink_path)
            else:
                shutil.rmtree(sink_path)
        s_env.set_parallelism(1)
        t = st_env.from_elements([(1, 'hi', 'hello'), (2, 'hi', 'hello')], ['a', 'b', 'c'])
        st_env.create_temporary_table('stream_sink', TableDescriptor.for_connector('filesystem').schema(Schema.new_builder().column('a', DataTypes.BIGINT()).column('b', DataTypes.STRING()).column('c', DataTypes.STRING()).build()).option('path', sink_path).format(FormatDescriptor.for_format('csv').option('field-delimiter', ',').build()).build())
        from pyflink.table.expressions import col
        t.select(col('a') + 1, col('b'), col('c')).execute_insert('stream_sink').wait()
        with open(os.path.join(sink_path, os.listdir(sink_path)[0]), 'r') as f:
            lines = f.read()
            self.assertEqual(lines, '2,hi,hello\n' + '3,hi,hello\n')