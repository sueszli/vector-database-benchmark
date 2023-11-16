from __future__ import annotations
import io
import yaml
from ansible.parsing.yaml.dumper import AnsibleDumper

class YamlTestUtils(object):
    """Mixin class to combine with a unittest.TestCase subclass."""

    def _loader(self, stream):
        if False:
            i = 10
            return i + 15
        'Vault related tests will want to override this.\n\n        Vault cases should setup a AnsibleLoader that has the vault password.'

    def _dump_stream(self, obj, stream, dumper=None):
        if False:
            while True:
                i = 10
        'Dump to a py2-unicode or py3-string stream.'
        return yaml.dump(obj, stream, Dumper=dumper)

    def _dump_string(self, obj, dumper=None):
        if False:
            while True:
                i = 10
        'Dump to a py2-unicode or py3-string'
        return yaml.dump(obj, Dumper=dumper)

    def _dump_load_cycle(self, obj):
        if False:
            print('Hello World!')
        string_from_object_dump = self._dump_string(obj, dumper=AnsibleDumper)
        stream_from_object_dump = io.StringIO(string_from_object_dump)
        loader = self._loader(stream_from_object_dump)
        obj_2 = loader.get_data()
        string_from_object_dump_2 = self._dump_string(obj_2, dumper=AnsibleDumper)
        self.assertEqual(string_from_object_dump, string_from_object_dump_2)
        self.assertEqual(obj, obj_2)
        stream_3 = io.StringIO(string_from_object_dump_2)
        loader_3 = self._loader(stream_3)
        obj_3 = loader_3.get_data()
        string_from_object_dump_3 = self._dump_string(obj_3, dumper=AnsibleDumper)
        self.assertEqual(obj, obj_3)
        self.assertEqual(obj_2, obj_3)
        self.assertEqual(string_from_object_dump, string_from_object_dump_3)