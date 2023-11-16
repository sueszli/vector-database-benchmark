import os
from caffe2.python import core
from caffe2.python.db_file_reader import DBFileReader
from caffe2.python.pipeline import pipe
from caffe2.python.task import Cluster, TaskGroup

class CachedReader(DBFileReader):
    default_name_suffix = 'cached_reader'
    "Reader with persistent in-file cache.\n\n    Example usage:\n    cached_reader = CachedReader(\n        reader,\n        db_path='/tmp/cache.db',\n        db_type='LevelDB',\n    )\n    build_cache_step = cached_reader.build_cache_step()\n    with LocalSession() as session:\n        session.run(build_cache_step)\n\n    Every time new CachedReader is created, it's expected that\n    db_path exists before calling .setup_ex(...) and .read(...).\n\n    If db_path doesn't exist, it's expected build_cache_step to be called\n    first to build a cache at db_path.\n\n    build_cache_step will check existence of provided db_path and in case\n    it's missing will initialize it by reading data from original reader.\n    All consequent attempts to read will ignore original reader\n    (i.e. no additional data will be read from it).\n\n    Args:\n        original_reader: Reader.\n            If provided, it's the original reader used to build the cache file.\n        db_path: str.\n\n    Optional Args:\n        db_type: str. DB type of file. A db_type is registed by\n            `REGISTER_CAFFE2_DB(<db_type>, <DB Class>)`.\n            Default to 'LevelDB'.\n        name: str or None. Name of CachedReader.\n            Optional name to prepend to blobs that will store the data.\n            Default to '<db_name>_<default_name_suffix>'.\n        batch_size: int.\n            How many examples are read for each time the read_net is run.\n            Defaults to 100.\n        loop_over: bool.\n            If True given, will go through examples in random order endlessly.\n            Defaults to False.\n    "

    def __init__(self, original_reader, db_path, db_type='LevelDB', name=None, batch_size=100, loop_over=False):
        if False:
            print('Hello World!')
        assert original_reader is not None, "original_reader can't be None"
        self.original_reader = original_reader
        super().__init__(db_path, db_type, name, batch_size, loop_over)

    def _init_reader_schema(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Prepare the reader schema.\n\n            Since an original reader is given,\n            use it's schema as ground truth.\n\n            Returns:\n                schema: schema.Struct. Used in Reader.__init__(...).\n        "
        return self.original_reader._schema

    def build_cache_step(self, overwrite=False):
        if False:
            print('Hello World!')
        'Build a step for generating cache DB file.\n\n            If self.db_path exists and not overwritting, build an empty step.\n            Overwise, build a step as follows.\n            Pipe original reader to the _DatasetWriter,\n            so that dataset field blobs are populated.\n            Then save these blobs into a file.\n\n            Args:\n                overwrite: bool. If true, ignore the existing file\n                    and build a new one overwritting the existing one anyway.\n\n            Returns:\n                build_cache_step: ExecutionStep.\n                    The step to be run for building a cache DB file.\n        '
        if os.path.exists(self.db_path) and (not overwrite):
            return core.execution_step('build_step', [])
        init_net = core.Net('init')
        self._init_field_blobs_as_empty(init_net)
        with Cluster(), core.NameScope(self.name), TaskGroup() as copy_tg:
            pipe(self.original_reader, self.ds.writer(), num_threads=16)
            copy_step = copy_tg.to_task().get_step()
        save_net = core.Net('save')
        self._save_field_blobs_to_db_file(save_net)
        return core.execution_step('build_cache', [init_net, copy_step, save_net])

    def _save_field_blobs_to_db_file(self, net):
        if False:
            i = 10
            return i + 15
        'Save dataset field blobs to a DB file at db_path'
        net.Save(self.ds.get_blobs(), [], db=self.db_path, db_type=self.db_type, blob_name_overrides=self.ds.field_names(), absolute_path=True)