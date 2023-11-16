from caffe2.python import core, scope, workspace, _import_c_extension as C
from caffe2.python.dataio import Reader
from caffe2.python.dataset import Dataset
from caffe2.python.schema import from_column_list
import os

class DBFileReader(Reader):
    default_name_suffix = 'db_file_reader'
    "Reader reads from a DB file.\n\n    Example usage:\n    db_file_reader = DBFileReader(db_path='/tmp/cache.db', db_type='LevelDB')\n\n    Args:\n        db_path: str.\n        db_type: str. DB type of file. A db_type is registed by\n            `REGISTER_CAFFE2_DB(<db_type>, <DB Class>)`.\n        name: str or None. Name of DBFileReader.\n            Optional name to prepend to blobs that will store the data.\n            Default to '<db_name>_<default_name_suffix>'.\n        batch_size: int.\n            How many examples are read for each time the read_net is run.\n        loop_over: bool.\n            If True given, will go through examples in random order endlessly.\n        field_names: List[str]. If the schema.field_names() should not in\n            alphabetic order, it must be specified.\n            Otherwise, schema will be automatically restored with\n            schema.field_names() sorted in alphabetic order.\n    "

    def __init__(self, db_path, db_type, name=None, batch_size=100, loop_over=False, field_names=None):
        if False:
            while True:
                i = 10
        assert db_path is not None, "db_path can't be None."
        assert db_type in C.registered_dbs(), 'db_type [{db_type}] is not available. \nChoose one of these: {registered_dbs}.'.format(db_type=db_type, registered_dbs=C.registered_dbs())
        self.db_path = os.path.expanduser(db_path)
        self.db_type = db_type
        self.name = name or '{db_name}_{default_name_suffix}'.format(db_name=self._extract_db_name_from_db_path(), default_name_suffix=self.default_name_suffix)
        self.batch_size = batch_size
        self.loop_over = loop_over
        super().__init__(self._init_reader_schema(field_names))
        self.ds = Dataset(self._schema, self.name + '_dataset')
        self.ds_reader = None

    def _init_name(self, name):
        if False:
            return 10
        return name or self._extract_db_name_from_db_path() + '_db_file_reader'

    def _init_reader_schema(self, field_names=None):
        if False:
            while True:
                i = 10
        'Restore a reader schema from the DB file.\n\n        If `field_names` given, restore scheme according to it.\n\n        Overwise, loade blobs from the DB file into the workspace,\n        and restore schema from these blob names.\n        It is also assumed that:\n        1). Each field of the schema have corresponding blobs\n            stored in the DB file.\n        2). Each blob loaded from the DB file corresponds to\n            a field of the schema.\n        3). field_names in the original schema are in alphabetic order,\n            since blob names loaded to the workspace from the DB file\n            will be in alphabetic order.\n\n        Load a set of blobs from a DB file. From names of these blobs,\n        restore the DB file schema using `from_column_list(...)`.\n\n        Returns:\n            schema: schema.Struct. Used in Reader.__init__(...).\n        '
        if field_names:
            return from_column_list(field_names)
        if self.db_type == 'log_file_db':
            assert os.path.exists(self.db_path), 'db_path [{db_path}] does not exist'.format(db_path=self.db_path)
        with core.NameScope(self.name):
            blob_prefix = scope.CurrentNameScope()
        workspace.RunOperatorOnce(core.CreateOperator('Load', [], [], absolute_path=True, db=self.db_path, db_type=self.db_type, load_all=True, add_prefix=blob_prefix))
        col_names = [blob_name[len(blob_prefix):] for blob_name in sorted(workspace.Blobs()) if blob_name.startswith(blob_prefix)]
        schema = from_column_list(col_names)
        return schema

    def setup_ex(self, init_net, finish_net):
        if False:
            for i in range(10):
                print('nop')
        'From the Dataset, create a _DatasetReader and setup a init_net.\n\n        Make sure the _init_field_blobs_as_empty(...) is only called once.\n\n        Because the underlying NewRecord(...) creats blobs by calling\n        NextScopedBlob(...), so that references to previously-initiated\n        empty blobs will be lost, causing accessibility issue.\n        '
        if self.ds_reader:
            self.ds_reader.setup_ex(init_net, finish_net)
        else:
            self._init_field_blobs_as_empty(init_net)
            self._feed_field_blobs_from_db_file(init_net)
            self.ds_reader = self.ds.random_reader(init_net, batch_size=self.batch_size, loop_over=self.loop_over)
            self.ds_reader.sort_and_shuffle(init_net)
            self.ds_reader.computeoffset(init_net)

    def read(self, read_net):
        if False:
            for i in range(10):
                print('nop')
        assert self.ds_reader, 'setup_ex must be called first'
        return self.ds_reader.read(read_net)

    def _init_field_blobs_as_empty(self, init_net):
        if False:
            return 10
        'Initialize dataset field blobs by creating an empty record'
        with core.NameScope(self.name):
            self.ds.init_empty(init_net)

    def _feed_field_blobs_from_db_file(self, net):
        if False:
            return 10
        'Load from the DB file at db_path and feed dataset field blobs'
        if self.db_type == 'log_file_db':
            assert os.path.exists(self.db_path), 'db_path [{db_path}] does not exist'.format(db_path=self.db_path)
        net.Load([], self.ds.get_blobs(), db=self.db_path, db_type=self.db_type, absolute_path=True, source_blob_names=self.ds.field_names())

    def _extract_db_name_from_db_path(self):
        if False:
            i = 10
            return i + 15
        'Extract DB name from DB path\n\n            E.g. given self.db_path=`/tmp/sample.db`, or\n            self.db_path = `dper_test_data/cached_reader/sample.db`\n            it returns `sample`.\n\n            Returns:\n                db_name: str.\n        '
        return os.path.basename(self.db_path).rsplit('.', 1)[0]