"""
@author:       Andrew Case
@license:      GNU General Public License 2.0
@contact:      atcuno@gmail.com
@organization: 
"""
import struct
from operator import attrgetter
import volatility.obj as obj
import volatility.debug as debug
import volatility.addrspace as addrspace
import volatility.plugins.linux.common as linux_common
import volatility.plugins.linux.pslist as linux_pslist
from volatility.renderers import TreeGrid
bash_hash_vtypes_32 = {'_pathdata': [8, {'path': [0, ['pointer', ['String', dict(length=1024)]]], 'flags': [4, ['int']]}], '_envdata': [8, {'name': [0, ['pointer', ['String', dict(length=1024)]]], 'value': [4, ['pointer', ['String', dict(length=1024)]]]}], 'bucket_contents': [20, {'next': [0, ['pointer', ['bucket_contents']]], 'key': [4, ['pointer', ['String', dict(length=1024)]]], 'data': [8, ['pointer', ['_pathdata']]], 'times_found': [16, ['int']]}], '_bash_hash_table': [12, {'bucket_array': [0, ['pointer', ['bucket_contents']]], 'nbuckets': [4, ['int']], 'nentries': [8, ['int']]}]}
bash_hash_vtypes_64 = {'_pathdata': [12, {'path': [0, ['pointer', ['String', dict(length=1024)]]], 'flags': [8, ['int']]}], '_envdata': [16, {'name': [0, ['pointer', ['String', dict(length=1024)]]], 'value': [8, ['pointer', ['String', dict(length=1024)]]]}], 'bucket_contents': [32, {'next': [0, ['pointer', ['bucket_contents']]], 'key': [8, ['pointer', ['String', dict(length=1024)]]], 'data': [16, ['pointer', ['_pathdata']]], 'times_found': [28, ['int']]}], '_bash_hash_table': [16, {'bucket_array': [0, ['pointer', ['bucket_contents']]], 'nbuckets': [8, ['int']], 'nentries': [12, ['int']]}]}

class _bash_hash_table(obj.CType):

    def is_valid(self):
        if False:
            return 10
        if not obj.CType.is_valid(self) or not self.bucket_array.is_valid() or (not self.nbuckets == 64) or (not self.nentries > 1):
            return False
        return True

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        if self.is_valid():
            seen = {}
            bucket_array = obj.Object(theType='Array', targetType='Pointer', offset=self.bucket_array, vm=self.nbuckets.obj_vm, count=64)
            for bucket_ptr in bucket_array:
                bucket = bucket_ptr.dereference_as('bucket_contents')
                while bucket.times_found > 0 and bucket.data.is_valid() and bucket.key.is_valid():
                    if bucket.v() in seen:
                        break
                    seen[bucket.v()] = 1
                    pdata = bucket.data
                    if pdata.path.is_valid() and 0 <= pdata.flags <= 2:
                        yield bucket
                    bucket = bucket.next

class BashHashTypes(obj.ProfileModification):
    conditions = {'os': lambda x: x in ['linux']}

    def modification(self, profile):
        if False:
            for i in range(10):
                print('nop')
        if profile.metadata.get('memory_model', '32bit') == '32bit':
            profile.vtypes.update(bash_hash_vtypes_32)
        else:
            profile.vtypes.update(bash_hash_vtypes_64)
        profile.object_classes.update({'_bash_hash_table': _bash_hash_table})

class linux_bash_hash(linux_pslist.linux_pslist):
    """Recover bash hash table from bash process memory"""

    def __init__(self, config, *args, **kwargs):
        if False:
            while True:
                i = 10
        linux_pslist.linux_pslist.__init__(self, config, *args, **kwargs)
        self._config.add_option('SCAN_ALL', short_option='A', default=False, help='scan all processes, not just those named bash', action='store_true')

    def calculate(self):
        if False:
            return 10
        linux_common.set_plugin_members(self)
        tasks = linux_pslist.linux_pslist(self._config).calculate()
        for task in tasks:
            proc_as = task.get_process_address_space()
            if not proc_as:
                continue
            if not (self._config.SCAN_ALL or str(task.comm) == 'bash'):
                continue
            for ent in task.bash_hash_entries():
                yield (task, ent)

    def unified_output(self, data):
        if False:
            for i in range(10):
                print('nop')
        return TreeGrid([('Pid', int), ('Name', str), ('Hits', int), ('Command', str), ('Path', str)], self.generator(data))

    def generator(self, data):
        if False:
            while True:
                i = 10
        for (task, bucket) in data:
            yield (0, [int(task.pid), str(task.comm), int(bucket.times_found), str(bucket.key.dereference()), str(bucket.data.path.dereference())])

    def render_text(self, outfd, data):
        if False:
            return 10
        self.table_header(outfd, [('Pid', '8'), ('Name', '20'), ('Hits', '6'), ('Command', '25'), ('Full Path', '')])
        for (task, bucket) in data:
            self.table_row(outfd, task.pid, task.comm, bucket.times_found, str(bucket.key.dereference()), str(bucket.data.path.dereference()))