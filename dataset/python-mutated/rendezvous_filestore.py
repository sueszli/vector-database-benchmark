from caffe2.python import core, workspace
from caffe2.python import dyndep
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:file_store_handler_ops')

def gen_rendezvous_ctx(self, model, dataset, is_train):
    if False:
        for i in range(10):
            print('nop')
    if self.opts['distributed']['num_shards'] < 2:
        return None
    workspace.RunOperatorOnce(core.CreateOperator('FileStoreHandlerCreate', [], ['store_handler'], path='/tmp', prefix='epoch.{}'.format(self.epoch)))
    rendezvous = dict(kv_handler='store_handler', shard_id=self.shard_id, num_shards=self.opts['distributed']['num_shards'], engine='GLOO', transport='tcp', interface=[], exit_nets=None) if is_train else None
    return rendezvous