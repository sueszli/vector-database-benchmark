import os
import time
import logging
import subprocess
import ray._private.services
import mxnet as mx
from mxnet import gluon
import copy
from bigdl.orca.ray.utils import to_list
from bigdl.dllib.utils.log4Error import *

class MXNetRunner(object):
    """Manages a MXNet model for training."""

    def setup_distributed(self, env, config, model_creator, loss_creator=None, validation_metrics_creator=None, eval_metrics_creator=None):
        if False:
            i = 10
            return i + 15
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger()
        self.config = config
        self.model_creator = model_creator
        self.loss_creator = loss_creator
        self.validation_metrics_creator = validation_metrics_creator
        self.eval_metrics_creator = eval_metrics_creator
        self.is_worker = False
        env['DMLC_NODE_HOST'] = self.get_node_ip()
        if env['DMLC_ROLE'] == 'worker':
            self.is_worker = True
        if self.is_worker:
            os.environ.update(env)
            self.kv = mx.kv.create('dist_sync')
            if 'seed' in self.config:
                mx.random.seed(self.config['seed'])
            self.model = self.model_creator(self.config)
            self.loss = self.loss_creator(self.config) if self.loss_creator else None
            self.eval_metrics = self.eval_metrics_creator(self.config) if self.eval_metrics_creator else None
            from mxnet.metric import CompositeEvalMetric
            if isinstance(self.eval_metrics, list):
                self.eval_metrics = CompositeEvalMetric(self.eval_metrics)
            self.val_metrics = self.validation_metrics_creator(self.config) if self.validation_metrics_creator else None
            if isinstance(self.val_metrics, list):
                self.val_metrics = CompositeEvalMetric(self.val_metrics)
            if not isinstance(self.model, mx.module.BaseModule):
                invalidInputError(self.loss, 'Loss not defined for gluon model, please specify loss_creator')
                self.trainer = gluon.Trainer(self.model.collect_params(), self.config['optimizer'], optimizer_params=self.config['optimizer_params'], kvstore=self.kv)
            else:
                self.trainer = None
        else:
            modified_env = os.environ.copy()
            modified_env.update(env)
            subprocess.Popen(['python', '-c', 'import mxnet'], shell=False, env=modified_env)

    def train(self, train_data, epochs=1, batch_size=32, validation_data=None, train_resize_batch_num=None):
        if False:
            for i in range(10):
                print('nop')
        'Train the model and update the model parameters.'
        stats = dict()
        if self.is_worker:
            config = copy.copy(self.config)
            if 'batch_size' not in config:
                config['batch_size'] = batch_size
            if train_resize_batch_num is not None:
                config['train_resize_batch_num'] = train_resize_batch_num
            train_data_iter = train_data(config, self.kv)
            val_data_iter = validation_data(config, self.kv) if validation_data else None
            start_time = time.time()
            if self.trainer:

                def cpu_context(target_data):
                    if False:
                        i = 10
                        return i + 15
                    if isinstance(target_data, list):
                        return [cpu_context(d) for d in target_data]
                    else:
                        return target_data.as_in_context(mx.cpu())
                for epoch in range(epochs):
                    if isinstance(train_data_iter, mx.io.DataIter):
                        train_data_iter.reset()
                    if self.eval_metrics:
                        self.eval_metrics.reset()
                    batch_start_time = time.time()
                    epoch_start_time = time.time()
                    for (i, batch) in enumerate(train_data_iter):
                        data = cpu_context(batch.data)
                        label = cpu_context(batch.label)
                        if not isinstance(data, list):
                            data = [data]
                        if not isinstance(label, list):
                            label = [label]
                        from mxnet import autograd as ag
                        with ag.record():
                            output = self.model(*data)
                            if not isinstance(output, list):
                                output = [output]
                            Ls = self.loss(*output, *label)
                            ag.backward(Ls)
                        self.trainer.step(batch_size)
                        if self.eval_metrics:
                            self.eval_metrics.update(label, output)
                        if not (i + 1) % self.config['log_interval']:
                            iteration_log = 'Epoch[%d] Batch[%d]  Speed: %f samples/sec  %s=%f' % (epoch, i, batch_size / (time.time() - batch_start_time), 'loss', Ls.asnumpy().mean())
                            if self.eval_metrics:
                                (names, accs) = self.eval_metrics.get()
                                (names, accs) = (to_list(names), to_list(accs))
                                for (name, acc) in zip(names, accs):
                                    iteration_log += '  %s=%f' % (name, acc)
                            self.logger.info(iteration_log)
                        batch_start_time = time.time()
                    self.logger.info('[Epoch %d] time cost: %f' % (epoch, time.time() - epoch_start_time))
                    if self.eval_metrics:
                        epoch_train_log = '[Epoch %d] training: ' % epoch
                        (names, accs) = self.eval_metrics.get()
                        (names, accs) = (to_list(names), to_list(accs))
                        for (name, acc) in zip(names, accs):
                            epoch_train_log += '%s=%f  ' % (name, acc)
                        self.logger.info(epoch_train_log)
                    if val_data_iter:
                        if isinstance(val_data_iter, mx.io.DataIter):
                            val_data_iter.reset()
                        self.val_metrics.reset()
                        for batch in val_data_iter:
                            data = cpu_context(batch.data)
                            label = cpu_context(batch.label)
                            if not isinstance(data, list):
                                data = [data]
                            if not isinstance(label, list):
                                label = [label]
                            output = self.model(*data)
                            if not isinstance(output, list):
                                output = [output]
                            self.val_metrics.update(label, output)
                        epoch_val_log = '[Epoch %d] validation: ' % epoch
                        (names, accs) = self.val_metrics.get()
                        (names, accs) = (to_list(names), to_list(accs))
                        for (name, acc) in zip(names, accs):
                            epoch_val_log += '%s=%f  ' % (name, acc)
                        self.logger.info(epoch_val_log)
                if self.eval_metrics:
                    (names, accs) = self.eval_metrics.get()
                    (names, accs) = (to_list(names), to_list(accs))
                    for (name, acc) in zip(names, accs):
                        stats[name] = acc
            else:
                if 'init' not in self.config:
                    from mxnet.initializer import Uniform
                    self.config['init'] = Uniform(0.01)
                if self.eval_metrics is None:
                    self.eval_metrics = 'acc'
                self.model.fit(train_data=train_data_iter, num_epoch=epochs, initializer=self.config['init'], kvstore=self.kv, optimizer=self.config['optimizer'], optimizer_params=self.config['optimizer_params'], eval_data=val_data_iter, eval_metric=self.eval_metrics, validation_metric=self.val_metrics, batch_end_callback=mx.callback.Speedometer(batch_size, self.config['log_interval']), epoch_end_callback=None if 'model' not in self.config else mx.callback.do_checkpoint(self.config['model']))
            epoch_time = time.time() - start_time
            stats['epoch_time'] = epoch_time
        return [stats]

    def shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        'Attempts to shut down the runner.'
        del self.logger
        if self.is_worker:
            del self.kv
            del self.model
            del self.trainer
            del self.loss
            del self.eval_metrics
            del self.val_metrics

    def get_node_ip(self):
        if False:
            i = 10
            return i + 15
        'Returns the IP address of the current node.'
        if 'node_ip' not in self.__dict__:
            self.node_ip = ray._private.services.get_node_ip_address()
        return self.node_ip

    def find_free_port(self):
        if False:
            while True:
                i = 10
        'Finds a free port on the current node.'
        if 'port' not in self.__dict__:
            from bigdl.orca.learn.mxnet.utils import find_free_port
            self.port = find_free_port()
        return self.port