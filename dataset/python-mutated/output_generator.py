from caffe2.python import timeout_guard

def fun_conclude_operator(self):
    if False:
        for i in range(10):
            print('nop')
    timeout_guard.EuthanizeIfNecessary(600.0)

def assembleAllOutputs(self):
    if False:
        while True:
            i = 10
    output = {}
    output['train_model'] = self.train_model
    output['test_model'] = self.test_model
    output['model'] = self.model_output
    output['metrics'] = self.metrics_output
    return output