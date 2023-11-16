import copy
import time
from buildbot.test.fake import httpclientservice as fakehttpclientservice
from buildbot.util.kubeclientservice import KubeError

class KubeClientService(fakehttpclientservice.HTTPClientService):

    def __init__(self, kube_config=None, *args, **kwargs):
        if False:
            print('Hello World!')
        c = kube_config.getConfig()
        super().__init__(c['master_url'], *args, **kwargs)
        self.namespace = c['namespace']
        self.addService(kube_config)
        self.pods = {}

    def createPod(self, namespace, spec):
        if False:
            while True:
                i = 10
        if 'metadata' not in spec:
            raise KubeError({'message': 'Pod "" is invalid: metadata.name: Required value: name or generateName is required'})
        name = spec['metadata']['name']
        pod = {'kind': 'Pod', 'metadata': copy.copy(spec['metadata']), 'spec': copy.deepcopy(spec['spec'])}
        self.pods[namespace + '/' + name] = pod
        return pod

    def deletePod(self, namespace, name, graceperiod=0):
        if False:
            i = 10
            return i + 15
        if namespace + '/' + name not in self.pods:
            raise KubeError({'message': 'Pod not found', 'reason': 'NotFound'})
        spec = self.pods[namespace + '/' + name]
        del self.pods[namespace + '/' + name]
        spec['metadata']['deletionTimestamp'] = time.ctime(time.time())
        return spec

    def waitForPodDeletion(self, namespace, name, timeout):
        if False:
            return 10
        if namespace + '/' + name in self.pods:
            raise TimeoutError(f'Did not see pod {name} terminate after {timeout}s')
        return {'kind': 'Status', 'reason': 'NotFound'}