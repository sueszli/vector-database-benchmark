class QueueKeys:
    """
    This class handles all keys parser logic.
    """

    def __init__(self, prefix: str='bull'):
        if False:
            return 10
        self.prefix = prefix

    def getKeys(self, name: str):
        if False:
            return 10
        names = ['', 'active', 'wait', 'waiting-children', 'paused', 'completed', 'failed', 'delayed', 'stalled', 'limiter', 'prioritized', 'id', 'stalled-check', 'meta', 'pc', 'events']
        keys = {}
        for name_type in names:
            keys[name_type] = self.toKey(name, name_type)
        return keys

    def toKey(self, name: str, name_type: str):
        if False:
            print('Hello World!')
        return f'{self.getQueueQualifiedName(name)}:{name_type}'

    def getQueueQualifiedName(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.prefix}:{name}'