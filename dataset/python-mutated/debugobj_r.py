from idlelib import rpc

def remote_object_tree_item(item):
    if False:
        i = 10
        return i + 15
    wrapper = WrappedObjectTreeItem(item)
    oid = id(wrapper)
    rpc.objecttable[oid] = wrapper
    return oid

class WrappedObjectTreeItem:

    def __init__(self, item):
        if False:
            i = 10
            return i + 15
        self.__item = item

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        value = getattr(self.__item, name)
        return value

    def _GetSubList(self):
        if False:
            return 10
        sub_list = self.__item._GetSubList()
        return list(map(remote_object_tree_item, sub_list))

class StubObjectTreeItem:

    def __init__(self, sockio, oid):
        if False:
            i = 10
            return i + 15
        self.sockio = sockio
        self.oid = oid

    def __getattr__(self, name):
        if False:
            return 10
        value = rpc.MethodProxy(self.sockio, self.oid, name)
        return value

    def _GetSubList(self):
        if False:
            print('Hello World!')
        sub_list = self.sockio.remotecall(self.oid, '_GetSubList', (), {})
        return [StubObjectTreeItem(self.sockio, oid) for oid in sub_list]
if __name__ == '__main__':
    from unittest import main
    main('idlelib.idle_test.test_debugobj_r', verbosity=2)