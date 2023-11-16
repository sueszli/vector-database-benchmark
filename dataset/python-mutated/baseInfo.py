import json

class BaseInfo:

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return self.dump_json()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return self.dump_json()

    def dump_json(self, flag=True):
        if False:
            for i in range(10):
                print('nop')
        item = self._dump_json()
        if flag:
            return json.dumps(item)
        else:
            return item

    def _dump_json(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()