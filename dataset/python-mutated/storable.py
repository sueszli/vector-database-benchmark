import json

class Storable:

    def get_state(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def set_state(self, state):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def save(self, path):
        if False:
            return 10
        state = self.get_state()
        state_json = json.dumps(state)
        with open(path, 'w') as f:
            f.write(state_json)
        return str(path)

    def load(self, path):
        if False:
            print('Hello World!')
        with open(path, 'r') as f:
            state_data = f.read()
        state = json.loads(state_data)
        self.set_state(state)