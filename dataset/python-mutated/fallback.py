from gradio.components.base import Component

class Fallback(Component):

    def preprocess(self, payload):
        if False:
            while True:
                i = 10
        return payload

    def postprocess(self, value):
        if False:
            print('Hello World!')
        return value

    def example_inputs(self):
        if False:
            for i in range(10):
                print('nop')
        return {'foo': 'bar'}

    def api_info(self):
        if False:
            for i in range(10):
                print('nop')
        return {'type': {}, 'description': 'any valid json'}