import requests
from starlette.requests import Request
from ray import serve

@serve.deployment
class EchoClass:

    def __init__(self, echo_str: str):
        if False:
            print('Hello World!')
        self.echo_str = echo_str

    def __call__(self, request: Request) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.echo_str
foo_node = EchoClass.bind('foo')
bar_node = EchoClass.bind('bar')
baz_node = EchoClass.bind('baz')
for (node, echo) in [(foo_node, 'foo'), (bar_node, 'bar'), (baz_node, 'baz')]:
    serve.run(node)
    assert requests.get('http://localhost:8000/').text == echo
import requests
response = requests.get('http://localhost:8000/')
echo = response.text
print(echo)
from ray import serve
from ray.serve.handle import DeploymentHandle

@serve.deployment
class LanguageClassifer:

    def __init__(self, spanish_responder: DeploymentHandle, french_responder: DeploymentHandle):
        if False:
            print('Hello World!')
        self.spanish_responder = spanish_responder
        self.french_responder = french_responder

    async def __call__(self, http_request):
        request = await http_request.json()
        (language, name) = (request['language'], request['name'])
        if language == 'spanish':
            response = self.spanish_responder.say_hello.remote(name)
        elif language == 'french':
            response = self.french_responder.say_hello.remote(name)
        else:
            return 'Please try again.'
        return await response

@serve.deployment
class SpanishResponder:

    def say_hello(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        return f'Hola {name}'

@serve.deployment
class FrenchResponder:

    def say_hello(self, name: str):
        if False:
            i = 10
            return i + 15
        return f'Bonjour {name}'
spanish_responder = SpanishResponder.bind()
french_responder = FrenchResponder.bind()
language_classifier = LanguageClassifer.bind(spanish_responder, french_responder)
serve.run(language_classifier)
import requests
response = requests.post('http://localhost:8000', json={'language': 'spanish', 'name': 'Dora'})
greeting = response.text
print(greeting)
assert greeting == 'Hola Dora'