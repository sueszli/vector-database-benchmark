import subprocess
import responder
api = responder.API()

@api.route('/fortune')
class GreetingResource:

    def on_request(self, req, resp):
        if False:
            return 10
        resp.headers.update({'X-Life': '42'})

    def on_get(self, req, resp):
        if False:
            print('Hello World!')
        resp.headers.update({'X-ArtificialLife': '400'})
        fortune = subprocess.check_output(['fortune']).decode()
        resp.media = {'fortune': fortune}
r = api.requests.get('http://;/fortune')
print(r.text)
print(r.headers)