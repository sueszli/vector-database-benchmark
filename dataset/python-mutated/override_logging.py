import logging
from sanic import Sanic, text
logging_format = '[%(asctime)s] %(process)d-%(levelname)s '
logging_format += '%(module)s::%(funcName)s():l%(lineno)d: '
logging_format += '%(message)s'
logging.basicConfig(format=logging_format, level=logging.DEBUG)
log = logging.getLogger()
app = Sanic('app')

@app.route('/')
def test(request):
    if False:
        return 10
    log.info("received request; responding with 'hey'")
    return text('hey')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)