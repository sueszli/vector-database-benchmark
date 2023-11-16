import random
try:
    from urllib import urlencode
except ImportError:
    from urllib.parse import urlencode
from flask import request
from .app import app

@app.route('/bench')
def bench_test():
    if False:
        while True:
            i = 10
    total = int(request.args.get('total', 10000))
    show = int(request.args.get('show', 20))
    nlist = [random.randint(1, total) for _ in range(show)]
    result = []
    result.append('<html><head></head><body>')
    args = dict(request.args)
    for nl in nlist:
        args['n'] = nl
        argstr = urlencode(sorted(args.items()), doseq=True)
        result.append("<a href='/bench?{0}'>follow {1}</a><br>".format(argstr, nl))
    result.append('</body></html>')
    return ''.join(result)