from execjs import compile

def hash33_token(t):
    if False:
        print('Hello World!')
    (e, n) = (0, len(t))
    for i in range(0, n):
        e += (e << 5) + ord(t[i])
    return 2147483647 & e

def hash33_bkn(skey):
    if False:
        i = 10
        return i + 15
    e = skey
    t = 5381
    for n in range(0, len(e)):
        t += (t << 5) + ord(e[n])
    return 2147483647 & t

def get_js(js_name):
    if False:
        print('Hello World!')
    with open(js_name, 'r', encoding='UTF-8') as f:
        js_data = f.read()
        return js_data

def get_sck(skey):
    if False:
        for i in range(10):
            print('nop')
    md5 = get_js('decrypt/md5.js')
    ctx = compile(md5)
    result = ctx.call('hex_md5', str(skey))
    return str(result)

def get_csrf_token(skey):
    if False:
        for i in range(10):
            print('nop')
    js = get_js('decrypt/getCSRFToken.js')
    ctx = compile(js)
    tmp_data = ctx.call('getCSRFToken', str(skey))
    js = get_js('decrypt/md5.js')
    ctx = compile(js)
    result = ctx.call('hex_md5', str(tmp_data))
    return result