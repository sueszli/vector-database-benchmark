from sanic import Sanic, response
app = Sanic('Example')

@app.route('/')
def handle_request(request):
    if False:
        for i in range(10):
            print('nop')
    return response.redirect('/redirect')

@app.route('/redirect')
async def test(request):
    return response.json({'Redirected': True})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)