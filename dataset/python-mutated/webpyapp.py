import web
urls = ('/', 'index')

class index:

    def GET(self):
        if False:
            print('Hello World!')
        return 'Hello, world!'
app = web.application(urls, globals()).wsgifunc()