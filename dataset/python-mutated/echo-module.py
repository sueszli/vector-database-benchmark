from http.server import BaseHTTPRequestHandler, HTTPServer
import json
PORT = 8888

class PostHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        if False:
            i = 10
            return i + 15
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        json_data = json.loads(post_data)
        new_json_data = process_data(json_data)
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(new_json_data).encode('utf8'))

def run(server_class=HTTPServer, handler_class=PostHandler, port=PORT):
    if False:
        while True:
            i = 10
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd on port {}'.format(port))
    httpd.serve_forever()

def process_data(data):
    if False:
        i = 10
        return i + 15
    return data
if __name__ == '__main__':
    from sys import argv
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()