import http.server
import socketserver
import io
import xlsxwriter

class Handler(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if False:
            i = 10
            return i + 15
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        worksheet = workbook.add_worksheet()
        worksheet.write(0, 0, 'Hello, world!')
        workbook.close()
        output.seek(0)
        self.send_response(200)
        self.send_header('Content-Disposition', 'attachment; filename=test.xlsx')
        self.send_header('Content-type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        self.end_headers()
        self.wfile.write(output.read())
        return
print('Server listening on port 8000...')
httpd = socketserver.TCPServer(('', 8000), Handler)
httpd.serve_forever()