"""

Tutorial: File upload and download

Uploads
-------

When a client uploads a file to a CherryPy application, it's placed
on disk immediately. CherryPy will pass it to your exposed method
as an argument (see "myFile" below); that arg will have a "file"
attribute, which is a handle to the temporary uploaded file.
If you wish to permanently save the file, you need to read()
from myFile.file and write() somewhere else.

Note the use of 'enctype="multipart/form-data"' and 'input type="file"'
in the HTML which the client uses to upload the file.


Downloads
---------

If you wish to send a file to the client, you have two options:
First, you can simply return a file-like object from your page handler.
CherryPy will read the file and serve it as the content (HTTP body)
of the response. However, that doesn't tell the client that
the response is a file to be saved, rather than displayed.
Use cherrypy.lib.static.serve_file for that; it takes four
arguments:

serve_file(path, content_type=None, disposition=None, name=None)

Set "name" to the filename that you expect clients to use when they save
your file. Note that the "name" argument is ignored if you don't also
provide a "disposition" (usually "attachement"). You can manually set
"content_type", but be aware that if you also use the encoding tool, it
may choke if the file extension is not recognized as belonging to a known
Content-Type. Setting the content_type to "application/x-download" works
in most cases, and should prompt the user with an Open/Save dialog in
popular browsers.

"""
import os
import os.path
import cherrypy
from cherrypy.lib import static
localDir = os.path.dirname(__file__)
absDir = os.path.join(os.getcwd(), localDir)

class FileDemo(object):

    @cherrypy.expose
    def index(self):
        if False:
            print('Hello World!')
        return '\n        <html><body>\n            <h2>Upload a file</h2>\n            <form action="upload" method="post" enctype="multipart/form-data">\n            filename: <input type="file" name="myFile" /><br />\n            <input type="submit" />\n            </form>\n            <h2>Download a file</h2>\n            <a href=\'download\'>This one</a>\n        </body></html>\n        '

    @cherrypy.expose
    def upload(self, myFile):
        if False:
            while True:
                i = 10
        out = '<html>\n        <body>\n            myFile length: %s<br />\n            myFile filename: %s<br />\n            myFile mime-type: %s\n        </body>\n        </html>'
        size = 0
        while True:
            data = myFile.file.read(8192)
            if not data:
                break
            size += len(data)
        return out % (size, myFile.filename, myFile.content_type)

    @cherrypy.expose
    def download(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(absDir, 'pdf_file.pdf')
        return static.serve_file(path, 'application/x-download', 'attachment', os.path.basename(path))
tutconf = os.path.join(os.path.dirname(__file__), 'tutorial.conf')
if __name__ == '__main__':
    cherrypy.quickstart(FileDemo(), config=tutconf)