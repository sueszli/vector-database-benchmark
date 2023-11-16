"""
Sample application that demonstrates how to use the App Engine Images API.

For more information, see README.md.
"""
from google.appengine.api import images
from google.appengine.ext import blobstore
import webapp2

class Thumbnailer(webapp2.RequestHandler):

    def get(self):
        if False:
            while True:
                i = 10
        blob_key = self.request.get('blob_key')
        if blob_key:
            blob_info = blobstore.get(blob_key)
            if blob_info:
                img = images.Image(blob_key=blob_key)
                img.resize(width=80, height=100)
                img.im_feeling_lucky()
                thumbnail = img.execute_transforms(output_encoding=images.JPEG)
                self.response.headers['Content-Type'] = 'image/jpeg'
                self.response.out.write(thumbnail)
                return
        self.error(404)

class ServingUrlRedirect(webapp2.RequestHandler):

    def get(self):
        if False:
            i = 10
            return i + 15
        blob_key = self.request.get('blob_key')
        if blob_key:
            blob_info = blobstore.get(blob_key)
            if blob_info:
                url = images.get_serving_url(blob_key, size=150, crop=True, secure_url=True)
                return webapp2.redirect(url)
        self.error(404)
app = webapp2.WSGIApplication([('/img', Thumbnailer), ('/redirect', ServingUrlRedirect)], debug=True)